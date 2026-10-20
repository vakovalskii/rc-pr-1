"""
Живая 3D-карта и карта проходимости: берёт уменьшенные RGBD-пары прямо с Pi 4 (depth_cam.py --serve
--color), считает одометрию тем же трекером, что slam.py, и шлёт в релей новые точки, позу и сетку
«где можно ехать» — пульт (окно «3D-карта») и /map3d.html рисуют, как карта наполняется.

    uv run --python 3.12 --with open3d --with aiohttp --with pillow python live_slam.py

Новая карта — при старте и при каждом нажатии «запись 3D» в пульте; обрыв связи карту не сбрасывает.
Каждый сеанс целиком пишется в data/live-<время>/: raw/ (пары 320x240 — можно пересобрать slam.py)
и map/ (map.png + map.json в формате map_build.py). Релей кладёт последнюю сетку в свою папку карты.

Проходимость — сетка занятости 5 см с лог-шансами: точки на высоте корпуса (4–35 см над полом) —
препятствие, клетки по лучу от камеры до них — свободно, проеханный путь — свободно точно.
«Вверх»: пока машинка не разъехалась — по наклону камеры (--pitch), потом — по плоскости траектории.

Сообщения в релей (/ws/map3d):
  binary  b"P" + uint32 n + int16[n,3] (мм, мир первого кадра: x вправо, y вниз, z вперёд) + uint8[n,3] цвет
  binary  b"G" + uint32 len + json {res, origin, width, height, R (строки: e1, e2, вверх), floor} + PNG сетки
  text    {"type":"pose","up":[3],"T":[16],"kf":,"lost":,"jumps":,"pts":,"frames":,"fps":,"lag_ms":}
          {"type":"reset","rec":папка записи или null}
"""
import argparse, asyncio, io, json, math, pathlib, time
import numpy as np
import open3d as o3d
import aiohttp
from PIL import Image
from slam import Tracker

ROOT = pathlib.Path(__file__).parent

def decode(b):
    L = int.from_bytes(b[4:8], "little"); meta = json.loads(b[8:8 + L]); o = 8 + L
    Lz = int.from_bytes(b[o:o + 4], "little")
    return meta, b[o + 4:o + 4 + Lz], b[o + 4 + Lz:]

class Grid:
    """Сетка занятости в выровненной системе: e1 — вперёд первой камеры по полу, e2 — влево, вверх — нормаль к полу."""
    HIT, MISS, LIM = 0.9, 0.6, 4.0

    def __init__(self, a):
        self.a, self.n = a, int(a.grid_m / a.res)
        self.L = np.zeros((self.n, self.n), np.float32); self.kfs = []; self.R = None; self.dirty = False

    def set_up(self, up):
        """Новое «вверх» → новые оси; если сильно повернулось — пересобрать сетку по всем ключевым кадрам."""
        up = np.asarray(up, float); up /= np.linalg.norm(up)
        if self.R is not None and math.degrees(math.acos(min(1, abs(float(up @ self.R[2]))))) < 2: return
        f = np.array([0, 0, 1.0]); e1 = f - (f @ up) * up; e1 /= np.linalg.norm(e1)
        self.R = np.stack([e1, np.cross(up, e1), up])
        self.L[:] = 0
        for T, pts in self.kfs: self._integrate(T, pts)
        self.dirty = True

    def cell(self, q):
        return np.floor(q[..., :2] / self.a.res).astype(int) + self.n // 2

    def add(self, T, pts):
        if len(pts) > 2500: pts = pts[np.random.default_rng(0).choice(len(pts), 2500, replace=False)]
        self.kfs.append((T.copy(), pts)); self._integrate(T, pts); self.dirty = True

    def _integrate(self, T, pts):
        a, R = self.a, self.R
        q, qc = pts @ R.T, R @ T[:3, 3]
        h = q[:, 2] + a.cam_height                       # высота над полом: камера первого кадра на cam_height
        band = (h > a.z_min) & (h < a.z_max)
        free = set()
        for p in q[band][:: max(1, band.sum() // 800)]:  # лучи от камеры до препятствий: по пути свободно
            d = np.linalg.norm(p[:2] - qc[:2]); k = max(int(d / a.res), 1)
            t = np.arange(k)[:, None] / k
            free.update(map(tuple, self.cell(qc[None, :2] + t * (p[None, :2] - qc[None, :2]))[:-1].tolist()))
        occ = set(map(tuple, self.cell(q[band]).tolist()))
        ok = lambda c: 0 <= c[0] < self.n and 0 <= c[1] < self.n
        for c in free - occ:
            if ok(c): self.L[c[1], c[0]] = max(-self.LIM, self.L[c[1], c[0]] - self.MISS)
        for c in occ:
            if ok(c): self.L[c[1], c[0]] = min(self.LIM, self.L[c[1], c[0]] + self.HIT)
        cx, cy = self.cell(qc[None])[0]; r = int(math.ceil(a.car_radius / a.res))   # где проехали — свободно точно
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; m = xx ** 2 + yy ** 2 <= r * r
        ys, xs = np.clip(cy + yy[m], 0, self.n - 1), np.clip(cx + xx[m], 0, self.n - 1)
        self.L[ys, xs] = np.minimum(self.L[ys, xs], -2.0)

    def render(self, kf_poses):
        """→ (meta в формате map_build.py, PNG): 0 — препятствие, 255 — свободно, 128 — неизвестно, север сверху."""
        a, L = self.a, self.L
        known = np.argwhere(np.abs(L) > 0.3)
        if not len(known): return None, None
        (y0, x0), (y1, x1) = known.min(0) - 6, known.max(0) + 7
        y0, x0 = max(y0, 0), max(x0, 0); y1, x1 = min(y1, self.n), min(x1, self.n)
        g = np.full(L.shape, 128, np.uint8); g[L < -0.5] = 255; g[L > 0.6] = 0
        g = g[y0:y1, x0:x1][::-1]
        buf = io.BytesIO(); Image.fromarray(g).save(buf, "PNG")
        org = [(x0 - self.n // 2) * a.res, (y0 - self.n // 2) * a.res]
        kfs = []
        for i, T in enumerate(kf_poses):
            q = self.R @ T[:3, 3]; f = self.R @ (T[:3, :3] @ [0, 0, 1.0])
            kfs.append({"i": i, "name": "", "x": round(float(q[0]), 3), "y": round(float(q[1]), 3), "yaw": round(math.atan2(f[1], f[0]), 3)})
        meta = {"res": a.res, "origin": org, "width": int(x1 - x0), "height": int(y1 - y0), "cam_height": a.cam_height,
                "spawn": kfs[0] if kfs else {"x": 0, "y": 0, "yaw": 0}, "keyframes": kfs[-400:], "source": "live_slam",
                "R": self.R.round(5).tolist(), "floor": -a.cam_height}
        return meta, buf.getvalue()

class Live:
    def __init__(self, a):
        self.a = a; self.reset(None)

    def reset(self, rec):
        self.tr, self.cam_i, self.seen, self.pts, self.path, self.n, self.t0 = None, None, set(), 0, [], 0, time.time()
        self.traj_up = None; self.grid = Grid(self.a); self.grid_at = 0; self.times = []
        self.dir = ROOT / "data" / f"live-{time.strftime('%Y%m%d-%H%M%S')}{'-' + rec if rec else ''}"
        (self.dir / "raw" / "depth").mkdir(parents=True, exist_ok=True); (self.dir / "raw" / "color").mkdir(exist_ok=True)
        print("сеанс:", self.dir)

    def rate(self):
        """к/с за последние ~20 кадров (время от запуска сюда не годится: машинка могла быть выключена)."""
        self.times.append(time.time()); self.times = self.times[-20:]
        return round((len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-3), 1) if len(self.times) > 1 else 0.0

    def up(self):
        if self.traj_up is not None: return self.traj_up
        th = math.radians(self.a.pitch)                   # камера смотрит вверх на pitch: «вверх» в её системе
        return np.array([0, -math.cos(th), math.sin(th)])

    def process(self, meta, zpng, cjpg):
        """→ (кусок облака bytes или None, сетка bytes или None, поза dict)."""
        a = self.a
        z = np.asarray(Image.open(io.BytesIO(zpng)), np.uint16); rgb = np.asarray(Image.open(io.BytesIO(cjpg)).convert("RGB"))
        if self.tr is None:
            self.cam_i = o3d.camera.PinholeCameraIntrinsic(meta["w"], meta["h"], meta["fx"], meta["fy"], meta["cx"], meta["cy"])
            self.tr = Tracker(self.cam_i, a.depth_max, key_m=a.key_m, key_deg=a.key_deg)
            (self.dir / "raw" / "intrinsics.json").write_text(json.dumps({"width": meta["w"], "height": meta["h"], "fx": meta["fx"],
                "fy": meta["fy"], "cx": meta["cx"], "cy": meta["cy"], "depth_scale": meta["depth_scale"]}))
            self.grid.set_up(self.up())
        (self.dir / "raw" / "depth" / f"{self.n:06d}.png").write_bytes(zpng)
        (self.dir / "raw" / "color" / f"{self.n:06d}.jpg").write_bytes(cjpg)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(np.ascontiguousarray(z)),
            depth_scale=1.0 / meta["depth_scale"], depth_trunc=a.depth_max, convert_rgb_to_intensity=False)
        T, is_key = self.tr.step(rgbd); self.n += 1
        self.path.append(T[:3, 3].copy())
        chunk = None
        if is_key:
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.cam_i, np.linalg.inv(T)).voxel_down_sample(a.voxel)
            xyz, col = np.asarray(pc.points), np.asarray(pc.colors)
            self.grid.add(T, xyz)
            k = np.floor(xyz / a.voxel).astype(np.int64)            # шлём только воксели, которых ещё не было
            keys = (k[:, 0] + 50000) * 10 ** 10 + (k[:, 1] + 50000) * 10 ** 5 + (k[:, 2] + 50000)
            new = np.array([q not in self.seen for q in keys.tolist()], bool)
            self.seen.update(keys[new].tolist())
            if new.any():
                mm = np.clip(xyz[new] * 1000, -32767, 32767).astype("<i2"); c8 = (col[new] * 255).astype(np.uint8)
                chunk = b"P" + int(new.sum()).to_bytes(4, "little") + mm.tobytes() + c8.tobytes()
                self.pts += int(new.sum())
        if self.n % 10 == 0 and len(self.path) > 20:          # «вверх» по плоскости траектории (как в slam.py)
            P = np.array(self.path); ev, evec = np.linalg.eigh(np.cov((P - P.mean(0)).T))
            if ev[2] > 0.3 ** 2 and ev[1] > 0.15 ** 2 and ev[0] < 0.03 ** 2:
                self.traj_up = evec[:, 0] * (1 if evec[1, 0] < 0 else -1); self.grid.set_up(self.traj_up)
        grid = None
        if self.grid.dirty and time.time() - self.grid_at > 1.0:
            gm, png = self.grid.render(self.tr.poses[::5])
            if gm:
                (self.dir / "map").mkdir(exist_ok=True)
                (self.dir / "map" / "map.png").write_bytes(png); (self.dir / "map" / "map.json").write_text(json.dumps(gm))
                js = json.dumps(gm).encode(); grid = b"G" + len(js).to_bytes(4, "little") + js + png
            self.grid.dirty, self.grid_at = False, time.time()
        tr = self.tr
        pose = {"type": "pose", "up": self.up().round(4).tolist(), "T": T.reshape(-1).round(4).tolist(), "kf": int(is_key),
                "lost": tr.lost, "jumps": tr.jumps, "pts": self.pts, "frames": self.n,
                "fps": self.rate(), "lag_ms": round((time.time() - meta["t"]) * 1000)}
        return chunk, grid, pose

async def depth_url(s, relay):
    async with s.ws_connect(f"{relay}/ws/depthview") as ws:
        async for m in ws:
            if m.type == aiohttp.WSMsgType.TEXT and '"depthsrc"' in m.data: return json.loads(m.data)["url"]

async def main(a):
    live, loop = Live(a), asyncio.get_running_loop()
    prev, announced = None, False
    async with aiohttp.ClientSession(trust_env=False) as s:
        while True:
            try:
                src = a.pi4 or await asyncio.wait_for(depth_url(s, a.relay), 10)
                async with s.ws_connect(src.replace("ws://", "http://"), max_msg_size=8 << 20) as cam, \
                           s.ws_connect(f"{a.relay}/ws/map3d", max_msg_size=32 << 20) as out:
                    await cam.send_str(json.dumps({"type": "raw"}))
                    print(f"камера {src} → релей {a.relay}/ws/map3d")
                    if not announced:                                # первая связь в этом запуске — чистый лист у зрителей
                        await out.send_str(json.dumps({"type": "reset", "rec": None})); announced = True
                    latest, ev = {"b": None}, asyncio.Event()

                    async def rx():                                  # держим только свежую пару — не копим очередь
                        async for m in cam:
                            if m.type == aiohttp.WSMsgType.BINARY and m.data[:4] == b"RGBD": latest["b"] = m.data; ev.set()
                    rt = asyncio.create_task(rx())
                    try:
                        while not rt.done():
                            try: await asyncio.wait_for(ev.wait(), 5)
                            except asyncio.TimeoutError: raise ConnectionError("5 с без кадров с Pi 4")
                            ev.clear(); meta, zpng, cjpg = decode(latest["b"])
                            cur = meta.get("rec")                            # папка записи на Pi 4 или None
                            if cur and cur != prev:                          # нажали «запись 3D» — новая карта
                                live.reset(cur); await out.send_str(json.dumps({"type": "reset", "rec": cur})); print("новая карта", cur)
                            prev = cur
                            chunk, grid, pose = await loop.run_in_executor(None, live.process, meta, zpng, cjpg)
                            if chunk: await out.send_bytes(chunk)
                            if grid: await out.send_bytes(grid)
                            await out.send_str(json.dumps(pose))
                            if live.n % 20 == 0:
                                print(f"кадров {live.n}, точек {live.pts}, потеряно {pose['lost']}, {pose['fps']} к/с, задержка {pose['lag_ms']} мс")
                    finally: rt.cancel()
            except Exception as e:
                print(f"нет связи ({e.__class__.__name__}: {e}), карта сохраняется, повтор через 2 с"); await asyncio.sleep(2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--relay", default="http://192.168.1.137:8080")
    p.add_argument("--pi4", help="адрес depth_cam, например ws://192.168.1.109:8090/ (по умолчанию спросить у релея)")
    p.add_argument("--voxel", type=float, default=0.04, help="м, шаг облака для показа")
    p.add_argument("--depth-max", type=float, default=2.5, help="м, дальше глубина D455 шумит и мажет карту")
    p.add_argument("--key-m", type=float, default=0.08)
    p.add_argument("--key-deg", type=float, default=8)
    p.add_argument("--pitch", type=float, default=25, help="°, на сколько камера смотрит вверх (пока нет траектории)")
    p.add_argument("--cam-height", type=float, default=0.10, help="м, высота RealSense над полом")
    p.add_argument("--z-min", type=float, default=0.04, help="м, ниже — пол")
    p.add_argument("--z-max", type=float, default=0.35, help="м, выше — не мешает проехать")
    p.add_argument("--car-radius", type=float, default=0.15, help="м, полоса вокруг пути, которая точно свободна")
    p.add_argument("--res", type=float, default=0.05, help="м, клетка сетки проходимости")
    p.add_argument("--grid-m", type=float, default=30, help="м, сторона сетки")
    asyncio.run(main(p.parse_args()))
