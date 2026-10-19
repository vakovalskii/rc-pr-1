"""
Живая 3D-карта: берёт уменьшенные RGBD-пары прямо с Pi 4 (depth_cam.py --serve --color), считает
одометрию тем же трекером, что slam.py, и шлёт в релей новые точки и позу — /map3d.html рисует,
как карта наполняется. Чистовая карта потом — slam.py по полной записи.

    uv run --python 3.12 --with open3d --with aiohttp --with pillow python live_slam.py

Новая карта начинается при старте и при каждом нажатии «запись 3D» в пульте.
Сообщения в релей (/ws/map3d):
  binary  b"P" + uint32 n + int16[n,3] (мм, мир первого кадра: x вправо, y вниз, z вперёд) + uint8[n,3] цвет
  text    {"type":"pose","up":[3] или null (вверх в мире первого кадра, когда машинка разъехалась),"T":[16],"path":[[x,y,z],...последние],"kf":,"lost":,"jumps":,"pts":,"fps":}
          {"type":"reset"}
"""
import argparse, asyncio, io, json, time
import numpy as np
import open3d as o3d
import aiohttp
from PIL import Image
from slam import Tracker

def decode(b):
    L = int.from_bytes(b[4:8], "little"); meta = json.loads(b[8:8 + L]); o = 8 + L
    Lz = int.from_bytes(b[o:o + 4], "little"); z = np.asarray(Image.open(io.BytesIO(b[o + 4:o + 4 + Lz])), np.uint16)
    rgb = np.asarray(Image.open(io.BytesIO(b[o + 4 + Lz:])).convert("RGB"))
    return meta, z, rgb

class Live:
    def __init__(self, a):
        self.a, self.tr, self.cam_i, self.seen = a, None, None, set()
        self.reset()

    def reset(self):
        self.tr, self.seen, self.pts, self.path, self.n, self.t0, self.up = None, set(), 0, [], 0, time.time(), None

    def process(self, meta, z, rgb):
        """→ (кусок облака bytes или None, поза dict)."""
        a = self.a
        if self.tr is None:
            self.cam_i = o3d.camera.PinholeCameraIntrinsic(meta["w"], meta["h"], meta["fx"], meta["fy"], meta["cx"], meta["cy"])
            self.tr = Tracker(self.cam_i, a.depth_max, key_m=a.key_m, key_deg=a.key_deg)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(np.ascontiguousarray(z)),
            depth_scale=1.0 / meta["depth_scale"], depth_trunc=a.depth_max, convert_rgb_to_intensity=False)
        T, is_key = self.tr.step(rgbd); self.n += 1
        self.path.append(T[:3, 3].round(3).tolist())
        chunk = None
        if is_key:
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.cam_i, np.linalg.inv(T)).voxel_down_sample(a.voxel)
            xyz, col = np.asarray(pc.points), np.asarray(pc.colors)
            k = np.floor(xyz / a.voxel).astype(np.int64)            # шлём только воксели, которых ещё не было
            keys = (k[:, 0] + 50000) * 10 ** 10 + (k[:, 1] + 50000) * 10 ** 5 + (k[:, 2] + 50000)
            new = np.array([q not in self.seen for q in keys.tolist()], bool)
            self.seen.update(keys[new].tolist())
            if new.any():
                mm = np.clip(xyz[new] * 1000, -32767, 32767).astype("<i2"); c8 = (col[new] * 255).astype(np.uint8)
                chunk = b"P" + int(new.sum()).to_bytes(4, "little") + mm.tobytes() + c8.tobytes()
                self.pts += int(new.sum())
        tr = self.tr
        if self.n % 10 == 0 and len(self.path) > 20:          # «вверх» по плоскости траектории (как в slam.py)
            P = np.array(self.path); ev, evec = np.linalg.eigh(np.cov((P - P.mean(0)).T))
            if ev[2] > 0.3 ** 2 and ev[1] > 0.15 ** 2 and ev[0] < 0.03 ** 2:
                self.up = (evec[:, 0] * (1 if evec[1, 0] < 0 else -1)).round(4).tolist()
        pose = {"up": self.up, "type": "pose", "T": T.reshape(-1).round(4).tolist(), "path": self.path[-1:], "kf": int(is_key),
                "lost": tr.lost, "jumps": tr.jumps, "pts": self.pts, "frames": self.n,
                "fps": round(self.n / max(time.time() - self.t0, 1e-3), 1), "lag_ms": round((time.time() - meta["t"]) * 1000)}
        return chunk, pose

async def depth_url(s, relay):
    async with s.ws_connect(f"{relay}/ws/depthview") as ws:
        async for m in ws:
            if m.type == aiohttp.WSMsgType.TEXT and '"depthsrc"' in m.data: return json.loads(m.data)["url"]

async def main(a):
    live, loop = Live(a), asyncio.get_running_loop()
    async with aiohttp.ClientSession(trust_env=False) as s:
        while True:
            try:
                src = a.pi4 or await asyncio.wait_for(depth_url(s, a.relay), 10)
                async with s.ws_connect(src.replace("ws://", "http://"), max_msg_size=8 << 20) as cam, \
                           s.ws_connect(f"{a.relay}/ws/map3d", max_msg_size=32 << 20) as out:
                    await cam.send_str(json.dumps({"type": "raw"}))
                    print(f"камера {src} → релей {a.relay}/ws/map3d")
                    latest, ev, prev, first = {"b": None}, asyncio.Event(), None, True

                    async def rx():                                  # держим только свежую пару — не копим очередь
                        async for m in cam:
                            if m.type == aiohttp.WSMsgType.BINARY and m.data[:4] == b"RGBD": latest["b"] = m.data; ev.set()
                    rt = asyncio.create_task(rx())
                    try:
                        while not rt.done():
                            await ev.wait(); ev.clear(); b = latest["b"]
                            meta, z, rgb = decode(b)
                            cur = meta.get("rec")                            # папка записи на Pi 4 или None
                            if first or (cur and cur != prev):               # старт или нажали «запись 3D» — новая карта
                                live.reset(); first = False
                                await out.send_str(json.dumps({"type": "reset", "rec": cur})); print("новая карта", cur or "")
                            prev = cur
                            chunk, pose = await loop.run_in_executor(None, live.process, meta, z, rgb)
                            if chunk: await out.send_bytes(chunk)
                            await out.send_str(json.dumps(pose))
                            if live.n % 20 == 0:
                                print(f"кадров {live.n}, точек {live.pts}, потеряно {pose['lost']}, {pose['fps']} к/с, задержка {pose['lag_ms']} мс")
                    finally: rt.cancel()
            except Exception as e:
                print(f"нет связи ({e.__class__.__name__}: {e}), повтор через 2 с"); await asyncio.sleep(2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--relay", default="http://192.168.1.137:8080")
    p.add_argument("--pi4", help="адрес depth_cam, например ws://192.168.1.109:8090/ (по умолчанию спросить у релея)")
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--depth-max", type=float, default=3.0)
    p.add_argument("--key-m", type=float, default=0.08)
    p.add_argument("--key-deg", type=float, default=8)
    asyncio.run(main(p.parse_args()))
