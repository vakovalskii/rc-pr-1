"""
Автопилот-исследователь: машинка сама ездит по комнате, не врезаясь, строит карту пола и препятствий
(как робот-пылесос) и учится по заездам. Газ — тот, что выставлен на ESP (потолок 18%), здесь не меняется.

    uv run --python 3.12 --with open3d --with aiohttp --with pillow python explore.py explore 60
    uv run ... python explore.py lap 120          # круг по комнате вдоль стены справа
    uv run ... python explore.py dry 20           # всё считает и пишет, но газ 0 — проверка восприятия

Что внутри:
  * SLAM — тот же Live из live_slam.py (RGBD-пары с Pi 4): поза, облако, сетка проходимости; всё
    публикуется в релей — в пульте во вкладке «3D-карта» видно, как машинка едет и рисует карту.
  * Память препятствий: профиль глубины (16 секторов, 15 раз в секунду, depth_cam.py) по текущей позе
    кладётся на свою сетку. D455 ближе ~0.4 м не видит — подъезжать ближе можно только по памяти.
  * Планировщик: дуги руля от -1 до 1, по каждой — сколько можно проехать, не задев препятствие корпусом;
    в «explore» дуги ещё тянет в неразведанное, в «lap» — к стене справа на нужном расстоянии.
  * Тупик: разворот в три приёма — выбрать сторону, где свободнее (и держать её весь манёвр), задом с рулём
    от неё (нос уходит туда), потом вперёд полным рулём в неё; топчемся на месте — задом дальше.
  * Обучение: после заезда — упоры, накат после «стоп», площадь, замкнутость круга; по ним правятся
    дистанция остановки и упреждение, всё копится в data/explore/params.json и log-*.jsonl.

Безопасность: нет цифр глубины 0.5 с — стоп; скрипт упал — ESP сама в нейтраль через 300 мс;
тронул стик в пульте — релей отдаёт управление человеку. Руль на время заезда — «по датчику» (mode 2),
в конце возвращается ручной (mode 1).
"""
import argparse, asyncio, io, json, math, pathlib, sys, time
import numpy as np
import aiohttp
from live_slam import Live, decode

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "data" / "explore"
DEF = {"stop_m": 0.45, "coast_m": 0.12, "slow_m": 0.8, "half_w": 0.16, "wall_m": 0.45, "steer_max_deg": 25,
       "wheelbase": 0.26, "speed": 0.3, "reverse_s": 1.3}

class Memory:
    """Сетка препятствий 5 см из профиля глубины: лог-шансы, по лучу до препятствия — свободно."""
    def __init__(self, res=0.05, size_m=30):
        self.res, self.n = res, int(size_m / res); self.L = np.zeros((self.n, self.n), np.float32)
        self.visited = np.zeros((self.n, self.n), bool)

    def ij(self, x, y):
        return int(math.floor(x / self.res)) + self.n // 2, int(math.floor(y / self.res)) + self.n // 2

    def add(self, x, y, yaw, prof):
        for b, r in enumerate(prof):
            a = math.radians(-44 + (b + .5) * 5.5)             # + вправо
            if r is None: continue                             # пусто: может, слишком близко для глубины — память не трогаем
            hit = r < 2.5
            rr = r if hit else 2.0
            th = yaw - a
            for k in range(1, int(rr / self.res)):
                i, j = self.ij(x + k * self.res * math.cos(th), y + k * self.res * math.sin(th))
                if 0 <= i < self.n and 0 <= j < self.n: self.L[j, i] = max(-3, self.L[j, i] - 0.3)
            if hit:
                i, j = self.ij(x + rr * math.cos(th), y + rr * math.sin(th))
                if 0 <= i < self.n and 0 <= j < self.n: self.L[j, i] = min(4, self.L[j, i] + 1.0)

    def occ(self, x, y):
        i, j = self.ij(x, y)
        return 0 <= i < self.n and 0 <= j < self.n and self.L[j, i] > 0.5

    def mark_visited(self, x, y, r=0.15):
        i, j = self.ij(x, y); k = int(r / self.res)
        self.visited[max(j - k, 0):j + k + 1, max(i - k, 0):i + k + 1] = True

class Pilot:
    def __init__(self, a, P):
        self.a, self.P = a, P
        self.live = Live(argparse.Namespace(voxel=0.04, depth_max=2.5, key_m=0.08, key_deg=8, pitch=25, cam_height=0.10,
                                            z_min=0.04, z_max=0.35, car_radius=0.15, res=0.05, grid_m=30, max_points=1_500_000))
        self.mem = Memory()
        self.pose = None; self.pose_t = 0; self.pose_hist = []
        self.stats = None; self.stats_t = 0
        self.tele = {}; self.state = "start"; self.state_t = time.time(); self.steer = 0.0; self.thr = 0.0; self.coast_from = None
        self.log = []; self.events = {"bumps": 0, "stops": 0, "reverses": 0, "lost_stops": 0, "coast": []}
        self.start = None; self.path = 0.0

    # --- поза в выровненной системе (x вперёд первой камеры по полу, y влево) ---
    def pose2d(self, T):
        g = self.live.grid
        q = g.R @ T[:3, 3]; f = g.R @ (T[:3, :3] @ [0, 0, 1.0])
        return float(q[0]), float(q[1]), math.atan2(f[1], f[0])

    def arc(self, s, length, step=0.05):
        """Точки дуги для руля s (+ вправо) от текущей позы; велосипедная модель."""
        x, y, yaw = self.pose
        d = math.radians(self.P["steer_max_deg"]) * s
        k = -math.tan(d) / self.P["wheelbase"]                 # кривизна: вправо — yaw убывает
        pts = []
        for i in range(1, int(length / step) + 1):
            l = i * step
            if abs(k) < 1e-4: px, py, pyaw = x + l * math.cos(yaw), y + l * math.sin(yaw), yaw
            else:
                pyaw = yaw + k * l
                px = x + (math.sin(pyaw) - math.sin(yaw)) / k; py = y - (math.cos(pyaw) - math.cos(yaw)) / k
            pts.append((px, py, pyaw, l))
        return pts

    def clearance(self, s, length=1.2):
        """Сколько метров по дуге s свободно с учётом ширины корпуса (память + свежий профиль)."""
        hw = self.P["half_w"]
        for px, py, pyaw, l in self.arc(s, length):
            for off in (-hw, 0, hw):
                ox, oy = px - off * math.sin(pyaw), py + off * math.cos(pyaw)
                if self.mem.occ(ox, oy): return l
        return length

    def novelty(self, s):
        """Сколько непосещённых клеток рядом с концом дуги — тянет туда, где ещё не были."""
        px, py, _, _ = self.arc(s, 0.8)[-1]
        i, j = self.mem.ij(px, py); k = 6
        win = self.mem.visited[max(j - k, 0):j + k + 1, max(i - k, 0):i + k + 1]
        return 1 - win.mean() if win.size else 0

    def wall_right(self, s):
        """Расстояние до препятствия справа от конца дуги (луч вбок по памяти), для режима «круг»."""
        px, py, pyaw, _ = self.arc(s, 0.6)[-1]
        for k in range(1, 30):
            d = k * 0.05
            if self.mem.occ(px + d * math.sin(pyaw), py - d * math.cos(pyaw)): return d
        return 1.5

    def open_side(self):
        """Куда разворачиваться: −1 влево, +1 вправо — где по памяти дуга полным рулём длиннее и профиль дальше."""
        p = self.stats.get("prof") or [] if self.stats else []
        side_m = lambda bins: np.mean([r if r is not None else 0.3 for r in bins]) if bins else 0
        left = self.clearance(-1, 1.0) + 0.5 * side_m(p[:6]); right = self.clearance(1, 1.0) + 0.5 * side_m(p[10:])
        return -1 if left > right else 1

    def front_now(self):
        """Ближнее препятствие прямо по ходу по свежему профилю (центральные 8 секторов) и дыра внизу."""
        p = self.stats.get("prof") or []
        mid = [r for r in p[4:12] if r is not None]
        return (min(mid) if mid else 9.0), self.stats.get("hole_lo", 0)

    def decide(self):
        P = self.P
        if self.pose is None or self.stats is None: return 0.0, 0.0, "жду SLAM"
        if time.time() - self.stats_t > 0.5: return 0.0, 0.0, "нет глубины"
        if self.live.tr and self.live.tr.miss >= 3: self.events["lost_stops"] += 1; return 0.0, 0.0, "SLAM потерян"
        front, hole = self.front_now()
        cands = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        best, best_sc, best_cl = 0.0, -1e9, 0
        for s in cands:
            cl = self.clearance(s)
            sc = cl * 2.0 - abs(s - self.steer) * 0.3
            if self.a.mode == "explore": sc += self.novelty(s) * 1.2
            if self.a.mode == "lap": sc -= abs(min(self.wall_right(s), 1.2) - P["wall_m"]) * 2.5
            if sc > best_sc: best, best_sc, best_cl = s, sc, cl
        stop_at = P["stop_m"] + P["coast_m"] * 0.5
        straight = self.clearance(0, 0.8)
        # совсем близко по свежему профилю, большая «дыра» внизу (ближе 0.4 м глубина не меряет) или все дуги короткие
        if front < P["stop_m"] - 0.05 or (hole > 0.55 and front < 0.6) or best_cl < stop_at:
            return 0.0, best, "тупик"
        if best_cl < P["slow_m"] and abs(best) < 0.5 and straight < P["slow_m"]:
            return 0.6, best, "осторожно"
        return 1.0, best, "еду"

async def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / "params.json"; P = {**DEF, **(json.loads(pf.read_text()) if pf.exists() else {})}
    print("параметры:", P)
    pil = Pilot(a, P); loop = asyncio.get_running_loop()
    async with aiohttp.ClientSession(trust_env=False) as s:
        cam = await s.ws_connect(a.pi4, max_msg_size=8 << 20)
        pult = await s.ws_connect(f"{a.relay}/ws/pult")
        out = await s.ws_connect(f"{a.relay}/ws/map3d", max_msg_size=32 << 20)
        await cam.send_str(json.dumps({"type": "raw"}))
        await out.send_str(json.dumps({"type": "reset", "rec": None}))
        latest = {"b": None}; ev = asyncio.Event()

        async def rx_cam():
            async for m in cam:
                if m.type == aiohttp.WSMsgType.TEXT:
                    j = json.loads(m.data)
                    if j.get("type") == "depth" and "prof" in j:
                        pil.stats, pil.stats_t = j, time.time()
                        if pil.pose: pil.mem.add(*pil.pose, j["prof"])
                elif m.type == aiohttp.WSMsgType.BINARY and m.data[:4] == b"RGBD": latest["b"] = m.data; ev.set()

        async def rx_pult():
            async for m in pult:
                if m.type == aiohttp.WSMsgType.TEXT:
                    j = json.loads(m.data)
                    if j.get("type") == "tele": pil.tele = j

        async def slam():
            while True:
                await ev.wait(); ev.clear()
                meta, zpng, cjpg = decode(latest["b"])
                chunk, grid, pose = await loop.run_in_executor(None, pil.live.process, meta, zpng, cjpg)
                T = np.array(pose["T"]).reshape(4, 4); x, y, yaw = pil.pose2d(T)
                if pil.pose: pil.path += math.hypot(x - pil.pose[0], y - pil.pose[1])
                pil.pose, pil.pose_t = (x, y, yaw), time.time(); pil.pose_hist.append((time.time(), x, y, yaw))
                pil.mem.mark_visited(x, y)
                if pil.start is None: pil.start = (x, y, yaw)
                pose.update({"auto": pil.state, "mode": a.mode})
                if chunk: await out.send_bytes(chunk)
                if grid: await out.send_bytes(grid)
                await out.send_str(json.dumps(pose))

        tasks = [asyncio.create_task(c()) for c in (rx_cam, rx_pult, slam)]
        await pult.send_str(json.dumps({"type": "esp", "cmd": "mode,2"}))
        i, t0, last_move = 0, time.time(), time.time()
        man = None                    # манёвр: {"phase": "назад"|"разворот", "until": t, "side": ±1}
        escapes = []                  # (время, x, y) начала разворотов — чтобы заметить, что топчемся
        def start_escape(now, why):
            side = pil.open_side()
            near = [e for e in escapes if now - e[0] < 15 and pil.pose and math.hypot(e[1] - pil.pose[0], e[2] - pil.pose[1]) < 0.4]
            back = P["reverse_s"] * (1.6 if len(near) >= 2 else 1.0)  # топчемся на месте — сдаём дальше
            escapes.append((now, *(pil.pose[:2] if pil.pose else (0, 0))))
            pil.events["reverses"] += 1
            return {"phase": "назад", "until": now + back, "side": side, "why": why}
        try:
            while time.time() - t0 < a.seconds:
                now = time.time()
                if man and man["phase"] == "назад":              # задом с рулём от выбранной стороны: нос уходит в неё
                    thr, st, why = -1.0, -man["side"], "назад"
                    if now > man["until"]: man = {"phase": "разворот", "until": now + 1.5, "side": man["side"]}
                elif man and man["phase"] == "разворот":         # вперёд полным рулём в выбранную сторону
                    thr, st, why = 1.0, man["side"], "разворот"
                    if now > man["until"] or pil.clearance(man["side"], 0.8) < P["stop_m"] - 0.1: man = None
                    elif pil.front_now()[0] < P["stop_m"] - 0.1: man = start_escape(now, "разворот упёрся")
                else:
                    thr, st, why = pil.decide()
                    if why == "тупик":
                        if pil.state != "тупик": pil.events["stops"] += 1
                        elif now - pil.state_t > 0.4: man = start_escape(now, "тупик"); thr, st, why = -1.0, -man["side"], "назад"
                # упор: газ вперёд, а SLAM-поза за 1.5 с почти не сдвинулась
                if thr > 0 and len(pil.pose_hist) > 2:
                    old = [h for h in pil.pose_hist if now - h[0] > 1.5]
                    if old and math.hypot(pil.pose[0] - old[-1][1], pil.pose[1] - old[-1][2]) > 0.05: last_move = now
                    elif old and now - last_move > 1.5:
                        pil.events["bumps"] += 1; last_move = now
                        man = start_escape(now, "упор"); thr, st, why = -1.0, -man["side"], "упор → назад"
                else: last_move = now
                if why != pil.state:
                    if pil.state in ("еду", "осторожно") and why == "тупик" and pil.pose:
                        pil.coast_from = (pil.pose, now)
                    pil.state, pil.state_t = why, now
                if a.mode == "dry": thr = 0.0
                if getattr(pil, "coast_from", None) and now - pil.coast_from[1] > 1.0 and pil.pose:  # накат после «стоп»
                    (cx, cy, _), _ = pil.coast_from; pil.events["coast"].append(round(math.hypot(pil.pose[0] - cx, pil.pose[1] - cy), 3))
                    pil.coast_from = None
                await pult.send_str(json.dumps({"type": "cmd", "id": i, "steer": round(st, 2), "throttle": round(thr, 2)})); i += 1
                pil.steer, pil.thr = st, thr
                rec = {"t": round(now - t0, 2), "pose": [round(v, 3) for v in pil.pose] if pil.pose else None, "state": why,
                       "thr": thr, "steer": st, "front": pil.front_now()[0] if pil.stats else None, "pot": pil.tele.get("pot")}
                pil.log.append(rec)
                if i % 10 == 0: print(f"{rec['t']:5.1f}с {why:12s} руль {st:+.2f} газ {thr:+.1f} впереди {rec['front']} поза {rec['pose']}")
                # круг замкнулся: проехали > 4 м и вернулись к старту
                if a.mode == "lap" and pil.start and pil.path > 4 and math.hypot(pil.pose[0] - pil.start[0], pil.pose[1] - pil.start[1]) < 0.5:
                    print("КРУГ ЗАМКНУТ"); pil.events["lap_closed"] = True; break
                await asyncio.sleep(0.1)
        finally:
            for _ in range(10):
                await pult.send_str(json.dumps({"type": "cmd", "id": i, "steer": 0, "throttle": 0})); i += 1; await asyncio.sleep(0.05)
            await pult.send_str(json.dumps({"type": "esp", "cmd": "mode,1"}))
            await asyncio.sleep(0.3)
            for t in tasks: t.cancel()
            # --- разбор заезда и правка параметров ---
            g = pil.live.grid
            free = int((g.L < -0.5).sum()) * g.a.res ** 2; occ = int((g.L > 0.6).sum()) * g.a.res ** 2
            ev_ = pil.events; coast = ev_["coast"]
            rep = {"mode": a.mode, "seconds": round(time.time() - t0, 1), "path_m": round(pil.path, 2), "free_m2": round(free, 2),
                   "obst_m2": round(occ, 2), "bumps": ev_["bumps"], "stops": ev_["stops"], "reverses": ev_["reverses"],
                   "slam_lost_stops": ev_["lost_stops"], "coast_m": coast, "lap_closed": ev_.get("lap_closed", False),
                   "slam_lost_frames": pil.live.tr.lost if pil.live.tr else None, "params": P}
            newP = dict(P)
            if coast: newP["coast_m"] = round(0.5 * P["coast_m"] + 0.5 * float(np.median(coast)), 3)
            if a.mode != "dry":
                if ev_["bumps"]: newP["stop_m"] = round(min(P["stop_m"] + 0.05 * ev_["bumps"], 0.8), 2)
                elif ev_["stops"] >= 2: newP["stop_m"] = round(max(P["stop_m"] - 0.03, 0.30), 2)   # без упоров — смелее, до 30 см
            rep["new_params"] = newP
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (OUT / f"log-{stamp}-{a.mode}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pil.log))
            (OUT / f"report-{stamp}-{a.mode}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
            if a.mode != "dry": pf.write_text(json.dumps(newP, indent=1))
            print(json.dumps({k: v for k, v in rep.items() if k != "params"}, ensure_ascii=False, indent=1))
            print("сеанс SLAM:", pil.live.dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["explore", "lap", "dry"])
    p.add_argument("seconds", type=float)
    p.add_argument("--relay", default="http://192.168.1.137:8080")
    p.add_argument("--pi4", default="http://192.168.1.109:8090/")
    asyncio.run(main(p.parse_args()))
