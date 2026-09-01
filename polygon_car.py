"""
Машинка в полигоне: свободная езда по карте, собранной из твоей съёмки.

Позиция (x, y, курс) живёт на плоскости пола, столкновения — по сетке
занятости из map_build.py. Картинка — ближайший по позе кадр съёмки,
довёрнутый на разницу курса (окно кадрирования едет внутри кадра) и
подтянутый по масштабу на разницу расстояния. Это ещё не сплат: между
кадрами картинка перескакивает, зато пиксели настоящие и съезжать
с траектории съёмки можно.

    python3 polygon_car.py data/synth
"""
import argparse, asyncio, functools, io, json, math, pathlib, time
import numpy as np
import websockets
from PIL import Image

def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

class PolygonCar:
    def __init__(self, root, out, zoom, max_speed, wheelbase, max_steer_deg, radius, allow_unknown):
        root = pathlib.Path(root)
        meta = json.loads((root / "map" / "map.json").read_text())
        self.grid = np.asarray(Image.open(root / "map" / "map.png"))[::-1].copy()   # строка 0 = y0
        self.res, (self.x0, self.y0) = meta["res"], meta["origin"]
        self.H, self.W = self.grid.shape
        self.frames = (root / "map" / meta["frames_dir"]).resolve()
        self.kf = meta["keyframes"]
        self.kx = np.array([k["x"] for k in self.kf]); self.ky = np.array([k["y"] for k in self.kf])
        self.kyaw = np.array([k["yaw"] for k in self.kf])
        cam = meta["camera"]
        self.src_w, self.src_h = cam["width"], cam["height"]
        self.hfov = math.radians(cam["hfov_deg"])
        self.f_src = self.src_w / (2 * math.tan(self.hfov / 2))
        self.out, self.zoom0 = out, zoom
        self.max_speed, self.L, self.max_steer = max_speed, wheelbase, math.radians(max_steer_deg)
        self.radius, self.allow_unknown = radius, allow_unknown
        self.spawn = meta["spawn"]
        self.collisions, self.collision_until = 0, 0.0
        self.cur_kf, self.cur_cost = 0, 1e9
        self.reset()
        print(f"карта {self.W}x{self.H}, кадров-ключей {len(self.kf)}, исходник {self.src_w}x{self.src_h}, HFOV {cam['hfov_deg']}°")

    def reset(self, x=None, y=None, yaw=None):
        self.x = self.spawn["x"] if x is None else x
        self.y = self.spawn["y"] if y is None else y
        self.yaw = self.spawn["yaw"] if yaw is None else yaw
        self.speed = 0.0; self.steer = self.throttle = 0.0
        self.last_cmd_at, self.last_cmd_id, self.failsafe = 0.0, -1, True
        self.odometer = 0.0

    def apply(self, cmd):
        t = cmd.get("type", "cmd")
        if t == "reset":
            self.reset(cmd.get("x"), cmd.get("y"), cmd.get("yaw")); return
        if t != "cmd": return
        self.steer = max(-1.0, min(1.0, float(cmd.get("steer", 0))))
        self.throttle = max(-1.0, min(1.0, float(cmd.get("throttle", 0))))
        self.last_cmd_id = cmd.get("id", -1)
        self.last_cmd_at = time.monotonic()

    def cell(self, x, y):
        cx, cy = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
        if cx < 0 or cy < 0 or cx >= self.W or cy >= self.H: return 0
        return int(self.grid[cy, cx])

    def blocked(self, x, y):
        for k in range(8):
            a = k * math.pi / 4
            v = self.cell(x + self.radius * math.cos(a), y + self.radius * math.sin(a))
            if v == 0 or (v == 128 and not self.allow_unknown): return True
        return False

    def step(self, dt):
        if time.monotonic() - self.last_cmd_at > 0.5:
            self.throttle = self.steer = 0.0; self.failsafe = True
        else:
            self.failsafe = False
        self.speed += (self.throttle * self.max_speed - self.speed) * min(dt * 3.0, 1.0)
        if abs(self.speed) < 1e-4: return
        px, py, pyaw = self.x, self.y, self.yaw
        # руль +1 = вправо = по часовой; yaw против часовой, поэтому минус
        self.yaw -= self.speed / self.L * math.tan(self.steer * self.max_steer) * dt
        self.x += math.cos(self.yaw) * self.speed * dt
        self.y += math.sin(self.yaw) * self.speed * dt
        if self.blocked(self.x, self.y):                      # упёрлись — откат и стоп
            self.x, self.y, self.yaw = px, py, pyaw
            self.speed = 0.0
            if not self.collision: self.collisions += 1        # один упор = один удар
            self.collision_until = time.monotonic() + 0.4
        else:
            self.odometer += abs(self.speed) * dt

    @property
    def collision(self):
        return time.monotonic() < self.collision_until

    def pick_keyframe(self):
        # ближайший по позиции кадр, смотрящий примерно туда же (иначе увидим стену за спиной).
        # Курс в выборе НЕ участвует: разница курса должна остаться видимой как панорама,
        # иначе сеть не увидит, что машинка отвернула от маршрута
        cost = (self.kx - self.x) ** 2 + (self.ky - self.y) ** 2 + np.where(np.abs(angdiff(self.yaw, self.kyaw)) > 1.0, 100.0, 0.0)
        best = int(cost.argmin())
        if best != self.cur_kf and cost[best] < cost[self.cur_kf] * 0.7:   # гистерезис против мигания
            self.cur_kf = best
        return self.cur_kf

    @functools.lru_cache(maxsize=96)
    def load(self, name):
        return Image.open(self.frames / name).convert("RGB")

    def render(self, quality):
        i = self.pick_keyframe(); k = self.kf[i]
        img = self.load(k["name"])
        Wd, Hd = img.size
        dyaw = angdiff(self.yaw, k["yaw"])
        # боковой снос от кадра тоже виден: сдвиг вбок на типичной глубине сцены ~2 м
        # выглядит как поворот на lat/2 радиан (влево от кадра = сцена уходит вправо)
        lat = -(self.x - k["x"]) * math.sin(k["yaw"]) + (self.y - k["y"]) * math.cos(k["yaw"])
        dyaw += lat / 2.0
        lim = self.hfov / 2 * 0.9
        pan = -self.f_src * math.tan(max(-lim, min(lim, dyaw))) * (Wd / self.src_w)
        fwd = (self.x - k["x"]) * math.cos(k["yaw"]) + (self.y - k["y"]) * math.sin(k["yaw"])
        zoom = self.zoom0 * max(0.75, min(1.6, 1 + fwd * 0.5))
        cw = int(Wd / zoom); ch = min(int(cw * self.out[1] / self.out[0]), Hd)
        cx = Wd / 2 + pan
        left = int(max(0, min(Wd - cw, cx - cw / 2))); top = int((Hd - ch) / 2)
        crop = img.crop((left, top, left + cw, top + ch)).resize(self.out, Image.BILINEAR)
        buf = io.BytesIO(); crop.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

async def run(url, car, fps, quality):
    async with websockets.connect(url, max_size=8 << 20, proxy=None) as ws:
        print(f"подключились к {url}")
        async def rx():
            async for raw in ws:
                try: car.apply(json.loads(raw))
                except Exception as e: print("плохая команда:", e)
        asyncio.create_task(rx())
        period, last, seq = 1.0 / fps, time.monotonic(), 0
        while True:
            now = time.monotonic()
            car.step(now - last); last = now
            await ws.send(car.render(quality))
            await ws.send(json.dumps({
                "type": "tele", "seq": seq, "ack": car.last_cmd_id,
                "speed": round(car.speed, 2), "failsafe": car.failsafe,
                "x": round(car.x, 3), "y": round(car.y, 3), "yaw": round(car.yaw, 3),
                "kf": car.cur_kf, "collision": car.collision, "collisions": car.collisions,
                "odo": round(car.odometer, 2),
            }))
            seq += 1
            await asyncio.sleep(max(0.0, period - (time.monotonic() - now)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="папка съёмки с map/ и frames/")
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/car")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--quality", type=int, default=60)
    p.add_argument("--zoom", type=float, default=1.4, help="запас окна на поворот взгляда")
    p.add_argument("--max-speed", type=float, default=1.2, help="м/с на полном газу")
    p.add_argument("--wheelbase", type=float, default=0.2)
    p.add_argument("--max-steer", type=float, default=30.0, help="градусов")
    p.add_argument("--radius", type=float, default=0.12, help="радиус машинки для столкновений, м")
    p.add_argument("--allow-unknown", action="store_true", help="пускать в неразведанные клетки")
    a = p.parse_args()
    car = PolygonCar(a.dir, (a.width, a.height), a.zoom, a.max_speed, a.wheelbase, a.max_steer, a.radius, a.allow_unknown)
    asyncio.run(run(a.url, car, a.fps, a.quality))
