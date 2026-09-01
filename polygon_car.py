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
import argparse, asyncio, functools, io, json, math, pathlib, random, time
import numpy as np
import websockets
from PIL import Image, ImageDraw

def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

class PolygonCar:
    def __init__(self, root, out, zoom, max_speed, wheelbase, max_steer_deg, radius, allow_unknown, obstacles=0, seed=None):
        root = pathlib.Path(root)
        self.rng = random.Random(seed)                        # расстановка конусов воспроизводима по сиду
        meta = json.loads((root / "map" / "map.json").read_text())
        self.cam_h = meta.get("cam_height", 0.12)
        self.n_obst, self.obstacles = obstacles, []           # препятствия: {x,y,r,h,kind}
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
        self.t = 0.0                                          # своё время: шаг двигает его на dt
        self.reset()
        print(f"карта {self.W}x{self.H}, кадров-ключей {len(self.kf)}, исходник {self.src_w}x{self.src_h}, HFOV {cam['hfov_deg']}°")

    def reset(self, x=None, y=None, yaw=None):
        self.x = self.spawn["x"] if x is None else x
        self.y = self.spawn["y"] if y is None else y
        self.yaw = self.spawn["yaw"] if yaw is None else yaw
        self.speed = 0.0; self.steer = self.throttle = 0.0
        self.last_cmd_at, self.last_cmd_id, self.failsafe = -1.0, -1, True
        self.odometer = 0.0
        self.collisions, self.collision_until = 0, 0.0        # сброс = новый заезд: счётчики с нуля
        self.place_obstacles()

    def place_obstacles(self):
        """Случайные конусы и коробки вдоль маршрута, не у старта и не вплотную друг к другу."""
        self.obstacles = []
        if not self.n_obst: return
        rng = self.rng
        for _ in range(200):
            if len(self.obstacles) >= self.n_obst: break
            i = rng.randrange(12, len(self.kf) - 12)
            k = self.kf[i]
            lat = rng.uniform(-0.25, 0.25)
            x = k["x"] - lat * math.sin(k["yaw"]); y = k["y"] + lat * math.cos(k["yaw"])
            if math.hypot(x - self.spawn["x"], y - self.spawn["y"]) < 1.2: continue
            if any(math.hypot(x - o["x"], y - o["y"]) < 1.3 for o in self.obstacles): continue
            if self.cell(x, y) != 255: continue
            kind = rng.choice(["cone", "box"])
            self.obstacles.append({"x": round(x, 3), "y": round(y, 3), "r": 0.12 if kind == "cone" else 0.16,
                                   "h": 0.32 if kind == "cone" else 0.28, "kind": kind})

    def apply(self, cmd):
        t = cmd.get("type", "cmd")
        if t == "reset":
            self.reset(cmd.get("x"), cmd.get("y"), cmd.get("yaw")); return
        if t != "cmd": return
        self.steer = max(-1.0, min(1.0, float(cmd.get("steer", 0))))
        self.throttle = max(-1.0, min(1.0, float(cmd.get("throttle", 0))))
        self.last_cmd_id = cmd.get("id", -1)
        self.last_cmd_at = self.t

    def cell(self, x, y):
        cx, cy = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
        if cx < 0 or cy < 0 or cx >= self.W or cy >= self.H: return 0
        return int(self.grid[cy, cx])

    def blocked(self, x, y):
        for o in self.obstacles:
            if math.hypot(x - o["x"], y - o["y"]) < o["r"] + self.radius: return True
        for k in range(8):
            a = k * math.pi / 4
            v = self.cell(x + self.radius * math.cos(a), y + self.radius * math.sin(a))
            if v == 0 or (v == 128 and not self.allow_unknown): return True
        return False

    def step(self, dt):
        self.t += dt
        if self.t - self.last_cmd_at > 0.5:
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
        if self.blocked(self.x, self.y):                      # упёрлись
            nx, ny = self.x, self.y
            if not self.collision: self.collisions += 1        # один упор = один удар
            self.collision_until = self.t + 0.4
            if not self.blocked(nx, py):                       # скользим вдоль препятствия по x
                self.x, self.y = nx, py; self.speed *= 0.3
            elif not self.blocked(px, ny):                     # или по y
                self.x, self.y = px, ny; self.speed *= 0.3
            else:                                              # клин — откат и стоп
                self.x, self.y, self.yaw = px, py, pyaw
                self.speed = 0.0
        else:
            self.odometer += abs(self.speed) * dt

    @property
    def collision(self):
        return self.t < self.collision_until

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

    def view(self):
        """Кадр-ключ и окно кадрирования под текущую позу."""
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
        return img, (left, top, left + cw, top + ch), cw

    def render_image(self, size=None):
        """PIL-картинка того, что видит машинка, размером size (по умолчанию --width x --height)."""
        size = size or self.out
        img, box, cw = self.view()
        im = img.crop(box).resize(size, Image.BILINEAR)
        if self.obstacles:
            self.draw_obstacles(im, self.f_src * size[0] / cw)     # фокус окна: cw исходных px -> size[0] выходных
        return im

    def render(self, quality):
        buf = io.BytesIO(); self.render_image().save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def draw_obstacles(self, im, f):
        """Билборд по пинхол-геометрии: машинка и препятствие на одном полу, камера на cam_h."""
        W, H = im.size; cx, cy = W / 2, H / 2
        d = ImageDraw.Draw(im)
        seen = []
        for o in self.obstacles:
            dx, dy = o["x"] - self.x, o["y"] - self.y
            fwd = dx * math.cos(self.yaw) + dy * math.sin(self.yaw)
            lat = -dx * math.sin(self.yaw) + dy * math.cos(self.yaw)         # + = слева
            if fwd < 0.2 or abs(lat) > fwd * 1.2: continue                    # сзади или вне поля зрения
            seen.append((fwd, lat, o))
        for fwd, lat, o in sorted(seen, reverse=True):                        # дальние первыми
            u = cx - f * lat / fwd
            v_base = cy + f * self.cam_h / fwd
            v_top = cy - f * (o["h"] - self.cam_h) / fwd
            half = f * o["r"] / fwd
            shade = max(0.55, min(1.0, 1.6 / (fwd + 1)))
            if o["kind"] == "cone":
                col = tuple(int(c * shade) for c in (255, 120, 20))
                d.polygon([(u, v_top), (u - half, v_base), (u + half, v_base)], fill=col, outline=(40, 20, 0))
                for t in (0.35, 0.65):                                       # белые полосы
                    yv = v_top + (v_base - v_top) * t; hw = half * t
                    d.line([(u - hw, yv), (u + hw, yv)], fill=(245, 245, 245), width=max(1, int(half * 0.18)))
            else:
                col = tuple(int(c * shade) for c in (60, 80, 170))
                d.rectangle([u - half, v_top, u + half, v_base], fill=col, outline=(20, 25, 60))
                d.line([(u - half, v_top), (u + half, v_base)], fill=(200, 210, 240), width=max(1, int(half * 0.12)))
                d.line([(u + half, v_top), (u - half, v_base)], fill=(200, 210, 240), width=max(1, int(half * 0.12)))

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
                "obstacles": [[o["x"], o["y"], o["r"]] for o in car.obstacles],
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
    p.add_argument("--obstacles", type=int, default=0, help="сколько случайных конусов/коробок ставить на маршрут (новые при каждом сбросе)")
    a = p.parse_args()
    car = PolygonCar(a.dir, (a.width, a.height), a.zoom, a.max_speed, a.wheelbase, a.max_steer, a.radius, a.allow_unknown, a.obstacles)
    asyncio.run(run(a.url, car, a.fps, a.quality))
