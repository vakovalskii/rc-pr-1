"""
Эксперт-водитель: ведёт машинку по маршруту съёмки и объезжает конусы.
Один и тот же класс крутится и в реальном времени через релей (auto_drive.py),
и в среде без сна (env.py, gen_data.py) — поэтому время он получает снаружи.

Управление — pure pursuit: цель на lookahead метров впереди по маршруту.
Объезд — сдвиг цели вбок на ширину конуса + машинки + запас, в ту сторону,
где по карте свободно. Возмущения (--noise) дёргают исполняемый руль,
а меткой остаётся чистая команда: в датасете появляются выезды с траектории
и правильная реакция на них.
"""
import json, math, pathlib, random
import numpy as np
from PIL import Image

def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

class Expert:
    def __init__(self, data_dir, lookahead=0.35, gain=1.6, throttle=0.6, noise=0.0, seed=None):
        d = pathlib.Path(data_dir) / "map"
        meta = json.loads((d / "map.json").read_text())
        self.W = np.array([[k["x"], k["y"]] for k in meta["keyframes"]])
        self.grid = np.asarray(Image.open(d / "map.png"))[::-1]
        self.res, (self.x0, self.y0) = meta["res"], meta["origin"]
        self.lookahead, self.gain, self.throttle, self.noise = lookahead, gain, throttle, noise
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.target, self.laps = 1, 0
        self.stuck_until, self.stuck_steer, self.stuck_phase = None, 0.0, 1
        self.noise_until, self.noise_val = -10.0, 0.0
        self.last_t = 0.0
        self.recovering = False

    def free(self, x, y, r=0.14):
        for k in range(8):
            cx = int((x + r * math.cos(k * 0.785) - self.x0) / self.res)
            cy = int((y + r * math.sin(k * 0.785) - self.y0) / self.res)
            if not (0 <= cx < self.grid.shape[1] and 0 <= cy < self.grid.shape[0]) or self.grid[cy, cx] != 255:
                return False
        return True

    def act(self, x, y, yaw, obstacles, collision, t):
        """-> (руль исполняемый, газ, руль-метка, круг завершён?)"""
        W, pos = self.W, np.array([x, y])
        d = np.linalg.norm(W - pos, axis=1)
        near = int(d.argmin())
        if self.target <= near <= self.target + 20:                 # петля замкнута: не прыгать в конец
            self.target = near
        j = self.target
        # у конуса смотрим дальше вперёд: тогда сдвиг цели вбок даёт плавную дугу, а не руль в упор
        ahead = self.lookahead
        for ox, oy, orad in obstacles:
            along = (ox - x) * math.cos(yaw) + (oy - y) * math.sin(yaw)
            lat = -(ox - x) * math.sin(yaw) + (oy - y) * math.cos(yaw)
            if -0.3 < along < 1.6 and abs(lat) < 0.8: ahead = max(ahead, 0.7)
        while j + 1 < len(W) and np.linalg.norm(W[j] - pos) < ahead:
            j += 1
        lap_done = False
        if j >= len(W) - 1 and np.linalg.norm(W[-1] - pos) < self.lookahead:
            self.laps += 1; self.target = 1; j = 1; lap_done = True
        self.target = j
        goal = W[j]
        err = angdiff(math.atan2(goal[1] - pos[1], goal[0] - pos[0]), yaw)
        near_obst = 0.0
        for ox, oy, orad in obstacles:
            m = min(range(max(1, j - 12), min(len(W) - 1, j + 14)), key=lambda q: np.hypot(W[q][0] - ox, W[q][1] - oy))
            path_yaw = math.atan2(W[m + 1][1] - W[m - 1][1], W[m + 1][0] - W[m - 1][0])
            lat_o = -(ox - W[m][0]) * math.sin(path_yaw) + (oy - W[m][1]) * math.cos(path_yaw)   # конус слева от маршрута: +
            along = (ox - x) * math.cos(yaw) + (oy - y) * math.sin(yaw)
            if not (-0.3 < along < 1.6) or abs(lat_o) > orad + 0.35:
                continue
            need = orad + 0.12 + 0.25                                  # конус + машинка + запас
            side = -1 if lat_o >= 0 else 1                             # конус слева -> уходим вправо
            ramp = min(1.0, max(0.0, (1.4 - along) / 0.7))             # сдвиг нарастает с 1.4 м, полный с 0.7 м
            off = max(-0.55, min(0.55, lat_o + side * need)) * ramp    # не дальше 55 см от маршрута: на повороте вылет
            gx, gy = goal[0] - off * math.sin(path_yaw), goal[1] + off * math.cos(path_yaw)
            if not self.free(gx, gy):                                  # там стена — с другой стороны
                off = (lat_o - side * need) * ramp
                gx, gy = goal[0] - off * math.sin(path_yaw), goal[1] + off * math.cos(path_yaw)
                if not self.free(gx, gy):                              # и там тоже — хотя бы полсдвига
                    off *= 0.5; gx, gy = goal[0] - off * math.sin(path_yaw), goal[1] + off * math.cos(path_yaw)
            goal = np.array([gx, gy])
            err = angdiff(math.atan2(goal[1] - pos[1], goal[0] - pos[0]), yaw)
            near_obst = max(near_obst, (1.6 - along) / 1.6)
        steer = max(-1.0, min(1.0, -err * self.gain))                  # цель слева (err>0) -> руль влево (минус)
        throttle = self.throttle * (1.0 - 0.35 * min(1.0, abs(err) / 0.8)) * (1.0 - 0.25 * near_obst)
        # упёрлись: по очереди — секунду назад с обратным рулём, потом секунду вперёд
        # с рулём в другую сторону; так не залипаем между конусом и краем коридора
        if collision and self.stuck_until is None:
            self.stuck_phase = (self.stuck_phase + 1) % 2
            self.stuck_until, self.stuck_steer = t + 1.0, (-steer if self.stuck_phase == 0 else (1.0 if steer <= 0 else -1.0))
        if self.stuck_until is not None:
            if t < self.stuck_until: steer, throttle = self.stuck_steer, (-0.5 if self.stuck_phase == 0 else 0.4)
            else: self.stuck_until = None
        label = steer
        if self.noise > 0 and self.stuck_until is None and near_obst == 0.0:   # у конуса не дёргаем
            if t > self.noise_until + 1.5 and self.rng.random() < 0.04:
                self.noise_until = t + self.rng.uniform(0.3, 0.7)
                self.noise_val = self.rng.choice([-1, 1]) * self.noise
            if t < self.noise_until:
                steer = max(-1.0, min(1.0, steer + self.noise_val))
        self.recovering = self.stuck_until is not None       # фаза выезда из упора: не образец езды
        return steer, throttle, label, lap_done
