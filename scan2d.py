"""
2D-SLAM поверх RGBD-одометрии — для машинки без IMU (D455 по USB 2 гироскоп не отдаёт).

Две идеи, как у TurtleBot с Kinect (depthimage_to_laserscan + gmapping/Hector):
  1. Плоское движение. Машинка едет по полу, камера закреплена жёстко (высота, наклон известны),
     значит у позы три степени свободы: x, y, курс. Подъём, крен и наклон из одометрии — её ошибка,
     их выбрасываем (planar()). Это убирает «взлетающий» путь и гнутую карту.
  2. Скан-матчинг. Глубина на высоте корпуса → «виртуальный лидар» (точки в плоскости пола). Каждый кадр
     подгоняется к уже накопленной 2D-карте перебором малых сдвигов и поворотов вокруг позы одометрии
     (корреляционный матчинг по полю правдоподобия). Ошибка одометрии не копится — кадр «пришпилен» к стенам.

Система «выровненная» (как у Grid в live_slam): e1 — вперёд первой камеры по полу, e2 — влево, вверх.
"""
import math
import numpy as np
from scipy import ndimage

def level_R(pitch_deg):
    """Оси выровненной системы в координатах камеры (строки e1, e2, вверх) при наклоне камеры вверх на pitch."""
    th = math.radians(pitch_deg)
    up = np.array([0, -math.cos(th), math.sin(th)])
    f = np.array([0, 0, 1.0]); e1 = f - (f @ up) * up; e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])

def planar(T, R):
    """Поза камеры в мире первого кадра (4x4) → та же поза, но строго на плоскости: (x, y, курс), высота и наклон как у первого кадра."""
    q = R @ T[:3, 3]; f = R @ (T[:3, :3] @ [0, 0, 1.0])
    yaw = math.atan2(f[1], f[0])
    return pose_T(q[0], q[1], yaw, R), (float(q[0]), float(q[1]), yaw)

def pose_T(x, y, yaw, R):
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T = np.eye(4); T[:3, :3] = R.T @ Rz @ R; T[:3, 3] = R.T @ [x, y, 0]
    return T

class Scan2D:
    def __init__(self, R, cam_height=0.10, band=(0.04, 0.35), res=0.05, size_m=30, max_range=3.0):
        self.R, self.h, self.band, self.res, self.n, self.max_range = R, cam_height, band, res, int(size_m / res), max_range
        self.hits = np.zeros((self.n, self.n), np.float32)
        self.field = None; self.dirty = True; self.frames = 0
        self.last = None                                      # последняя подгонка: (dx, dy, dyaw, score)

    def scan(self, z_m, fx, fy, cx, cy, step=2):
        """Глубина (м) → точки «лидара» в плоскости, в системе камеры, выровненной по полу: (вперёд, влево)."""
        zz = z_m[::step, ::step]; v, u = np.mgrid[0:z_m.shape[0]:step, 0:z_m.shape[1]:step]
        m = (zz > 0.15) & (zz < self.max_range)
        P = np.stack([(u[m] - cx) / fx * zz[m], (v[m] - cy) / fy * zz[m], zz[m]], 1)   # камера: x вправо, y вниз, z вперёд
        Q = P @ self.R.T                                      # в выровненной: [вперёд, влево, вверх]
        hgt = Q[:, 2] + self.h
        Q = Q[(hgt > self.band[0]) & (hgt < self.band[1])][:, :2]
        if len(Q) > 600: Q = Q[np.random.default_rng(self.frames).choice(len(Q), 600, replace=False)]
        return Q

    def _cells(self, pts):
        return np.floor(pts / self.res).astype(int) + self.n // 2

    def _world(self, pts, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)

    def _field(self):
        """Поле правдоподобия: близко к занятой клетке — высокое (гаусс по расстоянию, σ = 7 см)."""
        if self.dirty:
            occ = self.hits >= 2
            d = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(self.hits.shape, 9.0)
            self.field = np.exp(-(d ** 2) / (2 * 0.07 ** 2)); self.dirty = False
        return self.field

    def match(self, pts, x, y, yaw, win_xy=0.12, win_deg=6.0):
        """Ищет лучший сдвиг вокруг позы одометрии: грубо (3 см, 1.5°), потом точнее (1 см, 0.5°)."""
        if len(pts) < 40 or (self.hits >= 2).sum() < 150: return x, y, yaw, None
        F = self._field()
        best = (x, y, yaw, -1.0)
        for wxy, sxy, wdeg, sdeg in ((win_xy, 0.03, win_deg, 1.5), (0.03, 0.01, 1.5, 0.5)):
            bx, by, byaw = best[:3]
            dxs = np.arange(-wxy, wxy + 1e-9, sxy); das = np.radians(np.arange(-wdeg, wdeg + 1e-9, sdeg))
            for da in das:
                yw = byaw + da; c, s = math.cos(yw), math.sin(yw)
                W = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)
                for dx in dxs:
                    cx = np.floor((W[:, 0] + bx + dx) / self.res).astype(int) + self.n // 2
                    for dy in dxs:
                        cy = np.floor((W[:, 1] + by + dy) / self.res).astype(int) + self.n // 2
                        ok = (cx >= 0) & (cx < self.n) & (cy >= 0) & (cy < self.n)
                        sc = F[cy[ok], cx[ok]].mean() if ok.any() else 0
                        if sc > best[3]: best = (bx + dx, by + dy, yw, sc)
        # слабое совпадение (мало знакомых стен в кадре) — веры нет, остаёмся на одометрии
        if best[3] < 0.35: return x, y, yaw, best[3]
        self.last = (best[0] - x, best[1] - y, best[2] - yaw, best[3])
        return best[0], best[1], best[2], best[3]

    def add(self, pts, x, y, yaw):
        c = self._cells(self._world(pts, x, y, yaw))
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.n) & (c[:, 1] >= 0) & (c[:, 1] < self.n)
        np.add.at(self.hits, (c[ok, 1], c[ok, 0]), 1.0)
        np.minimum(self.hits, 20, out=self.hits)
        self.frames += 1; self.dirty = True
