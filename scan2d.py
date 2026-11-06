"""
2D-SLAM поверх RGBD-одометрии — для машинки без IMU (D455 по USB 2 гироскоп не отдаёт).

Две идеи, как у TurtleBot с Kinect (depthimage_to_laserscan + gmapping/Hector):
  1. Плоское движение. Машинка едет по полу, камера закреплена жёстко (высота, наклон известны),
     значит у позы три степени свободы: x, y, курс. Подъём, крен и наклон из одометрии — её ошибка,
     их выбрасываем (planar()). Это убирает «взлетающий» путь и гнутую карту.
  2. Скан-матчинг. Глубина на высоте корпуса → «виртуальный лидар» (точки в плоскости пола). Каждый кадр
     подгоняется к уже накопленной 2D-карте перебором малых сдвигов и поворотов вокруг позы одометрии
     (корреляционный матчинг по полю правдоподобия). Ошибка одометрии не копится — кадр «пришпилен» к стенам.

Система «выровненная» (как у Grid ниже): e1 — вперёд первой камеры по полу, e2 — влево, вверх.
"""
import io, math
import numpy as np
from PIL import Image
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
            occ = self.hits >= 4                      # одиночные попадания — шум (одна камера), стена — где били не раз
            d = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(self.hits.shape, 9.0)
            self.field = np.exp(-(d ** 2) / (2 * 0.07 ** 2)); self.dirty = False
        return self.field

    def match(self, pts, x, y, yaw, win_xy=0.12, win_deg=6.0):
        """Ищет лучший сдвиг вокруг позы одометрии: грубо (3 см, 1.5°), потом точнее (1 см, 0.5°)."""
        if len(pts) < 40 or (self.hits >= 4).sum() < 150: return x, y, yaw, None
        F = self._field()
        # Отсчёт — сама поза одометрии; уйти от неё можно, только если совпадение заметно лучше: штраф растёт к краю окна
        # (до 0.05). Без этого на ровном поле (стоим, стена вдоль, оценка везде 1.0) argmax брал первый вариант — угол
        # окна −6° — и поза каждый кадр уползала на шаг: −800° за минуту.
        sc0 = self._score(F, pts, x, y, yaw)
        best = (x, y, yaw, sc0, sc0)                      # x, y, курс, оценка, оценка со штрафом
        coarse = (win_xy, 0.03, win_deg, 1.5) if win_xy <= 0.15 else (win_xy, 0.05, win_deg, 3.0)   # широкое окно — грубее шаг
        for wxy, sxy, wdeg, sdeg in (coarse, (0.05, 0.01, 2.0, 0.5)):
            bx, by, byaw = best[:3]
            dxs = np.arange(-wxy, wxy + 1e-9, sxy); das = np.radians(np.arange(-wdeg, wdeg + 1e-9, sdeg))
            for da in das:                                # все сдвиги одного поворота разом: (dy, dx, точка)
                yw = byaw + da; c, s = math.cos(yw), math.sin(yw)
                W = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)
                cx = np.floor((W[None, :, 0] + bx + dxs[:, None]) / self.res).astype(int) + self.n // 2
                cy = np.floor((W[None, :, 1] + by + dxs[:, None]) / self.res).astype(int) + self.n // 2
                okx, oky = (cx >= 0) & (cx < self.n), (cy >= 0) & (cy < self.n)
                v = F[np.clip(cy, 0, self.n - 1)[:, None, :], np.clip(cx, 0, self.n - 1)[None, :, :]] * (oky[:, None, :] & okx[None, :, :])
                sc = v.mean(2)                            # за краем сетки — ноль (раньше такие точки выбрасывались — почти не бывает)
                pen = 0.025 * (np.hypot(bx + dxs[None, :] - x, by + dxs[:, None] - y) / win_xy + abs(yw - yaw) / math.radians(win_deg))
                iy, ix = np.unravel_index(int((sc - pen).argmax()), sc.shape)
                if sc[iy, ix] - pen[iy, ix] > best[4]: best = (bx + dxs[ix], by + dxs[iy], yw, float(sc[iy, ix]), float(sc[iy, ix] - pen[iy, ix]))
        # слабое совпадение (мало знакомых стен в кадре) — веры нет, остаёмся на одометрии
        if best[3] < 0.35: return x, y, yaw, best[3]
        self.last = (best[0] - x, best[1] - y, best[2] - yaw, best[3])
        return best[0], best[1], best[2], best[3]

    def _score(self, F, pts, x, y, yaw):
        c = self._cells(self._world(pts, x, y, yaw))
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.n) & (c[:, 1] >= 0) & (c[:, 1] < self.n)
        return float((F[c[ok, 1], c[ok, 0]].sum()) / max(len(pts), 1))

    def add(self, pts, x, y, yaw):
        c = self._cells(self._world(pts, x, y, yaw))
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.n) & (c[:, 1] >= 0) & (c[:, 1] < self.n)
        np.add.at(self.hits, (c[ok, 1], c[ok, 0]), 1.0)
        np.minimum(self.hits, 20, out=self.hits)
        self.frames += 1; self.dirty = True

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
