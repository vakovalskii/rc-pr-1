"""
SLAM на одной камере: глубина — Depth Anything V2 (метрическая, для помещений) на ПК, движение — Холл на заднем
правом колесе + потенциометр руля (одометрия pi_car, поля odo_* в телеметрии). Строит 2D-карту помещения
(сетка проходимости 5 см) и цветное облако; шлёт их в релей тем же протоколом, что live_slam.py, — пульт
(окно «3D-карта») и /map3d.html рисуют, как карта наполняется.

    uv run --python 3.12 --with transformers --with opencv-python-headless --with aiohttp --with scipy --with pillow \\
        python mono_slam.py --relay http://192.168.1.109:8080
    ... python mono_slam.py --replay data/mono-20261106-120000      # тот же заезд ещё раз, без машинки

Как работает кадр:
  1. Рыбий глаз → обычная камера (модель равноудалённая, поле по диагонали --fov-diag или калибровка --calib).
  2. Depth Anything → глубина. Метры у неё для камеры в полутора метрах над полом: нашу (12 см) она «поднимает»
     до ~0.8 м, и дальности до стен врут в разы. Поэтому нейронке доверяем только форму: где пол, а где нет.
  3. Пол — плоскость RANSAC по низу кадра. В каждом столбце снизу вверх идём по полу до первого «не пола» —
     это основание препятствия. Расстояние до него — геометрия: высота камеры (--cam-height) и наклон (--pitch,
     мерить: --calib-wall) дают, где луч пикселя упирается в пол. Это 384 точки на кадр, как у лидара.
     У горизонта геометрия теряет точность (1° наклона на 2 м — это 50 см): там берём глубину нейронки,
     домноженную на поправку, которую дают ближние основания (геометрия / нейронка).
  4. Поза: одометрия с Холла даёт сдвиг с прошлого кадра, скан подгоняется к уже накопленной карте (scan2d.Scan2D)
     в окне ±12 см / ±6° вокруг неё — так не копится ни ошибка руля, ни пробуксовка.
  5. Карта пополняется, когда проехали 5 см или повернулись на 3°: стоя на месте одна и та же ошибка нейронки
     не вбивается в карту сотней одинаковых кадров.

Наклон камеры: поставь машинку ровно на D метров от стены носом в неё и запусти с --calib-wall D —
скрипт найдёт основание стены и скажет, какой --pitch ставить.

Каждый сеанс пишется в data/mono-<время>/: frames/*.jpg (как с камеры) + tele.jsonl — можно прогнать заново
с другими настройками (--replay); map/map.png + map.json (формат map_build.py) и preview.png с путём.
"""
import argparse, asyncio, io, json, math, pathlib, time
import numpy as np
from PIL import Image, ImageDraw
from scan2d import Grid, Scan2D, pose_T

ROOT = pathlib.Path(__file__).parent

class Undistort:
    """Рыбий глаз → прямолинейная камера с полем hfov. Без калибровки — равноудалённая модель r = f·θ по полю диагонали."""
    def __init__(self, w, h, a):
        import cv2
        self.cv2 = cv2
        if a.calib:
            c = json.loads(pathlib.Path(a.calib).read_text())
            K, D = np.array(c["K"], float), np.array(c["D"], float).reshape(4, 1)
            K[:2] *= w / c["width"]                       # калибровка могла быть на другом размере кадра
        else:
            f = math.hypot(w, h) / 2 / math.radians(a.fov_diag / 2)
            K, D = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), np.zeros((4, 1))
        self.w, self.h = a.out_w, round(a.out_w * h / w)
        self.fx = self.fy = self.w / 2 / math.tan(math.radians(a.hfov / 2))
        self.cx, self.cy = self.w / 2, self.h / 2
        P = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.m1, self.m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (self.w, self.h), cv2.CV_16SC2)
        mask = np.full((h, w), 255, np.uint8); mask[int(h * (1 - a.mask_bottom)):] = 0   # свой бампер внизу кадра
        self.mask = cv2.remap(mask, self.m1, self.m2, cv2.INTER_NEAREST, borderValue=0) > 0

    def __call__(self, rgb):
        return self.cv2.remap(rgb, self.m1, self.m2, self.cv2.INTER_LINEAR)

class Depth:
    """Depth Anything V2: RGB (uint8 HxWx3) → глубина в метрах того же размера. cuda → mps → cpu."""
    def __init__(self, a):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        name = {"s": "Small", "b": "Base", "l": "Large"}[a.model]
        repo = f"depth-anything/Depth-Anything-V2-Metric-Indoor-{name}-hf"
        self.proc = AutoImageProcessor.from_pretrained(repo, size={"height": a.input, "width": a.input})
        self.net = AutoModelForDepthEstimation.from_pretrained(repo).to(self.dev).eval()
        self.half = self.dev == "cuda"
        if self.half: self.net.half()
        print(f"глубина: {repo} на {self.dev}, вход {a.input}")

    def __call__(self, rgb):
        t = self.torch
        x = self.proc(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"].to(self.dev)
        with t.inference_mode():
            d = self.net(pixel_values=x.half() if self.half else x).predicted_depth[:, None].float()
            d = t.nn.functional.interpolate(d, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        return d.cpu().numpy()

def floor_plane(P, up0, rng, iters=80, tol=0.02):
    """RANSAC по точкам низа кадра → (нормаль «вверх» в осях камеры, расстояние от камеры до пола) или None.
    Нормаль должна быть не дальше 25° от ожидаемой up0: стена внизу кадра полом не считается."""
    if len(P) < 200: return None
    best, bn = 0, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9: continue
        n /= L
        if n @ up0 < 0: n = -n
        if n @ up0 < math.cos(math.radians(25)): continue
        d = -(n @ a); inl = np.abs(P @ n + d) < tol
        if inl.sum() > best: best, bn = int(inl.sum()), inl
    if bn is None or best < 0.4 * len(P): return None
    Q = P[bn]; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][2]       # уточнить по всем inlier-ам
    if n @ up0 < 0: n = -n
    if n @ up0 < math.cos(math.radians(25)): return None            # уточнение увело (точки в линию) — не верим
    return n, float(-(n @ c))                                        # камера над полом: n·p + h = 0 → h = −n·c

def level_axes(up):
    """Строки e1 (вперёд по полу), e2 (влево), вверх — в осях камеры (x вправо, y вниз, z вперёд)."""
    up = up / np.linalg.norm(up); f = np.array([0, 0, 1.0])
    e1 = f - (f @ up) * up; e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])

def font(size):
    """Шрифт с кириллицей: встроенный в PIL её не знает."""
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()

class VORot:
    """Поворот камеры между кадрами по оптическому потоку (LK) + существенная матрица (RANSAC) → курс в плоскости пола.
    Не зависит от масштаба и колёс: на кафеле колёса скользят, руль не откалиброван — а точки на картинке не врут.
    + — влево (как курс одометрии). None — мало точек или кадр смазан/однотонный."""
    def __init__(self, und):
        import cv2
        self.cv2, self.prev = cv2, None
        self.K = np.array([[und.fx, 0, und.cx], [0, und.fy, und.cy], [0, 0, 1.0]])
        self.mask = und.mask.astype(np.uint8) * 255
        self.inl = 0

    def step(self, rgb, R):
        cv2 = self.cv2; g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY); prev, self.prev = self.prev, g
        if prev is None: return None
        p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=self.mask)
        if p0 is None or len(p0) < 30: return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1; a0, a1 = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        if len(a0) < 30: return None
        E, m = cv2.findEssentialMat(a0, a1, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3): return None
        self.inl, Rm, _, _ = cv2.recoverPose(E, a0, a1, self.K, mask=m)
        if self.inl < 25: return None
        Rl = R @ Rm @ R.T                                                # точки повернулись на −курс в осях «вперёд, влево, вверх»
        d = -math.atan2(Rl[1, 0], Rl[0, 0])
        return d if abs(d) < math.radians(20) else None

class Mono:
    def __init__(self, a, out_dir):
        self.a, self.dir = a, out_dir
        self.und = self.depth = None; self.rng = np.random.default_rng(0)
        self.ups, self.walls = [], []
        self.R = self.scan = self.grid = None; self.plane = None; self.k_far = None; self.k_fars = []
        self.x = self.y = self.yaw = 0.0; self.odo = None; self.added = None
        self.kf_T, self.path, self.seen, self.pts = [], [], set(), 0
        self.n, self.times, self.scores, self.grid_at = 0, [], [], 0
        self.no_odo_warned = False; self.vo = None; self.vo_yaw = 0.0; self.yaw_src = '—'; self.rgb = self.geo_now = None; self.wall_pitch = None; self.lid = None; self.lid_t = 0.0
        self.vlm, self.vlm_t0, self.vlm_ms = None, 0.0, None
        self.dec_p, self.dec_t, self.dec_ms, self.p_vlm, self.p_want, self.p_arcs = None, 0.0, None, None, None, None
        self.rx_n = 0; self.z_vis = self.floor_vis = self.vlm_img = None; self.vlm_frame_age = None
        import collections
        self.vlm_hist = collections.deque(maxlen=40)                     # (пришёл, мс, ответ)
        self.ctl_t = collections.deque(maxlen=50); self.proc_ms = self.depth_ms = self.dec_ms = None

    def rate(self):
        self.times.append(time.time()); self.times = self.times[-20:]
        return round((len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-3), 1) if len(self.times) > 1 else 0.0

    def odo_delta(self, tele):
        """Сдвиг с прошлого кадра по одометрии в осях машинки на прошлом кадре: (вперёд, влево, поворот)."""
        if "odo_x" not in tele:
            if not self.no_odo_warned: print("в телеметрии нет odo_* — едем только по стенам (обнови pi_car.py)"); self.no_odo_warned = True
            return 0.0, 0.0, 0.0
        cur = (tele["odo_x"], tele["odo_y"], math.radians(tele["odo_yaw"]))
        prev, self.odo = self.odo, cur
        if prev is None: return 0.0, 0.0, 0.0
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]; c, s = math.cos(prev[2]), math.sin(prev[2])
        if math.hypot(dx, dy) > 0.5:                                     # pi_car перезапустился (одометрия с нуля) или
            return 0.0, 0.0, 0.0                                         # пачка ложных импульсов — за кадр столько не проехать
        return c * dx + s * dy, -s * dx + c * dy, (cur[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi

    def cam_xy(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * self.a.cam_x, self.y + s * self.a.cam_x

    def boundary(self, P, ok):
        """Основания препятствий: для каждого столбца — строка первого «не пола» снизу (или −1, если пол до горизонта).
        Пол — пиксели в пределах 8% высоты камеры от плоскости пола нейронки."""
        n, h = self.plane
        floor = ok & (np.abs(P @ n - (-h)) < 0.08 * h); self.floor_vis = floor
        H, W = floor.shape
        bottom = int(H * (1 - self.a.mask_bottom * 1.25))              # чуть выше бампера (он после выпрямления изогнут)
        rows = np.full(W, -1)
        f = floor[:bottom][::-1]                                         # снизу вверх
        # первый пол снизу, потом первая «дыра» в полу длиной от 3 строк — там основание
        started = np.maximum.accumulate(f, axis=0)
        notf = started & ~f
        run = notf[:-2] & notf[1:-1] & notf[2:]
        has = run.any(0)
        first = run.argmax(0)
        rows[has] = bottom - 1 - first[has]
        rows[~f.any(0)] = bottom - 1                                     # пола в столбце нет вовсе — препятствие вплотную
        # блики на кафеле дают одиночные «препятствия» посреди пола: настоящее основание тянется по соседним столбцам
        from scipy import ndimage
        return ndimage.median_filter(rows, size=9, mode="nearest")

    def ranges(self, rows, P):
        """Строки оснований → точки в плоскости пола от камеры (вперёд, влево) и откуда дальность: геометрия или нейронка."""
        a, u = self.a, self.und
        cols = np.nonzero(rows >= 0)[0]
        if not len(cols): return np.zeros((0, 2)), np.zeros(0, bool)
        r = np.stack([(cols - u.cx) / u.fx, (rows[cols] - u.cy) / u.fy, np.ones(len(cols))], 1) @ self.R.T   # лучи: вперёд, влево, вверх
        down = -r[:, 2] / np.hypot(r[:, 0], r[:, 1])                    # тангенс угла под горизонт
        geo = down > math.tan(math.radians(a.geo_min_deg))
        d_geo = np.where(geo, a.cam_height / np.maximum(down, 1e-6), np.inf)
        Pn = P[rows[cols], cols] @ self.R.T; d_net = np.hypot(Pn[:, 0], Pn[:, 1])
        near = geo & (d_geo < a.geo_trust_m)
        if near.sum() >= 15:                                             # поправка нейронки по ближним основаниям
            k = float(np.median(d_geo[near] / d_net[near])); self.k_fars.append(k)
            self.k_far = k if self.k_far is None else 0.8 * self.k_far + 0.2 * k
        d = np.where(geo, d_geo, d_net * (self.k_far or 1.0))
        dirxy = r[:, :2] / np.hypot(r[:, 0], r[:, 1])[:, None]
        keep = d < a.max_range
        pts, g = (dirxy * d[:, None])[keep], geo[keep]
        # Нависание: пол уходит под кровать, основание далеко — а край кровати висит на высоте машинки ближе. Его
        # «основание» не видно (оно в воздухе), поэтому берём всё «не пол» выше найденного основания, но ниже крыши
        # машинки (--car-h) и ближе основания — по глубине нейронки с поправкой k_far. В каждом столбце — ближайшее.
        self.over_px = []
        if self.k_far:
            H, W = P.shape[:2]; v = np.arange(H)[:, None]
            above = (v < rows[None, :]) & (rows[None, :] >= 0)
            Q = (P * self.k_far) @ self.R.T; hgt = Q[..., 2] + a.cam_height; dist = np.hypot(Q[..., 0], Q[..., 1])
            m = above & ~self.floor_vis & (hgt > a.z_min) & (hgt < a.car_h) & (dist < 1.5) & self.und.mask
            dd = np.where(m, dist, np.inf); best = dd.argmin(0); dcol = dd.min(0)
            # шум нейронки — россыпь по отдельным столбцам и кадрам; нависание настоящее, если держится
            # в соседних столбцах (медиана по 9) и в 2 из 3 последних кадров (медиана по времени)
            from scipy import ndimage
            dcol = ndimage.median_filter(np.where(np.isfinite(dcol), dcol, 9.0), size=9, mode="nearest")
            self.over_hist = (getattr(self, "over_hist", []) + [dcol])[-3:]
            dcol = np.median(np.stack(self.over_hist), 0)
            cols = np.nonzero(dcol < 1.5)[0]
            if len(cols):
                q = Q[best[cols], cols][:, :2]; q *= (dcol[cols] / np.maximum(np.hypot(q[:, 0], q[:, 1]), 1e-3))[:, None]
                self.over_px = list(zip(cols.tolist(), best[cols].tolist()))
                pts = np.concatenate([pts, q]); g = np.concatenate([g, np.zeros(len(q), bool)])
        return pts, g

    def process(self, jpg, tele):
        """→ (кусок облака или None, сетка или None, поза dict или None)."""
        a = self.a
        raw = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
        if self.und is None:
            self.und = Undistort(raw.shape[1], raw.shape[0], a); self.depth = Depth(a)
            u = self.und; v, uu = np.mgrid[0:u.h, 0:u.w]
            self.ray = np.stack([(uu - u.cx) / u.fx, (v - u.cy) / u.fy, np.ones_like(uu, float)], -1)   # z = 1
        t_in = time.time(); rgb = self.und(raw); z = self.depth(rgb); self.depth_ms = round((time.time() - t_in) * 1000)
        if self.vo is None and a.vo: self.vo = VORot(self.und)
        self.z_vis, self.frame_rx = z, getattr(self, "rx_at", t_in)
        ok = self.und.mask & (z > 0.05)
        P = self.ray * z[..., None]                                    # точки в осях камеры, единицы нейронки

        th = math.radians(a.pitch if a.pitch is not None else 0)
        up0 = np.array([0, -math.cos(th), math.sin(th)]) if self.R is None else self.R[2]
        low = ok.copy(); low[: int(self.und.h * 0.6)] = False
        fl = floor_plane(P[low][:: max(1, low.sum() // 1500)], up0, self.rng, tol=0.05)
        if fl: self.plane = fl
        if self.R is None:
            if fl: self.ups.append(fl[0])
            if a.pitch is None and len(self.ups) < a.init_frames and self.n < a.init_frames * 4:
                self.n += 1; return None, None, None
            # «вверх»: наклон задан (--pitch, лучше всего — из --calib-wall) или по нормали пола нейронки
            self.R = level_axes(up0 if a.pitch is not None or not self.ups else np.mean(self.ups, 0))
            self.scan = Scan2D(self.R, a.cam_height, (a.z_min, a.z_max), a.res, a.grid_m, a.max_range)
            self.grid = Grid(argparse.Namespace(res=a.res, grid_m=a.grid_m, cam_height=a.cam_height, z_min=a.z_min, z_max=a.z_max,
                                                car_radius=a.car_radius))
            self.grid.set_up(self.R[2])
            pitch = math.degrees(math.asin(max(-1, min(1, float(self.R[2] @ [0, 0, 1.0])))))
            print(f"наклон камеры {pitch:+.1f}° ({'задан' if a.pitch is not None else 'по полу нейронки — лучше померить --calib-wall'})")
        if self.plane is None: self.n += 1; return None, None, None
        rows = self.boundary(P, ok)
        self.rows_hist = (getattr(self, "rows_hist", []) + [rows])[-3:]
        if len(self.rows_hist) == 3: rows = np.median(np.stack(self.rows_hist), 0).astype(int)
        lid, geo = self.ranges(rows, P)
        self.last_rows, self.rgb, self.geo_now = rows, rgb, geo
        u = self.und; mid = rows[int(u.w * 0.4):int(u.w * 0.6)]; mid = mid[mid >= 0]   # стена по носу на wall_m: какой наклон
        self.wall_pitch = math.degrees(-math.atan(a.cam_height / a.wall_m) + math.atan((float(np.median(mid)) - u.cy) / u.fy)) \
            if len(mid) > 10 else None

        if a.calib_wall:                                                # стоим носом в стену на calib_wall м
            u = self.und; mid = rows[int(u.w * 0.4):int(u.w * 0.6)]; mid = mid[mid >= 0]
            if len(mid) > 10:
                v = float(np.median(mid)); e_cam = -math.atan((v - u.cy) / u.fy)
                self.walls.append(-math.atan(a.cam_height / a.calib_wall) - e_cam)
                print(f"основание стены в строке {v:.0f} → --pitch {math.degrees(np.median(self.walls)):+.1f} "
                      f"(по {len(self.walls)} кадрам, разброс {math.degrees(np.std(self.walls)):.1f}°)")
            self.n += 1; return None, None, None

        lid = lid + [a.cam_x, 0]                                       # от камеры — к середине задней оси
        self.lid, self.lid_t = lid, time.time(); self.proc_ms = round((self.lid_t - t_in) * 1000)
        mid = lid[(np.abs(lid[:, 1]) < 0.10) & geo] if len(lid) else lid    # что прямо по носу, по геометрии: для тестов пути
        self.front_d = float(np.median(mid[:, 0])) if len(mid) >= 10 else None
        fx, fy, dyaw = self.odo_delta(tele); self.hall_last = tele.get("hall_n")
        vo = self.vo.step(rgb, self.R) if self.vo else None
        if vo is not None: self.vo_yaw += vo                             # для проверки: курс по камере, копится всегда
        moving = bool(fx or fy) or abs(float(tele.get("thr", 0) or 0)) > 0.05
        if vo is not None and (moving or abs(vo) > math.radians(1.5)):   # стоим — шум потока не копим (люди ходят)
            dyaw = vo; self.yaw_src = "камера"
        else: self.yaw_src = "колесо+руль" if dyaw else self.yaw_src
        d = math.copysign(math.hypot(fx, fy), fx)                        # путь — по колесу, вдоль курса середины шага
        mid = self.yaw + dyaw / 2
        self.x += d * math.cos(mid); self.y += d * math.sin(mid); self.yaw += dyaw
        sc = None
        if fx or fy or dyaw:                                           # стоим — позу не трогаем: подгонять нечего
            self.x, self.y, self.yaw, sc = self.scan.match(lid, self.x, self.y, self.yaw)
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        if sc is not None: self.scores.append(sc)
        self.n += 1
        xc, yc = self.cam_xy(); T = pose_T(xc, yc, self.yaw, self.R)
        self.path.append((self.x, self.y))

        chunk = None
        moved = self.added is None or math.hypot(self.x - self.added[0], self.y - self.added[1]) > a.key_m \
            or abs((self.yaw - self.added[2] + math.pi) % (2 * math.pi) - math.pi) > math.radians(a.key_deg) \
            or len(self.kf_T) < 3
        if moved and len(lid) and (fx or fy or dyaw or len(self.kf_T) < 3):
            self.added = (self.x, self.y, self.yaw)
            self.scan.add(lid, self.x, self.y, self.yaw)
            cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
            L = lid - [a.cam_x, 0]                                     # обратно от камеры: лучи сетки идут из неё
            hq = (a.z_min + a.z_max) / 2 - a.cam_height                # основание — препятствие на высоте корпуса
            W = np.stack([xc + cy_ * L[:, 0] - sy_ * L[:, 1], yc + sy_ * L[:, 0] + cy_ * L[:, 1], np.full(len(L), hq)], 1)
            self.grid.add(T, W @ self.R)                               # q-мир → мир первой камеры
            self.kf_T.append(T)
            # облако для глаз: глубина нейронки × поправка, только ближе max_range
            Q = (P[ok] * (self.k_far or 1.0)) @ self.R.T; col = rgb[ok]
            keep = np.hypot(Q[:, 0], Q[:, 1]) < a.max_range; Q, col = Q[keep], col[keep]
            Wc = np.stack([xc + cy_ * Q[:, 0] - sy_ * Q[:, 1], yc + sy_ * Q[:, 0] + cy_ * Q[:, 1], Q[:, 2]], 1) @ self.R
            v = np.floor(Wc / a.voxel).astype(np.int64)                # в облако — только новые воксели
            keys = (v[:, 0] + 50000) * 10 ** 10 + (v[:, 1] + 50000) * 10 ** 5 + (v[:, 2] + 50000)
            keys, first = np.unique(keys, return_index=True)
            new = np.array([q not in self.seen for q in keys.tolist()], bool)
            idx = first[new]
            if len(idx) and self.pts < a.max_points and a.cloud:
                self.seen.update(keys[new].tolist())
                mm = np.clip(Wc[idx] * 1000, -32767, 32767).astype("<i2")
                chunk = b"P" + len(idx).to_bytes(4, "little") + mm.tobytes() + col[idx].astype(np.uint8).tobytes()
                self.pts += len(idx)
        grid = None
        if self.grid.dirty and time.time() - self.grid_at > 1.0:
            grid = self.save(); self.grid_at = time.time()
        pose = {"type": "pose", "up": self.R[2].round(4).tolist(), "T": T.reshape(-1).round(4).tolist(), "kf": int(moved),
                "lost": 0, "jumps": 0, "pts": self.pts, "frames": self.n, "fps": self.rate(),
                "k_far": None if self.k_far is None else round(self.k_far, 3), "match": None if sc is None else round(sc, 2),
                "scan": int(len(lid)), "geo": int(geo.sum()),
                "x": round(self.x, 2), "y": round(self.y, 2), "yaw": round(math.degrees(self.yaw), 1)}
        return chunk, grid, pose

    def view(self, pose):
        """JPEG «что сейчас видит SLAM»: слева выпрямленный кадр с найденными основаниями (зелёные — по геометрии,
        оранжевые — по нейронке), справа карта 6×6 м вокруг машинки: стены, путь, текущий скан."""
        a, u = self.a, self.und
        cam = Image.fromarray(self.rgb).copy(); d = ImageDraw.Draw(cam)
        cols = np.nonzero(self.last_rows >= 0)[0]
        g = iter(self.geo_now.tolist()) if self.geo_now is not None else None
        for c in cols:
            v = int(self.last_rows[c]); d.rectangle([c, v - 1, c, v + 1], fill=(40, 220, 60))
        hz = u.cy + math.tan(math.asin(max(-1, min(1, float(self.R[2] @ [0, 0, 1.0]))))) * u.fy   # смотрит вверх — горизонт ниже центра if self.R is not None else None
        if hz is not None: d.line([(0, hz), (u.w, hz)], fill=(80, 160, 255))        # горизонт по текущему наклону
        S, half = u.h, 3.0; px = S / (2 * half)
        mp = Image.new("RGB", (S, S), (128, 128, 128)); md = ImageDraw.Draw(mp)
        if self.scan is not None:
            r = int(half / a.res); cx, cy = (np.floor(np.array([self.x, self.y]) / a.res).astype(int) + self.scan.n // 2)
            win = self.scan.hits[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
            if self.grid is not None:
                Lw = self.grid.L[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
                img = np.full(Lw.shape, 128, np.uint8); img[Lw < -0.5] = 235; img[Lw > 0.6] = 20
            else: img = np.where(win >= 2, 20, 128).astype(np.uint8)
            if img.size == 0: img = np.full((2 * r, 2 * r), 128, np.uint8)   # машинка за краем сетки
            im = Image.fromarray(img[::-1]).resize((round(img.shape[1] * a.res * px), round(img.shape[0] * a.res * px)), Image.NEAREST)
            ox = max(cx - r, 0) - (cx - r); oy = (cy + r) - min(cy + r, self.scan.n)
            mp.paste(im.convert("RGB"), (round(ox * a.res * px), round(oy * a.res * px)))
            md = ImageDraw.Draw(mp)
            to = lambda x, y: (S / 2 + (x - self.x) * px, S / 2 - (y - self.y) * px)
            if len(self.path) > 1: md.line([to(*q) for q in self.path[-600::2]], fill=(220, 40, 40), width=2)
            c_, s_ = math.cos(self.yaw), math.sin(self.yaw)
            for f, l in getattr(self, "lid", np.zeros((0, 2)))[::2]:
                X, Y = to(self.x + c_ * f - s_ * l, self.y + s_ * f + c_ * l); md.rectangle([X - 1, Y - 1, X + 1, Y + 1], fill=(40, 220, 60))
            body = [(-a.rear_m, -a.half_w), (a.cam_x + a.front_m, -a.half_w), (a.cam_x + a.front_m, a.half_w), (-a.rear_m, a.half_w)]
            md.polygon([to(self.x + c_ * f - s_ * l, self.y + s_ * f + c_ * l) for f, l in body], outline=(30, 110, 230), width=2)
            X, Y = to(self.x, self.y); md.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=(30, 110, 230))
            md.line([(X, Y), to(self.x + 0.35 * c_, self.y + 0.35 * s_)], fill=(30, 110, 230), width=3)
            md.line([(8, S - 10), (8 + px, S - 10)], fill=(0, 0, 0), width=3); md.text((12, S - 28), "1 м", fill=(0, 0, 0), font=font(13))
        H2 = u.h + 44                                                       # второй ряд: что видит каждая модель
        out = Image.new("RGB", (2 * u.w + 230, S + H2 + 50), (24, 26, 30)); out.paste(cam, (0, 40)); out.paste(mp, (u.w, 40))
        y2 = 40 + S + 22
        if self.z_vis is not None:
            iz = 1 / np.maximum(self.z_vis, 0.05); lo, hi = np.percentile(iz, [2, 98]); q = np.clip((iz - lo) / (hi - lo + 1e-6), 0, 1)
            stops = np.array([[30, 30, 90], [40, 120, 200], [80, 200, 120], [250, 220, 60], [240, 70, 40]], float)  # далеко → близко
            k = q * (len(stops) - 1); i0 = np.floor(k).astype(int).clip(0, len(stops) - 2); fr = (k - i0)[..., None]
            out.paste(Image.fromarray((stops[i0] * (1 - fr) + stops[i0 + 1] * fr).astype(np.uint8)), (0, y2))
        if self.floor_vis is not None:
            fv = self.rgb.astype(np.float32) * 0.45; fv[self.floor_vis] = fv[self.floor_vis] * 0.5 + np.array([40, 200, 90]) * 0.6
            fv = fv.astype(np.uint8); cols = np.nonzero(self.last_rows >= 0)[0]
            fv[np.clip(self.last_rows[cols], 0, u.h - 1), cols] = (255, 60, 60)
            for (c_, r_) in getattr(self, "over_px", []): fv[max(r_ - 1, 0):r_ + 2, c_] = (255, 140, 0)
            out.paste(Image.fromarray(fv), (u.w, y2))
        if self.vlm_img is not None:
            out.paste(Image.fromarray(self.vlm_img).resize((216, round(216 * u.h / u.w))), (2 * u.w + 8, y2))
        t2 = ImageDraw.Draw(out); f12 = font(13)
        t2.text((6, y2 - 18), f"Depth Anything: глубина (жёлтое — близко), {self.depth_ms} мс", fill=(255, 210, 90), font=f12)
        t2.text((u.w + 6, y2 - 18), "пол — зел., основания — красн., нависания — оранж.", fill=(120, 240, 140), font=f12)
        t2.text((2 * u.w + 8, y2 - 18), "кадр, который видела VLM", fill=(140, 220, 255), font=f12)
        if self.vlm_frame_age is not None:
            t2.text((2 * u.w + 8, y2 + round(216 * u.h / u.w) + 4), f"кадру было {self.vlm_frame_age} мс,\nкогда пришёл ответ", fill=(140, 220, 255), font=f12)
        t2.text((6, 12), "камера + основания препятствий (зелёные), горизонт (синий)", fill=(235, 235, 235), font=f12)
        t2.text((u.w + 6, 12), "карта 6×6 м: стены, путь, скан", fill=(235, 235, 235), font=f12)
        out = out.crop((0, 0, 2 * u.w + 230, y2 + u.h + 4))
        buf = io.BytesIO(); out.save(buf, "JPEG", quality=80); return buf.getvalue()


    def prob_lines(self):
        """Шансы: по каждой дуге руля (из скана + «куда хочется») и мнения моделей — проценты, полоски."""
        bar = lambda p: "█" * round(p * 20) + "·" * (20 - round(p * 20))
        name = {-1.0: "влево до упора", -0.6: "влево", -0.3: "чуть влево", 0.0: "прямо", 0.3: "чуть вправо", 0.6: "вправо", 1.0: "вправо до упора"}
        L = ["", "руль (шанс выбора · свободно по дуге):"]
        for s_, (p, c) in (self.p_arcs or {}).items(): L.append(f"  {name[s_]:16s} {bar(p)} {p * 100:3.0f}%  {c:.2f} м")
        fmt = lambda d: "  ".join(f"{k} {d[k] * 100:.0f}%" for k in ("left", "straight", "right", "back")) if d else "—"
        L += ["", f"VLM за 3 с:        {fmt(self.p_vlm)}",
              f"{self.a.decide} ({self.dec_ms} мс): {fmt(self.dec_p if time.time() - self.dec_t < 1.5 else None)}",
              f"итог «куда хочется»: {fmt(self.p_want)}", ""]
        return L

    def state(self):
        """Всё, что видно на странице текстом (его можно выделить и скопировать): цифры, решения, ответы VLM, модели."""
        a, now = self.a, time.time()
        rx = getattr(self, "rx_hist", None)
        if rx is None: rx = self.rx_hist = __import__("collections").deque(maxlen=60)
        rx.append((now, self.rx_n, self.n))
        dt = rx[-1][0] - rx[0][0] if len(rx) > 1 else 0; got = rx[-1][1] - rx[0][1]; done = rx[-1][2] - rx[0][2]
        hz = lambda q: (len(q) - 1) / max(q[-1] - q[0], 1e-3) if len(q) > 1 else 0.0
        vms = sorted(h[1] for h in self.vlm_hist if now - h[0] < 10)
        pitch = None if self.R is None else round(math.degrees(math.asin(float(self.R[2] @ [0, 0, 1.0]))), 1)
        dev = self.depth.dev if self.depth else "—"
        f = lambda ta: time.strftime("%H:%M:%S", time.localtime(ta)) + f".{int(ta * 10) % 10}"
        return {
            "front_d": getattr(self, "front_d", None), "hall_n": getattr(self, "hall_last", None),
            "now": [f"решение: {getattr(self, 'ex_state', None) or '— (автопилот выключен)'}"] + self.prob_lines() + [
                    f"поза ({self.x:+.2f}, {self.y:+.2f}) м, курс {math.degrees(self.yaw):+.0f}° (источник: {self.yaw_src}), кадров {self.n}",
                    f"курс по камере всего: {math.degrees(self.vo_yaw):+.0f}° (точек в RANSAC {self.vo.inl if self.vo else 0}) — проверка: поверни руками на 360°",
                    f"скан: {0 if self.lid is None else len(self.lid)} точек, подгонка к карте {self.scores[-1] if self.scores else '—'}",
                    f"наклон камеры {pitch}°, высота {a.cam_height} м; если стена ровно в {a.wall_m:g} м по носу — наклон был бы "
                    f"{'—' if self.wall_pitch is None else f'{self.wall_pitch:+.1f}°'}"],
            "timing": [f"кадры: пришло {got / max(dt, 1e-3):.1f}/с, обработано {done / max(dt, 1e-3):.1f}/с, пропущено "
                       f"{max(0, got - done) / max(got, 1) * 100:.0f}% (берём самый свежий)",
                       f"кадру сейчас {round((now - getattr(self, 'frame_rx', now)) * 1000)} мс",
                       f"кадр → скан {self.proc_ms} мс, из них Depth Anything {self.depth_ms} мс",
                       f"скану в момент решения {self.dec_ms} мс, решений {hz(list(self.ctl_t)):.1f}/с",
                       f"VLM: {sum(1 for h in self.vlm_hist if now - h[0] < 10) / 10:.1f} ответа/с, задержка медиана "
                       f"{vms[len(vms) // 2] if vms else '—'} мс (мин {vms[0] if vms else '—'}, макс {vms[-1] if vms else '—'}), "
                       f"кадру при ответе {self.vlm_frame_age} мс"],
            "vlm": [f"{f(ta)}  {ms:4d} мс  → {v.get('go', '?'):8s} {v.get('s', '')} | слева {v.get('l', '—')} | центр {v.get('c', '—')} | "
                    f"справа {v.get('r', '—')}{' | ЛЮДИ' if v.get('p') else ''}" for ta, ms, v in list(self.vlm_hist)[::-1][:15]],
            "decisions": [f"{f(ta)}  {txt}" for ta, txt in list(getattr(self, "dec_log", []))[::-1][:25]],
            "models": [f"Depth Anything V2 Metric Indoor {a.model.upper()} — {dev}, вход {a.input}, {self.depth_ms} мс/кадр → где пол, где нет",
                       f"Пол: RANSAC-плоскость + геометрия (высота {a.cam_height} м, наклон {a.pitch}°) → дальность до оснований",
                       f"Нависания: «не пол» ниже {a.car_h} м по глубине нейронки × {self.k_far and round(self.k_far, 2)}",
                       "Скан → карта: scan2d, подгонка к карте + сетка 5 см",
                       "Одометрия: Холл на заднем правом колесе + потенциометр руля (pi_car на Pi 4)",
                       f"VLM: {a.vlm} (neuraldeep), {a.vlm_par} запроса в полёте, temperature 0, + история 4 ответов → куда интереснее + вероятности",
                       f"Решатель: {a.decide} (neuraldeep /v1/decisions) по тексту скана и VLM, ~4 раза/с → вероятности left/straight/right/back",
                       f"Автопилот: 7 дуг руля, стоп {a.stop_m} м, назад только по карте, газ ESP {a.esp_max / 10:g}%",
                       f"Корпус (во всех расчётах): от задней оси вперёд до бампера {a.cam_x + a.front_m:.2f} м (камера в {a.cam_x} м), "
                       f"назад до кормы {a.rear_m} м, ширина {2 * a.half_w:.2f} м, высота {a.car_h} м; перёд — нос с камерой, "
                       f"бампер внизу кадра вырезан (низ {a.mask_bottom:.0%})"]}

    def save(self):
        """Сетка → map/ (map.png + map.json + preview.png с путём); → сообщение b"G" для релея или None."""
        if self.grid is None: return None
        gm, png = self.grid.render(self.kf_T)
        if not gm: return None
        m = self.dir / "map"; m.mkdir(parents=True, exist_ok=True)
        (m / "map.png").write_bytes(png); (m / "map.json").write_text(json.dumps(gm))
        im = Image.open(io.BytesIO(png)).convert("RGB"); k = 4
        im = im.resize((im.width * k, im.height * k), Image.NEAREST); dr = ImageDraw.Draw(im)
        res, (ox, oy), H = gm["res"], gm["origin"], gm["height"]
        px = lambda x, y: ((x - ox) / res * k, (H - (y - oy) / res) * k)   # сетка перевёрнута: север сверху
        if len(self.path) > 1: dr.line([px(*p) for p in self.path[::2]], fill=(220, 40, 40), width=3)
        x0, y0 = px(self.x, self.y); dr.ellipse([x0 - 6, y0 - 6, x0 + 6, y0 + 6], fill=(30, 110, 230))
        dr.line([(x0, y0), px(self.x + 0.3 * math.cos(self.yaw), self.y + 0.3 * math.sin(self.yaw))], fill=(30, 110, 230), width=3)
        im.save(m / "preview.png")
        js = json.dumps(gm).encode()
        return b"G" + len(js).to_bytes(4, "little") + js + png

    def summary(self):
        s = np.array(self.k_fars) if self.k_fars else np.array([np.nan])
        sc = np.array(self.scores) if self.scores else np.array([np.nan])
        print(f"кадров {self.n}, точек облака {self.pts}, путь {sum(math.dist(p, q) for p, q in zip(self.path, self.path[1:])):.1f} м, "
              f"поправка нейронки вдали ×{np.nanmedian(s):.2f} (разброс {np.nanstd(s):.2f}), "
              f"подгонка к карте: средняя оценка {np.nanmean(sc):.2f}, принято {(sc >= 0.35).sum()} из {len(self.scores)}")

class Recorder:
    def __init__(self, d):
        (d / "frames").mkdir(parents=True, exist_ok=True); self.d, self.i = d, 0
        self.f = (d / "tele.jsonl").open("a")

    def put(self, jpg, tele):
        (self.d / "frames" / f"{self.i:06d}.jpg").write_bytes(jpg)
        self.f.write(json.dumps({"i": self.i, "t": round(time.time(), 3), **tele}, ensure_ascii=False) + "\n"); self.f.flush()
        self.i += 1

VLM_PROMPT = ('Ты камера на носу RC-багги 1:16 в квартире (не на улице). Камера в 12 см над полом и смотрит вперёд, '
              'сзади камеры нет. Машинка 40 см в длину, 28 см в ширину, 20 см в высоту: под мебель ниже 20 см не пролезет. '
              'Чёрная дуга внизу кадра — её собственный передний бампер, это не препятствие. Ответь коротким JSON без пробелов: '
              '{"s":"сцена до 6 слов","l":"что слева","c":"что по центру","r":"что справа",'
              '"go":"left|straight|right|back — куда ехать, чтобы исследовать и не врезаться",'
              '"pr":{"left":0..1,"straight":0..1,"right":0..1,"back":0..1} — уверенность в каждом варианте,"p":есть ли люди true/false}')

def history(mono, n=4):
    """Короткая текстовая память для VLM: что она говорила последние кадры, что делал автопилот, где машинка.
    Каждый кадр без неё — с чистого листа: модель не знает, что слева уже были, и что задний ход не едет."""
    now = time.time(); h = [x for x in mono.vlm_hist if now - x[0] < 15][-n:]
    if not h and not getattr(mono, "ex_state", None): return ""
    lines = [f"{now - ta:.0f} с назад: {v.get('s', '')}, советовала {v.get('go', '?')}" for ta, _, v in h]
    return ("\nКонтекст (не повторяй совет, если он не помог):\n" + "\n".join(lines) +
            f"\nАвтопилот сейчас: {getattr(mono, 'ex_state', '—')}. Поза ({mono.x:+.1f}, {mono.y:+.1f}) м, курс {math.degrees(mono.yaw):+.0f}°.")

async def vlm_loop(a, mono, d):
    """Наблюдатель: последний кадр → VLM (neuraldeep, OpenAI-совместимый) → mono.vlm. Держит --vlm-par запросов в полёте,
    так ответы идут чаще, чем длится один (~0.9 с напрямую; через системный прокси втрое дольше — поэтому мимо него).
    Ключ — переменная NDT_KEY или ~/.config/neuraldeep/key."""
    import aiohttp, base64, os
    key = os.environ.get("NDT_KEY") or pathlib.Path("~/.config/neuraldeep/key").expanduser().read_text().strip()
    log = (d / "vlm.jsonl").open("a")
    async with aiohttp.ClientSession(trust_env=False) as s:
        async def one():
            while True:
                if mono.rgb is None: await asyncio.sleep(0.2); continue
                img = mono.rgb; buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=70); t0 = time.time()
                frame_at = getattr(mono, "frame_rx", t0)                  # когда этот кадр пришёл с машинки
                body = {"model": a.vlm, "response_format": {"type": "json_object"}, "max_tokens": 120, "temperature": 0,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": VLM_PROMPT + history(mono)},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
                try:
                    async with s.post(f"{a.vlm_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"},
                                      timeout=aiohttp.ClientTimeout(total=15)) as r:
                        j = await r.json()
                    txt = j["choices"][0]["message"]["content"]; v = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
                    if t0 > mono.vlm_t0:                                  # ответы могут прийти не по порядку — берём свежий кадр
                        mono.vlm, mono.vlm_t0, mono.vlm_ms = v, t0, round((time.time() - t0) * 1000)
                    mono.vlm_hist.append((time.time(), round((time.time() - t0) * 1000), v))
                    if t0 >= mono.vlm_t0: mono.vlm_img, mono.vlm_frame_age = img, round((time.time() - frame_at) * 1000)
                    log.write(json.dumps({"t": round(t0, 3), "ms": round((time.time() - t0) * 1000), **v}, ensure_ascii=False) + "\n"); log.flush()
                except Exception as e:
                    print(f"VLM: {e.__class__.__name__}: {str(e)[:120]}"); await asyncio.sleep(1)
        print(f"VLM: {a.vlm}, {a.vlm_par} запроса в полёте")
        await asyncio.gather(*[one() for _ in range(a.vlm_par)])

DEC_CHOICES = [{"value": "left"}, {"value": "straight"}, {"value": "right"}, {"value": "back"}]

def situation(mono):
    """Текст для модели решений: что видит скан по сторонам, что сзади по карте, что говорит VLM."""
    pa = getattr(mono, "p_arcs", None) or {}
    side = lambda ks: max((pa[k][1] for k in ks if k in pa), default=0.0)
    v = mono.vlm or {}
    ex = getattr(mono, "explorer", None)
    back = ex.back_free() if ex else 0.0
    return (f"RC-машинка 40×28 см в квартире, камера только спереди. Свободно по дугам руля: влево {side((-1.0, -0.6)):.2f} м, "
            f"прямо {side((-0.3, 0.0, 0.3)):.2f} м, вправо {side((0.6, 1.0)):.2f} м; сзади по карте свободно {back:.2f} м. "
            f"Камера: {v.get('s', '—')}; слева {v.get('l', '—')}, по центру {v.get('c', '—')}, справа {v.get('r', '—')}. "
            f"Сейчас: {getattr(mono, 'ex_state', '—')}.")

async def decide_loop(a, mono):
    """Второе мнение: модель решений neuraldeep (/v1/decisions) по тексту ситуации → вероятности left/straight/right/back
    за ~0.2 с. Картинку не видит — только то, что уже поняли скан и VLM."""
    import aiohttp, os
    key = os.environ.get("NDT_KEY") or pathlib.Path("~/.config/neuraldeep/key").expanduser().read_text().strip()
    async with aiohttp.ClientSession(trust_env=False) as s:
        print(f"решатель: {a.decide}")
        while True:
            await asyncio.sleep(0.25)
            if not getattr(mono, "p_arcs", None): continue
            body = {"model": a.decide, "input": situation(mono), "questions": [{"id": "go", "type": "choice",
                    "instructions": "Куда ехать, чтобы исследовать комнату и не врезаться?", "choices": DEC_CHOICES}]}
            t0 = time.time()
            try:
                async with s.post(f"{a.vlm_url}/decisions", json=body, headers={"Authorization": f"Bearer {key}"},
                                  timeout=aiohttp.ClientTimeout(total=3)) as r:
                    j = await r.json()
                pr = {x["value"]: x["probability"] for x in j["answers"][0]["probabilities"]}
                mono.dec_p, mono.dec_t, mono.dec_ms = pr, time.time(), round((time.time() - t0) * 1000)
            except Exception as e:
                print(f"решатель: {e.__class__.__name__}: {str(e)[:100]}"); await asyncio.sleep(1)

class Explorer:
    """Сам ездит по комнате и не бьётся. Решение 9 раз в секунду, по порядку (tick):
      1. сдаём назад — до намеченного числа импульсов колеса; 0.7 с без импульса — сзади упор, назад 6 с не ездим;
      2. дуги руля −1…+1: сколько проедем, пока нос не упрётся (скан камеры + память слепой зоны под бампером);
      3. куда хочется: большинство голосов VLM за 3 с (или середина комнаты, когда исследование кончилось);
      4. упор вперёд: газ есть, колесо 2.5 с стоит — камера не видит то, что вплотную;
      5. впереди ≥ stop_m — едем по лучшей дуге (без рывков, газ 1.0 = потолок ESP);
      6. тупик — сначала вперёд на полном упоре руля, если там ≥ 25 см: туда смотрит камера;
      7. иначе назад, но только по карте: туда, где пол видели или сами проехали, и не дальше 45 см.
         Камеры сзади нет — вслепую назад не едем;
      8. сзади карта пустая — осторожно назад на 1 импульс (~13 см): упор заметим через 0.7 с;
      9. и это не вышло — стоим и просим помочь стиком.
    Человек тронул стик в пульте — автопилот выключается."""
    STEERS = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)

    def __init__(self, a, mono):
        self.a, self.m = a, mono
        self.mem = []                                    # (время, точки в осях одометрии)
        self.state, self.until, self.side = "еду", 0.0, 1.0
        self.want, self.want_t = 0.0, 0.0
        self.prev_s, self.hall, self.hall_t, self.fwd_t, self.no_rev = 0.0, None, time.time(), time.time(), 0.0
        self.rev_n0, self.rev_pulses = 0, 1
        self.sent, self.sent_t, self.override = (0.0, 0.0), time.time(), None
        self.on = True
        self.goal, self.free_hist, self.t0 = None, [], time.time()

    def _odo(self, tele):
        return (tele.get("odo_x", 0.0), tele.get("odo_y", 0.0), math.radians(tele.get("odo_yaw", 0.0)))

    def obstacles(self, tele, now):
        """Скан + память последних 3 с — в осях машинки сейчас (задняя ось, x вперёд, y влево)."""
        ox, oy, oyaw = self._odo(tele); c, s = math.cos(oyaw), math.sin(oyaw)
        if self.m.lid is not None and self.m.lid_t > (self.mem[-1][0] if self.mem else 0):
            L = self.m.lid; self.mem.append((self.m.lid_t, np.stack([ox + c * L[:, 0] - s * L[:, 1], oy + s * L[:, 0] + c * L[:, 1]], 1)))
        self.mem = [(t, P) for t, P in self.mem if now - t < 3.0]
        if not self.mem: return np.zeros((0, 2))
        W = np.concatenate([P for _, P in self.mem]) - [ox, oy]
        O = np.stack([c * W[:, 0] + s * W[:, 1], -s * W[:, 0] + c * W[:, 1]], 1)
        # память — только там, где камера сейчас не видит: вплотную (низ кадра под бампером) и за краями поля зрения.
        # Остальное камера видит сама — старые точки там только врут (одометрия по Холлу грубая, назад пропускает).
        rel = O - [self.a.cam_x, 0]; d = np.hypot(rel[:, 0], rel[:, 1])
        seen = (d > self.a.blind_m) & (np.abs(np.arctan2(rel[:, 1], rel[:, 0])) < math.radians(self.a.hfov / 2 - 5))
        cur = self.m.lid if self.m.lid is not None and now - self.m.lid_t < 0.6 else np.zeros((0, 2))
        return np.concatenate([cur, O[~seen]])

    def clearance(self, O, steer, length=1.2):
        a = self.a; k = -steer * 1.25                     # руль вправо (+) — кривизна минус
        front = a.cam_x + a.front_m
        for l in np.arange(0.0, length + 1e-9, 0.05):
            if abs(k) > 1e-3: th = k * l; x, y = math.sin(th) / k, (1 - math.cos(th)) / k
            else: th, x, y = 0.0, l, 0.0
            fx, fy = x + front * math.cos(th), y + front * math.sin(th)
            if len(O) and (np.hypot(O[:, 0] - fx, O[:, 1] - fy) < a.half_w).any(): return l
        return length

    def check_done(self, now):
        """Исследование кончилось, когда свободного пола на карте за done_s секунд прибавилось меньше 3%.
        Тогда цель — середина: клетка свободного пола, дальше всех от препятствий и неизвестного."""
        g = self.m.grid
        if self.goal is not None or g is None or now - self.t0 < self.a.min_explore_s: return
        free = int((g.L < -0.5).sum()); self.free_hist.append((now, free))
        self.free_hist = [(t, f) for t, f in self.free_hist if now - t < self.a.done_s + 5]
        old = [f for t, f in self.free_hist if now - t >= self.a.done_s]
        if not old or free > old[-1] * 1.03: return
        from scipy import ndimage
        D = ndimage.distance_transform_edt(g.L < -0.5)
        iy, ix = np.unravel_index(int(D.argmax()), D.shape)
        self.goal = ((ix - g.n // 2 + 0.5) * self.a.res, (iy - g.n // 2 + 0.5) * self.a.res)
        print(f"  исследовано: свободно {free * self.a.res ** 2:.1f} м², новое не появляется {self.a.done_s:.0f} с — "
              f"еду в середину ({self.goal[0]:+.2f}, {self.goal[1]:+.2f}) м, до стен там {D.max() * self.a.res:.2f} м")

    def tick(self, tele, now):
        """→ (газ, руль, что делаю)."""
        a = self.a
        if now - self.m.lid_t > 0.6: return 0.0, 0.0, "нет скана"
        O = self.obstacles(tele, now)
        n = tele.get("hall_n")
        if n != self.hall: self.hall, self.hall_t = n, now
        # ── 1. уже сдаём назад: до намеченного числа импульсов колеса, и только пока колесо крутится
        if self.state == "назад":
            moved = abs((n or 0) - self.rev_n0)
            if now - self.hall_t > 0.7:                                  # 0.7 с без импульса — упёрлись сзади
                self.state, self.no_rev = "еду", now + 6; print("  сзади упор — назад 6 с не езжу")
            elif moved >= self.rev_pulses or now > self.until: self.state = "еду"
            else: return -1.0, -self.side, f"назад {moved}/{self.rev_pulses} имп"
            self.hall_t = now
        # ── 2. сколько проедем вперёд по каждой дуге руля (скан + память слепой зоны)
        cl = {s: self.clearance(O, s) for s in self.STEERS}
        self.check_done(now)
        # ── 3. куда хочется: цель «середина» после исследования, иначе большинство голосов VLM за 3 с
        if self.goal is not None:
            gx, gy = self.goal; dist = math.hypot(gx - self.m.x, gy - self.m.y)
            if dist < a.goal_tol: self.on = False; return 0.0, 0.0, "стою в середине"
            bear = (math.atan2(gy - self.m.y, gx - self.m.x) - self.m.yaw + math.pi) % (2 * math.pi) - math.pi
            self.want, self.want_t = float(np.clip(-bear / 0.6, -1, 1)), now   # цель слева (+) — руль влево (−)
        v = self.m.vlm if now - self.m.vlm_t0 < 2.5 else None
        DIRS = ("left", "straight", "right", "back")
        def norm(d):
            t = sum(max(0.0, float(d.get(k, 0) or 0)) for k in DIRS); return {k: max(0.0, float(d.get(k, 0) or 0)) / t for k in DIRS} if t > 0 else None
        vl = [norm(h[2].get("pr") or {h[2].get("go"): 1.0}) for h in self.m.vlm_hist if now - h[0] < 3.0]
        vl = [x for x in vl if x]
        p_vlm = {k: sum(x[k] for x in vl) / len(vl) for k in DIRS} if vl else None   # среднее за 3 с: один ответ шумит
        p_dec = self.m.dec_p if now - self.m.dec_t < 1.5 else None
        mix = [x for x in (p_vlm, p_dec) if x]
        p_want = {k: sum(x[k] for x in mix) / len(mix) for k in DIRS} if mix else None
        self.m.p_vlm, self.m.p_want = p_vlm, p_want
        if self.goal is None and p_want:
            fwd = {k: p_want[k] for k in DIRS[:3]}; t = sum(fwd.values()) or 1
            self.want = (fwd["right"] - fwd["left"]) / t * 0.8; self.want_t = now   # «куда хочется» — взвешенно, не да/нет
            w = 0.25 * max(fwd.values()) / t                                         # чем увереннее — тем сильнее тянет
        else:
            if self.goal is None and now - self.want_t > 6: self.want, self.want_t = float(np.random.default_rng().choice([-0.6, 0.0, 0.6])), now
            w = 0.5 if self.goal is not None else 0.1
        score = {s: min(cl[s], 1.0) - 0.25 * abs(s - self.prev_s) - w * abs(s - self.want) for s in self.STEERS}
        z = np.array([score[s] for s in self.STEERS]) / 0.08; z = np.exp(z - z.max()); z /= z.sum()
        self.m.p_arcs = {s: (float(p), cl[s]) for s, p in zip(self.STEERS, z)}   # шанс каждой дуги руля (softmax оценки)
        best = max(score, key=score.get)
        # ── 4. упор вперёд: газ есть, а колесо 1.2 с не крутится (камера не видит то, что вплотную под бампером)
        if self.sent[0] <= 0: self.fwd_t = now
        # импульс колеса — раз в 13 см: на малом ходу их нет и по секунде, поэтому ждём 2.5 с
        stuck = self.sent[0] > 0 and now - self.hall_t > 2.5 and now - self.fwd_t > 2.5
        # ── 5. свободно — едем по лучшей дуге, без рывков (решение 9 раз в секунду успевает)
        if cl[best] >= a.stop_m and not stuck:
            self.prev_s = best
            thr = 1.0 if cl[best] > 0.8 and not (v and v.get("p")) else 0.75   # люди в кадре или стена близко — тише
            return thr, best, f"еду {cl[best]:.2f} м"
        # ── 6. тупик или упор. Сначала вперёд на полном упоре руля — туда камера смотрит
        side = 1.0 if cl[1.0] >= cl[-1.0] else -1.0
        if stuck: side = -1.0 if self.prev_s > 0 else 1.0                 # упёрлись — крутим в другую сторону, чем ехали
        if cl[side] >= 0.25 and not (stuck and now - self.fwd_t > 5):
            self.prev_s = side; return 0.75, side, f"разворот вперёд {'вправо' if side > 0 else 'влево'} {cl[side]:.2f} м"
        # ── 7. назад — только туда, где карта знает свободный пол (сами проехали или видели), и не дальше
        back = self.back_free()
        if now >= self.no_rev and back >= 0.15:
            self.side = side; self.state = "назад"; self.rev_n0 = n or 0; self.hall_t = self.fwd_t = now
            self.rev_pulses = max(1, int(min(back, 0.45) / self.m.a.mpp)); self.until = now + 2.5
            return -1.0, -side, f"{'упор' if stuck else 'тупик'} — назад до {min(back, 0.45):.2f} м (сзади свободно {back:.2f})"
        if now >= self.no_rev:                                           # карта сзади пустая (только начали): один импульс
            self.side = side; self.state = "назад"; self.rev_n0 = n or 0; self.hall_t = self.fwd_t = now   # ~13 см, упрёмся —
            self.rev_pulses, self.until = 1, now + 1.5                       # колесо встанет, через 0.7 с стоп
            return -1.0, -side, "осторожно назад на 1 импульс: сзади не видно"
        return 0.0, 0.0, f"застряла: впереди {cl[best]:.2f} м, сзади по карте {back:.2f} м — помоги стиком"

    def back_free(self):
        """Сколько метров за кормой свободно по карте (полоса шириной корпуса). Неизвестное — не свободно:
        камеры сзади нет, верим только тому, что видели раньше или где сами проехали."""
        g = self.m.grid
        if g is None: return 0.0
        c, s = math.cos(self.m.yaw), math.sin(self.m.yaw)
        for l in np.arange(0.05, 0.6, 0.05):
            for off in (-0.1, 0.0, 0.1):
                x = self.m.x - (self.a.rear_m + l) * c - off * s; y = self.m.y - (self.a.rear_m + l) * s + off * c
                j, i = (np.floor(np.array([x, y]) / self.a.res).astype(int) + g.n // 2)
                if not (0 <= i < g.n and 0 <= j < g.n) or g.L[i, j] >= -0.5: return l - 0.05
        return 0.55

async def explore_loop(a, mono, ws, st):
    """10 раз в секунду: решение → команда в релей как автопилот. Выход/обрыв — нейтраль и ручной руль."""
    ex, i, last_log = Explorer(a, mono), 0, 0
    mono.explorer = ex
    await ws.send_str(json.dumps({"type": "hello", "role": "pilot"}))
    await ws.send_str(json.dumps({"type": "esp", "cmd": "mode,2"}))
    await ws.send_str(json.dumps({"type": "esp", "cmd": f"max,{a.esp_max}"}))   # потолок газа ESP на время исследования
    print("исследую: еду сам (стик в пульте — перехват)")
    try:
        while ex.on:
            now = time.time(); tele = st["tele"]
            thr, steer, why = ex.tick(tele, now)
            # человек взял управление: телеметрия говорит не то, что шлём, дольше секунды
            if abs(ex.sent[0]) > 0 and now - ex.sent_t > 0.5 and abs(float(tele.get("thr", 0)) - ex.sent[0]) > 0.3:
                ex.override = ex.override or now
                if now - ex.override > 1.0: print("человек взял управление — автопилот выключен"); break
            else: ex.override = None
            await ws.send_str(json.dumps({"type": "cmd", "id": i, "steer": round(steer, 2), "throttle": round(thr, 2)})); i += 1
            if abs(thr - ex.sent[0]) > 0.05: ex.sent_t = now
            ex.sent = (thr, steer)
            if why.split(" ")[0] != (getattr(mono, "ex_state", "") or "").split(" ")[0]:
                if not hasattr(mono, "dec_log"): mono.dec_log = __import__("collections").deque(maxlen=200)
                mono.dec_log.append((now, f"{why}, руль {steer:+.1f}, газ {thr:+.1f}"))
            mono.ex_state = f"{why}, руль {steer:+.1f}, газ {thr:+.1f}"
            mono.ctl_t.append(now); mono.dec_ms = round((now - mono.lid_t) * 1000)   # сколько скану было, когда решали
            if now - last_log > 2: print(f"  {why}, руль {steer:+.1f}, газ {thr:+.1f}"); last_log = now
            await asyncio.sleep(0.1)
    finally:
        try:
            for _ in range(5):
                await ws.send_str(json.dumps({"type": "cmd", "id": i, "steer": 0, "throttle": 0})); i += 1; await asyncio.sleep(0.05)
            await ws.send_str(json.dumps({"type": "esp", "cmd": "mode,1"}))
        except Exception: pass

async def send(out, chunk, grid, pose):
    if out is None or out.closed: return
    if chunk: await out.send_bytes(chunk)
    if grid: await out.send_bytes(grid)
    if pose: await out.send_str(json.dumps(pose))

async def live(a):
    import aiohttp
    d = ROOT / "data" / f"mono-{time.strftime('%Y%m%d-%H%M%S')}"
    mono, rec, loop = Mono(a, d), Recorder(d), asyncio.get_running_loop()
    print("сеанс:", d)
    announced = False
    view = {"jpg": None, "ev": asyncio.Event(), "n": 0, "mono": mono}
    await serve_view(a, view)
    if a.vlm != "none": asyncio.create_task(vlm_loop(a, mono, d))
    if a.decide != "none": asyncio.create_task(decide_loop(a, mono))
    async with aiohttp.ClientSession(trust_env=False) as s:       # системный прокси мимо: релей в локалке
        while True:
            try:
                async with s.ws_connect(f"{a.relay}/ws/pult", max_msg_size=8 << 20) as cam, \
                           s.ws_connect(f"{a.relay}/ws/map3d", max_msg_size=32 << 20) as out:
                    await cam.send_str(json.dumps({"type": "view", "shrink": 0}))   # кадры как с камеры, без пережатия
                    if not announced: await out.send_str(json.dumps({"type": "reset", "rec": None})); announced = True
                    print(f"камера и одометрия: {a.relay}/ws/pult → карта: {a.relay}/ws/map3d")
                    st = {"jpg": None, "tele": {}}; ev = asyncio.Event()

                    async def rx():                                  # только смотрим: команд в пульт не шлём
                        async for m in cam:
                            if m.type == aiohttp.WSMsgType.BINARY: st["jpg"] = m.data; mono.rx_n += 1; mono.rx_at = time.time(); ev.set()
                            elif m.type == aiohttp.WSMsgType.TEXT and m.data.startswith('{"type": "tele"'): st["tele"] = json.loads(m.data)
                    rt = asyncio.create_task(rx())
                    et = asyncio.create_task(explore_loop(a, mono, cam, st)) if a.explore else None
                    try:
                        while not rt.done():
                            try: await asyncio.wait_for(ev.wait(), 5)
                            except asyncio.TimeoutError: raise ConnectionError("5 с без кадров")
                            ev.clear(); jpg, tele = st["jpg"], st["tele"]   # свежий кадр, пока считали прошлый — остальные пропущены
                            rec.put(jpg, tele)
                            chunk, grid, pose = await loop.run_in_executor(None, mono.process, jpg, tele)
                            await send(out, chunk, grid, pose)
                            if pose and mono.n % 30 == 0: log(pose)
                            if view["n"] and mono.rgb is not None:   # кто-то смотрит /
                                try: view["jpg"] = await loop.run_in_executor(None, mono.view, pose); view["ev"].set()
                                except Exception as e: print(f"картинка: {e.__class__.__name__}: {e}")
                    finally:
                        rt.cancel()
                        if et: et.cancel(); await asyncio.gather(et, return_exceptions=True)
            except (KeyboardInterrupt, asyncio.CancelledError): raise
            except Exception as e:
                print(f"нет связи ({e.__class__.__name__}: {e}), карта сохраняется, повтор через 2 с"); await asyncio.sleep(2)

PAGE = """<!doctype html><html lang=ru><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Mono SLAM</title>
<style>
:root{--bg:#15171b;--panel:#1e2126;--text:#e6e6e3;--dim:#9aa0a6;--gold:#f2c94c;--green:#6fdc8c;--blue:#8cc8f0;--line:#2c3036}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,"Segoe UI",sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 520px;gap:12px;padding:12px;max-width:1700px;margin:auto}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:6px;background:#000;display:block}
.side{display:flex;flex-direction:column;gap:10px}
section{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
h2{margin:0 0 6px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--dim);display:flex;justify-content:space-between;align-items:center}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.55 ui-monospace,Menlo,monospace;user-select:text}
#now pre{font-size:14px;color:var(--green)}#vlm pre{color:var(--blue)}#models pre{color:var(--gold)}
button{background:#2c3036;color:var(--text);border:1px solid #3a3f46;border-radius:4px;padding:3px 10px;font:12px sans-serif;cursor:pointer}
button:hover{background:#373c43}.bar{display:flex;gap:8px;align-items:center;margin-bottom:2px}
#pause.on{background:#5a3a1a}
</style>
<div class=wrap>
 <div><img src="/stream" alt="камера, карта и что видят модели"></div>
 <div class=side>
  <div class=bar><button id=copy>Копировать всё</button><button id=pause>Пауза текста</button><span id=st style="color:var(--dim)"></span></div>
  <section id=now><h2>Сейчас</h2><pre></pre></section>
  <section id=timing><h2>Задержки и частоты</h2><pre></pre></section>
  <section id=decisions><h2>Решения автопилота (смены)</h2><pre></pre></section>
  <section id=vlm><h2>Ответы VLM</h2><pre></pre></section>
  <section id=models><h2>Что работает</h2><pre></pre></section>
 </div>
</div>
<script>
let paused=false, last=null;
const keys=['now','timing','decisions','vlm','models'];
async function tick(){
  if(!paused){try{const r=await fetch('/state',{cache:'no-store'});last=await r.json();
    for(const k of keys){document.querySelector('#'+k+' pre').textContent=(last[k]||[]).join('\\n')}
    document.getElementById('st').textContent='обновлено '+new Date().toLocaleTimeString();
  }catch(e){document.getElementById('st').textContent='нет связи со скриптом'}}
  setTimeout(tick,500)}
tick();
document.getElementById('pause').onclick=e=>{paused=!paused;e.target.classList.toggle('on',paused);e.target.textContent=paused?'Продолжить':'Пауза текста'};
document.getElementById('copy').onclick=async()=>{
  const t=keys.map(k=>'## '+document.querySelector('#'+k+' h2').textContent+'\\n'+document.querySelector('#'+k+' pre').textContent).join('\\n\\n');
  try{await navigator.clipboard.writeText(t);document.getElementById('st').textContent='скопировано'}catch(e){
    const ta=document.createElement('textarea');ta.value=t;document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();
    document.getElementById('st').textContent='скопировано'}};
</script>"""

async def serve_view(a, view):
    """http://<этот ПК>:view_port/ — живая картинка SLAM (MJPEG): кадр с основаниями препятствий и карта вокруг."""
    from aiohttp import web
    async def page(_): return web.Response(text=PAGE, content_type="text/html")
    async def stream(req):
        r = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=f", "Cache-Control": "no-store"})
        await r.prepare(req); view["n"] += 1
        try:
            while True:
                await view["ev"].wait(); view["ev"].clear(); j = view["jpg"]
                await r.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
        except (ConnectionResetError, asyncio.CancelledError): pass
        finally: view["n"] -= 1
        return r
    async def state(_):
        m = view.get("mono")
        return web.json_response(m.state() if m else {}, dumps=lambda o: json.dumps(o, ensure_ascii=False, default=str))
    app = web.Application(); app.add_routes([web.get("/", page), web.get("/stream", stream), web.get("/state", state)])
    rn = web.AppRunner(app); await rn.setup(); await web.TCPSite(rn, "0.0.0.0", a.view_port).start()
    print(f"смотреть: http://localhost:{a.view_port}/")

def log(p):
    print(f"кадр {p['frames']}: {p['fps']} к/с, поза ({p['x']:+.2f}, {p['y']:+.2f}) м {p['yaw']:+.0f}°, "
          f"скан {p['scan']} точек ({p['geo']} по геометрии), поправка вдали ×{p['k_far']}, подгонка {p['match']}")

async def replay(a):
    src = pathlib.Path(a.replay)
    d = src / f"replay-{time.strftime('%H%M%S')}"; mono = Mono(a, d)
    tele = [json.loads(l) for l in (src / "tele.jsonl").read_text().splitlines() if l.strip()]
    out = None
    if a.relay != "none":
        import aiohttp
        s = aiohttp.ClientSession(trust_env=False); out = await s.ws_connect(f"{a.relay}/ws/map3d", max_msg_size=32 << 20)
        await out.send_str(json.dumps({"type": "reset", "rec": None}))
    t0 = time.time()
    for t in tele:
        r = mono.process((src / "frames" / f"{t['i']:06d}.jpg").read_bytes(), t)
        await send(out, *r)
        if r[2] and mono.n % 30 == 0: log(r[2])
    mono.save(); print(f"{time.time() - t0:.0f} с, результат: {d / 'map'}"); mono.summary()
    if out is not None: await out.close(); await s.close()

def main(a):
    if a.replay: asyncio.run(replay(a)); return
    try: asyncio.run(live(a))
    except KeyboardInterrupt: print("стоп")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--relay", default="http://192.168.1.109:8080", help="релей на Pi 4; при --replay можно none")
    p.add_argument("--replay", help="папка data/mono-… — прогнать записанный заезд")
    p.add_argument("--model", choices=("s", "b", "l"), default="s", help="Depth Anything V2 Small / Base / Large")
    p.add_argument("--input", type=int, default=392, help="сторона входа нейронки (кратно 14): больше — точнее и медленнее")
    p.add_argument("--device", help="cuda / mps / cpu (по умолчанию что есть)")
    p.add_argument("--fov-diag", type=float, default=140, help="°, поле объектива по диагонали (без калибровки)")
    p.add_argument("--calib", help="json {width, K, D} из cv2.fisheye.calibrate")
    p.add_argument("--hfov", type=float, default=100, help="°, поле выпрямленного кадра по горизонтали")
    p.add_argument("--out-w", type=int, default=384, help="ширина выпрямленного кадра")
    p.add_argument("--mask-bottom", type=float, default=0.16, help="доля низа кадра под бампером — не смотрим")
    p.add_argument("--cam-height", type=float, default=0.12, help="м, высота объектива над полом — от неё весь масштаб")
    p.add_argument("--cam-x", type=float, default=0.20, help="м, камера впереди задней оси (там считает одометрия)")
    p.add_argument("--pitch", type=float, help="°, наклон камеры вверх (из --calib-wall); не задан — по полу нейронки")
    p.add_argument("--explore", action="store_true", help="ездить самому: исследовать комнату, не врезаясь")
    p.add_argument("--esp-max", type=int, default=190, help="потолок газа ESP, ‰ (190 = 19%%)")
    p.add_argument("--done-s", type=float, default=60, help="с без нового свободного пола — исследование окончено")
    p.add_argument("--min-explore-s", type=float, default=120, help="с, раньше не заканчивать")
    p.add_argument("--goal-tol", type=float, default=0.3, help="м, «доехал до середины»")
    p.add_argument("--car-h", type=float, default=0.20, help="м, высота машинки: ниже — нависание, не проехать")
    p.add_argument("--rear-m", type=float, default=0.10, help="м, корма позади задней оси")
    p.add_argument("--mpp", type=float, default=0.13, help="м на импульс колеса (для «сколько сдать назад»)")
    p.add_argument("--stop-m", type=float, default=0.30, help="м, свободно впереди меньше — назад")
    p.add_argument("--half-w", type=float, default=0.14, help="м, полширины корпуса с запасом")
    p.add_argument("--front-m", type=float, default=0.10, help="м, нос впереди камеры")
    p.add_argument("--vlm", default="qwen3.6-35b-a3b-noreason", help="VLM-наблюдатель (none — без него)")
    p.add_argument("--decide", default="clef-flash", help="модель решений neuraldeep по тексту (none — без неё)")
    p.add_argument("--vlm-url", default="https://api.neuraldeep.ru/v1")
    p.add_argument("--vlm-par", type=int, default=2, help="запросов к VLM одновременно")
    p.add_argument("--cloud", action="store_true", help="слать цветное облако из глубины нейронки (масштаб у неё врёт — по умолчанию нет)")
    p.add_argument("--blind-m", type=float, default=0.45, help="м от камеры: ближе пол не виден — там верим памяти")
    p.add_argument("--no-vo", dest="vo", action="store_false", help="курс не по камере, а только по колесу и рулю")
    p.add_argument("--view-port", type=int, default=8091, help="живая картинка SLAM на этом ПК")
    p.add_argument("--wall-m", type=float, default=1.0, help="м, для подсказки наклона на картинке: стена по носу на таком расстоянии")
    p.add_argument("--calib-wall", type=float, help="м до стены прямо по носу: только померить --pitch и выйти")
    p.add_argument("--geo-min-deg", type=float, default=3, help="°, основание выше этого угла под горизонтом — дальность по геометрии")
    p.add_argument("--geo-trust-m", type=float, default=1.2, help="м, ближе — геометрия точная, по ней поправка нейронки")
    p.add_argument("--init-frames", type=int, default=5, help="кадров с полом до старта карты")
    p.add_argument("--max-range", type=float, default=3.0, help="м, дальше глубина из одной камеры уже гадание")
    p.add_argument("--z-min", type=float, default=0.04, help="м, ниже — пол")
    p.add_argument("--z-max", type=float, default=0.35, help="м, выше — не мешает проехать")
    p.add_argument("--key-m", type=float, default=0.05, help="м, проехали столько — кадр в карту")
    p.add_argument("--key-deg", type=float, default=3, help="°, повернулись на столько — кадр в карту")
    p.add_argument("--car-radius", type=float, default=0.15)
    p.add_argument("--res", type=float, default=0.05, help="м, клетка сетки")
    p.add_argument("--grid-m", type=float, default=30)
    p.add_argument("--voxel", type=float, default=0.03, help="м, шаг облака для показа")
    p.add_argument("--max-points", type=int, default=1_500_000)
    return p

if __name__ == "__main__":
    main(build_parser().parse_args())
