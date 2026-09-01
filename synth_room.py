"""
Синтетическая комната с точно известными позами камеры — тестовый полигон.

Нужна, чтобы прогнать весь конвейер (кадры → COLMAP → позы → карта → езда)
до появления настоящей съёмки и померить ошибку против правды: здесь позы
камеры и план комнаты известны точно, а не восстановлены.

Рендер — трассировка лучей по коробкам на numpy: комната-коробка изнутри,
мебель-коробки, у каждой грани своя шумовая текстура (без повторов —
иначе SfM путает похожие места).

    python3 synth_room.py data/synth --frames 150
    python3 synth_room.py data/synth --frames 150 --video   # + synth.mp4 для prep_video.py
"""
import argparse, json, math, pathlib, subprocess
import numpy as np
from PIL import Image

# --- сцена: метры, z вверх, пол z=0 -------------------------------------------
ROOM = (5.0, 4.0, 2.6)                       # ширина x, глубина y, высота z
OBSTACLES = [                                 # (имя, min xyz, max xyz)
    ("sofa",    (0.5, 3.5, 0.0), (2.3, 3.95, 0.8)),
    ("shelf",   (3.0, 3.5, 0.0), (4.0, 3.95, 1.2)),
    ("cabinet", (4.55, 0.3, 0.0), (4.95, 1.8, 1.9)),
    ("table",   (2.0, 1.6, 0.0), (3.0, 2.4, 0.75)),
    ("box",     (0.15, 1.2, 0.0), (0.55, 1.6, 0.4)),
]
# маршрут камеры: скруглённый прямоугольник вокруг стола, петля замыкается
PATH_RECT = (0.9, 0.9, 4.1, 3.1)
PATH_R = 0.5

def rounded_rect_path(n, rect, r):
    x0, y0, x1, y1 = rect
    segs = []                                  # (тип, параметры, длина)
    corners = [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]
    straights = [((x0 + r, y0), (x1 - r, y0)), ((x1, y0 + r), (x1, y1 - r)),
                 ((x1 - r, y1), (x0 + r, y1)), ((x0, y1 - r), (x0, y0 + r))]
    for s, c in zip(straights, corners):
        (ax, ay), (bx, by) = s
        segs.append(("line", (ax, ay, bx, by), math.hypot(bx - ax, by - ay)))
        segs.append(("arc", c, math.pi / 2 * r))
    total = sum(s[2] for s in segs)
    out = []
    for k in range(n):
        d = total * k / n
        for kind, p, L in segs:
            if d <= L:
                u = d / L
                if kind == "line":
                    ax, ay, bx, by = p
                    x, y = ax + (bx - ax) * u, ay + (by - ay) * u
                    yaw = math.atan2(by - ay, bx - ax)
                else:
                    cx, cy, a0 = p
                    a = math.radians(a0) + u * math.pi / 2
                    x, y = cx + r * math.cos(a), cy + r * math.sin(a)
                    yaw = a + math.pi / 2
                out.append((x, y, yaw))
                break
            d -= L
    return out

# --- текстуры -----------------------------------------------------------------
def value_noise(h, w, rng, octaves=((6, .45), (24, .3), (90, .15), (300, .1))):
    acc = np.zeros((h, w), np.float32)
    for cells, amp in octaves:
        g = rng.random((max(2, int(cells * h / max(h, w))) + 1, max(2, int(cells * w / max(h, w))) + 1)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += amp * (np.asarray(im, np.float32) / 255.0)
    acc -= acc.min(); acc /= max(acc.max(), 1e-6)
    return acc

def make_texture(w_m, h_m, base, rng, ppm=400, kind="wall"):
    w, h = int(min(2048, max(64, w_m * ppm))), int(min(2048, max(64, h_m * ppm)))
    n = value_noise(h, w, rng)
    tint = value_noise(h, w, rng, ((3, .6), (12, .4)))
    base = np.array(base, np.float32)
    tex = base[None, None, :] * (0.55 + 0.7 * n[..., None])
    tex[..., 0] *= 0.85 + 0.3 * tint
    tex[..., 2] *= 0.85 + 0.3 * (1 - tint)
    if kind == "floor":                        # доски: полосы + шов
        xs = np.arange(w)[None, :] / ppm
        plank = ((xs % 0.6) < 0.02).astype(np.float32)
        tex *= (1 - 0.5 * plank)[..., None]
        rows = np.arange(h)[:, None] / ppm
        tex *= (1 - 0.25 * ((rows % 1.4) < 0.015))[..., None]
    else:                                      # "постеры" — сильные углы и цвет
        for _ in range(int(2 + w_m * h_m * 1.5)):
            pw, ph = rng.integers(w // 14, w // 4), rng.integers(h // 14, h // 4)
            px, py = rng.integers(0, w - pw), rng.integers(0, h - ph)
            col = rng.random(3) * 200 + 30
            patch = value_noise(ph, pw, rng, ((4, .5), (30, .5)))
            tex[py:py + ph, px:px + pw] = col[None, None, :] * (0.5 + 0.7 * patch[..., None])
    return np.clip(tex, 0, 255).astype(np.uint8)

# --- рендер ---------------------------------------------------------------------
class Scene:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        W, D, H = ROOM
        self.boxes = [np.array([[0, 0, 0], [W, D, H]], np.float64)]      # 0 — комната (изнутри)
        self.names = ["room"]
        for name, mn, mx in OBSTACLES:
            self.boxes.append(np.array([mn, mx], np.float64)); self.names.append(name)
        self.boxes = np.stack(self.boxes)                                  # (M,2,3)
        palette = {"room": [(215, 205, 190), (225, 210, 190), (200, 210, 220), (210, 200, 205), (150, 110, 70), (235, 235, 230)],
                   "sofa": [(90, 110, 160)] * 6, "shelf": [(160, 120, 80)] * 6, "cabinet": [(120, 90, 60)] * 6,
                   "table": [(170, 140, 100)] * 6, "box": [(190, 170, 130)] * 6}
        self.tex = {}                                                        # (box, axis, side) -> tex
        for b in range(len(self.boxes)):
            mn, mx = self.boxes[b]
            size = mx - mn
            for axis in range(3):
                a1, a2 = [i for i in range(3) if i != axis]
                for side in (0, 1):
                    kind = "floor" if (b == 0 and axis == 2 and side == 0) else "wall"
                    base = palette[self.names[b]][axis * 2 + side]
                    self.tex[(b, axis, side)] = make_texture(size[a1], size[a2], base, rng, kind=kind)
        self.light = np.array([W / 2, D / 2, H - 0.1])

    def render(self, C, R_wc, w, h, f, ss=2):
        """C — центр камеры, R_wc — оси камеры в мире (столбцы: right, down, forward)."""
        W, Hh = w * ss, h * ss
        fs = f * ss
        u = (np.arange(W) + 0.5 - W / 2) / fs
        v = (np.arange(Hh) + 0.5 - Hh / 2) / fs
        uu, vv = np.meshgrid(u, v)
        d_cam = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3)
        d = d_cam @ R_wc.T
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        d_safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
        P = d.shape[0]
        best_t = np.full(P, np.inf); best_face = np.full(P, -1, np.int32)
        for b in range(len(self.boxes)):
            mn, mx = self.boxes[b]
            t1 = (mn[None, :] - C[None, :]) / d_safe
            t2 = (mx[None, :] - C[None, :]) / d_safe
            tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
            tmin, tmax = tn.max(1), tf.min(1)
            if b == 0:                                       # изнутри: выход из коробки
                t = tmax; axis = tf.argmin(1); side = (d[np.arange(P), axis] > 0).astype(int)
                hit = tmax > 0
            else:
                t = tmin; axis = tn.argmax(1); side = (d[np.arange(P), axis] < 0).astype(int)
                hit = (tmax > np.maximum(tmin, 0)) & (tmin > 0)
            better = hit & (t < best_t)
            best_t = np.where(better, t, best_t)
            best_face = np.where(better, b * 6 + axis * 2 + side, best_face)
        pts = C[None, :] + d * best_t[:, None]
        rgb = np.zeros((P, 3), np.float32)
        for face in np.unique(best_face):
            if face < 0: continue
            b, axis, side = face // 6, (face % 6) // 2, face % 2
            m = best_face == face
            mn, mx = self.boxes[b]
            a1, a2 = [i for i in range(3) if i != axis]
            tex = self.tex[(b, axis, side)]
            th, tw = tex.shape[:2]
            uf = (pts[m, a1] - mn[a1]) / (mx[a1] - mn[a1]); vf = (pts[m, a2] - mn[a2]) / (mx[a2] - mn[a2])
            ix = np.clip((uf * (tw - 1)).astype(int), 0, tw - 1); iy = np.clip((vf * (th - 1)).astype(int), 0, th - 1)
            col = tex[iy, ix].astype(np.float32)
            n = np.zeros(3); n[axis] = 1 if (side == 0) else -1            # нормаль внутрь сцены
            if b == 0: n = -n
            L = self.light[None, :] - pts[m]
            dist = np.linalg.norm(L, axis=1) + 1e-6
            lam = np.clip((L / dist[:, None]) @ n, 0, 1)
            shade = 0.45 + 0.65 * lam / (1 + 0.04 * dist ** 2)
            rgb[m] = col * shade[:, None]
        img = np.clip(rgb, 0, 255).astype(np.uint8).reshape(Hh, W, 3)
        im = Image.fromarray(img)
        if ss > 1:
            im = im.resize((w, h), Image.LANCZOS)
        return im

def pose_matrix(x, y, z, yaw):
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    R_wc = np.stack([r, dn, f], 1)
    return np.array([x, y, z]), R_wc

def rotmat_to_qvec(R):
    """COLMAP-порядок (w, x, y, z)."""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q[0] = 0.25 * s; q[1] = (R[2, 1] - R[1, 2]) / s; q[2] = (R[0, 2] - R[2, 0]) / s; q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        qi = 0.25 * s; qj = (R[j, i] + R[i, j]) / s; qk = (R[k, i] + R[i, k]) / s; qw = (R[k, j] - R[j, k]) / s
        q[0] = qw; q[[1, 2, 3][i]] = qi; q[[1, 2, 3][j]] = qj; q[[1, 2, 3][k]] = qk
    return q

def main():
    p = argparse.ArgumentParser()
    p.add_argument("outdir")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--hfov", type=float, default=70.0, help="градусов по горизонтали")
    p.add_argument("--cam-height", type=float, default=0.12)
    p.add_argument("--wobble", type=float, default=12.0, help="покачивание взгляда, градусов")
    p.add_argument("--ss", type=int, default=2, help="суперсэмплинг против алиасинга")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--video", action="store_true", help="ещё и собрать synth.mp4")
    a = p.parse_args()

    out = pathlib.Path(a.outdir); frames = out / "frames"; frames.mkdir(parents=True, exist_ok=True)
    f = (a.width / 2) / math.tan(math.radians(a.hfov / 2))
    cx, cy = a.width / 2, a.height / 2
    print(f"сцена {ROOM[0]}x{ROOM[1]} м, {len(OBSTACLES)} препятствий, камера {a.width}x{a.height} f={f:.1f}")
    scene = Scene(a.seed)
    path = rounded_rect_path(a.frames, PATH_RECT, PATH_R)
    poses = []
    for i, (x, y, yaw0) in enumerate(path):
        yaw = yaw0 + math.radians(a.wobble) * math.sin(i * 2 * math.pi / 37.0)
        C, R_wc = pose_matrix(x, y, a.cam_height, yaw)
        im = scene.render(C, R_wc, a.width, a.height, f, a.ss)
        name = f"{i:05d}.jpg"
        im.save(frames / name, "JPEG", quality=92)
        R_cw = R_wc.T                                  # COLMAP: мир -> камера
        t = -R_cw @ C
        poses.append({"name": name, "x": x, "y": y, "z": a.cam_height, "yaw": yaw,
                      "qvec": rotmat_to_qvec(R_cw).tolist(), "tvec": t.tolist()})
        if i % 25 == 0: print(f"  кадр {i}/{a.frames}  ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f}°")
    (out / "poses_gt.json").write_text(json.dumps({
        "fx": f, "fy": f, "cx": cx, "cy": cy, "width": a.width, "height": a.height,
        "cam_height": a.cam_height, "frames": poses}, indent=1))
    (out / "scene.json").write_text(json.dumps({
        "room": ROOM, "obstacles": [{"name": n, "min": mn, "max": mx} for n, mn, mx in OBSTACLES],
        "path": [[x, y] for x, y, _ in path]}, indent=1))
    (out / "manifest.json").write_text(json.dumps({"frames": len(poses), "width": a.width, "height": a.height,
                                                    "source": "synth_room.py", "synthetic": True}, indent=1))
    print(f"готово: {len(poses)} кадров -> {frames}, правда в poses_gt.json и scene.json")
    if a.video:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(frames / "%05d.jpg"),
                        "-pix_fmt", "yuv420p", "-crf", "18", str(out / "synth.mp4")], check=True)
        print(f"видео: {out / 'synth.mp4'}")

if __name__ == "__main__":
    main()
