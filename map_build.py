"""
Карта полигона из облака точек: сетка занятости для столкновений и минимапы.

    python3 map_build.py data/synth --res 0.05

Читает <dir>/recon/{points.npz, poses.json}, пишет <dir>/map/:
  map.png       0 — препятствие, 255 — свободно, 128 — неизвестно (север = +y сверху)
  map.json      привязка (origin, res), ключевые кадры с позами, точка спавна
  map_view.png  то же, но для глаз: траектория, кадры, правда сцены, если есть

Препятствие — где есть точки на высоте корпуса машинки (по умолчанию 4–50 см
над полом). Свободно — где лежат точки пола или где проезжала камера.
"""
import argparse, json, math, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--res", type=float, default=0.05, help="клетка, м")
    p.add_argument("--z-min", type=float, default=0.04)
    p.add_argument("--z-max", type=float, default=0.5)
    p.add_argument("--min-pts", type=int, default=3, help="точек в клетке, чтобы считать её препятствием")
    p.add_argument("--reach", type=float, default=2.5, help="сколько метров вокруг траектории брать в карту")
    p.add_argument("--cam-radius", type=float, default=0.18, help="радиус, который камера точно проехала: свободно всегда")
    p.add_argument("--roam", type=float, default=1.2, help="коридор вокруг маршрута, где можно ездить, если там ничего не видно (стены и мебель блокируют всё равно)")
    a = p.parse_args()

    root = pathlib.Path(a.dir); rec = root / "recon"; out = root / "map"; out.mkdir(exist_ok=True)
    poses = json.loads((rec / "poses.json").read_text())
    kf = poses["frames"]
    pts = np.load(rec / "points.npz")
    xyz = pts["xyz"]
    cams = np.array([[f["x"], f["y"]] for f in kf])
    x0, y0 = cams.min(0) - a.reach; x1, y1 = cams.max(0) + a.reach
    W, H = int(math.ceil((x1 - x0) / a.res)), int(math.ceil((y1 - y0) / a.res))

    def to_cell(x, y):
        return ((x - x0) / a.res).astype(int), ((y - y0) / a.res).astype(int)

    inside = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
    band = inside & (xyz[:, 2] >= a.z_min) & (xyz[:, 2] <= a.z_max)
    floor = inside & (xyz[:, 2] >= -0.05) & (xyz[:, 2] < a.z_min)
    obs = np.zeros((H, W), np.int32); fl = np.zeros((H, W), np.int32)
    cx, cy = to_cell(xyz[band, 0], xyz[band, 1]); np.add.at(obs, (cy, cx), 1)
    cx, cy = to_cell(xyz[floor, 0], xyz[floor, 1]); np.add.at(fl, (cy, cx), 1)

    grid = np.full((H, W), 128, np.uint8)                       # неизвестно
    grid[fl >= 1] = 255                                          # пол виден — свободно
    grid[obs >= a.min_pts] = 0                                   # препятствие
    # где проехала камера — точно свободно, что бы ни говорил шум точек;
    # коридор пошире вокруг маршрута открываем только там, где ничего не видно
    # (неизвестно), реконструированные препятствия он не стирает
    yy, xx = np.mgrid[0:H, 0:W]
    r_hard, r_roam = a.cam_radius / a.res, a.roam / a.res
    for x, y in cams:
        cxi, cyi = (x - x0) / a.res, (y - y0) / a.res
        d2 = (xx - cxi) ** 2 + (yy - cyi) ** 2
        grid[d2 <= r_hard * r_hard] = 255
        grid[(d2 <= r_roam * r_roam) & (grid == 128)] = 255
    # затянуть мелкие дыры в свободной зоне
    free = grid == 255
    neigh = ndimage.convolve(free.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - free
    grid[(grid == 128) & (neigh >= 5)] = 255

    img = Image.fromarray(grid[::-1])                           # север сверху
    img.save(out / "map.png")

    spawn = {"x": kf[0]["x"], "y": kf[0]["y"], "yaw": kf[0]["yaw"]}
    cam_h = json.loads((rec / "report.json").read_text()).get("cam_height_m", 0.12) if (rec / "report.json").exists() else 0.12
    meta = {"res": a.res, "origin": [float(x0), float(y0)], "width": W, "height": H, "cam_height": cam_h,
            "camera": poses["camera"], "frames_dir": "../frames", "spawn": spawn,
            "keyframes": [{"i": i, "name": f["name"], "x": f["x"], "y": f["y"], "yaw": f["yaw"]} for i, f in enumerate(kf)]}
    (out / "map.json").write_text(json.dumps(meta, indent=1))

    # --- картинка для глаз -----------------------------------------------------
    S = 4
    view = Image.new("RGB", (W * S, H * S), (40, 44, 52))
    px = view.load(); g = grid[::-1]
    col = {128: (40, 44, 52), 255: (225, 228, 232), 0: (20, 20, 24)}
    for yy_ in range(H):
        for xx_ in range(W):
            c = col[int(g[yy_, xx_])]
            for dy in range(S):
                for dx in range(S):
                    px[xx_ * S + dx, yy_ * S + dy] = c
    d = ImageDraw.Draw(view)
    def P(x, y): return ((x - x0) / a.res * S, (H - (y - y0) / a.res) * S)
    scene_f = root / "scene.json"; rep_f = rec / "report.json"
    if scene_f.exists() and rep_f.exists() and "gt" in json.loads(rep_f.read_text()):
        sim = json.loads(rep_f.read_text())["gt"]["sim_to_gt"]
        s, R, t = sim["s"], np.array(sim["R"]), np.array(sim["t"])
        def gt2est(x, y):                                        # est = R^T (gt - t) / s
            v = R.T @ (np.array([x, y, 0.12]) - t) / s
            return v[0], v[1]
        sc = json.loads(scene_f.read_text())
        Wm, Dm, _ = sc["room"]
        for poly in [[(0, 0), (Wm, 0), (Wm, Dm), (0, Dm)]] + [
                [(o["min"][0], o["min"][1]), (o["max"][0], o["min"][1]), (o["max"][0], o["max"][1]), (o["min"][0], o["max"][1])]
                for o in sc["obstacles"]]:
            pts_ = [P(*gt2est(x, y)) for x, y in poly]
            d.polygon(pts_, outline=(240, 80, 80))
    d.line([P(x, y) for x, y in cams], fill=(80, 140, 255), width=2)
    for f in kf:
        x, y = P(f["x"], f["y"]); d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(80, 140, 255))
    sx, sy = P(spawn["x"], spawn["y"]); d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], outline=(80, 220, 120), width=2)
    view.save(out / "map_view.png")

    n_obs, n_free, n_unk = (grid == 0).sum(), (grid == 255).sum(), (grid == 128).sum()
    print(f"карта {W}x{H} клеток по {a.res} м ({(x1 - x0):.1f}x{(y1 - y0):.1f} м)")
    print(f"препятствий {n_obs}, свободно {n_free}, неизвестно {n_unk}  |  точек в полосе {band.sum()}, на полу {floor.sum()}")
    print(f"кадров-ключей {len(kf)}, спавн ({spawn['x']:.2f}, {spawn['y']:.2f})")
    print(f"готово: {out}/map.png, map.json, map_view.png")

if __name__ == "__main__":
    main()
