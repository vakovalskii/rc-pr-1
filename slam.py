"""
3D-карта места по записи RealSense (кнопка «запись 3D» в пульте): RGBD-одометрия Open3D
(против ключевого кадра, CPU), облако из ключевых кадров → позы, облако, сетка → формат recon/ для map_build.py.

    rsync -a pi@192.168.1.109:rec/20261020-011500/ data/room/raw/
    uv run --python 3.12 --with open3d --with scipy --with pillow python slam.py data/room
    python3 map_build.py data/room

Вход: <dir>/raw/{depth/*.png, color/*.jpg, intrinsics.json} — глубина уже выровнена по цветной камере.
Выход в <dir>/recon/: poses.json, points.npz, report.json (как у recon.py), cloud.ply;
<dir>/frames — ссылка на цветные кадры.

Масштаб метрический сразу (это глубина, не COLMAP). Система координат как у recon.py:
«вверх» — нормаль к плоскости траектории, пол — RANSAC (или --cam-height), z вверх, пол z=0, первая камера в (0,0) и смотрит вдоль +x.
"""
import argparse, json, math, pathlib, time
import numpy as np
import open3d as o3d

def _deg(M):
    """Угол поворота 4x4 в градусах."""
    return math.degrees(math.acos(max(-1, min(1, (np.trace(M[:3, :3]) - 1) / 2))))

class Tracker:
    """Одометрия против ключевого кадра (а не соседнего) — меньше копится дрейф. Новый ключ — когда
    отъехали на key_m или повернулись на key_deg. Начальное приближение — движение прошлого шага.
    Шаг больше max_step_m / max_step_deg за кадр считаем ошибкой (было: одна плохая оценка разогнала
    модель скорости, и поза за минуту «уехала» на 236 м). Потеряли трекинг — стоим на месте; потеряли reset_after раз подряд (уехали далеко от ключа) —
    текущий кадр становится новым ключом в старой позе: карта не обрывается, но здесь возможен скачок."""
    def __init__(self, cam_i, depth_max, key_m=0.10, key_deg=10, reset_after=5, max_step_m=0.25, max_step_deg=25):
        self.cam_i, self.key_m, self.key_deg, self.reset_after = cam_i, key_m, key_deg, reset_after
        self.max_step_m, self.max_step_deg = max_step_m, max_step_deg
        self.opt = o3d.pipelines.odometry.OdometryOption(depth_diff_max=0.07, depth_min=0.2, depth_max=depth_max)
        self.jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self.T, self.key, self.Tkey, self.dT = np.eye(4), None, np.eye(4), np.eye(4)
        self.poses, self.lost, self.miss, self.jumps = [], 0, 0, 0

    def step(self, rgbd):
        """→ (поза камеры в мире первого кадра 4x4, ключевой ли это кадр — его точки надо добавить в облако)."""
        if self.key is None:
            self.key = rgbd; self.poses.append(self.T.copy()); return self.T, True
        init = np.linalg.inv(self.Tkey) @ self.T @ self.dT
        ok, rel, _ = o3d.pipelines.odometry.compute_rgbd_odometry(rgbd, self.key, self.cam_i, init, self.jac, self.opt)
        if ok:   # шаг больше, чем машинка может проехать за кадр, — ошибка одометрии, а не движение
            step = np.linalg.inv(self.T) @ (self.Tkey @ rel)
            ok = np.linalg.norm(step[:3, 3]) <= self.max_step_m and _deg(step) <= self.max_step_deg
        if not ok:
            self.dT = np.eye(4)                          # иначе плохая скорость сама себя разгоняет
            self.lost += 1; self.miss += 1; self.poses.append(self.T.copy())
            if self.miss >= self.reset_after:
                self.key, self.Tkey, self.dT, self.miss = rgbd, self.T.copy(), np.eye(4), 0; self.jumps += 1
            return self.T, False
        self.miss = 0
        Tn = self.Tkey @ rel; self.dT = np.linalg.inv(self.T) @ Tn; self.T = Tn
        self.poses.append(self.T.copy())
        d = np.linalg.inv(self.Tkey) @ self.T
        if np.linalg.norm(d[:3, 3]) > self.key_m or _deg(d) > self.key_deg:
            self.key, self.Tkey = rgbd, self.T.copy(); return self.T, True
        return self.T, False

    def correct(self, T_new):
        """Поправка позы снаружи (плоскость, скан-матчинг): сдвигаем и ключевой кадр — одометрия дальше считает от исправленной позы."""
        D = T_new @ np.linalg.inv(self.T)
        self.T, self.Tkey = T_new.copy(), D @ self.Tkey
        if self.poses: self.poses[-1] = T_new.copy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--step", type=int, default=2, help="брать каждый N-й кадр (запись 15 к/с)")
    p.add_argument("--voxel", type=float, default=0.02, help="м, прореживание облака")
    p.add_argument("--depth-max", type=float, default=3.0, help="м, дальше глубина D455 на 640x480 шумная")
    p.add_argument("--pitch", type=float, default=25, help="°, на сколько камера смотрит вверх — если «вверх» не взять из траектории")
    p.add_argument("--cam-height", type=float, default=0.10, help="м, высота RealSense над полом, если пол не виден")
    a = p.parse_args()

    root = pathlib.Path(a.dir); raw = root / "raw"; out = root / "recon"; out.mkdir(exist_ok=True)
    K = json.loads((raw / "intrinsics.json").read_text())
    dfiles = sorted((raw / "depth").glob("*.png"))[::a.step]
    names = [f.stem + ".jpg" for f in dfiles]
    cam_i = o3d.camera.PinholeCameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    scale = 1.0 / K["depth_scale"]
    pcd = o3d.geometry.PointCloud()      # ScalableTSDFVolume в open3d 0.20 на маке пустой — копим облака ключевых кадров

    def load(i):
        c = o3d.io.read_image(str(raw / "color" / names[i])); d = o3d.io.read_image(str(dfiles[i]))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(c, d, depth_scale=scale, depth_trunc=a.depth_max,
                                                                  convert_rgb_to_intensity=False)

    tr, t0 = Tracker(cam_i, a.depth_max), time.time()
    for i in range(len(dfiles)):
        cur = load(i)
        T, is_key = tr.step(cur)
        if is_key:                                       # ключевой кадр: его точки — в облако
            pcd += o3d.geometry.PointCloud.create_from_rgbd_image(cur, cam_i, np.linalg.inv(T))
            pcd = pcd.voxel_down_sample(a.voxel)
        if i % 50 == 0: print(f"  кадр {i}/{len(dfiles)}, потеряно {tr.lost}, {time.time() - t0:.0f} с")
    poses, lost = tr.poses, tr.lost

    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    xyz = np.asarray(pcd.points); rgb = np.asarray(pcd.colors)

    # --- «вверх»: машинка едет по полу, значит центры камеры лежат в плоскости, параллельной ему
    # (камера на D455 смотрит заметно вверх, пол в кадре бывает только краем). Короткая запись без
    # разъезда — берём «вверх» первой камеры.
    P = np.array([q[:3, 3] for q in poses])
    th = math.radians(a.pitch); cam_up = np.array([0, -math.cos(th), math.sin(th)])   # камера смотрит вверх на pitch
    ev, evec = np.linalg.eigh(np.cov((P - P.mean(0)).T)) if len(P) > 3 else (np.zeros(3), np.eye(3))
    if ev[2] > 0.3 ** 2 and ev[1] > 0.15 ** 2 and ev[0] < 0.05 ** 2:     # разъезд в две стороны, по высоте ровно
        n = evec[:, 0] * (1 if evec[:, 0] @ cam_up > 0 else -1); up_src = "траектория"
    else:
        n = cam_up; up_src = f"наклон камеры {a.pitch}° (разъезда мало или траектория не плоская)"
    # пол: плоскость с этой нормалью ниже камеры на 3–40 см; не нашли — высота камеры из параметра
    rest, D, floor_src = pcd.voxel_down_sample(0.03), a.cam_height, "параметр --cam-height"
    for _ in range(6):
        if len(rest.points) < 300: break
        (A, B, C, d), idx = rest.segment_plane(0.02, 3, 1000)
        m = np.array([A, B, C]); sg = 1 if m @ n > 0 else -1
        if m @ n * sg > math.cos(math.radians(15)) and 0.03 < d * sg < 0.4:
            D, floor_src = float(d * sg), f"RANSAC, {len(idx)} точек"; break
        rest = rest.select_by_index(idx, invert=True)
    fwd = np.array([0, 0, 1.0]); xw = fwd - (fwd @ n) * n; xw /= np.linalg.norm(xw)
    R = np.stack([xw, np.cross(n, xw), n]); o = -D * n           # строки — оси мира в координатах первой камеры
    W = lambda P: (P - o) @ R.T

    Fw = np.array([q[:3, :3] @ fwd for q in poses]) @ R.T
    Pw = W(P)
    frames = [{"name": names[i], "x": float(Pw[i, 0]), "y": float(Pw[i, 1]), "z": float(Pw[i, 2]),
               "yaw": float(math.atan2(Fw[i, 1], Fw[i, 0]))} for i in range(len(poses))]
    xyz_w = W(xyz)
    np.savez_compressed(out / "points.npz", xyz=xyz_w.astype(np.float32), rgb=(rgb * 255).astype(np.uint8))
    pcd.points = o3d.utility.Vector3dVector(xyz_w); o3d.io.write_point_cloud(str(out / "cloud.ply"), pcd)

    path = float(np.linalg.norm(np.diff(Pw[:, :2], axis=0), axis=1).sum())
    loop = float(np.linalg.norm(Pw[-1, :2] - Pw[0, :2]))
    cam = {"model": "PINHOLE", "width": K["width"], "height": K["height"], "params": [K["fx"], K["fy"], K["cx"], K["cy"]],
           "hfov_deg": round(math.degrees(2 * math.atan(K["width"] / (2 * K["fx"]))), 1)}
    ratio = 1 - lost / max(len(dfiles) - 1, 1)
    report = {"method": "open3d rgbd odometry, keyframe cloud", "frames": len(dfiles), "lost": lost, "tracked_ratio": round(ratio, 3),
              "points": len(xyz),
              "path_m": round(path, 2), "end_to_start_m": round(loop, 2), "cam_height_m": round(float(D), 3), "floor": floor_src, "up": up_src,
              "cam_z_spread_m": round(float(Pw[:, 2].std()), 3), "camera": cam, "seconds": round(time.time() - t0)}
    report["verdict"] = ("ГОДНАЯ" if ratio >= 0.95 else "СОЙДЁТ" if ratio >= 0.8 else
                         "ПЕРЕСНЯТЬ: трекинг часто терялся — медленнее, светлее, без рывков")
    (out / "poses.json").write_text(json.dumps({"camera": cam, "frames": frames}, indent=1))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    fr = root / "frames"
    if not fr.exists(): fr.symlink_to("raw/color")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"результат: {out}/poses.json, points.npz, cloud.ply; дальше: python3 map_build.py {root}")

if __name__ == "__main__":
    main()
