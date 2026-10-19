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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--step", type=int, default=2, help="брать каждый N-й кадр (запись 15 к/с)")
    p.add_argument("--voxel", type=float, default=0.02, help="м, прореживание облака")
    p.add_argument("--depth-max", type=float, default=3.0, help="м, дальше глубина D455 на 640x480 шумная")
    p.add_argument("--cam-height", type=float, default=0.10, help="м, высота RealSense над полом, если пол не виден")
    a = p.parse_args()

    root = pathlib.Path(a.dir); raw = root / "raw"; out = root / "recon"; out.mkdir(exist_ok=True)
    K = json.loads((raw / "intrinsics.json").read_text())
    dfiles = sorted((raw / "depth").glob("*.png"))[::a.step]
    names = [f.stem + ".jpg" for f in dfiles]
    cam_i = o3d.camera.PinholeCameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    scale = 1.0 / K["depth_scale"]
    pcd = o3d.geometry.PointCloud()      # ScalableTSDFVolume в open3d 0.20 на маке пустой — копим облака ключевых кадров
    opt = o3d.pipelines.odometry.OdometryOption(depth_diff_max=0.07, depth_min=0.2, depth_max=a.depth_max)
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    def load(i):
        c = o3d.io.read_image(str(raw / "color" / names[i])); d = o3d.io.read_image(str(dfiles[i]))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(c, d, depth_scale=scale, depth_trunc=a.depth_max,
                                                                  convert_rgb_to_intensity=False)

    # Одометрия против ключевого кадра (а не соседнего) — меньше копится дрейф. Новый ключ — когда
    # отъехали на 10 см или повернулись на 10°. Начальное приближение — движение прошлого шага.
    poses, lost, t0 = [], 0, time.time()
    T, key, Tkey, step = np.eye(4), load(0), np.eye(4), np.eye(4)
    for i in range(len(dfiles)):
        cur = load(i) if i else key
        if i:
            init = np.linalg.inv(Tkey) @ T @ step
            ok, rel, _ = o3d.pipelines.odometry.compute_rgbd_odometry(cur, key, cam_i, init, jac, opt)
            if not ok:                                   # потеряли: стоим на месте, кадр не вклеиваем
                lost += 1; poses.append(T.copy()); continue
            Tn = Tkey @ rel; step = np.linalg.inv(T) @ Tn; T = Tn
            d = np.linalg.inv(Tkey) @ T
            if not (np.linalg.norm(d[:3, 3]) > 0.10 or math.degrees(math.acos(min(1, (np.trace(d[:3, :3]) - 1) / 2))) > 10):
                poses.append(T.copy()); continue
            key, Tkey = cur, T.copy()
        poses.append(T.copy())                           # ключевой кадр: его точки — в облако
        pcd += o3d.geometry.PointCloud.create_from_rgbd_image(cur, cam_i, np.linalg.inv(T))
        pcd = pcd.voxel_down_sample(a.voxel)
        if i % 50 == 0: print(f"  кадр {i}/{len(dfiles)}, потеряно {lost}, {time.time() - t0:.0f} с")

    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    xyz = np.asarray(pcd.points); rgb = np.asarray(pcd.colors)

    # --- «вверх»: машинка едет по полу, значит центры камеры лежат в плоскости, параллельной ему
    # (камера на D455 смотрит заметно вверх, пол в кадре бывает только краем). Короткая запись без
    # разъезда — берём «вверх» первой камеры.
    P = np.array([q[:3, 3] for q in poses])
    cam_up = np.array([0, -1.0, 0])
    ev, evec = np.linalg.eigh(np.cov((P - P.mean(0)).T)) if len(P) > 3 else (np.zeros(3), np.eye(3))
    if ev[2] > 0.3 ** 2 and ev[1] > 0.15 ** 2 and ev[0] < 0.03 ** 2:     # разъезд в две стороны, по высоте ровно
        n = evec[:, 0] * (1 if evec[:, 0] @ cam_up > 0 else -1); up_src = "траектория"
    else:
        n = cam_up; up_src = "первая камера (разъезда мало)"
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
