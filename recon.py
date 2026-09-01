"""
Реконструкция съёмки через COLMAP: кадры → позы камеры → облако точек.
Результат переводится в метры и в систему координат пола: z вверх, пол z=0,
первая камера в (0,0) и смотрит вдоль +x.

    python3 recon.py data/synth                       # берёт data/synth/frames
    python3 recon.py data/dom --cam-height 0.12       # высота камеры над полом при съёмке
    python3 recon.py data/dom --matcher sequential --loop   # длинная съёмка + замыкание петли

Выход в <dir>/recon/: poses.json, points.npz, report.json. В конце — вердикт:
годная съёмка или переснимать, и по каким числам.

Масштаб COLMAP не знает. Берём его из одного известного числа — высоты камеры
над полом: камера ехала на постоянной высоте, значит её центры лежат в
плоскости, параллельной полу; расстояние от этой плоскости до пола и есть
cam_height.
"""
import argparse, json, math, pathlib, shutil, subprocess, sys, time, urllib.request
import numpy as np

VOCAB_URL = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
VOCAB = pathlib.Path.home() / ".cache/colmap/vocab_tree_flickr100K_words32K.bin"

def run(cmd, log):
    print("  $", " ".join(str(c) for c in cmd[:2]), "...")
    t = time.time()
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    print(f"    {time.time() - t:.0f} с, код {r.returncode}")
    if r.returncode != 0:
        sys.exit(f"COLMAP упал, смотри {log}")

# --- разбор текстовой модели ------------------------------------------------
def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def read_model(d):
    cams = {}
    for line in (d / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        cams[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]), "params": [float(v) for v in p[4:]]}
    imgs = []
    lines = [l for l in (d / "images.txt").read_text().splitlines() if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        q = np.array([float(v) for v in p[1:5]]); t = np.array([float(v) for v in p[5:8]])
        R = qvec2rot(q)
        imgs.append({"id": int(p[0]), "name": p[9], "cam": int(p[8]), "R": R, "t": t,
                     "C": -R.T @ t, "n2d": len(lines[i + 1].split()) // 3})
    pts, rgb, err, track = [], [], [], []
    for line in (d / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        pts.append([float(p[1]), float(p[2]), float(p[3])]); rgb.append([int(p[4]), int(p[5]), int(p[6])])
        err.append(float(p[7])); track.append((len(p) - 8) // 2)
    return cams, imgs, np.array(pts), np.array(rgb, np.uint8), np.array(err), np.array(track)

def umeyama(src, dst):
    """Подобие dst ≈ s·R·src + t. Возвращает s, R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    s = np.trace(np.diag(sig) @ Dm) / (S ** 2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s

def align_to_floor(imgs, pts, err, track, cam_height, path_length=None):
    """Плоскость через центры камер → z вверх; пол → z=0; масштаб из высоты камеры."""
    C = np.stack([im["C"] for im in imgs])
    mu = C.mean(0)
    _, _, Vt = np.linalg.svd(C - mu)
    n = Vt[2]
    ups = np.stack([im["R"].T @ np.array([0, -1, 0]) for im in imgs])    # "верх" камеры в мире
    if (ups @ n).mean() < 0: n = -n
    fwd0 = imgs[0]["R"].T @ np.array([0, 0, 1.0])
    x = fwd0 - (fwd0 @ n) * n; x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Rw = np.stack([x, y, n])                                              # мир COLMAP -> пол
    good = (track >= 3) & (err <= 2.0)
    P = (pts[good] - mu) @ Rw.T
    below = P[:, 2][(P[:, 2] < 0)]
    if len(below) < 50:
        sys.exit("под камерами почти нет точек — пол не найден; камера была слишком высоко или пол без текстуры")
    # Пол — самый нижний ПЛОТНЫЙ слой точек под камерами. Сам пол под острым углом
    # часто не реконструируется, но стены и мебель стоят на нём — их низ и есть пол.
    lo = np.percentile(below, 1)
    hist, edges = np.histogram(below, bins=60, range=(lo, 0))
    dense = np.nonzero(hist >= 0.35 * hist.max())[0]
    k = dense[0]
    zf = 0.5 * (edges[k] + edges[k + 1])
    scale = cam_height / (-zf)
    if path_length:                                                        # ручной масштаб точнее любого пола
        est_len = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
        scale_p = path_length / est_len
        print(f"  масштаб по полу {1 / scale:.4f} ед/м, по длине маршрута {1 / scale_p:.4f} ед/м "
              f"(расходятся в {scale / scale_p:.2f} раз) — беру маршрут")
        scale = scale_p
        zf = -cam_height / scale
    def tf_point(p): return ((p - mu) @ Rw.T - np.array([0, 0, zf])) * scale
    def tf_rot(R): return R @ Rw.T                                          # R_cw в новом мире
    C0 = tf_point(imgs[0]["C"])
    shift = np.array([C0[0], C0[1], 0.0])
    return (lambda p: tf_point(p) - shift), tf_rot, scale, zf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="папка с frames/ (после prep_video.py или synth_room.py)")
    p.add_argument("--cam-height", type=float, default=0.12, help="высота камеры над полом при съёмке, м")
    p.add_argument("--path-length", type=float, help="сколько метров прошла камера за съёмку — задаёт масштаб надёжнее пола")
    p.add_argument("--matcher", choices=["auto", "sequential", "exhaustive"], default="auto")
    p.add_argument("--overlap", type=int, default=15)
    p.add_argument("--loop", action="store_true", help="замыкание петли через vocab tree (скачает ~150 МБ)")
    p.add_argument("--max-size", type=int, default=1600, help="длинная сторона кадра для SIFT")
    p.add_argument("--camera", default="SIMPLE_RADIAL",
                   help="SIMPLE_PINHOLE для синтетики. Только модели с ОДНИМ фокусом: при езде на постоянной высоте "
                        "без наклонов PINHOLE/OPENCV не могут оценить fy и сплющивают мир по вертикали в разы")
    p.add_argument("--gpu", action="store_true", help="есть CUDA (на маке нет)")
    p.add_argument("--fresh", action="store_true", help="снести прошлую реконструкцию")
    a = p.parse_args()

    root = pathlib.Path(a.dir); frames = root / "frames"
    files = sorted(frames.glob("*.jpg"))
    if not files: sys.exit(f"нет кадров в {frames}")
    out = root / "recon"
    if a.fresh and out.exists(): shutil.rmtree(out)
    out.mkdir(exist_ok=True)
    db, sparse, log = out / "db.db", out / "sparse", out / "colmap.log"
    gpu = "1" if a.gpu else "0"
    n = len(files)
    print(f"кадров {n}, камера {a.camera}, GPU {'да' if a.gpu else 'нет'}")

    if not (sparse / "0" / "images.bin").exists():
        if db.exists(): db.unlink()
        run(["colmap", "feature_extractor", "--database_path", db, "--image_path", frames,
             "--ImageReader.single_camera", "1", "--ImageReader.camera_model", a.camera,
             "--FeatureExtraction.use_gpu", gpu, "--FeatureExtraction.max_image_size", a.max_size], log)
        matcher = a.matcher if a.matcher != "auto" else ("exhaustive" if n <= 250 else "sequential")
        if matcher == "exhaustive":
            run(["colmap", "exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", gpu], log)
        else:
            cmd = ["colmap", "sequential_matcher", "--database_path", db, "--FeatureMatching.use_gpu", gpu,
                   "--SequentialMatching.overlap", a.overlap, "--SequentialMatching.quadratic_overlap", "1"]
            if a.loop:
                if not VOCAB.exists():
                    VOCAB.parent.mkdir(parents=True, exist_ok=True)
                    print("  качаю vocab tree ...")
                    try: urllib.request.urlretrieve(VOCAB_URL, VOCAB)
                    except Exception as e: print("  не скачался, петля без loop detection:", e)
                if VOCAB.exists():
                    cmd += ["--SequentialMatching.loop_detection", "1", "--SequentialMatching.vocab_tree_path", VOCAB]
            run(cmd, log)
        sparse.mkdir(exist_ok=True)
        run(["colmap", "mapper", "--database_path", db, "--image_path", frames, "--output_path", sparse], log)
    models = sorted([d for d in sparse.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not models: sys.exit("mapper не собрал ни одной модели — съёмка не годится (мало перекрытия или текстуры)")
    txt = out / "txt"
    best, best_n = None, -1
    for m in models:
        t = txt / m.name; t.mkdir(parents=True, exist_ok=True)
        run(["colmap", "model_converter", "--input_path", m, "--output_path", t, "--output_type", "TXT"], log)
        k = sum(1 for l in (t / "images.txt").read_text().splitlines() if l and not l.startswith("#")) // 2
        if k > best_n: best, best_n = t, k
    cams, imgs, pts, rgb, err, track = read_model(best)
    imgs.sort(key=lambda im: im["name"])
    cam = cams[imgs[0]["cam"]]
    fx = cam["params"][0]

    tf_point, tf_rot, scale, zf = align_to_floor(imgs, pts, err, track, a.cam_height, a.path_length)
    poses = []
    for im in imgs:
        C = tf_point(im["C"]); R = tf_rot(im["R"])
        fwd = R.T @ np.array([0, 0, 1.0])
        poses.append({"name": im["name"], "x": float(C[0]), "y": float(C[1]), "z": float(C[2]),
                      "yaw": float(math.atan2(fwd[1], fwd[0])), "R": R.tolist(), "n2d": im["n2d"]})
    good = (track >= 2) & (err <= 3.0)
    P = tf_point(pts[good])
    np.savez_compressed(out / "points.npz", xyz=P.astype(np.float32), rgb=rgb[good], err=err[good], track=track[good])

    zs = np.array([p["z"] for p in poses])
    report = {
        "frames": n, "registered": len(imgs), "registered_ratio": round(len(imgs) / n, 3),
        "models": len(models), "points": int(good.sum()), "mean_reproj_px": round(float(err.mean()), 3),
        "mean_track": round(float(track.mean()), 2), "scale_units_per_m": round(1 / scale, 4),
        "cam_height_m": a.cam_height, "cam_z_spread_m": round(float(zs.std()), 4),
        "camera": {"model": cam["model"], "width": cam["width"], "height": cam["height"], "params": cam["params"],
                   "hfov_deg": round(math.degrees(2 * math.atan(cam["width"] / (2 * fx))), 1)},
    }
    gt_file = root / "poses_gt.json"
    if gt_file.exists():
        gt = {f["name"]: f for f in json.loads(gt_file.read_text())["frames"]}
        est = np.array([[p["x"], p["y"], p["z"]] for p in poses if p["name"] in gt])
        ref = np.array([[gt[p["name"]]["x"], gt[p["name"]]["y"], gt[p["name"]]["z"]] for p in poses if p["name"] in gt])
        s, R, t = umeyama(est, ref)
        rmse = float(np.sqrt((((est @ R.T) * s + t - ref) ** 2).sum(1).mean()))
        report["gt"] = {"rmse_m": round(rmse, 4), "scale_check": round(float(s), 4),
                        "sim_to_gt": {"s": float(s), "R": R.tolist(), "t": t.tolist()}}
    ratio, rp = report["registered_ratio"], report["mean_reproj_px"]
    if ratio >= 0.9 and rp <= 1.0 and len(models) == 1: verdict = "ГОДНАЯ: почти все кадры сели, ошибка меньше пикселя"
    elif ratio >= 0.6 and rp <= 1.5: verdict = "СОЙДЁТ: часть кадров выпала — пройдись по комнате ещё раз, медленнее"
    else: verdict = "ПЕРЕСНЯТЬ: мало кадров зарегистрировалось или ошибка велика — смаз, мало текстуры или рывки"
    report["verdict"] = verdict
    (out / "poses.json").write_text(json.dumps({"camera": report["camera"], "frames": poses}, indent=1))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print()
    print(f"зарегистрировано {len(imgs)}/{n} кадров ({ratio * 100:.0f}%), моделей {len(models)}")
    print(f"точек {report['points']}, ошибка репроекции {rp} px, длина трека {report['mean_track']}")
    print(f"масштаб: {report['scale_units_per_m']} ед. COLMAP на метр, разброс высоты камер {report['cam_z_spread_m']} м")
    if "gt" in report:
        print(f"против правды: RMSE позиций {report['gt']['rmse_m']} м, масштаб по высоте камеры ошибся в {report['gt']['scale_check']:.3f} раз")
    print(f"вердикт: {verdict}")
    print(f"результат: {out}/poses.json, points.npz, report.json")

if __name__ == "__main__":
    main()
