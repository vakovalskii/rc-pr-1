"""
Карта из записи «запись 3D» (~/rec/<время>/ на Pi 4: depth/*.png, color/*.jpg, frames.csv, imu.csv) — тем же
кодом, что живая карта (live_slam.Live): одометрия, плоская поза, 2D-скан-матчинг, сетка проходимости.

IMU (imu.csv, те же часы, что кадры): гравитация — акселерометр через ФНЧ (τ≈0.25 с) → «вверх» в осях камеры;
курс — гироскоп вдоль «вверх», проинтегрированный до времени кадра. Пропуски в IMU (датчик на RSUSB иногда
застывает, сторож его поднимает) просто не добавляют поворота — скан-матчинг доберёт по стенам.

    python rec_replay.py ~/rec/20261025-125909            # с IMU
    python rec_replay.py ~/rec/20261025-125909 --no-imu   # как раньше — для сравнения
    python rec_replay.py ~/rec/A ~/rec/B                  # несколько кусков подряд в одну карту

Итог: data/live-<время>-<имя записи>/map/map.png + map.json и строчка со счётчиками.
"""
import argparse, csv, io, json, math, pathlib, time
import numpy as np
from PIL import Image
from live_slam import Live

def load_imu(d):
    """→ (t_ms, up[n,3], yaw_deg[n]) на каждом замере акселерометра/гироскопа, или None, если IMU не писался."""
    f = d / "imu.csv"
    if not f.exists() or f.stat().st_size < 100: return None
    rows = [r for r in csv.reader(f.open()) if r and r[0] != "t_ms"]
    rows.sort(key=lambda r: float(r[0]))
    t_out, up_out, yaw_out = [], [], []
    up = None; yaw = 0.0; tg = None
    for r in rows:
        t = float(r[0]); v = np.array([float(r[2]), float(r[3]), float(r[4])])
        if r[1] == "a":
            up = v if up is None else up + (v - up) * 0.02
        elif up is not None:
            u = up / np.linalg.norm(up)
            if tg is not None and 0 < t - tg < 100: yaw += float(v @ u) * (t - tg) / 1000
            tg = t
        if up is not None:
            t_out.append(t); up_out.append(up.copy()); yaw_out.append(math.degrees(yaw))
    return np.array(t_out), np.array(up_out), np.array(yaw_out)

def imu_at(imu, t):
    ts, ups, yaws = imu
    i = int(np.searchsorted(ts, t)) - 1
    if i < 0 or t - ts[i] > 2000: return None          # IMU в этот момент молчал — кадр без подсказки
    up = ups[i]; g = float(np.linalg.norm(up))
    return {"up": (up / g).tolist(), "g": g, "yaw": float(yaws[i])}

def replay(live, d, step):
    intr = json.loads((d / "intrinsics.json").read_text())
    imu = load_imu(d)
    frames = [(int(r[0]), float(r[1])) for r in csv.reader((d / "frames.csv").open()) if r]
    print(f"{d.name}: кадров {len(frames)}, IMU {'есть' if imu is not None else 'нет'}")
    for k, (i, ts) in enumerate(frames[::step]):
        zp, cp = d / "depth" / f"{i:06d}.png", d / "color" / f"{i:06d}.jpg"
        if not zp.exists() or not cp.exists(): continue
        z = np.asarray(Image.open(zp))[::2, ::2]                     # как живая карта: пары 320x240
        bz = io.BytesIO(); Image.fromarray(z).save(bz, "PNG", compress_level=1)
        bc = io.BytesIO(); Image.open(cp).reduce(2).save(bc, "JPEG", quality=85)
        meta = {"w": z.shape[1], "h": z.shape[0], "fx": intr["fx"] / 2, "fy": intr["fy"] / 2, "cx": intr["cx"] / 2,
                "cy": intr["cy"] / 2, "depth_scale": intr["depth_scale"], "t": time.time(),
                "imu": imu_at(imu, ts) if imu is not None else None}
        _, _, pose = live.process(meta, bz.getvalue(), bc.getvalue())
        if k % 100 == 0: print(f"  {k}: потери {pose['lost']}, скачки {pose['jumps']}, поправок по гироскопу {pose['imu_fix']}")
    return pose

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--step", type=int, default=1, help="брать каждый N-й кадр")
    p.add_argument("--no-imu", dest="imu", action="store_false")
    a = p.parse_args()
    cfg = argparse.Namespace(voxel=0.04, depth_max=2.5, key_m=0.08, key_deg=8, pitch=25, cam_height=0.10, z_min=0.04,
                             z_max=0.35, car_radius=0.15, max_points=1_500_000, planar=True, scan=True, imu=a.imu,
                             res=0.05, grid_m=30)
    dirs = [pathlib.Path(x).expanduser() for x in a.dirs]
    live = Live(cfg, freeze_up=True); live.reset(dirs[0].name + ("" if a.imu else "-noimu"))
    t0 = time.time()
    for d in dirs: pose = replay(live, d, a.step)
    P = np.array(live.path); L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0
    sc = f", совпадение со стенами ср {np.mean(live.match_scores):.2f}" if live.match_scores else ""
    print(f"ИТОГ {'с IMU' if a.imu else 'без IMU'}: кадров {live.n}, путь {L:.1f} м, потери {pose['lost']}, скачки {pose['jumps']}, "
          f"поправок по гироскопу {pose['imu_fix']}, «вверх» {'из IMU' if live.imu_up is not None else 'по --pitch'}{sc} "
          f"| {time.time() - t0:.0f} с | карта: {live.dir / 'map' / 'map.png'}")
