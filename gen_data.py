"""
Быстрый сбор датасета: эксперт ездит в среде без сна в N процессах сразу.
То, что через релей заняло бы полчаса, здесь занимает секунды.

    .venv/bin/python gen_data.py data/synth --out synth_obst --workers 8 --laps 4 --noise 0.4

Кладёт datasets/<out>/w<k>/ (кадры 320x240 + log.jsonl) — тот же формат,
что пишет релей; train_bc.py принимает их списком: datasets/synth_obst/*
"""
import argparse, json, os, pathlib, time
from multiprocessing import Pool
import numpy as np
from PIL import Image

def worker(args):
    data_dir, out, k, laps, noise, obstacles, size = args
    from env import PolygonEnv
    from expert import Expert
    env = PolygonEnv(data_dir, obstacles=obstacles, obs=size, laps=1, max_collisions=3, max_steps=1200, seed=k)
    ex = Expert(data_dir, noise=noise, seed=k)
    d = pathlib.Path("datasets") / out / f"w{k}"; (d / "frames").mkdir(parents=True, exist_ok=True)
    log = (d / "log.jsonl").open("w")
    i, done_laps, hits, t0 = 0, 0, 0, time.time()
    for episode in range(laps * 3):                                  # эпизод = один круг с новыми конусами
        if done_laps >= laps: break
        obs = env.reset(); ex.reset(); done = False
        while not done:
            c = env.car
            steer, throttle, label, lap = ex.act(c.x, c.y, c.yaw, [[o["x"], o["y"], o["r"]] for o in c.obstacles], c.collision, c.t)
            Image.fromarray(obs).save(d / "frames" / f"{i:06d}.jpg", quality=82)
            log.write(json.dumps({"i": i, "t": round(c.t, 3), "steer": round(steer, 3), "throttle": round(throttle, 3),
                                  "steer_label": round(label, 3), "speed": round(c.speed, 2), "x": round(c.x, 3),
                                  "y": round(c.y, 3), "yaw": round(c.yaw, 3), "kf": c.cur_kf, "collision": c.collision,
                                  "collisions": c.collisions, "failsafe": False, "episode": episode, "recover": ex.recovering}) + "\n")
            obs, _, done, info = env.step(steer, throttle)
            i += 1
            if lap or info["laps"] >= 1: done = True; done_laps += 1
        hits += env.car.collisions
    log.close()
    return k, i, hits, time.time() - t0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--laps", type=int, default=4)
    p.add_argument("--noise", type=float, default=0.4)
    p.add_argument("--obstacles", type=int, default=4)
    p.add_argument("--size", type=int, nargs=2, default=(320, 240))
    a = p.parse_args()
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(worker, [(a.dir, a.out, k, a.laps, a.noise, a.obstacles, tuple(a.size)) for k in range(a.workers)])
    tot = sum(r[1] for r in res); dt = time.time() - t0
    for k, n, hits, s in res:
        print(f"  w{k}: {n} кадров, {a.laps} кругов, ударов {hits}, {s:.1f} с")
    print(f"итого {tot} кадров за {dt:.1f} с = {tot / dt:.0f} кадров/с на {a.workers} процессах -> datasets/{a.out}/")
