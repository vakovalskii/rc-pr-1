"""
Полигон как среда в памяти: без сна, без сокета, без JPEG — шаг за миллисекунды.
Ровно та же машинка и та же картинка, что идут в пульт, только в N раз быстрее
и в N процессов. Это дорога к RL: миллион шагов здесь — минуты, а не сутки.

    .venv/bin/python env.py data/synth --bench                 # сколько шагов в секунду
    .venv/bin/python env.py data/synth --eval models/bc.pt     # прогнать политику без релея

API в духе gymnasium:
    env = PolygonEnv("data/synth", obstacles=4)
    obs = env.reset()                      # uint8 (120,160,3)
    obs, reward, done, info = env.step(steer, throttle)

Награда: продвижение по маршруту в метрах, минус 1 за удар, минус штраф за
дёрганый руль. Эпизод кончается после laps кругов, 5 ударов или max_steps.
"""
import argparse, math, time
import numpy as np
from polygon_car import PolygonCar

class PolygonEnv:
    def __init__(self, data_dir, obstacles=4, obs=(160, 120), dt=1 / 15, max_steps=2000, laps=1,
                 max_collisions=5, seed=None, zoom=1.4, max_speed=1.2):
        self.car = PolygonCar(data_dir, (640, 480), zoom, max_speed, 0.2, 30.0, 0.12, False, obstacles, seed)
        self.obs_size, self.dt, self.max_steps, self.laps_target, self.max_coll = obs, dt, max_steps, laps, max_collisions
        W = np.array([[k["x"], k["y"]] for k in self.car.kf])
        self.arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))])
        self.W = W
        if seed is not None:
            import random; random.seed(seed); np.random.seed(seed)
        self.n = 0

    def _progress(self):
        d = np.linalg.norm(self.W - np.array([self.car.x, self.car.y]), axis=1)
        near = int(d.argmin())
        if self.near <= near <= self.near + 20:
            self.near = near
        elif near < 5 and self.near > len(self.W) - 25:                # замкнули круг
            self.laps += 1; self.near = near
        return self.laps * self.arc[-1] + self.arc[self.near]

    def reset(self):
        self.car.reset()
        self.n, self.laps, self.near, self.prev_steer = 0, 0, 0, 0.0
        self.prog = self._progress()
        return self.obs()

    def obs(self):
        return np.asarray(self.car.render_image(self.obs_size))

    def step(self, steer, throttle):
        self.car.apply({"type": "cmd", "steer": steer, "throttle": throttle, "id": self.n})
        coll0 = self.car.collisions
        self.car.step(self.dt)
        self.n += 1
        prog = self._progress()
        hit = self.car.collisions - coll0
        reward = (prog - self.prog) - 1.0 * hit - 0.02 * abs(steer - self.prev_steer)
        self.prog, self.prev_steer = prog, steer
        done = self.laps >= self.laps_target or self.car.collisions >= self.max_coll or self.n >= self.max_steps
        info = {"x": self.car.x, "y": self.car.y, "yaw": self.car.yaw, "speed": self.car.speed,
                "collision": bool(hit), "collisions": self.car.collisions, "odo": self.car.odometer,
                "progress": prog, "laps": self.laps, "kf": self.car.cur_kf, "t": self.car.t,
                "obstacles": [[o["x"], o["y"], o["r"]] for o in self.car.obstacles]}
        return self.obs(), reward, done, info

def bench(data_dir, steps=600):
    env = PolygonEnv(data_dir, obstacles=4); env.reset()
    t0 = time.perf_counter()
    for i in range(steps):
        _, _, done, _ = env.step(math.sin(i / 20) * 0.5, 0.6)
        if done: env.reset()
    dt = time.perf_counter() - t0
    print(f"{steps} шагов за {dt:.2f} с = {steps / dt:.0f} шагов/с в одном процессе (стенд через релей даёт 15)")

def evaluate(data_dir, model_path, episodes, obstacles, max_throttle):
    import torch
    from train_bc import BCNet, device
    dev = device(); ck = torch.load(model_path, map_location="cpu")
    net = BCNet(); net.load_state_dict(ck["state"]); net.to(dev).eval()
    env = PolygonEnv(data_dir, obstacles=obstacles, laps=2)
    tot_prog, tot_hits, t0 = 0.0, 0, time.perf_counter()
    for ep in range(episodes):
        obs, done, R = env.reset(), False, 0.0
        stall, back_until, back_steer = 0, -1, 0.0
        while not done:
            x = torch.from_numpy(obs).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
            with torch.no_grad(): s, t = net(x)[0].tolist()
            s, t = max(-1, min(1, s)), max(-1, min(max_throttle, t if t > 0 else t * 0.3))
            # тот же рефлекс упора, что в pilot.py: секунда без движения -> секунда назад
            stall = stall + 1 if (t > 0.1 and abs(env.car.speed) < 0.05) else 0
            if stall > 15 and env.n > back_until:
                back_until, back_steer, stall = env.n + 15, (1.0 if s <= 0 else -1.0), 0
            if env.n < back_until: s, t = back_steer, -0.5
            obs, r, done, info = env.step(s, t)
            R += r
        tot_prog += info["progress"]; tot_hits += info["collisions"]
        print(f"  эпизод {ep + 1}: кругов {info['laps']}, продвижение {info['progress']:.1f} м, ударов {info['collisions']}, "
              f"шагов {env.n}, награда {R:.1f}")
    n = episodes
    print(f"итог {model_path}: в среднем {tot_prog / n:.1f} м и {tot_hits / n:.1f} ударов за эпизод, {time.perf_counter() - t0:.1f} с")
    log_eval({"kind": "env", "model": str(model_path), "obstacles": obstacles, "episodes": n,
              "progress_m": round(tot_prog / n, 2), "collisions": round(tot_hits / n, 2), "seconds": round(time.perf_counter() - t0, 1)})

def log_eval(rec):
    """История прогонов политики — для страницы /train.html."""
    import json, pathlib
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    p = pathlib.Path("models"); p.mkdir(exist_ok=True)
    with (p / "evals.jsonl").open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--eval", help="путь к модели: прогнать политику в среде")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--obstacles", type=int, default=4)
    p.add_argument("--max-throttle", type=float, default=0.6)
    a = p.parse_args()
    if a.bench: bench(a.dir)
    if a.eval: evaluate(a.dir, a.eval, a.episodes, a.obstacles, a.max_throttle)
