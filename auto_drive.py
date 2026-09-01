"""
Эксперт-водитель: ведёт машинку по путевым точкам (кадрам съёмки) через релей,
как будто это очень аккуратный человек с пультом. Нужен для двух вещей:
проверить физику и столкновения полигона без рук и нагенерить датасет
«кадр → команда» для поведенческого клонирования.

    .venv/bin/python auto_drive.py data/synth --laps 2 --record synth_expert

Управление — pure pursuit: смотрим на точку маршрута в lookahead метрах
впереди и рулим на неё. Знает позицию из телеметрии полигона.
"""
import argparse, asyncio, json, math, pathlib, random, time, urllib.request
# без прокси: системный HTTPS_PROXY ловит даже localhost и отвечает 502
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
import numpy as np
import websockets

def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

async def main(a):
    meta = json.loads((pathlib.Path(a.dir) / "map" / "map.json").read_text())
    W = np.array([[k["x"], k["y"]] for k in meta["keyframes"]])
    http = a.url.replace("ws://", "http://").split("/ws/")[0]
    if a.record:
        print(NOPROXY.open(f"{http}/rec/start?name={a.record}").read().decode())
    tele = {}
    async with websockets.connect(a.url, max_size=8 << 20, proxy=None) as ws:
        await ws.send(json.dumps({"type": "reset"}))
        async def rx():
            async for msg in ws:
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("type") == "tele": tele.update(m)
        asyncio.create_task(rx())
        target, laps, cid, t0 = 1, 0, 0, time.monotonic()
        stuck_until, stuck_steer = 0.0, 0.0
        noise_until, noise_val = 0.0, 0.0
        last_rep = 0
        while laps < a.laps and time.monotonic() - t0 < a.timeout:
            cid += 1
            if "x" in tele:
                pos = np.array([tele["x"], tele["y"]])
                # ближайшая точка маршрута, затем lookahead вперёд по маршруту
                d = np.linalg.norm(W - pos, axis=1)
                near = int(d.argmin())
                if target <= near <= target + 20: target = near      # петля замкнута: не прыгать в конец
                j = target
                while j + 1 < len(W) and np.linalg.norm(W[j] - pos) < a.lookahead:
                    j += 1
                if j >= len(W) - 1 and np.linalg.norm(W[-1] - pos) < a.lookahead:
                    laps += 1; target = 1; j = 1
                    print(f"круг {laps} за {time.monotonic() - t0:.0f} с, пробег {tele.get('odo')} м, удары {tele.get('collisions')}")
                    if laps >= a.laps: break
                target = j
                goal = W[j]
                err = angdiff(math.atan2(goal[1] - pos[1], goal[0] - pos[0]), tele["yaw"])
                steer = max(-1.0, min(1.0, -err * a.gain))          # цель слева (err>0) -> руль влево (минус)
                throttle = a.throttle * (1.0 - 0.5 * min(1.0, abs(err) / 0.6))
                # упёрлись: сдаём назад с обратным рулём секунду, потом снова вперёд
                if tele.get("collision") and not stuck_until:
                    stuck_until = time.monotonic() + 1.0; stuck_steer = -steer
                if stuck_until:
                    if time.monotonic() < stuck_until: steer, throttle = stuck_steer, -0.5
                    else: stuck_until = 0.0
                # возмущения: иногда дёргаем руль в сторону, чтобы в датасете были
                # выезды с траектории; меткой остаётся чистая команда эксперта
                label = steer
                now = time.monotonic()
                if a.noise > 0 and not stuck_until:
                    if now > noise_until + 1.5 and random.random() < 0.04:
                        noise_until = now + random.uniform(0.3, 0.7); noise_val = random.choice([-1, 1]) * a.noise
                    if now < noise_until: steer = max(-1.0, min(1.0, steer + noise_val))
            else:
                steer, throttle, label = 0.0, 0.0, 0.0
            await ws.send(json.dumps({"type": "cmd", "id": cid, "steer": round(steer, 3), "throttle": round(throttle, 3),
                                      "steer_label": round(label, 3)}))
            if time.monotonic() - last_rep > 3 and "x" in tele:
                last_rep = time.monotonic()
                print(f"  ({tele['x']:.2f},{tele['y']:.2f}) цель {target}/{len(W)} руль {steer:+.2f} газ {throttle:.2f} "
                      f"v {tele.get('speed')} удары {tele.get('collisions')}")
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "cmd", "id": cid + 1, "steer": 0, "throttle": 0}))
    if a.record:
        print(NOPROXY.open(f"{http}/rec/stop").read().decode())
    print(f"итог: кругов {laps}, пробег {tele.get('odo')} м, ударов {tele.get('collisions')}, {time.monotonic() - t0:.0f} с")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/pult")
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--lookahead", type=float, default=0.35)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--throttle", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--record", help="имя датасета для записи")
    p.add_argument("--noise", type=float, default=0.0, help="амплитуда случайных дёрганий руля (0.4 — норм для датасета)")
    asyncio.run(main(p.parse_args()))
