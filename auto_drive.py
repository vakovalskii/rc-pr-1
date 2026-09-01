"""
Эксперт-водитель в реальном времени: ведёт машинку через релей по маршруту
съёмки, объезжает конусы, при желании пишет датасет. Логика — в expert.py,
здесь только сокет и печать. Быстрый сбор данных без релея — gen_data.py.

    .venv/bin/python auto_drive.py data/synth --laps 8 --noise 0.4 --reshuffle --record zal
"""
import argparse, asyncio, json, time, urllib.request
import websockets
from expert import Expert
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost мимо системного прокси

async def main(a):
    ex = Expert(a.dir, a.lookahead, a.gain, a.throttle, a.noise)
    http = a.url.replace("ws://", "http://").split("/ws/")[0]
    if a.record:
        print(NOPROXY.open(f"{http}/rec/start?name={a.record}").read().decode())
    tele = {}
    async with websockets.connect(a.url, max_size=8 << 20, proxy=None) as ws:
        await ws.send(json.dumps({"type": "hello", "role": "pilot"}))
        await ws.send(json.dumps({"type": "reset"}))
        async def rx():
            async for msg in ws:
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("type") == "tele": tele.update(m)
        asyncio.create_task(rx())
        cid, t0, last_rep = 0, time.monotonic(), 0.0
        while ex.laps < a.laps and time.monotonic() - t0 < a.timeout:
            cid += 1
            if "x" in tele:
                steer, throttle, label, lap = ex.act(tele["x"], tele["y"], tele["yaw"], tele.get("obstacles", []),
                                                    tele.get("collision", False), time.monotonic())
                if lap:
                    print(f"круг {ex.laps} за {time.monotonic() - t0:.0f} с, пробег {tele.get('odo')} м, удары {tele.get('collisions')}")
                    if ex.laps >= a.laps: break
                    if a.reshuffle: await ws.send(json.dumps({"type": "reset"}))   # новые конусы на новый круг
            else:
                steer, throttle, label = 0.0, 0.0, 0.0
            await ws.send(json.dumps({"type": "cmd", "id": cid, "steer": round(steer, 3), "throttle": round(throttle, 3),
                                      "steer_label": round(label, 3), "recover": ex.recovering}))
            if time.monotonic() - last_rep > 3 and "x" in tele:
                last_rep = time.monotonic()
                print(f"  ({tele['x']:.2f},{tele['y']:.2f}) цель {ex.target}/{len(ex.W)} руль {steer:+.2f} газ {throttle:.2f} "
                      f"v {tele.get('speed')} удары {tele.get('collisions')}")
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "cmd", "id": cid + 1, "steer": 0, "throttle": 0}))
    if a.record:
        print(NOPROXY.open(f"{http}/rec/stop").read().decode())
    print(f"итог: кругов {ex.laps}, пробег {tele.get('odo')} м, ударов {tele.get('collisions')}, {time.monotonic() - t0:.0f} с")

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
    p.add_argument("--reshuffle", action="store_true", help="после каждого круга сброс на старт: конусы перетасуются")
    asyncio.run(main(p.parse_args()))
