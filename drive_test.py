"""
Тест пути: сколько машинка проехала вперёд и назад — по колесу (Холл) против камеры (дальность до стены по носу).

Машинку ставим носом в ровную стену в 1.5–2.5 м, сзади свободно ~1 м. Должен работать mono_slam.py (без --explore):
он даёт дальность до стены (/state → front_d). Каждый заход: вперёд на N импульсов прямо, пауза, назад на N импульсов.

    .venv/bin/python drive_test.py --pulses 4 --runs 3

Итог по заходам: импульсы, на сколько приблизилась/отодвинулась стена по камере → см на импульс вперёд и назад.
Страховка: стена ближе --min-wall — стоп; заход не дольше 6 с; колесо 1 с не крутится — стоп.
"""
import argparse, asyncio, json, statistics, time, urllib.request
import aiohttp

def state(view):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.loads(op.open(f"{view}/state", timeout=2).read())

async def settle(view, sec=1.5):
    """Средняя дальность до стены и счётчик колеса за sec секунд стоя."""
    d, n = [], None
    t_end = time.time() + sec
    while time.time() < t_end:
        s = state(view)
        if s.get("front_d") is not None: d.append(s["front_d"])
        n = s.get("hall_n", n)
        await asyncio.sleep(0.1)
    return (statistics.median(d) if d else None), n

async def drive(ws, view, a, sign, i0):
    """Едем прямо (газ sign) до N импульсов колеса. → сколько импульсов насчитали."""
    _, n0 = await settle(view, 0.3); i = i0; t0 = last_move = time.time(); n_prev = n0
    while True:
        s = state(view); n = s.get("hall_n", n_prev)
        if n != n_prev: last_move, n_prev = time.time(), n
        done = abs((n or 0) - (n0 or 0))
        why = None
        if done >= a.pulses: why = "набрали импульсы"
        elif sign > 0 and s.get("front_d") is not None and s["front_d"] < a.min_wall: why = f"стена {s['front_d']:.2f} м — стоп"
        elif time.time() - t0 > 6: why = "6 с вышло"
        elif time.time() - last_move > 1.0 and time.time() - t0 > 1.0: why = "колесо стоит 1 с"
        if why: break
        await ws.send_str(json.dumps({"type": "cmd", "id": i, "steer": 0, "throttle": sign * a.thr})); i += 1
        await asyncio.sleep(0.05)
    for _ in range(6):
        await ws.send_str(json.dumps({"type": "cmd", "id": i, "steer": 0, "throttle": 0})); i += 1; await asyncio.sleep(0.05)
    return done, why, i

async def main(a):
    rows = []
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.ws_connect(f"{a.relay}/ws/pult") as ws:
            await ws.send_str(json.dumps({"type": "hello", "role": "pilot"}))
            await ws.send_str(json.dumps({"type": "esp", "cmd": "mode,2"}))
            i = 950000
            try:
                for r in range(a.runs):
                    for sign, name in ((1, "вперёд"), (-1, "назад")):
                        d0, _ = await settle(a.view)
                        done, why, i = await drive(ws, a.view, a, sign, i)
                        await asyncio.sleep(0.8)                         # докатилась
                        d1, _ = await settle(a.view)
                        cam = None if d0 is None or d1 is None else (d0 - d1) * sign   # вперёд: стена ближе — плюс
                        cm = None if not done or cam is None else cam / done * 100
                        rows.append((r + 1, name, done, d0, d1, cam, cm, why))
                        print(f"заход {r + 1} {name}: {done} имп ({why}); стена {d0 and round(d0, 2)} → {d1 and round(d1, 2)} м; "
                              f"по камере {cam and round(cam * 100)} см → {cm and round(cm, 1)} см/имп")
            finally:
                for _ in range(6):
                    await ws.send_str(json.dumps({"type": "cmd", "id": i, "steer": 0, "throttle": 0})); i += 1; await asyncio.sleep(0.05)
                await ws.send_str(json.dumps({"type": "esp", "cmd": "mode,1"}))
    for name in ("вперёд", "назад"):
        v = [r[6] for r in rows if r[1] == name and r[6] is not None]
        if v: print(f"{name}: {statistics.median(v):.1f} см на импульс (разброс {min(v):.1f}…{max(v):.1f}, заходов {len(v)})")
    f = [r[2] for r in rows if r[1] == "вперёд"]; b = [r[2] for r in rows if r[1] == "назад"]
    print(f"импульсов вперёд {sum(f)}, назад {sum(b)} — если колесо честное, а машинка вернулась к старту, суммы равны")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--relay", default="http://192.168.1.109:8080")
    p.add_argument("--view", default="http://localhost:8091", help="mono_slam.py: оттуда дальность до стены")
    p.add_argument("--pulses", type=int, default=4, help="импульсов за заход (2 магнита: ~13 см каждый)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--thr", type=float, default=0.8, help="газ (1.0 = потолок ESP)")
    p.add_argument("--min-wall", type=float, default=0.75, help="м от задней оси до стены — ближе не подъезжаем")
    asyncio.run(main(p.parse_args()))
