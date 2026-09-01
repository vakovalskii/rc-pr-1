"""
Автопилот: подключается к релею как пульт, смотрит на кадры, рулит моделью.

    .venv/bin/python pilot.py --seconds 60 --max-throttle 0.5
    .venv/bin/python pilot.py --record auto1      # заодно пишет свой заезд в датасет

Пока едет — считает кадры/с инференса, RTT петли, пробег и удары; в конце
печатает сводку. Телефонный пульт на это время закрой: две руки на одном
руле перемешают команды.
"""
import argparse, asyncio, io, json, time, urllib.request
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost мимо системного прокси
import torch, websockets
from PIL import Image
from train_bc import BCNet, preprocess, device

async def main(a):
    dev = device()
    ck = torch.load(a.model, map_location="cpu")
    model = BCNet(); model.load_state_dict(ck["state"]); model.to(dev).eval()
    http = a.url.replace("ws://", "http://").split("/ws/")[0]
    if a.record:
        print(NOPROXY.open(f"{http}/rec/start?name={a.record}").read().decode())
    state = {"steer": 0.0, "throttle": 0.0, "frames": 0, "infer_ms": 0.0, "rtt": None,
             "x": None, "y": None, "odo": 0.0, "collisions": 0, "last": None}
    pending = {}
    async with websockets.connect(a.url, max_size=8 << 20, proxy=None) as ws:
        print(f"подключились к {a.url}, модель {a.model}, устройство {dev}")
        await ws.send(json.dumps({"type": "hello", "role": "pilot"}))   # релей перестанет слушать пустой пульт
        if a.reset: await ws.send(json.dumps({"type": "reset"}))
        t_end = time.monotonic() + a.seconds

        async def rx():
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    t0 = time.perf_counter()
                    x = preprocess(Image.open(io.BytesIO(msg))).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        s, t = model(x)[0].tolist()
                    state["steer"] = max(-1, min(1, s))
                    state["throttle"] = max(-1, min(a.max_throttle, t if t > 0 else t * 0.3))
                    state["frames"] += 1; state["infer_ms"] += (time.perf_counter() - t0) * 1000
                else:
                    m = json.loads(msg)
                    if m.get("type") != "tele": continue
                    t0 = pending.pop(m.get("ack"), None)
                    if t0: state["rtt"] = (time.perf_counter() - t0) * 1000 if state["rtt"] is None else state["rtt"] * 0.8 + (time.perf_counter() - t0) * 1000 * 0.2
                    for k in ("x", "y", "odo", "collisions"):
                        if k in m: state[k] = m[k]
                    state["last"] = m

        asyncio.create_task(rx())
        cid, t_rep = 0, time.monotonic()
        while time.monotonic() < t_end:
            cid += 1; pending[cid] = time.perf_counter()
            if len(pending) > 40: pending.clear()
            await ws.send(json.dumps({"type": "cmd", "id": cid, "steer": round(state["steer"], 3), "throttle": round(state["throttle"], 3)}))
            if time.monotonic() - t_rep > 2:
                t_rep = time.monotonic(); n = max(state["frames"], 1)
                print(f"  руль {state['steer']:+.2f} газ {state['throttle']:+.2f} | инференс {state['infer_ms'] / n:.1f} мс | "
                      f"RTT {state['rtt'] and round(state['rtt'])} мс | пробег {state['odo']} м | удары {state['collisions']}")
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "cmd", "id": cid + 1, "steer": 0, "throttle": 0}))
    if a.record:
        print(NOPROXY.open(f"{http}/rec/stop").read().decode())
    n = max(state["frames"], 1)
    print(f"\nитог за {a.seconds} с: кадров {state['frames']} ({state['frames'] / a.seconds:.1f}/с), инференс {state['infer_ms'] / n:.1f} мс, "
          f"пробег {state['odo']} м, ударов {state['collisions']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/bc.pt")
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/pult")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--max-throttle", type=float, default=0.5)
    p.add_argument("--record", help="имя датасета — записать заезд автопилота")
    p.add_argument("--reset", action="store_true", help="сначала поставить машинку на старт")
    asyncio.run(main(p.parse_args()))
