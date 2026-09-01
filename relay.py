"""
Релей: сводит машинку и пульт. Единственная точка, доступная обоим.

Машинка сама подключается сюда (исходящее соединение) — поэтому CGNAT
оператора не мешает: внутрь машинки стучаться не нужно.

    машинка ──WS /ws/car──► релей ◄──WS /ws/pult── браузер
"""
import argparse, asyncio, json, time, pathlib, datetime
from aiohttp import web, WSMsgType

WEB = pathlib.Path(__file__).parent / "web"
DATASETS = pathlib.Path(__file__).parent / "datasets"
# что из телеметрии машинки кладём рядом с кадром в датасет
TELE_KEYS = ("speed", "pos", "yaw", "x", "y", "kf", "collision", "collisions", "failsafe")

class Hub:
    def __init__(self):
        self.car = None
        self.pults = set()
        self.stats = {"frames": 0, "bytes": 0, "since": time.time()}
        self.last_cmd = {"steer": 0.0, "throttle": 0.0}   # с чем спаривать кадр
        self.rec = None                                    # {dir, log, n}

    # --- запись датасета для имитационного обучения -------------------------
    def rec_start(self, name=None):
        self.rec_stop()
        name = name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = DATASETS / name
        (d / "frames").mkdir(parents=True, exist_ok=True)
        self.rec = {"dir": d, "log": (d / "log.jsonl").open("a"), "n": 0, "name": name}
        print(f"[rec] пишу в {d}")
        return name

    def rec_stop(self):
        if self.rec:
            self.rec["log"].close()
            print(f"[rec] остановлено, кадров {self.rec['n']}")
            self.rec = None

    def rec_frame(self, jpeg, tele):
        """Кадр + команда, которая его вызвала. Формат как у DonkeyCar."""
        if not self.rec:
            return
        i = self.rec["n"]
        (self.rec["dir"] / "frames" / f"{i:06d}.jpg").write_bytes(jpeg)
        self.rec["log"].write(json.dumps({
            "i": i, "t": round(time.time(), 3),
            "steer": self.last_cmd.get("steer", 0.0),
            "throttle": self.last_cmd.get("throttle", 0.0),
            **({"steer_label": self.last_cmd["steer_label"]} if "steer_label" in self.last_cmd else {}),
            **{k: tele.get(k) for k in TELE_KEYS if k in tele},
        }, ensure_ascii=False) + "\n")
        self.rec["log"].flush()
        self.rec["n"] = i + 1

    async def to_pults(self, msg, binary=False):
        dead = []
        for ws in self.pults:
            try:
                await (ws.send_bytes(msg) if binary else ws.send_str(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.pults.discard(ws)

hub = Hub()

async def ws_car(request):
    ws = web.WebSocketResponse(max_msg_size=8 << 20)
    await ws.prepare(request)
    hub.car = ws
    pending = None
    print("[car] подключилась")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:            # кадр JPEG
                hub.stats["frames"] += 1
                hub.stats["bytes"] += len(msg.data)
                pending = msg.data
                await hub.to_pults(msg.data, binary=True)
            elif msg.type == WSMsgType.TEXT:            # телеметрия следом за кадром
                if pending is not None:
                    try:
                        hub.rec_frame(pending, json.loads(msg.data))
                    except Exception as e:
                        print("[rec] сбой записи:", e)
                    pending = None
                await hub.to_pults(msg.data)
    finally:
        hub.car = None
        print("[car] отвалилась")
    return ws

async def ws_pult(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    hub.pults.add(ws)
    print(f"[pult] подключился, всего {len(hub.pults)}")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    hub.last_cmd = json.loads(msg.data)
                except Exception:
                    pass
                if hub.car is not None:
                    await hub.car.send_str(msg.data)    # команда на борт
    finally:
        hub.pults.discard(ws)
    return ws

async def index(request):
    return web.FileResponse(WEB / "pult.html")

async def map_page(request):
    return web.FileResponse(WEB / "map.html")

async def status(request):
    s = hub.stats
    dt = max(time.time() - s["since"], 1e-6)
    return web.json_response({
        "car_online": hub.car is not None,
        "pults": len(hub.pults),
        "fps": round(s["frames"] / dt, 1),
        "kbit_s": round(s["bytes"] * 8 / dt / 1000, 1),
        "rec": hub.rec["name"] if hub.rec else None,
        "rec_frames": hub.rec["n"] if hub.rec else 0,
    })

async def rec_start(request):
    return web.json_response({"rec": hub.rec_start(request.query.get("name"))})

async def rec_stop(request):
    n = hub.rec["n"] if hub.rec else 0
    hub.rec_stop()
    return web.json_response({"stopped": True, "frames": n})

def make_app(map_dir=None):
    app = web.Application()
    routes = [
        web.get("/", index),
        web.get("/map.html", map_page),
        web.get("/status", status),
        web.get("/rec/start", rec_start),
        web.get("/rec/stop", rec_stop),
        web.get("/ws/car", ws_car),
        web.get("/ws/pult", ws_pult),
        web.static("/web", WEB),
    ]
    if map_dir:                                  # карта полигона для минимапы и /map.html
        routes.append(web.static("/map", pathlib.Path(map_dir)))
    app.add_routes(routes)
    return app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--map", help="папка с map.png/map.json (из map_build.py), чтобы пульт видел карту")
    a = p.parse_args()
    web.run_app(make_app(a.map), host="0.0.0.0", port=a.port)
