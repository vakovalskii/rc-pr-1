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
MODELS = pathlib.Path(__file__).parent / "models"
# что из телеметрии машинки кладём рядом с кадром в датасет
TELE_KEYS = ("speed", "pos", "yaw", "x", "y", "kf", "collision", "collisions", "failsafe")

class Hub:
    def __init__(self):
        self.car = None
        self.pults, self.busy = set(), set()
        self.pilots = set()                                # пульты, объявившие себя автопилотом
        self.stats = {"frames": 0, "bytes": 0, "since": time.time()}
        self.last_cmd = {"steer": 0.0, "throttle": 0.0}   # с чем спаривать кадр
        self.driver, self.driver_at = None, 0.0            # пульт, который сейчас реально рулит
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
            **{k: self.last_cmd[k] for k in ("steer_label", "recover") if k in self.last_cmd},
            **{k: tele.get(k) for k in TELE_KEYS if k in tele},
        }, ensure_ascii=False) + "\n")
        self.rec["log"].flush()
        self.rec["n"] = i + 1

    async def _send(self, ws, msg, binary):
        self.busy.add(ws)
        try: await (ws.send_bytes(msg) if binary else ws.send_str(msg))
        except Exception: self.pults.discard(ws)
        finally: self.busy.discard(ws)

    async def to_pults(self, msg, binary=False):
        # Каждому пульту — отдельно и без ожидания: медленный пульт (или плохой Wi-Fi до него)
        # не тормозит остальных, а кадр, пришедший пока он занят, пропускаем — очередь не копится.
        for ws in list(self.pults):
            if binary and ws in self.busy: continue
            asyncio.ensure_future(self._send(ws, msg, binary))

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
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
            except Exception:
                continue
            if m.get("type") == "hello":                  # {"type":"hello","role":"pilot"}
                if m.get("role") == "pilot":
                    hub.pilots.add(ws)
                    await hub.to_pults(json.dumps({"type": "pilot", "online": True}))
                continue
            # Пока едет автопилот, пустые сердцебиения человеческого пульта (нули)
            # не пропускаем — иначе они перебивают автопилот 20 раз в секунду.
            # Ненулевая команда с пульта проходит: человек перехватил управление.
            if (m.get("type") == "cmd" and ws not in hub.pilots and hub.pilots
                    and not m.get("steer") and not m.get("throttle")):
                continue
            if m.get("type", "cmd") == "cmd":
                # Несколько открытых пультов (вкладки, телефон + ПК) шлют сердцебиение нулями
                # и перебивают того, кто рулит. Нули от остальных глушим секунду после
                # последней ненулевой команды водителя.
                now = time.time()
                if m.get("steer") or m.get("throttle"):
                    hub.driver, hub.driver_at = ws, now
                elif hub.driver is not None and ws is not hub.driver and now - hub.driver_at < 1.0:
                    continue
                hub.last_cmd = m
            if hub.car is not None:
                await hub.car.send_str(msg.data)        # команда на борт
    finally:
        hub.pults.discard(ws)
        if ws in hub.pilots:
            hub.pilots.discard(ws)
            await hub.to_pults(json.dumps({"type": "pilot", "online": bool(hub.pilots)}))
    return ws

# --- камера глубины (RealSense на отдельной малине): свой поток, пульт показывает вторым окном
# Картинку Pi 4 может отдавать пультам сама (depth_cam.py --serve): тогда она шлёт сюда только
# {"type":"depthsrc","port":N}, а релей подсказывает пультам прямой адрес — кадры не идут через эфир Zero дважды.
depth_viewers, depth_busy = set(), set()
depth_src = {"msg": None}

async def _depth_send(ws, data, binary):
    depth_busy.add(ws)
    try: await (ws.send_bytes(data) if binary else ws.send_str(data))
    except Exception: depth_viewers.discard(ws)
    finally: depth_busy.discard(ws)

async def ws_depth(request):                       # источник: depth_cam.py на Pi 4
    ws = web.WebSocketResponse(max_msg_size=8 << 20)
    await ws.prepare(request)
    print("[depth] камера глубины подключилась")
    try:
        async for msg in ws:
            binary = msg.type == WSMsgType.BINARY
            if not binary and msg.type != WSMsgType.TEXT: continue
            if not binary and '"depthsrc"' in msg.data:
                port = json.loads(msg.data).get("port")
                depth_src["msg"] = json.dumps({"type": "depthsrc", "url": f"ws://{request.remote}:{port}/"})
                print("[depth] прямой поток:", depth_src["msg"])
                for v in list(depth_viewers): asyncio.ensure_future(_depth_send(v, depth_src["msg"], False))
                continue
            for v in list(depth_viewers):
                if binary and v in depth_busy: continue     # медленный зритель: кадр пропускаем, очередь не копим
                asyncio.ensure_future(_depth_send(v, msg.data, binary))
    finally:
        print("[depth] камера глубины отвалилась")
        depth_src["msg"] = None
        for v in list(depth_viewers):
            asyncio.ensure_future(_depth_send(v, json.dumps({"type": "depth", "online": False}), False))
    return ws

async def ws_depthview(request):                   # зрители: пульты
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    depth_viewers.add(ws)
    if depth_src["msg"]: await ws.send_str(depth_src["msg"])
    try:
        async for _ in ws: pass
    finally:
        depth_viewers.discard(ws)
    return ws

# --- живая 3D-карта: live_slam.py (на маке) шлёт сюда куски облака и позу, страницы /map3d.html смотрят.
# Куски облака копим, чтобы вкладка, открытая посреди проезда, сразу получила всё, что уже собрано.
map3d = {"chunks": [], "pose": None, "viewers": set()}

async def _m3_send(ws, data, binary):
    try: await (ws.send_bytes(data) if binary else ws.send_str(data))
    except Exception: map3d["viewers"].discard(ws)

async def ws_map3d(request):                       # источник: live_slam.py
    ws = web.WebSocketResponse(max_msg_size=32 << 20)
    await ws.prepare(request)
    print("[map3d] SLAM подключился")
    async for msg in ws:
        binary = msg.type == WSMsgType.BINARY
        if not binary and msg.type != WSMsgType.TEXT: continue
        if binary: map3d["chunks"].append(msg.data)
        else:
            m = json.loads(msg.data)
            if m.get("type") == "reset": map3d["chunks"].clear()
            if m.get("type") == "pose": map3d["pose"] = msg.data
        for v in list(map3d["viewers"]): asyncio.ensure_future(_m3_send(v, msg.data, binary))
    print("[map3d] SLAM отключился")
    return ws

async def ws_map3dview(request):                   # зрители: /map3d.html
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    for c in list(map3d["chunks"]): await ws.send_bytes(c)
    if map3d["pose"]: await ws.send_str(map3d["pose"])
    map3d["viewers"].add(ws)
    try:
        async for _ in ws: pass
    finally:
        map3d["viewers"].discard(ws)
    return ws

async def map3d_page(request):
    return web.FileResponse(WEB / "map3d.html")

async def index(request):
    return web.FileResponse(WEB / "pult.html")

async def map_page(request):
    return web.FileResponse(WEB / "map.html")

async def train_page(request):
    return web.FileResponse(WEB / "train.html")

async def status(request):
    s = hub.stats
    dt = max(time.time() - s["since"], 1e-6)
    return web.json_response({
        "car_online": hub.car is not None,
        "pults": len(hub.pults),
        "pilot": bool(hub.pilots),
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
        web.get("/train.html", train_page),
        web.get("/status", status),
        web.get("/rec/start", rec_start),
        web.get("/rec/stop", rec_stop),
        web.get("/ws/car", ws_car),
        web.get("/ws/pult", ws_pult),
        web.get("/ws/depth", ws_depth),
        web.get("/ws/depthview", ws_depthview),
        web.get("/map3d.html", map3d_page),
        web.get("/ws/map3d", ws_map3d),
        web.get("/ws/map3dview", ws_map3dview),
        web.static("/web", WEB),
    ]
    if map_dir:                                  # карта полигона для минимапы и /map.html
        routes.append(web.static("/map", pathlib.Path(map_dir)))
    MODELS.mkdir(exist_ok=True)
    routes.append(web.static("/models", MODELS))  # кривые обучения для /train.html
    app.add_routes(routes)
    return app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--map", help="папка с map.png/map.json (из map_build.py), чтобы пульт видел карту")
    a = p.parse_args()
    web.run_app(make_app(a.map), host="0.0.0.0", port=a.port)
