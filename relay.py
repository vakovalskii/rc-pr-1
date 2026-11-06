"""
Релей: сводит машинку и пульт. Единственная точка, доступная обоим.

Машинка сама подключается сюда (исходящее соединение) — поэтому CGNAT
оператора не мешает: внутрь машинки стучаться не нужно.

    машинка ──WS /ws/car──► релей ◄──WS /ws/pult── браузер
"""
import argparse, asyncio, base64, collections, fcntl, io, struct, termios, hmac, ipaddress, json, socket, time, pathlib, datetime
import aiohttp
from aiohttp import web, WSMsgType

WEB = pathlib.Path(__file__).parent / "web"
DATASETS = pathlib.Path(__file__).parent / "datasets"
MODELS = pathlib.Path(__file__).parent / "models"
# что из телеметрии машинки кладём рядом с кадром в датасет
TELE_KEYS = ("speed", "pos", "yaw", "x", "y", "kf", "collision", "collisions", "failsafe")

def lowat(request, n=16 << 10):
    """Не держать в ядре секунды данных к зрителю: на медленном канале (отдача 4G ~1 Мбит/с) буфер отправки,
    который ядро само растит до мегабайт, копил 3–5 с видео. Маленький SO_SNDBUF (ядро удваивает: ~48 КБ ≈ 0.35 с)
    + NOTSENT_LOWAT: send() сразу ждёт, зритель попадает в busy, лишние кадры пропускаются, а не копятся."""
    sock = request.transport.get_extra_info("socket") if request.transport else None
    if sock is not None:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 24 << 10)
            sock.setsockopt(socket.IPPROTO_TCP, 25, n)   # TCP_NOTSENT_LOWAT
        except OSError: pass

SHRINK = {1: ((480, 360), 50), 2: ((256, 192), 30)}   # уровни сжатия для зрителя из интернета; 0 — кадр как с камеры

def shrink(jpg, size=(256, 192), q=30):
    """Кадр для зрителя из интернета: 640x480 ~10 КБ → 256x192 q30 ~1.7 КБ (~10 мс на Pi 4). Для руления хватает,
    а 20 к/с — это ~0.3 Мбит/с, четверть отдачи 4G: очередь не копится, задержка ≈ пинг."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(jpg)); im.draft("RGB", size)
        if im.size != size: im = im.convert("RGB").resize(size)
        out = io.BytesIO(); im.save(out, "JPEG", quality=q, optimize=True); return out.getvalue()
    except Exception: return jpg

class Box:
    """Ящик одного зрителя: последний кадр и последняя телеметрия (свежие вытесняют старые) + очередь прочих
    сообщений. Шлёт одна задача по порядку: прочее, телеметрия, кадр. На канале 1 Мбит/с отставание — один кадр."""
    def __init__(self, ws, on_fail, sock=None, small=False):
        self.ws, self.on_fail, self.sock, self.small = ws, on_fail, sock, small
        self.frame = self.tele = None; self.other = collections.deque(maxlen=200)
        self.ev = asyncio.Event(); self.task = asyncio.ensure_future(self.run())

    def put(self, msg, binary):
        if binary: self.frame = msg
        elif msg.startswith('{"type": "tele"'): self.tele = msg
        else: self.other.append(msg)
        self.ev.set()

    async def run(self):
        try:
            while True:
                await self.ev.wait(); self.ev.clear()
                while self.other or self.tele is not None or self.frame is not None:
                    if self.other: await self.ws.send_str(self.other.popleft())
                    elif self.tele is not None: t, self.tele = self.tele, None; await self.ws.send_str(t)
                    elif self.outq() > 3000: await asyncio.sleep(0.01)   # прошлый кадр ещё в пути — ждём, кадр тем временем свежеет
                    else:
                        f, self.frame = self.frame, None
                        if self.small: f = await asyncio.get_running_loop().run_in_executor(None, shrink, f, *SHRINK[self.small])
                        await self.ws.send_bytes(f)
        except asyncio.CancelledError: pass
        except Exception: self.on_fail()

    def outq(self):
        """Сколько байт ещё не подтвердил зритель (отправлено + не отправлено) — TIOCOUTQ."""
        if self.sock is None: return 0
        try: return struct.unpack("i", fcntl.ioctl(self.sock.fileno(), termios.TIOCOUTQ, b"\0\0\0\0"))[0]
        except OSError: return 0

    def close(self): self.task.cancel()

class Hub:
    def __init__(self):
        self.car = None
        self.pults, self.boxes = set(), {}
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

    def add_pult(self, ws, sock=None, small=False):
        self.boxes[ws] = Box(ws, lambda: self.pults.discard(ws), sock, small)
        self.pults.add(ws)

    def drop_pult(self, ws):
        self.pults.discard(ws)
        b = self.boxes.pop(ws, None)
        if b: b.close()

    async def to_pults(self, msg, binary=False):
        # Каждому пульту — свой ящик (Box): медленный пульт (4G, плохой Wi-Fi) не тормозит остальных,
        # а пока он занят, кадр и телеметрия заменяются свежими — очередь не копится.
        for ws in list(self.pults):
            b = self.boxes.get(ws)
            if b: b.put(msg, binary)

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
    await ws.prepare(request); lowat(request)
    remote = not is_local(request)
    hub.add_pult(ws, request.transport.get_extra_info("socket") if request.transport else None, small=2 if remote else 0)
    hub.boxes[ws].put(json.dumps({"type": "view", "remote": remote, "shrink": hub.boxes[ws].small}), False)
    if modem["last"]: hub.boxes[ws].put(json.dumps(modem["last"], ensure_ascii=False), False)
    print(f"[pult] подключился, всего {len(hub.pults)}")
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
            except Exception:
                continue
            if m.get("type") == "view":                   # {"type":"view","shrink":0|1|2} — сжатие кадров только этому пульту
                b = hub.boxes.get(ws)
                if b and m.get("shrink") in (0, 1, 2):
                    b.small = m["shrink"]; b.put(json.dumps({"type": "view", "remote": remote, "shrink": b.small}), False)
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
        hub.drop_pult(ws)
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
                depth_src["port"] = port
                depth_src["msg"] = json.dumps({"type": "depthsrc", "url": f"ws://{request.remote}:{port}/"})
                print("[depth] прямой поток:", depth_src["msg"])
                for v in list(depth_viewers): asyncio.ensure_future(_depth_send(v, depth_src["msg"], False))
                continue
            for v in list(depth_viewers):
                if binary and v in depth_busy: continue     # медленный зритель: кадр пропускаем, очередь не копим
                asyncio.ensure_future(_depth_send(v, msg.data, binary))
    finally:
        print("[depth] камера глубины отвалилась")
        depth_src["msg"] = None; depth_src["port"] = None
        for v in list(depth_viewers):
            asyncio.ensure_future(_depth_send(v, json.dumps({"type": "depth", "online": False}), False))
    return ws

async def ws_depthdirect(request):                 # прямой поток Pi 4 через релей: для пульта из интернета (туннель, https)
    if not depth_src.get("port"): raise web.HTTPServiceUnavailable(text="камера глубины не на связи")
    ws = web.WebSocketResponse(max_msg_size=8 << 20)
    await ws.prepare(request); lowat(request)
    try:
        async with aiohttp.ClientSession() as s, s.ws_connect(f"ws://127.0.0.1:{depth_src['port']}/", max_msg_size=8 << 20) as up:
            busy, remote, last = set(), not is_local(request), [0.0]
            sock = request.transport.get_extra_info("socket") if request.transport else None
            gate = Box(ws, lambda: None, sock); gate.task.cancel()     # только ради outq()
            async def send(data):
                try:
                    if remote: data = await asyncio.get_running_loop().run_in_executor(None, shrink, data)
                    await ws.send_bytes(data)
                finally: busy.clear()
            async def down():
                async for m in up:
                    if ws.closed: break
                    if m.type == WSMsgType.BINARY:
                        if busy: continue                    # зритель не успевает — кадр пропускаем
                        # из интернета (отдача 4G ~1 Мбит/с): картинка глубины 256x192, не чаще ~7 к/с и только
                        # когда прошлая дошла — иначе она забивает канал и тормозит видео и управление
                        if remote and (time.time() - last[0] < 0.14 or gate.outq() > 3000): continue
                        busy.add(1); last[0] = time.time(); asyncio.ensure_future(send(m.data))
                    elif m.type == WSMsgType.TEXT: await ws.send_str(m.data)
            t = asyncio.ensure_future(down())
            async for m in ws:
                if m.type == WSMsgType.TEXT: await up.send_str(m.data)
            t.cancel()
    except Exception as e: print("[depth] прямой поток через релей:", e)
    return ws

async def ws_depthview(request):                   # зрители: пульты
    ws = web.WebSocketResponse()
    await ws.prepare(request); lowat(request)
    depth_viewers.add(ws)
    if depth_src["msg"]: await ws.send_str(depth_src["msg"])
    try:
        async for _ in ws: pass
    finally:
        depth_viewers.discard(ws)
    return ws

# --- живая 3D-карта: live_slam.py (на маке) шлёт сюда куски облака и позу, страницы /map3d.html смотрят.
# Куски облака копим, чтобы вкладка, открытая посреди проезда, сразу получила всё, что уже собрано.
# Сетка проходимости (b"G") — только последняя, и она же пишется в папку карты: map.html и минимапа видят её сразу.
map3d = {"chunks": [], "bytes": 0, "pose": None, "grid": None, "plan": None, "path": [], "gen": 0, "rec": None, "viewers": set(), "dir": None}
MAP3D_MAX = 20 << 20                               # ~2 млн точек

def save_grid(b):
    if not map3d["dir"]: return
    L = int.from_bytes(b[1:5], "little"); d = pathlib.Path(map3d["dir"])
    (d / "map.png").write_bytes(b[5 + L:]); (d / "map.json").write_bytes(b[5:5 + L])

class M3View:
    """Зритель карты. Облако — журнал кусков: у каждого зрителя свой курсор, при обрыве и переподключении досылается
    ровно то, чего у него нет (страница сообщает gen и сколько кусков уже получила). Поза, сетка, план — только свежие:
    после провала канала незачем догонять старые."""
    def __init__(self, ws, gen, have):
        self.ws, self.gen, self.cur = ws, gen, have
        self.grid = self.pose = self.plan = None; self.grid_t = 0.0; self.ev = asyncio.Event(); self.ev.set()

    async def run(self, small):
        ws = self.ws
        while not ws.closed:
            await self.ev.wait(); self.ev.clear()
            if self.gen != map3d["gen"]:                 # новая карта (или зритель с чужой/старой) — с чистого листа
                self.gen, self.cur, self.grid = map3d["gen"], 0, None
                await ws.send_str(json.dumps({"type": "reset", "rec": map3d["rec"], "gen": self.gen}))
                if map3d["path"]: await ws.send_str(json.dumps({"type": "path", "p": map3d["path"]}))
            while self.cur < len(map3d["chunks"]) and self.gen == map3d["gen"]:
                await ws.send_bytes(map3d["chunks"][self.cur]); self.cur += 1
                await self._fresh(small)                 # между кусками — свежая поза, чтобы машинка на карте не стояла
            await self._fresh(small)
            if map3d["grid"] is not self.grid and (not small or time.time() - self.grid_t > 2):
                self.grid, self.grid_t = map3d["grid"], time.time(); await ws.send_bytes(self.grid)
            elif map3d["grid"] is not self.grid: asyncio.get_running_loop().call_later(2, self.ev.set)

    async def _fresh(self, small):
        for k in ("pose", "plan"):
            if map3d[k] is not None and map3d[k] is not getattr(self, k):
                setattr(self, k, map3d[k]); await self.ws.send_str(map3d[k])

async def ws_map3d(request):                       # источник: live_slam.py
    ws = web.WebSocketResponse(max_msg_size=32 << 20)
    await ws.prepare(request)
    map3d["producer"] = ws
    print("[map3d] SLAM подключился")
    async for msg in ws:
        binary = msg.type == WSMsgType.BINARY
        if not binary and msg.type != WSMsgType.TEXT: continue
        if binary and msg.data[:1] == b"G":
            map3d["grid"] = msg.data
            try: save_grid(msg.data)
            except Exception as e: print("[map3d] не записал карту:", e)
        elif binary:
            if map3d["bytes"] + len(msg.data) > MAP3D_MAX: continue    # дальше не копим
            map3d["chunks"].append(msg.data); map3d["bytes"] += len(msg.data)
        else:
            m = json.loads(msg.data)
            if m.get("type") == "reset":
                map3d.update(chunks=[], grid=None, bytes=0, pose=None, plan=None, path=[], rec=m.get("rec"), gen=map3d["gen"] + 1)
            if m.get("type") == "pose":
                map3d["pose"] = msg.data; t = m["T"]; q = [round(t[3], 3), round(t[7], 3), round(t[11], 3)]
                P = map3d["path"]
                if not P or sum((q[i] - P[-3 + i]) ** 2 for i in range(3)) > 0.02 ** 2: P.extend(q)
            if m.get("type") == "plan": map3d["plan"] = msg.data
        for v in list(map3d["viewers"]): v.ev.set()
    if map3d.get("producer") is ws: map3d["producer"] = None
    print("[map3d] SLAM отключился")
    return ws

async def ws_map3dview(request):                   # зрители: /map3d.html?gen=&have= — что у страницы уже есть
    ws = web.WebSocketResponse()
    await ws.prepare(request); lowat(request)
    try: gen, have = int(request.query.get("gen", -1)), int(request.query.get("have", 0))
    except ValueError: gen, have = -1, 0
    if gen != map3d["gen"] or not 0 <= have <= len(map3d["chunks"]): gen, have = -1, 0
    v = M3View(ws, gen, have); map3d["viewers"].add(v)
    task = asyncio.ensure_future(v.run(not is_local(request)))
    try:
        async for msg in ws:                       # кнопки автопилота и «езжай сюда» — в автопилот (live_slam/autopilot)
            if msg.type == WSMsgType.TEXT and map3d.get("producer") is not None:
                try: await map3d["producer"].send_str(msg.data)
                except Exception: pass
    finally:
        map3d["viewers"].discard(v); task.cancel()
    return ws

async def map3d_page(request):
    return web.FileResponse(WEB / "map3d.html")

async def index(request):
    return web.FileResponse(WEB / "pult.html")

async def map_page(request):
    return web.FileResponse(WEB / "map.html")

async def train_page(request):
    return web.FileResponse(WEB / "train.html")

# Сигнал 4G: модем ZTE (MF79U) отдаёт его без пароля через веб-API. Опрашиваем раз в 5 с и рассылаем пультам.
MODEM_URL = ("http://192.168.0.1/goform/goform_get_cmd_process?multi_data=1&isTest=false"
             "&cmd=network_type,network_provider,signalbar,lte_rsrp,lte_rsrq,lte_snr,lte_rssi,ppp_status")
modem = {"last": None}

async def modem_poll(app):
    async def loop():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as s:
            while True:
                m = None
                try:
                    async with s.get(MODEM_URL, headers={"Referer": "http://192.168.0.1/index.html"}) as r:
                        j = json.loads(await r.text())
                    if j.get("network_type") is not None:          # это наш ZTE, а не чужой роутер на .0.1
                        num = lambda k: float(j[k]) if str(j.get(k, "")).lstrip("-").replace(".", "", 1).isdigit() else None
                        m = {"type": "modem", "net": j.get("network_type"), "op": j.get("network_provider"),
                             "bars": int(j["signalbar"]) if str(j.get("signalbar", "")).isdigit() else None,
                             "rsrp": num("lte_rsrp"), "rsrq": num("lte_rsrq"), "snr": num("lte_snr"), "rssi": num("lte_rssi"),
                             "online": j.get("ppp_status") == "ppp_connected", "t": round(time.time())}
                except Exception: pass
                modem["last"] = m
                await hub.to_pults(json.dumps(m or {"type": "modem", "net": None}, ensure_ascii=False))
                await asyncio.sleep(5)
    app["modem_task"] = asyncio.ensure_future(loop())

async def status(request):
    s = hub.stats
    dt = max(time.time() - s["since"], 1e-6)
    return web.json_response({
        "car_online": hub.car is not None,
        "pults": len(hub.pults),
        "pilot": bool(hub.pilots),
        "modem": modem["last"],
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

@web.middleware
async def no_cache(request, handler):
    """Страницы и карта всегда свежие: после выкладки телефон не должен показывать старый пульт из кэша."""
    resp = await handler(request)
    path = request.path
    if path == "/" or path.endswith((".html", ".json", ".png")) and not path.startswith("/models"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

# Пароль на пульт: через туннель в интернет страница открыта всем, кто знает адрес. Без пароля пускаем только
# свои: плату по USB-кабелю (10.55.0.x — машинка), сам хост и домашние сети. Интернет приходит через L2TP с 10.66.6.1.
AUTH = {"pass": None}
LOCAL = tuple(ipaddress.ip_network(n) for n in ("10.55.0.0/24", "127.0.0.0/8", "192.168.0.0/16"))   # кабель, сам хост, домашние Wi-Fi

def is_local(request):
    """Свой: плата по кабелю, локальный процесс или домашняя сеть — без прокси-заголовков и не через туннель в офис."""
    if any(h in request.headers for h in ("Cf-Connecting-Ip", "X-Forwarded-For", "X-Real-Ip")): return False
    try: return any(ipaddress.ip_address(request.remote) in n for n in LOCAL)
    except ValueError: return False

def auth_token():
    return hmac.new(AUTH["pass"].encode(), b"rc-pult", "sha256").hexdigest()

def authed(request):
    if hmac.compare_digest(request.cookies.get("rc_auth", ""), auth_token()): return True
    want = "Basic " + base64.b64encode(f"rc:{AUTH['pass']}".encode()).decode()   # для скриптов: curl -u rc:пароль
    return hmac.compare_digest(request.headers.get("Authorization", ""), want)

def login_page(err=""):
    return web.Response(text=(WEB / "login.html").read_text().replace("__ERR__", err), content_type="text/html")

async def login(request):
    """Своя страница входа вместо окна браузера: пароль → cookie на 30 дней."""
    if request.method == "POST":
        pw = (await request.post()).get("password", "")
        if AUTH["pass"] and hmac.compare_digest(pw, AUTH["pass"]):
            resp = web.HTTPFound("/")
            resp.set_cookie("rc_auth", auth_token(), max_age=30 * 86400, httponly=True, samesite="Lax")
            raise resp
        await asyncio.sleep(1)                           # перебор паролей — не быстрее раза в секунду
        return login_page("Неверный пароль")
    return login_page()

@web.middleware
async def auth(request, handler):
    if AUTH["pass"] and request.path != "/login" and not is_local(request) and not authed(request):
        if request.path == "/" or request.path.endswith(".html"): raise web.HTTPFound("/login")
        raise web.HTTPUnauthorized(text="нужен вход: /login")
    return await handler(request)

def make_app(map_dir=None):
    app = web.Application(middlewares=[auth, no_cache])
    app.on_startup.append(modem_poll)
    routes = [
        web.get("/", index),
        web.route("*", "/login", login),
        web.get("/map.html", map_page),
        web.get("/train.html", train_page),
        web.get("/status", status),
        web.get("/rec/start", rec_start),
        web.get("/rec/stop", rec_stop),
        web.get("/ws/car", ws_car),
        web.get("/ws/pult", ws_pult),
        web.get("/ws/depth", ws_depth),
        web.get("/ws/depthview", ws_depthview),
        web.get("/ws/depthdirect", ws_depthdirect),
        web.get("/map3d.html", map3d_page),
        web.get("/ws/map3d", ws_map3d),
        web.get("/ws/map3dview", ws_map3dview),
        web.static("/web", WEB),
    ]
    if map_dir:                                  # карта полигона для минимапы и /map.html
        pathlib.Path(map_dir).mkdir(parents=True, exist_ok=True); map3d["dir"] = map_dir
        routes.append(web.static("/map", pathlib.Path(map_dir)))
    MODELS.mkdir(exist_ok=True)
    routes.append(web.static("/models", MODELS))  # кривые обучения для /train.html
    app.add_routes(routes)
    return app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--map", help="папка с map.png/map.json (из map_build.py), чтобы пульт видел карту")
    p.add_argument("--pass-file", help="файл с паролем пульта (логин rc); без него пульт открыт всем")
    a = p.parse_args()
    if a.pass_file and pathlib.Path(a.pass_file).exists(): AUTH["pass"] = pathlib.Path(a.pass_file).read_text().strip()
    print("пароль на пульт:", "включён" if AUTH["pass"] else "выключен")
    web.run_app(make_app(a.map), host="0.0.0.0", port=a.port)
