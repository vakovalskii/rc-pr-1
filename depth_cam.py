"""
Камера глубины RealSense (D455 на Pi 4) -> релей -> пульт вторым окном.

    ~/rs-venv/bin/python depth_cam.py --url ws://pizero2w.local:8080/ws/depth

Шлёт два вида сообщений:
  * JPEG раскрашенной глубины (ближе — тёплое, дальше — холодное), по кадру в пути, без очереди;
  * JSON {"type":"depth", "center": м, "ahead": м, ...} — расстояние в центре кадра и
    до ближайшего препятствия в полосе по курсу (центральная треть ширины, средняя часть высоты).

pyrealsense2 под Python 3.13 (trixie) колёс нет — запускается из venv на Python 3.12 (uv).

Запись для SLAM (--color): пока существует файл ~/rc/rec.on (его ставит и убирает кнопка «запись 3D» в пульте), кадры глубины, выровненные
по цветной камере, и сами цветные кадры пишутся в ~/rec/<время>/{depth/*.png, color/*.jpg}
плюс intrinsics.json и frames.csv. Дальше — slam.py на машине с GPU.

Надёжность: камера на USB может пропасть и вернуться под новым номером — старый
pipeline тогда мёртв навсегда. Поэтому любой сбой камеры или 5 с без кадров =
выход с кодом ошибки, а systemd (rc-depth.service) перезапускает процесс без
ограничения попыток. При старте ждём камеру и делаем ей аппаратный сброс.
"""
import argparse, asyncio, io, json, os, pathlib, queue, threading, time
import numpy as np
import pyrealsense2 as rs
from PIL import Image

def wait_camera(timeout=60):
    """Ждём, пока RealSense появится на USB (после переподключения или включения)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        devs = rs.context().query_devices()
        if len(devs): return devs[0]
        time.sleep(1)
    print("RealSense не найдена на USB за минуту — выходим, systemd перезапустит"); os._exit(4)

def open_camera(a):
    dev = wait_camera()
    if a.reset:                                    # аппаратный сброс: камера после сбоя бывает «полуживой»
        print("аппаратный сброс камеры…"); dev.hardware_reset(); time.sleep(3); wait_camera()
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    if a.color: cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    prof = pipe.start(cfg)
    dev = prof.get_device()
    scale = dev.first_depth_sensor().get_depth_scale()
    global DI
    DI = prof.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    print(f"RealSense {dev.get_info(rs.camera_info.name)}, USB {usb}, глубина {a.width}x{a.height}@{a.fps}")
    return pipe, scale, usb

def colorize(z, near, far):
    """Своя палитра без OpenCV: близко — красный/оранжевый, далеко — синий, нет данных — чёрный."""
    t = np.clip((z - near) / (far - near), 0, 1)
    r = np.clip(1.6 - 2.2 * t, 0, 1); g = np.clip(1.2 - np.abs(t - .45) * 2.6, 0, 1); b = np.clip(2.0 * t - .4, 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    rgb[z <= 0] = (12, 14, 18)
    return rgb

REC_FLAG = pathlib.Path.home() / "rc" / "rec.on"

class Recorder:
    """Пишет пары (глубина по цветной камере, цвет) в отдельном потоке; не успевает диск — кадр пропускаем."""
    def __init__(self):
        self.dir, self.n, self.dropped, self.q = None, 0, 0, queue.Queue(maxsize=30)
        self.align = rs.align(rs.stream.color)
        threading.Thread(target=self._writer, daemon=True).start()

    def feed(self, fs, scale, want_raw=False):
        """Пишет кадр, если идёт запись; с want_raw ещё и возвращает уменьшенную пару для живой 3D-карты."""
        on = REC_FLAG.exists()
        if not on and self.dir: print(f"запись остановлена: {self.dir}, {self.n} кадров, пропущено {self.dropped}"); self.dir = None
        if not on and not want_raw: return None
        al = self.align.process(fs)
        d, c = al.get_depth_frame(), al.get_color_frame()
        if not d or not c: return None
        if not on: return self._raw(d, c, scale, None) if want_raw else None
        if not self.dir:
            self.dir = pathlib.Path.home() / "rec" / time.strftime("%Y%m%d-%H%M%S")
            (self.dir / "depth").mkdir(parents=True); (self.dir / "color").mkdir()
            i = c.profile.as_video_stream_profile().intrinsics
            (self.dir / "intrinsics.json").write_text(json.dumps({"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
                                                                  "cx": i.ppx, "cy": i.ppy, "depth_scale": scale}))
            self.n = self.dropped = 0; self.t0 = time.time(); print(f"запись: {self.dir}")
        try:
            self.q.put_nowait((self.dir, self.n, c.get_timestamp(), np.asanyarray(d.get_data()).copy(), np.asanyarray(c.get_data()).copy()))
            self.n += 1
        except queue.Full: self.dropped += 1
        return self._raw(d, c, scale, self.dir.name) if want_raw else None

    @staticmethod
    def _raw(d, c, scale, rec_name):
        """RGBD пополам (320x240): b"RGBD" + json интринсик + PNG глубины + JPEG цвета, длины uint32."""
        z = np.asanyarray(d.get_data())[::2, ::2].copy()
        rgb = Image.fromarray(np.asanyarray(c.get_data())).reduce(2)
        i = c.profile.as_video_stream_profile().intrinsics
        meta = json.dumps({"w": z.shape[1], "h": z.shape[0], "fx": i.fx / 2, "fy": i.fy / 2, "cx": i.ppx / 2, "cy": i.ppy / 2,
                           "depth_scale": scale, "t": time.time(), "rec": rec_name}).encode()
        bz = io.BytesIO(); Image.fromarray(z).save(bz, "PNG", compress_level=1)
        bc = io.BytesIO(); rgb.save(bc, "JPEG", quality=80)
        bz = bz.getvalue()
        return b"RGBD" + len(meta).to_bytes(4, "little") + meta + len(bz).to_bytes(4, "little") + bz + bc.getvalue()

    def _writer(self):
        while True:
            dr, i, ts, z, rgb = self.q.get()
            Image.fromarray(z).save(dr / "depth" / f"{i:06d}.png", compress_level=1)
            Image.fromarray(rgb).save(dr / "color" / f"{i:06d}.jpg", quality=92)
            with open(dr / "frames.csv", "a") as f: f.write(f"{i},{ts:.1f}\n")

DI, PROF = None, None

def profile(z, a):
    """Профиль препятствий для автопилота: 16 секторов по горизонтали, в каждом — ближняя дистанция (м)
    до точек на высоте корпуса. Камера наклонена вверх на a.pitch и стоит на a.cam_height над полом.
    Ещё доля «дыр» (нет глубины) внизу по центру: D455 ближе ~0.4 м не видит, и большая дыра там —
    признак, что препятствие уже совсем рядом."""
    global PROF
    if PROF is None:                                   # лучи пикселей считаем один раз (каждый 4-й пиксель)
        v, u = np.mgrid[0:z.shape[0]:4, 0:z.shape[1]:4]
        th = np.radians(a.pitch)
        PROF = {"rx": (u - DI.ppx) / DI.fx, "ry": (v - DI.ppy) / DI.fy, "s": np.sin(th), "c": np.cos(th)}
    P = PROF; zz = z[::4, ::4]
    Y = P["ry"] * zz
    h = a.cam_height - Y * P["c"] + zz * P["s"]        # высота над полом
    d = Y * P["s"] + zz * P["c"]                       # вперёд по полу
    x = P["rx"] * zz                                   # вбок (вправо +)
    m = (zz > 0) & (h > a.body_min) & (h < a.body_max) & (d > 0.05)
    ang = np.degrees(np.arctan2(x[m], d[m])); rng = np.hypot(x[m], d[m])
    bins = np.clip(((ang + 44) / 88 * 16).astype(int), 0, 15)
    prof = []
    for b in range(16):
        r = rng[bins == b]
        prof.append(round(float(np.percentile(r, 5)), 2) if r.size >= 8 else None)
    lo = zz[zz.shape[0] * 2 // 3:, zz.shape[1] // 4: zz.shape[1] * 3 // 4]
    return prof, round(float((lo == 0).mean()), 2)

rec = None
raw_viewers, raw_at = set(), 0.0                   # live_slam.py: уменьшенная RGBD-пара несколько раз в секунду

def grab(pipe, scale, a, filt):
    global raw_at
    fs = pipe.wait_for_frames(2000)
    want = bool(raw_viewers) and time.time() - raw_at >= 1 / a.raw_fps
    rawmsg = rec.feed(fs, scale, want) if rec else None
    if rawmsg: raw_at = time.time()
    d = fs.get_depth_frame()
    for f in filt: d = f.process(d)
    z = np.asanyarray(d.get_data()).astype(np.float32) * scale
    h, w = z.shape
    c = z[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4]; c = c[c > 0]
    band = z[int(h * .30):int(h * .75), w // 3:2 * w // 3]; band = band[(band > a.near) & (band < a.far)]
    stats = {"type": "depth", "online": True,
             "center": round(float(np.median(c)), 2) if c.size else None,
             "ahead": round(float(np.percentile(band, 3)), 2) if band.size > 50 else None,   # 3-й перцентиль: шум не считаем препятствием
             "valid": round(float((z > 0).mean()), 2), "w": w, "h": h,
             **dict(zip(("prof", "hole_lo"), profile(z, a))),
             **({"rec": rec.n, "rec_drop": rec.dropped, "rec_s": round(time.time() - rec.t0, 1)} if rec and rec.dir else {})}
    buf = io.BytesIO(); Image.fromarray(colorize(z, a.near, a.far)).save(buf, "JPEG", quality=a.quality)
    return buf.getvalue(), stats, rawmsg

last_frame = time.time()

def watchdog(limit):
    """Нет кадров дольше limit секунд — процесс завершается, systemd поднимает заново."""
    while True:
        time.sleep(1)
        if time.time() - last_frame > limit:
            print(f"СТОРОЖ: {limit} с без кадров глубины — перезапуск"); os._exit(3)

def grab_checked(pipe, scale, a, filt):
    global last_frame
    try:
        r = grab(pipe, scale, a, filt)
    except Exception as e:                         # камера отвалилась: pipeline уже не оживёт
        print(f"сбой камеры: {e.__class__.__name__}: {e} — перезапуск"); os._exit(2)
    last_frame = time.time()
    return r

viewers, busy = set(), set()

async def serve_viewer(ws):                        # пульт подключился напрямую (--serve)
    viewers.add(ws)
    try:
        async for raw in ws:                       # кнопка «запись 3D» в пульте
            try: m = json.loads(raw)
            except Exception: continue
            if m.get("type") == "raw":             # это live_slam: ему только сырые пары, без картинки
                viewers.discard(ws); raw_viewers.add(ws)
            if m.get("type") == "rec":
                if m.get("on"): REC_FLAG.touch()
                else: REC_FLAG.unlink(missing_ok=True)
    finally: viewers.discard(ws); raw_viewers.discard(ws)

async def _send(ws, data):
    busy.add(ws)
    try: await ws.send(data)
    except Exception: viewers.discard(ws); raw_viewers.discard(ws)
    finally: busy.discard(ws)

def to_viewers(jpeg, stats):
    for v in list(viewers):
        if v in busy: continue                     # медленный зритель: кадр пропускаем, очередь не копим
        asyncio.ensure_future(_send(v, jpeg)); asyncio.ensure_future(_send(v, stats))

async def main(a):
    import websockets
    global last_frame
    if a.serve:
        await websockets.serve(serve_viewer, "0.0.0.0", a.serve, max_size=1 << 16)
        print(f"отдаю кадры пультам напрямую на :{a.serve}")
    global rec
    pipe, scale, usb = open_camera(a)
    if a.color: rec = Recorder()
    last_frame = time.time()
    threading.Thread(target=watchdog, args=(a.watchdog,), daemon=True).start()
    filt = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(1)] if a.filters else []
    loop = asyncio.get_running_loop()
    while True:
        try:
            try: ws = await websockets.connect(a.url, max_size=8 << 20, proxy=None, open_timeout=5)
            except TypeError: ws = await websockets.connect(a.url, max_size=8 << 20, open_timeout=5)
            print(f"на связи с {a.url}")
            if a.serve: await ws.send(json.dumps({"type": "depthsrc", "port": a.serve}))
            n, t0, view_at = 0, time.time(), 0.0
            while True:
                jpeg, stats, rawmsg = await loop.run_in_executor(None, grab_checked, pipe, scale, a, filt)
                js_raw = None
                for v in list(raw_viewers):                # автопилот: цифры каждого кадра + сырые пары по расписанию
                    if v in busy: continue
                    js_raw = js_raw or json.dumps(stats)
                    asyncio.ensure_future(_send(v, js_raw))
                    if rawmsg: asyncio.ensure_future(_send(v, rawmsg))
                n += 1
                stats["fps"] = round(n / max(time.time() - t0, 1e-3), 1); stats["usb"] = usb
                if time.time() - t0 > 5: n, t0 = 0, time.time()
                js = json.dumps(stats)
                if a.serve:                                # картинку зрителям — не чаще view_fps: эфир 2.4 ГГц не резиновый
                    if time.time() - view_at >= 1 / a.view_fps: to_viewers(jpeg, js); view_at = time.time()
                    await ws.send(js)                      # через релей — только цифры
                else: await ws.send(jpeg); await ws.send(js)
        except Exception as e:                     # сюда попадают только сбои связи: камера падает через os._exit
            print(f"релей недоступен ({e.__class__.__name__}: {e}), повтор через 2 с")
            await asyncio.sleep(2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://pizero2w.local:8080/ws/depth")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--near", type=float, default=0.3, help="м, ближе — самый тёплый цвет")
    p.add_argument("--far", type=float, default=4.0, help="м, дальше — самый холодный")
    p.add_argument("--quality", type=int, default=70)
    p.add_argument("--filters", action="store_true", help="сглаживание и заполнение дыр (дороже по CPU)")
    p.add_argument("--watchdog", type=float, default=5, help="с без кадров до перезапуска")
    p.add_argument("--serve", type=int, default=0, help="порт: отдавать кадры пультам напрямую, релею — только адрес и цифры")
    p.add_argument("--color", action="store_true", help="включить цветную камеру и запись для SLAM по флагу ~/rc/rec.on")
    p.add_argument("--view-fps", type=float, default=8, help="к/с картинки глубины пультам (цифры — каждый кадр)")
    p.add_argument("--raw-fps", type=float, default=4, help="к/с уменьшенных RGBD-пар для живой 3D-карты (live_slam.py)")
    p.add_argument("--pitch", type=float, default=25, help="°, камера смотрит вверх (для профиля препятствий)")
    p.add_argument("--cam-height", type=float, default=0.10, help="м, высота камеры над полом")
    p.add_argument("--body-min", type=float, default=0.04, help="м, ниже — пол")
    p.add_argument("--body-max", type=float, default=0.35, help="м, выше — машинка проедет под")
    p.add_argument("--reset", action="store_true", help="аппаратный сброс камеры при старте")
    asyncio.run(main(p.parse_args()))
