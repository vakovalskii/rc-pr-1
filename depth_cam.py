"""
Камера глубины RealSense (D455 на Pi 4) -> релей -> пульт вторым окном.

    ~/rs-venv/bin/python depth_cam.py --url ws://192.168.1.137:8080/ws/depth

Шлёт два вида сообщений:
  * JPEG раскрашенной глубины (ближе — тёплое, дальше — холодное), по кадру в пути, без очереди;
  * JSON {"type":"depth", "center": м, "ahead": м, ...} — расстояние в центре кадра и
    до ближайшего препятствия в полосе по курсу (центральная треть ширины, средняя часть высоты).

pyrealsense2 под Python 3.13 (trixie) колёс нет — запускается из venv на Python 3.12 (uv).

Надёжность: камера на USB может пропасть и вернуться под новым номером — старый
pipeline тогда мёртв навсегда. Поэтому любой сбой камеры или 5 с без кадров =
выход с кодом ошибки, а systemd (rc-depth.service) перезапускает процесс без
ограничения попыток. При старте ждём камеру и делаем ей аппаратный сброс.
"""
import argparse, asyncio, io, json, os, threading, time
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
    prof = pipe.start(cfg)
    dev = prof.get_device()
    scale = dev.first_depth_sensor().get_depth_scale()
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

def grab(pipe, scale, a, filt):
    fs = pipe.wait_for_frames(2000)
    d = fs.get_depth_frame()
    for f in filt: d = f.process(d)
    z = np.asanyarray(d.get_data()).astype(np.float32) * scale
    h, w = z.shape
    c = z[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4]; c = c[c > 0]
    band = z[int(h * .30):int(h * .75), w // 3:2 * w // 3]; band = band[(band > a.near) & (band < a.far)]
    stats = {"type": "depth", "online": True,
             "center": round(float(np.median(c)), 2) if c.size else None,
             "ahead": round(float(np.percentile(band, 3)), 2) if band.size > 50 else None,   # 3-й перцентиль: шум не считаем препятствием
             "valid": round(float((z > 0).mean()), 2), "w": w, "h": h}
    buf = io.BytesIO(); Image.fromarray(colorize(z, a.near, a.far)).save(buf, "JPEG", quality=a.quality)
    return buf.getvalue(), stats

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
    try: await ws.wait_closed()
    finally: viewers.discard(ws)

async def _send(ws, data):
    busy.add(ws)
    try: await ws.send(data)
    except Exception: viewers.discard(ws)
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
    pipe, scale, usb = open_camera(a)
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
            n, t0 = 0, time.time()
            while True:
                jpeg, stats = await loop.run_in_executor(None, grab_checked, pipe, scale, a, filt)
                n += 1
                stats["fps"] = round(n / max(time.time() - t0, 1e-3), 1); stats["usb"] = usb
                if time.time() - t0 > 5: n, t0 = 0, time.time()
                js = json.dumps(stats)
                if a.serve: to_viewers(jpeg, js); await ws.send(js)       # через релей — только цифры
                else: await ws.send(jpeg); await ws.send(js)
        except Exception as e:                     # сюда попадают только сбои связи: камера падает через os._exit
            print(f"релей недоступен ({e.__class__.__name__}: {e}), повтор через 2 с")
            await asyncio.sleep(2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://192.168.1.137:8080/ws/depth")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--near", type=float, default=0.3, help="м, ближе — самый тёплый цвет")
    p.add_argument("--far", type=float, default=4.0, help="м, дальше — самый холодный")
    p.add_argument("--quality", type=int, default=70)
    p.add_argument("--filters", action="store_true", help="сглаживание и заполнение дыр (дороже по CPU)")
    p.add_argument("--watchdog", type=float, default=5, help="с без кадров до перезапуска")
    p.add_argument("--serve", type=int, default=0, help="порт: отдавать кадры пультам напрямую, релею — только адрес и цифры")
    p.add_argument("--reset", action="store_true", help="аппаратный сброс камеры при старте")
    asyncio.run(main(p.parse_args()))
