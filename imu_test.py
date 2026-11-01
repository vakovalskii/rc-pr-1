"""
Проверка IMU D455 на Pi 4: глубина + цвет 640x480@15 (как в depth_cam.py) и гироскоп/акселерометр, N секунд.
Считает дыры по часам самого датчика (между замерами гироскопа > 50 мс) — сколько курса теряем.

    python imu_test.py --mode sensor     # датчик движения открыт отдельно (как depth_cam.py сейчас)
    python imu_test.py --mode pipeline   # всё в одном pipeline с колбэком
Бэкенд выбирается тем, какой pyrealsense2 импортируется: RSUSB (PYTHONPATH=~/rs-rsusb) или колёсный (через ядро, нужны hid-sensor-*).
"""
import argparse, threading, time
import pyrealsense2 as rs

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["sensor", "pipeline"], default="pipeline")
p.add_argument("--secs", type=float, default=60)
a = p.parse_args()

st = {"g": 0, "a": 0, "frames": 0, "holes": 0, "hole_ms": 0.0, "max_ms": 0.0, "tg": None}
lock = threading.Lock()

def motion(f):
    m = f.as_motion_frame(); t = f.get_timestamp()
    with lock:
        if m.get_profile().stream_type() == rs.stream.gyro:
            st["g"] += 1
            if st["tg"] is not None and t - st["tg"] > 50:
                st["holes"] += 1; st["hole_ms"] += t - st["tg"]; st["max_ms"] = max(st["max_ms"], t - st["tg"])
            st["tg"] = t
        else: st["a"] += 1

ctx = rs.context(); dev = ctx.query_devices()[0]
print(dev.get_info(rs.camera_info.name), "прошивка", dev.get_info(rs.camera_info.firmware_version),
      "USB", dev.get_info(rs.camera_info.usb_type_descriptor), "| датчики:", [s.get_info(rs.camera_info.name) for s in dev.query_sensors()])
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
pipe = rs.pipeline(ctx)
if a.mode == "pipeline":
    cfg.enable_stream(rs.stream.accel); cfg.enable_stream(rs.stream.gyro)
    def cb(f):
        if f.is_frameset():
            with lock: st["frames"] += 1
        elif f.is_motion_frame(): motion(f)
    pipe.start(cfg, cb)
else:
    pipe.start(cfg)
    ms = next(s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "Motion Module")
    pick = {}
    for pr in ms.get_stream_profiles():
        if pr.stream_type() in (rs.stream.accel, rs.stream.gyro) and (pr.stream_type() not in pick or pr.fps() < pick[pr.stream_type()].fps()):
            pick[pr.stream_type()] = pr
    ms.open(list(pick.values())); ms.start(motion)
    def grab():
        while time.time() - t0 < a.secs:
            try: pipe.wait_for_frames(2000)
            except Exception as e: print("кадр:", e); continue
            with lock: st["frames"] += 1
    t0 = time.time(); threading.Thread(target=grab, daemon=True).start()
t0 = time.time()
while time.time() - t0 < a.secs:
    time.sleep(10)
    with lock: print(f"{time.time() - t0:4.0f} с: гиро {st['g']} акс {st['a']} кадров {st['frames']} дыр {st['holes']} потеряно {st['hole_ms']:.0f} мс (макс {st['max_ms']:.0f})")
el = time.time() - t0
print(f"ИТОГ {a.mode}: гироскоп {st['g'] / el:.0f} Гц, кадров {st['frames'] / el:.1f}/с, дыр {st['holes']}, потеряно {st['hole_ms'] / el / 10:.1f}% времени")
pipe.stop()
