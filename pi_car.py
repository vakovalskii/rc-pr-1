"""
Настоящая машинка на Raspberry Pi 4: камера + руль + ход тем же протоколом,
что fake_car.py и polygon_car.py, поэтому релей, пульт, запись датасета и
автопилот работают без изменений.

    python3 pi_car.py --url ws://192.168.1.84:8080/ws/car          # на Pi
    python3 pi_car.py --dry-run --url ws://127.0.0.1:8080/ws/car   # на маке, без железа
    python3 pi_car.py --neutral                                    # руль в центр, газ в ноль и выйти
    python3 pi_car.py --esp auto --url ws://...                    # ход и руль через ESP8266 по USB

Что делает:
  * камера через picamera2, аппаратный MJPEG, полный угол сенсора (1296x972 -> 640x480);
  * ШИМ 50 Гц аппаратный (dtoverlay pwm-2chan): ESC на GPIO12, серва на GPIO13;
  * FAILSAFE: нет команд 500 мс или пропала связь -> газ в нейтраль, руль в центр;
  * при старте 2 с нейтрали — ESC без неё не взводится;
  * газ ограничен --max-throttle (по умолчанию 0.3): первые выезды без сюрпризов;
  * в телеметрии температура SoC и недовольтаж — главная беда Pi 4 на батарее.

Два способа рулить:
  * --esp: ESP8266 (firmware/rc_esp) по USB. Pi шлёт ему `c,id,руль,газ` 50 раз в секунду,
    а ESP сам делает рампу газа, руль по потенциометру и свой failsafe 300 мс —
    если Pi зависнет или выдернут кабель, машинка встанет без участия Linux;
  * без --esp: аппаратный ШИМ малины. Он переживает смерть процесса и держит
    последнее значение, поэтому systemd после остановки вызывает `pi_car.py --neutral`
    (см. pi/rc-car.service).
"""
import argparse, asyncio, io, json, math, pathlib, subprocess, time

# ----------------------------------------------------------------- ШИМ ---
class HwPwm:
    """Аппаратный ШИМ через sysfs. pwm-2chan: канал 0 = GPIO12, канал 1 = GPIO13."""
    PERIOD = 20_000_000                                   # 50 Гц, в наносекундах

    def __init__(self, chip=None):
        base = pathlib.Path("/sys/class/pwm")
        chips = sorted(base.glob("pwmchip*")) if chip is None else [base / f"pwmchip{chip}"]
        chips = [c for c in chips if int((c / "npwm").read_text()) >= 2]
        if not chips:
            raise SystemExit("нет pwmchip с двумя каналами: в /boot/firmware/config.txt нужен "
                             "dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4 и перезагрузка")
        self.chip = chips[0]
        for ch in (0, 1):
            d = self.chip / f"pwm{ch}"
            if not d.exists():
                (self.chip / "export").write_text(str(ch))
                for _ in range(50):                        # udev выставляет права не сразу
                    if (d / "period").exists(): break
                    time.sleep(0.02)
            if int((d / "period").read_text() or 0) != self.PERIOD:
                (d / "duty_cycle").write_text("0")
                (d / "period").write_text(str(self.PERIOD))
            (d / "enable").write_text("1")

    def set_us(self, ch, us):
        (self.chip / f"pwm{ch}" / "duty_cycle").write_text(str(int(us * 1000)))

class FakePwm:
    def __init__(self): self.us = {0: 1500, 1: 1500}
    def set_us(self, ch, us): self.us[ch] = round(us)

ESC, SERVO = 0, 1                                         # каналы: GPIO12 ход, GPIO13 руль

class Drive:
    def __init__(self, pwm, a):
        self.pwm, self.a = pwm, a
        self.steer = self.throttle = 0.0

    def us(self):
        a = self.a
        s = -self.steer if a.steer_invert else self.steer
        t = -self.throttle if a.esc_invert else self.throttle
        t = max(-a.max_reverse, min(a.max_throttle, t))
        return 1500 + a.esc_trim + t * 500, 1500 + a.steer_trim + s * a.steer_range

    def apply(self, steer, throttle):
        self.steer, self.throttle = steer, throttle
        esc_us, servo_us = self.us()
        self.pwm.set_us(ESC, esc_us); self.pwm.set_us(SERVO, servo_us)

    def neutral(self):
        self.apply(0.0, 0.0)

class EspDrive:
    """Ход и руль через ESP8266 по USB-серийнику. Тот же интерфейс, что у Drive."""
    def __init__(self, port, a):
        import glob, threading, serial
        if port == "auto":
            ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/cu.usbserial*"))
            if not ports: raise SystemExit("ESP не найден: нет /dev/ttyUSB* — проверь USB-кабель (нужен с данными)")
            port = ports[0]
        self.port, self.a = port, a
        self.steer = self.throttle = 0.0
        self.id, self.tele = 0, {}
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout, self.ser.write_timeout = port, 115200, 0.2, 0.1
        self.ser.dtr = self.ser.rts = False                   # не дёргать reset NodeMCU при открытии
        self.ser.open()
        threading.Thread(target=self._rx, daemon=True).start()

    def _rx(self):
        # t,ack,pot,esc_us,motor,fs,mode,potL,potC,potR,inv,max,src,acc,brk
        while True:
            try: line = self.ser.readline().decode(errors="replace").strip()
            except Exception: time.sleep(0.5); continue
            p = line.split(",")
            if p[0] == "t" and len(p) >= 13:
                try: self.tele = {"esp_ack": int(p[1]), "pot": int(p[2]), "esc_us": int(p[3]), "steer_pwm": int(p[4]),
                                  "esp_failsafe": p[5] == "1", "steer_mode": int(p[6]), "esp_src": p[12]}
                except ValueError: pass
            elif line and not line.startswith("t,"):
                print(f"ESP: {line}")

    def us(self):
        return self.tele.get("esc_us", 1500), 1500

    def apply(self, steer, throttle):
        a = self.a
        self.steer, self.throttle = steer, throttle
        s = -steer if a.steer_invert else steer
        t = -throttle if a.esc_invert else throttle
        t = max(-a.max_reverse, min(a.max_throttle, t))
        self.id = (self.id + 1) % 1_000_000
        try: self.ser.write(f"c,{self.id},{round(s * 1000)},{round(t * 1000)}\n".encode())
        except Exception as e: print(f"ESP: запись не прошла ({e.__class__.__name__})")

    def neutral(self):
        self.apply(0.0, 0.0)

# --------------------------------------------------------------- камера ---
class Latest:
    """Слот на один кадр: старые выбрасываем, иначе задержка копится очередью."""
    def __init__(self, loop):
        self.loop, self.frame, self.ev = loop, None, asyncio.Event()
    def put(self, jpeg):
        def f():
            self.frame = jpeg; self.ev.set()
        self.loop.call_soon_threadsafe(f)
    async def get(self):
        await self.ev.wait(); self.ev.clear()
        return self.frame

def start_picamera(sink, a):
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput

    class Out(io.BufferedIOBase):
        def write(self, buf):
            sink.put(bytes(buf)); return len(buf)

    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (a.width, a.height), "format": "YUV420"},
        sensor={"output_size": (1296, 972)},               # биннинг всего сенсора: весь угол 160°
        controls={"FrameRate": a.fps}, buffer_count=4)
    cam.configure(cfg)
    cam.start_recording(MJPEGEncoder(bitrate=a.bitrate), FileOutput(Out()))
    return cam

async def fake_camera(sink, a, drive):
    from PIL import Image, ImageDraw
    seq = 0
    while True:
        img = Image.new("RGB", (a.width, a.height), (30, 34, 40))
        d = ImageDraw.Draw(img)
        x = a.width / 2 + drive.steer * a.width * 0.4
        d.line([a.width / 2, a.height - 20, x, a.height * 0.4], fill=(96, 165, 250), width=6)
        d.text((12, 10), f"DRY RUN  #{seq}  steer {drive.steer:+.2f}  thr {drive.throttle:+.2f}", fill=(230, 230, 235))
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=60); sink.put(buf.getvalue())
        seq += 1
        await asyncio.sleep(1 / a.fps)

# --------------------------------------------------------------- здоровье ---
def health():
    out = {}
    try: out["temp"] = round(int(pathlib.Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except Exception: pass
    try:
        v = int(subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1)
                .stdout.strip().split("=")[1], 16)
        out["undervolt"] = bool(v & 0x1)                  # прямо сейчас
        out["undervolt_seen"] = bool(v & 0x10000)          # было с момента загрузки
        out["throttled"] = bool(v & 0x4)
    except Exception: pass
    return out

# ---------------------------------------------------------------- связь ---
async def ws_connect(url):
    import websockets
    try:                                                  # proxy=None есть только в websockets >= 15
        return await websockets.connect(url, max_size=8 << 20, proxy=None, open_timeout=5)
    except TypeError:
        return await websockets.connect(url, max_size=8 << 20, open_timeout=5)

async def main(a):
    loop = asyncio.get_running_loop()
    if a.esp:
        drive = EspDrive(a.esp, a); drive.neutral()
        print(f"ход и руль: ESP8266 на {drive.port}. Нейтраль 2 с — взводим ESC ...")
    else:
        pwm = FakePwm() if a.dry_run else HwPwm(a.pwmchip)
        drive = Drive(pwm, a); drive.neutral()
        print(f"ШИМ: {'фейк' if a.dry_run else pwm.chip}. Нейтраль 2 с — взводим ESC ...")
    await asyncio.sleep(0 if a.dry_run else 2.0)

    sink = Latest(loop)
    if a.dry_run: asyncio.create_task(fake_camera(sink, a, drive))
    else: cam = start_picamera(sink, a)

    state = {"last_cmd": -1.0, "id": -1, "failsafe": True, "health": {}, "online": False}

    async def control():                                  # 50 Гц: failsafe живёт здесь, а не в сети
        while True:
            if time.monotonic() - state["last_cmd"] > 0.5 or not state["online"]:
                if not state["failsafe"]: print("FAILSAFE: нет команд — нейтраль")
                state["failsafe"] = True; drive.neutral()
            elif a.esp:
                drive.apply(drive.steer, drive.throttle)   # ESP ждёт команду чаще 300 мс, иначе свой failsafe
            await asyncio.sleep(0.02)

    async def doctor():
        while True:
            state["health"] = health()
            if state["health"].get("undervolt"): print("НЕДОВОЛЬТАЖ: питание Pi проседает")
            await asyncio.sleep(2)

    asyncio.create_task(control()); asyncio.create_task(doctor())
    backoff = 0.5
    while True:
        try:
            ws = await ws_connect(a.url)
        except Exception as e:
            print(f"релей {a.url} недоступен ({e.__class__.__name__}), повтор через {backoff:.1f} с")
            await asyncio.sleep(backoff); backoff = min(backoff * 2, 5); continue
        backoff = 0.5; state["online"] = True
        print(f"на связи с {a.url}")

        async def rx():
            async for raw in ws:
                try: m = json.loads(raw)
                except Exception: continue
                if m.get("type", "cmd") != "cmd": continue
                state["last_cmd"], state["id"], state["failsafe"] = time.monotonic(), m.get("id", -1), False
                drive.apply(max(-1.0, min(1.0, float(m.get("steer", 0)))), max(-1.0, min(1.0, float(m.get("throttle", 0)))))

        async def tx():
            seq = 0
            while True:
                jpeg = await sink.get()
                await ws.send(jpeg)
                esc_us, servo_us = drive.us()
                await ws.send(json.dumps({"type": "tele", "seq": seq, "ack": state["id"], "failsafe": state["failsafe"],
                                          "thr": round(drive.throttle, 2), "steer": round(drive.steer, 2),
                                          "esc_us": round(esc_us), "servo_us": round(servo_us),
                                          **getattr(drive, "tele", {}), **state["health"]}))
                seq += 1

        tasks = [asyncio.create_task(rx()), asyncio.create_task(tx())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending: t.cancel()
        state["online"] = False; drive.neutral()
        try: await ws.close()
        except Exception: pass
        print("связь потеряна — нейтраль, переподключаюсь")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/car")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--bitrate", type=int, default=2_000_000, help="MJPEG, бит/с")
    p.add_argument("--max-throttle", type=float, default=0.3, help="потолок газа вперёд, 0..1")
    p.add_argument("--max-reverse", type=float, default=0.3, help="потолок заднего хода, 0..1")
    p.add_argument("--steer-range", type=float, default=400, help="мкс от центра до упора руля")
    p.add_argument("--steer-trim", type=float, default=0, help="сдвиг центра руля, мкс")
    p.add_argument("--esc-trim", type=float, default=0, help="сдвиг нейтрали ESC, мкс")
    p.add_argument("--steer-invert", action="store_true")
    p.add_argument("--esc-invert", action="store_true")
    p.add_argument("--esp", help="порт ESP8266 (/dev/ttyUSB0) или auto: ход и руль через него, а не ШИМ малины")
    p.add_argument("--pwmchip", type=int, help="номер pwmchip, если автоопределение ошиблось")
    p.add_argument("--dry-run", action="store_true", help="без камеры и ШИМ: проверить протокол на маке")
    p.add_argument("--neutral", action="store_true", help="выставить нейтраль и выйти (для systemd)")
    a = p.parse_args()
    if a.neutral and a.esp:
        d = EspDrive(a.esp, a); d.neutral(); print("нейтраль отправлена в ESP")
    elif a.neutral:
        d = Drive(HwPwm(a.pwmchip), a); d.neutral(); print("нейтраль выставлена")
    else:
        asyncio.run(main(a))
