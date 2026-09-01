"""
Машинка, которая едет по твоей реальной съёмке.

Кадры настоящие — значит конвейер зрения видит ровно ту фактуру, тени и шум,
которые будут в бою. Это ещё не свободный полигон (съехать с траектории съёмки
нельзя), но пиксели честные.

  газ  -> движение вперёд/назад ВДОЛЬ отснятого маршрута
  руль -> поворот взгляда: окно кадрирования едет внутри широкого кадра

    python3 video_car.py --frames data/dom/frames
"""
import argparse, asyncio, io, json, pathlib, time
import websockets
from PIL import Image

class VideoCar:
    def __init__(self, frames, out_w, out_h, zoom, yaw_rate, speed_fps):
        self.files = sorted(pathlib.Path(frames).glob("*.jpg"))
        if not self.files:
            raise SystemExit(f"в {frames} нет кадров — сначала прогони prep_video.py")
        self.W, self.H = Image.open(self.files[0]).size
        self.out = (out_w, out_h)
        # окно кадрирования: уже исходника, разница и есть запас на поворот взгляда
        self.cw = int(self.W / zoom)
        self.ch = int(self.cw * out_h / out_w)
        self.ch = min(self.ch, self.H)
        self.pan_max = (self.W - self.cw) / 2      # предел поворота взгляда, пиксели
        self.yaw_rate, self.speed_fps = yaw_rate, speed_fps
        self.pos = 0.0            # позиция вдоль маршрута, в кадрах
        self.yaw = 0.0            # -1..1, куда смотрим
        self.speed = 0.0
        self.steer = self.throttle = 0.0
        self.last_cmd_at, self.last_cmd_id = 0.0, -1
        self.failsafe = True
        self._cache = (None, None)

    def apply(self, cmd):
        self.steer = max(-1.0, min(1.0, float(cmd.get("steer", 0))))
        self.throttle = max(-1.0, min(1.0, float(cmd.get("throttle", 0))))
        self.last_cmd_id = cmd.get("id", -1)
        self.last_cmd_at = time.monotonic()

    def step(self, dt):
        if time.monotonic() - self.last_cmd_at > 0.5:      # тот же failsafe
            self.throttle = self.steer = 0.0
            self.failsafe = True
        else:
            self.failsafe = False
        self.speed += (self.throttle * self.speed_fps - self.speed) * min(dt * 3.0, 1.0)
        self.pos = max(0.0, min(len(self.files) - 1.0, self.pos + self.speed * dt))
        # руль крутит взгляд, отпустил — взгляд плавно возвращается вперёд
        self.yaw += self.steer * self.yaw_rate * dt
        if abs(self.steer) < 0.02:
            self.yaw *= max(0.0, 1.0 - dt * 1.5)
        self.yaw = max(-1.0, min(1.0, self.yaw))

    def render(self, quality):
        idx = int(round(self.pos))
        if self._cache[0] != idx:
            self._cache = (idx, Image.open(self.files[idx]).convert("RGB"))
        img = self._cache[1]
        cx = self.W / 2 + self.yaw * self.pan_max
        left = int(cx - self.cw / 2)
        top = int((self.H - self.ch) / 2)
        crop = img.crop((left, top, left + self.cw, top + self.ch)).resize(self.out, Image.BILINEAR)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

async def run(url, car, fps, quality):
    async with websockets.connect(url, max_size=8 << 20, proxy=None) as ws:
        print(f"подключились к {url}")
        print(f"кадров {len(car.files)}, исходник {car.W}x{car.H}, окно {car.cw}x{car.ch}, "
              f"запас поворота ±{int(car.pan_max)} px")

        async def rx():
            async for raw in ws:
                try:
                    car.apply(json.loads(raw))
                except Exception as e:
                    print("плохая команда:", e)

        asyncio.create_task(rx())
        period, last, seq = 1.0 / fps, time.monotonic(), 0
        while True:
            now = time.monotonic()
            car.step(now - last); last = now
            await ws.send(car.render(quality))
            await ws.send(json.dumps({
                "type": "tele", "seq": seq, "ack": car.last_cmd_id,
                "speed": round(car.speed, 2), "failsafe": car.failsafe,
                "pos": round(car.pos, 1), "yaw": round(car.yaw, 2),
                "of": len(car.files) - 1,
            }))
            seq += 1
            await asyncio.sleep(max(0.0, period - (time.monotonic() - now)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/car")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--quality", type=int, default=60)
    p.add_argument("--zoom", type=float, default=1.6,
                   help="во сколько раз окно уже кадра; больше = шире обзор по сторонам")
    p.add_argument("--yaw-rate", type=float, default=1.2)
    p.add_argument("--speed", type=float, default=12.0, help="кадров/с на полном газу")
    a = p.parse_args()
    car = VideoCar(a.frames, a.width, a.height, a.zoom, a.yaw_rate, a.speed)
    asyncio.run(run(a.url, car, a.fps, a.quality))
