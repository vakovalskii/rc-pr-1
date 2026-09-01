"""
Машинка-симулятор. Говорит ровно тем протоколом, которым потом заговорит
настоящая — поэтому её можно будет подменить, не трогая пульт и релей.

Что имитирует честно:
  * поток JPEG-кадров с заданным fps и разрешением;
  * простую физику (газ -> скорость, руль -> курс);
  * FAILSAFE: нет команд дольше 500 мс -> газ в ноль;
  * эхо id команды, чтобы пульт мог померить RTT петли.
"""
import asyncio, json, math, time, io, argparse
import websockets
from PIL import Image, ImageDraw

class Car:
    def __init__(self):
        self.x, self.y, self.heading = 0.0, 0.0, 0.0
        self.speed = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.last_cmd_at = 0.0
        self.last_cmd_id = -1
        self.failsafe = True

    def apply(self, cmd):
        self.steer = max(-1.0, min(1.0, float(cmd.get("steer", 0))))
        self.throttle = max(-1.0, min(1.0, float(cmd.get("throttle", 0))))
        self.last_cmd_id = cmd.get("id", -1)
        self.last_cmd_at = time.monotonic()

    def step(self, dt):
        # FAILSAFE — то же правило, что уедет в прошивку
        if time.monotonic() - self.last_cmd_at > 0.5:
            self.throttle, self.steer = 0.0, 0.0
            self.failsafe = True
        else:
            self.failsafe = False
        target = self.throttle * 3.0                 # м/с
        self.speed += (target - self.speed) * min(dt * 3.0, 1.0)
        self.heading += self.steer * self.speed * dt * 1.2
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt

    def render(self, w, h, quality):
        img = Image.new("RGB", (w, h), (26, 30, 36))
        d = ImageDraw.Draw(img)
        # горизонт едет от курса — видно, что руль реально работает
        hz = h * 0.45 + math.sin(self.heading) * h * 0.18
        d.rectangle([0, hz, w, h], fill=(38, 52, 44))
        off = (self.heading * 140) % 80
        for i in range(-1, w // 80 + 2):             # разметка убегает от скорости
            gx = i * 80 - off
            d.line([gx, hz, gx + (gx - w / 2) * 0.7, h], fill=(70, 92, 78), width=2)
        for i in range(6):
            yy = hz + (h - hz) * ((i + (time.time() * self.speed * 0.4) % 1) / 6) ** 2
            d.line([w * 0.5 - 3, yy, w * 0.5 + 3, yy], fill=(200, 200, 120), width=3)
        d.text((12, 10), f"speed {self.speed:5.2f} m/s", fill=(230, 230, 235))
        d.text((12, 26), f"steer {self.steer:+.2f}  thr {self.throttle:+.2f}", fill=(230, 230, 235))
        if self.failsafe:
            d.text((12, 44), "FAILSAFE", fill=(255, 90, 90))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

async def run(url, fps, w, h, quality):
    car = Car()
    # proxy=None — иначе websockets лезет через системный HTTPS_PROXY даже на localhost
    async with websockets.connect(url, max_size=8 << 20, proxy=None) as ws:
        print(f"подключились к {url}")

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
            car.step(now - last)
            last = now
            await ws.send(car.render(w, h, quality))          # кадр
            await ws.send(json.dumps({                        # телеметрия + эхо
                "type": "tele", "seq": seq, "ack": car.last_cmd_id,
                "speed": round(car.speed, 2), "failsafe": car.failsafe,
            }))
            seq += 1
            await asyncio.sleep(max(0.0, period - (time.monotonic() - now)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/car")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--quality", type=int, default=60)
    a = p.parse_args()
    asyncio.run(run(a.url, a.fps, a.width, a.height, a.quality))
