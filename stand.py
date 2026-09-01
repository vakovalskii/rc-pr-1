"""
Запускальщик стенда: поднимает релей и машинку фоном, показывает статус и логи.

    python3 stand.py start                         # релей + синтетический симулятор
    python3 stand.py start --car polygon --data data/synth   # полигон по карте
    python3 stand.py start --car video --data data/dom       # езда по кадрам съёмки
    python3 stand.py status
    python3 stand.py logs car
    python3 stand.py rec start zal   /   python3 stand.py rec stop
    python3 stand.py stop

PID-файлы и логи лежат в .stand/. Питон берётся из .venv, если он есть.
"""
import argparse, json, os, pathlib, signal, socket, subprocess, sys, time, urllib.request
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost мимо системного прокси

ROOT = pathlib.Path(__file__).parent.resolve()
RUN = ROOT / ".stand"
PY = ROOT / ".venv/bin/python"
PY = str(PY) if PY.exists() else sys.executable

def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"

def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False

def pids():
    out = {}
    for f in RUN.glob("*.pid"):
        pid = int(f.read_text().strip() or 0)
        if alive(pid): out[f.stem] = pid
        else: f.unlink()
    return out

def spawn(name, cmd):
    RUN.mkdir(exist_ok=True)
    log = open(RUN / f"{name}.log", "a")
    log.write(f"\n=== {time.strftime('%H:%M:%S')} $ {' '.join(cmd)}\n"); log.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (RUN / f"{name}.pid").write_text(str(p.pid))
    return p.pid

def stop_all():
    for name, pid in pids().items():
        os.killpg(os.getpgid(pid), signal.SIGTERM); print(f"остановил {name} ({pid})")
        (RUN / f"{name}.pid").unlink(missing_ok=True)

def status(port):
    ps = pids()
    for n in ("relay", "car", "pilot"):
        print(f"{n:6s} {'работает, pid ' + str(ps[n]) if n in ps else 'не запущен'}")
    try:
        s = json.loads(NOPROXY.open(f"http://127.0.0.1:{port}/status", timeout=2).read())
        print(f"релей: машинка {'на связи' if s['car_online'] else 'НЕТ'}, пультов {s['pults']}, {s['fps']} к/с, {s['kbit_s']} кбит/с, "
              f"запись {s['rec'] or 'нет'}{' (' + str(s['rec_frames']) + ' кадров)' if s['rec'] else ''}")
    except Exception as e:
        print(f"релей не отвечает на :{port} ({e.__class__.__name__})")
    print(f"пульт:  http://{lan_ip()}:{port}      карта: http://{lan_ip()}:{port}/map.html")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start")
    st.add_argument("--car", choices=["fake", "video", "polygon", "none"], default="fake")
    st.add_argument("--data", help="папка съёмки: data/<имя> (для video и polygon)")
    st.add_argument("--port", type=int, default=8080)
    st.add_argument("--fps", type=int, default=15)
    st.add_argument("extra", nargs="*", help="доп. аргументы машинке, например --zoom 1.6")
    sub.add_parser("stop"); s2 = sub.add_parser("status"); s2.add_argument("--port", type=int, default=8080)
    lg = sub.add_parser("logs"); lg.add_argument("what", choices=["relay", "car"], nargs="?", default="car"); lg.add_argument("-n", type=int, default=30)
    rc = sub.add_parser("rec"); rc.add_argument("action", choices=["start", "stop"]); rc.add_argument("name", nargs="?"); rc.add_argument("--port", type=int, default=8080)
    sub.add_parser("ip")
    pl = sub.add_parser("pilot", help="запустить автопилот фоном (stop гасит и его)")
    pl.add_argument("--seconds", type=float, default=3600); pl.add_argument("--model", default="models/bc.pt")
    pl.add_argument("--max-throttle", type=float, default=0.6); pl.add_argument("--port", type=int, default=8080)
    a = p.parse_args()

    if a.cmd == "start":
        stop_all()
        if a.car in ("video", "polygon") and not a.data:
            sys.exit("нужна --data data/<имя>")
        relay = [PY, "-u", "relay.py", "--port", str(a.port)]
        map_dir = ROOT / a.data / "map" if a.data else None
        if map_dir and (map_dir / "map.json").exists():
            relay += ["--map", str(map_dir)]
        spawn("relay", relay)
        for _ in range(40):                                   # ждём, пока релей начнёт отвечать
            try: NOPROXY.open(f"http://127.0.0.1:{a.port}/status", timeout=1); break
            except Exception: time.sleep(0.25)
        else: sys.exit("релей не поднялся, смотри: python3 stand.py logs relay")
        url = f"ws://127.0.0.1:{a.port}/ws/car"
        car = {"fake": [PY, "fake_car.py"], "video": [PY, "video_car.py", "--frames", f"{a.data}/frames"],
               "polygon": [PY, "polygon_car.py", a.data], "none": None}[a.car]
        if car:
            spawn("car", [car[0], "-u"] + car[1:] + ["--url", url, "--fps", str(a.fps)] + a.extra); time.sleep(2.0)
        status(a.port)
    elif a.cmd == "stop":
        stop_all()
    elif a.cmd == "status":
        status(a.port)
    elif a.cmd == "logs":
        f = RUN / f"{a.what}.log"
        print(f.read_text().splitlines()[-a.n:] and "\n".join(f.read_text().splitlines()[-a.n:]) if f.exists() else "лога нет")
    elif a.cmd == "rec":
        q = f"?name={a.name}" if (a.action == "start" and a.name) else ""
        print(NOPROXY.open(f"http://127.0.0.1:{a.port}/rec/{a.action}{q}").read().decode())
    elif a.cmd == "ip":
        print(lan_ip())
    elif a.cmd == "pilot":
        for name, pid in pids().items():
            if name == "pilot": os.killpg(os.getpgid(pid), signal.SIGTERM); (RUN / "pilot.pid").unlink(missing_ok=True)
        spawn("pilot", [PY, "-u", "pilot.py", "--model", a.model, "--seconds", str(a.seconds), "--reset",
                        "--max-throttle", str(a.max_throttle), "--url", f"ws://127.0.0.1:{a.port}/ws/pult"])
        time.sleep(3); print((RUN / "pilot.log").read_text().splitlines()[-1]); status(a.port)

if __name__ == "__main__":
    main()
