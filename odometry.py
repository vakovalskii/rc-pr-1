"""
Скорость и поворот машинки по датчику Холла на заднем кардане и потенциометру руля (ESP шлёт оба в телеметрии).

Модель «велосипеда»: путь — импульсы Холла × метров на импульс, кривизна траектории — положение руля
× кривизна на полном упоре (отдельно влево и вправо: упоры у сервы несимметричные). Курс копится как
кривизна × пройденный путь, позиция — по курсу. Без гироскопа курс плывёт на пробуксовке и заносе;
придёт BNO085 — курс возьмём с него, а отсюда останется скорость и путь.

Живьём считает pi_car.py (поля odo_* в телеметрии) и пишет журнал data/odo/<время>.csv. Здесь — разбор
журналов и калибровка, результат в odometry.json (pi_car перечитывает его сам, перезапуск не нужен):

    python3 odometry.py report data/odo/20261105-221500.csv          # скорость, путь, повороты, картинка
    python3 odometry.py calib-dist data/odo/....csv --meters 5         # проехал ровно 5 м → метров на импульс
    python3 odometry.py calib-turn data/odo/....csv --turns 1          # на полном упоре развернулся на 360° → кривизна
"""
import argparse, csv, json, math, pathlib, time

PARAMS = pathlib.Path(__file__).with_name("odometry.json")
LOG_DIR = pathlib.Path(__file__).with_name("data") / "odo"
# До калибровки — прикидка: датчик на заднем правом полуосевом валу (крутится с колесом), колесо ~8.5 см,
# 2 магнита → ~13 см на импульс; радиус разворота на упоре ~0.8 м.
# wheel_y — где колесо с датчиком относительно середины задней оси (+ влево): правое — минус полколеи.
# В повороте внешнее колесо едет быстрее середины: путь середины = путь колеса / (1 − кривизна × wheel_y).
DEFAULTS = {"m_per_pulse": 0.13, "k_left": 1.25, "k_right": 1.25, "wheel_y": -0.09, "calibrated": False}
FIELDS = ("t", "n", "dt_us", "pot", "esc_us", "thr", "steer", "v", "s", "kappa", "x", "y", "yaw")

def load_params():
    try: return {**DEFAULTS, **json.loads(PARAMS.read_text())}
    except Exception: return dict(DEFAULTS)

def steer_norm(pot, potL, potC, potR):
    """Потенциометр → руль −1 (упор влево) … 0 … +1 (упор вправо); None, если руль не откалиброван."""
    if min(potL, potC, potR) < 0 or abs(potR - potL) < 60: return None
    d = pot - potC
    if (d > 0) == (potR > potC): return max(0.0, min(1.0, d / (potR - potC))) if potR != potC else 0.0
    return -max(0.0, min(1.0, d / (potL - potC))) if potL != potC else 0.0

class Odometry:
    """Копит путь, курс и позицию из телеметрии ESP. Ось x — куда машинка смотрела на старте, y — влево, курс против часовой."""
    def __init__(self, log=True):
        self.p, self.p_at = load_params(), self._mtime()
        self.n0 = self.t0 = None
        self.dist = self.odo = self.x = self.y = self.yaw = 0.0
        self.v = self.s = self.kappa = 0.0
        self.vmax = 0.0
        self.f = self.w = None
        if log:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            # часов нет: до NTP время после загрузки — сохранённое при выключении, имя может совпасть со старым журналом
            stem, i = time.strftime("%Y%m%d-%H%M%S"), 0
            while (LOG_DIR / f"{stem}{f'-{i}' if i else ''}.csv").exists(): i += 1
            self.path = LOG_DIR / f"{stem}{f'-{i}' if i else ''}.csv"
            self.f = self.path.open("w", newline=""); self.w = csv.writer(self.f); self.w.writerow(FIELDS)
        self.last_n, self.last_flush = None, 0.0

    @staticmethod
    def _mtime():
        try: return PARAMS.stat().st_mtime
        except OSError: return 0.0

    def update(self, t, n, dt_us, pot, potL, potC, potR, esc_us=1500, thr=0.0, steer=0.0):
        if self._mtime() != self.p_at: self.p, self.p_at = load_params(), self._mtime()   # калибровка применилась на лету
        mpp = self.p["m_per_pulse"]
        if self.last_n is None: self.last_n, self.t0 = n, t
        dn, self.last_n = n - self.last_n, n
        s = steer_norm(pot, potL, potC, potR)
        self.s = 0.0 if s is None else s
        # руль вправо (+) — кривизна отрицательная: курс против часовой
        self.kappa = -self.s * (self.p["k_right"] if self.s > 0 else self.p["k_left"])
        mpp /= max(0.5, 1 - self.kappa * self.p["wheel_y"])   # колесо с датчиком сбоку — к середине оси
        ds = dn * mpp
        sign = 1 if dn > 0 else -1 if dn < 0 else (1 if self.v >= 0 else -1)
        self.v = sign * mpp / (dt_us * 1e-6) if dt_us else 0.0
        self.vmax = max(self.vmax, abs(self.v))
        if ds:
            dyaw = self.kappa * ds
            mid = self.yaw + dyaw / 2                    # середина дуги: точнее, чем курс на начале шага
            self.x += ds * math.cos(mid); self.y += ds * math.sin(mid)
            self.yaw = (self.yaw + dyaw + math.pi) % (2 * math.pi) - math.pi
            self.dist += ds; self.odo += abs(ds)
        if self.w and (dn or dt_us or t - self.last_flush > 1.0):   # стоим — строка раз в секунду, едем — каждая
            self.w.writerow([round(t, 3), n, dt_us, pot, esc_us, round(thr, 3), round(steer, 3), round(self.v, 3),
                             round(self.s, 3), round(self.kappa, 4), round(self.x, 3), round(self.y, 3), round(self.yaw, 4)])
            if t - self.last_flush > 1.0: self.f.flush(); self.last_flush = t
        return self.state()

    def state(self):
        r = 1 / abs(self.kappa) if abs(self.kappa) > 1e-3 else None
        return {"odo_v": round(self.v, 2), "odo_kmh": round(self.v * 3.6, 1), "odo_vmax_kmh": round(self.vmax * 3.6, 1),
                "odo_m": round(self.odo, 1), "odo_steer": round(self.s, 2),
                "odo_r_m": None if r is None else round(r, 2),
                "odo_yaw_rate": round(math.degrees(self.kappa * self.v), 1),        # °/с, + влево
                "odo_x": round(self.x, 2), "odo_y": round(self.y, 2), "odo_yaw": round(math.degrees(self.yaw), 1),
                "odo_cal": self.p.get("calibrated", False)}

# ------------------------------------------------------------ разбор журналов ---
def read_log(path):
    rows = list(csv.DictReader(open(path)))
    if not rows: raise SystemExit(f"{path}: пустой журнал")
    return [{k: float(v) for k, v in r.items()} for r in rows]

def report(a):
    rows = read_log(a.log); p = load_params()
    t = [r["t"] - rows[0]["t"] for r in rows]
    pulses = rows[-1]["n"] - rows[0]["n"]
    odo = sum(abs(b["n"] - a_["n"]) for a_, b in zip(rows, rows[1:])) * p["m_per_pulse"]
    moving = [r for r in rows if abs(r["v"]) > 0.05]
    move_s = sum(b["t"] - a_["t"] for a_, b in zip(rows, rows[1:]) if abs(b["v"]) > 0.05)
    vmax = max((abs(r["v"]) for r in rows), default=0)
    turns = [r for r in moving if abs(r["s"]) > 0.3]
    rmin = min((1 / abs(r["kappa"]) for r in turns if abs(r["kappa"]) > 1e-3), default=None)
    left = sum(1 for r in turns if r["kappa"] > 0); right = len(turns) - left
    yaw_total = sum(r["kappa"] * (b["n"] - r["n"]) * p["m_per_pulse"] for r, b in zip(rows, rows[1:]))
    print(f"{pathlib.Path(a.log).name}: {t[-1]:.0f} с, в движении {move_s:.0f} с")
    print(f"  импульсов {pulses:+.0f} (со знаком), путь {odo:.1f} м, смещение от старта "
          f"{math.hypot(rows[-1]['x'], rows[-1]['y']):.1f} м")
    print(f"  скорость: макс {vmax * 3.6:.1f} км/ч, средняя в движении {odo / move_s * 3.6 if move_s else 0:.1f} км/ч")
    print(f"  в поворотах {len(turns) / max(len(moving), 1) * 100:.0f}% времени езды (влево {left}, вправо {right} отсчётов), "
          f"мин. радиус {rmin:.2f} м" if rmin else "  поворотов не было")
    print(f"  поворот курса за заезд {math.degrees(yaw_total):+.0f}° (+ влево)")
    if not p.get("calibrated"): print("  ВНИМАНИЕ: калибровки ещё нет — метры и радиусы по прикидке (calib-dist, calib-turn)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("  картинки не будет: нет matplotlib (pip install matplotlib)"); return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax[0].plot([r["x"] for r in rows], [r["y"] for r in rows], lw=2)
    ax[0].plot(0, 0, "go", label="старт"); ax[0].plot(rows[-1]["x"], rows[-1]["y"], "rs", label="финиш")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend(); ax[0].set_title("Траектория, м (по Холлу и рулю)")
    ax[1].plot(t, [r["v"] * 3.6 for r in rows], label="скорость, км/ч")
    ax[1].plot(t, [r["s"] * 10 for r in rows], alpha=.7, label="руль ×10 (+ вправо)")
    ax[1].plot(t, [r["thr"] * 10 for r in rows], alpha=.5, label="газ ×10")
    ax[1].grid(alpha=.3); ax[1].legend(); ax[1].set_xlabel("с"); ax[1].set_title("Скорость, руль и газ")
    out = a.png or str(pathlib.Path(a.log).with_suffix(".png"))
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"  картинка: {out}")

def save_params(upd):
    p = {**load_params(), **upd}
    PARAMS.write_text(json.dumps(p, ensure_ascii=False, indent=1)); return p

def calib_dist(a):
    rows = read_log(a.log)
    pulses = abs(rows[-1]["n"] - rows[0]["n"])
    if pulses < 10: raise SystemExit(f"за журнал всего {pulses:.0f} импульсов — датчик не видит магниты?")
    mpp = a.meters / pulses
    p = save_params({"m_per_pulse": round(mpp, 5), "calibrated": True})
    print(f"{pulses:.0f} импульсов на {a.meters} м → {mpp * 100:.2f} см на импульс; записано в {PARAMS.name}: {p}")

def calib_turn(a):
    rows = read_log(a.log); p = load_params()
    seg = [(r, b) for r, b in zip(rows, rows[1:]) if abs(r["s"]) > 0.9]          # только где руль на упоре
    if not seg: raise SystemExit("в журнале нет езды с рулём на упоре (|руль| > 0.9)")
    length = sum(abs(b["n"] - r["n"]) for r, b in seg) * p["m_per_pulse"]
    side = "k_right" if sum(r["s"] for r, _ in seg) > 0 else "k_left"
    # колесо с датчиком сбоку: его круг на wheel_y шире или уже круга середины оси
    rw = length / (2 * math.pi * a.turns)
    k = 1 / (rw + p["wheel_y"] if side == "k_left" else rw - p["wheel_y"])
    p = save_params({side: round(k, 4)})
    print(f"на упоре {'вправо' if side == 'k_right' else 'влево'} проехал {length:.2f} м за {a.turns} оборот(а) → "
          f"радиус {1 / k:.2f} м; записано в {PARAMS.name}: {p}")
    if not p.get("calibrated"): print("ВНИМАНИЕ: сначала calib-dist — радиус считается в метрах по импульсам")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("report", help="скорость, путь, повороты, картинка"); r.add_argument("log"); r.add_argument("--png")
    r.set_defaults(fn=report)
    c = sp.add_parser("calib-dist", help="метров на импульс по проезду известной длины"); c.add_argument("log")
    c.add_argument("--meters", type=float, required=True); c.set_defaults(fn=calib_dist)
    c = sp.add_parser("calib-turn", help="кривизна на упоре по развороту на 360°"); c.add_argument("log")
    c.add_argument("--turns", type=float, default=1); c.set_defaults(fn=calib_turn)
    a = ap.parse_args(); a.fn(a)
