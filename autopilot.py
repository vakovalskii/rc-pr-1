"""
Автопилот машинки как служба: всегда строит живую карту (как live_slam.py), а по кнопкам из пульта
(окно «3D-карта») сам ездит — исследует комнату, объезжает её кругом, едет в точку по касанию карты,
возвращается домой. Газ — потолок ESP (18%), здесь не меняется.

    uv run --python 3.12 --with open3d --with aiohttp --with pillow --with scipy python autopilot.py
    ... python autopilot.py --start explore        # сразу начать исследовать (или lap / home)

Стек — как у Nav2 в ROS 2, только лёгкий:
  * SLAM: Live из live_slam.py (RGBD-пары с Pi 4) — поза, облако, сетка проходимости; оси сетки заморожены.
  * Память препятствий: профиль глубины (16 секторов, 15 раз в секунду) по позе на свою сетку.
    D455 ближе ~0.4 м не видит — вплотную машинка ориентируется по памяти.
  * Стоимостная карта: препятствия раздуты на полкорпуса, рядом со стенами дороже, неизвестное — чуть дороже.
  * Исследование по фронтирам: граница «свободно / неизвестно», кластеры, выбор по длине пути и размеру.
  * Путь — Дейкстра/A* по стоимостной карте, следование — pure pursuit, страховка — дуги руля по памяти.
  * Круг: кольцо свободного места в ~0.45 м от препятствий, точки по кругу, едем по ним до замыкания.
  * Тупик/упор: разворот в три приёма. Человек тронул стик — автопилот сразу отдаёт управление.

Команды (текстом от /map3d.html через релей): {"type":"auto","cmd":"explore"|"lap"|"home"|"stop"|"goto","x":..,"y":..}.
Отдаёт кроме карты {"type":"plan","state","mode","path":[[x,y]],"target":[x,y],"R","floor",...}.
Разборы заездов и выученные параметры — data/explore/.
"""
import argparse, asyncio, heapq, json, math, pathlib, time
import numpy as np
import aiohttp
from scipy import ndimage
from live_slam import Live, decode

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "data" / "explore"
DEF = {"stop_m": 0.45, "coast_m": 0.15, "half_w": 0.16, "robot_r": 0.18, "steer_max_deg": 25, "wheelbase": 0.26,
       "reverse_s": 1.3, "lookahead": 0.4, "ring_m": 0.45}

class Memory:
    """Сетка препятствий из профиля глубины, те же клетки, что у сетки SLAM (5 см, центр в n/2)."""
    def __init__(self, res=0.05, size_m=30):
        self.res, self.n = res, int(size_m / res); self.L = np.zeros((self.n, self.n), np.float32)
        self.visited = np.zeros((self.n, self.n), bool)

    def ij(self, x, y):
        return int(math.floor(x / self.res)) + self.n // 2, int(math.floor(y / self.res)) + self.n // 2

    def add(self, x, y, yaw, prof):
        for b, r in enumerate(prof):
            if r is None: continue                             # пусто: может, слишком близко для глубины — память не трогаем
            a = math.radians(-44 + (b + .5) * 5.5)             # + вправо
            hit = r < 2.5; rr = r if hit else 2.0; th = yaw - a
            for k in range(1, int(rr / self.res)):
                i, j = self.ij(x + k * self.res * math.cos(th), y + k * self.res * math.sin(th))
                if 0 <= i < self.n and 0 <= j < self.n: self.L[j, i] = max(-3, self.L[j, i] - 0.3)
            if hit:
                i, j = self.ij(x + rr * math.cos(th), y + rr * math.sin(th))
                if 0 <= i < self.n and 0 <= j < self.n: self.L[j, i] = min(4, self.L[j, i] + 1.0)

    def decay(self, dt):
        """Забывание: препятствие, которое глубина давно не подтверждала, тает (~40 с от полной уверенности).
        Иначе дрейф позы копит «призраков», и скоро всё вокруг кажется занятым."""
        pos = self.L > 0
        self.L[pos] = np.maximum(self.L[pos] - 0.1 * dt, 0)

    def occ(self, x, y):
        i, j = self.ij(x, y)
        return 0 <= i < self.n and 0 <= j < self.n and self.L[j, i] > 0.5

    def mark_visited(self, x, y, r=0.15):
        i, j = self.ij(x, y); k = int(r / self.res)
        self.visited[max(j - k, 0):j + k + 1, max(i - k, 0):i + k + 1] = True

class Planner:
    """Стоимостная карта, фронтиры, путь. Работает на вырезке вокруг известной области."""
    def __init__(self, P, grid, mem):
        self.P, self.grid, self.mem, self.res = P, grid, mem, mem.res

    def build(self, pose):
        G, M = self.grid.L, self.mem.L
        occ = (G > 0.6) | (M > 0.5)
        free = ((G < -0.5) | (M < -0.5) | self.mem.visited) & ~occ
        known = occ | free
        ys, xs = np.nonzero(known)
        ci, cj = self.mem.ij(pose[0], pose[1])
        if len(xs) == 0: xs, ys = np.array([ci]), np.array([cj])
        m = 12
        self.i0, self.j0 = max(min(xs.min(), ci) - m, 0), max(min(ys.min(), cj) - m, 0)
        i1, j1 = min(max(xs.max(), ci) + m + 1, self.mem.n), min(max(ys.max(), cj) + m + 1, self.mem.n)
        sl = (slice(self.j0, j1), slice(self.i0, i1))
        self.occ, self.free = occ[sl], free[sl]
        self.unknown = ~self.occ & ~self.free
        d = ndimage.distance_transform_edt(~self.occ) * self.res
        self.dist = d
        self.lethal = d < self.P["robot_r"]
        self.cost = 1 + 4 * np.exp(-np.maximum(d - self.P["robot_r"], 0) / 0.12) + 1.5 * self.unknown
        self.start = (cj - self.j0, ci - self.i0)
        # стоим вплотную (по раздутию «внутри» препятствия) — даём выехать из своей клетки
        sj, si = self.start
        if 0 <= sj < self.lethal.shape[0] and 0 <= si < self.lethal.shape[1]:
            k = int(self.P["robot_r"] / self.res) + 1
            self.lethal[max(sj - k, 0):sj + k + 1, max(si - k, 0):si + k + 1] &= self.occ[max(sj - k, 0):sj + k + 1, max(si - k, 0):si + k + 1]

    def to_xy(self, j, i):
        return ((i + self.i0 - self.mem.n // 2) + .5) * self.res, ((j + self.j0 - self.mem.n // 2) + .5) * self.res

    def to_cell(self, x, y):
        i, j = self.mem.ij(x, y); return j - self.j0, i - self.i0

    def dijkstra(self, goal=None):
        H, W = self.cost.shape; dist = np.full((H, W), np.inf); par = -np.ones((H, W, 2), int)
        s = self.start
        if not (0 <= s[0] < H and 0 <= s[1] < W): return dist, par
        dist[s] = 0; pq = [(0.0, s)]
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        while pq:
            dcur, (j, i) = heapq.heappop(pq)
            if dcur > dist[j, i]: continue
            if goal is not None and (j, i) == goal: break
            for dj, di, w in nb:
                nj, ni = j + dj, i + di
                if 0 <= nj < H and 0 <= ni < W and not self.lethal[nj, ni]:
                    nd = dcur + w * self.cost[nj, ni]
                    if nd < dist[nj, ni]: dist[nj, ni] = nd; par[nj, ni] = (j, i); heapq.heappush(pq, (nd, (nj, ni)))
        return dist, par

    def path_to(self, par, goal):
        out, c = [], goal
        while c[0] >= 0 and c != self.start:
            out.append(self.to_xy(*c)); c = tuple(par[c])
            if len(out) > 5000: break
        return out[::-1]

    def frontier_goal(self):
        """Ближний выгодный фронтир: (путь, цель, число кластеров) или (None, None, 0) — исследовать нечего."""
        fr = self.free & ndimage.binary_dilation(self.unknown, np.ones((3, 3))) & ~self.lethal
        lab, n = ndimage.label(fr, np.ones((3, 3)))
        if n == 0: return None, None, 0
        dist, par = self.dijkstra()
        best, best_sc = None, np.inf
        sj, si = self.start
        for k in range(1, n + 1):
            cells = np.argwhere(lab == k)
            if len(cells) < 4: continue
            cells = cells[np.hypot(cells[:, 0] - sj, cells[:, 1] - si) * self.res >= 0.5]   # вплотную — камера нового не увидит
            cells = cells[self.dist[cells[:, 0], cells[:, 1]] >= 0.3]      # у самой стены неизвестное остаётся всегда — не цель
            if not len(cells): continue
            dd = dist[cells[:, 0], cells[:, 1]]
            if not np.isfinite(dd).any(): continue
            c = tuple(cells[np.argmin(dd)])
            sc = dd.min() * self.res - 0.03 * len(cells)          # ближе и крупнее — лучше
            if sc < best_sc: best, best_sc = c, sc
        if best is None: return None, None, n
        return self.path_to(par, best), self.to_xy(*best), n

    def goal_path(self, x, y):
        g = self.to_cell(x, y)
        H, W = self.cost.shape
        if not (0 <= g[0] < H and 0 <= g[1] < W): return None
        if self.lethal[g]:                                         # цель в препятствии — ближайшая проходимая клетка
            free = np.argwhere(~self.lethal)
            if not len(free): return None
            g = tuple(free[np.argmin(((free - np.array(g)) ** 2).sum(1))])
        dist, par = self.dijkstra(goal=g)
        return self.path_to(par, g) if np.isfinite(dist[g]) else None

    def ring(self):
        """Точки круга по комнате: кольцо свободного места в ring_m от препятствий, по углу вокруг центра."""
        free = self.free & ~self.lethal
        if free.sum() < 20: return []
        lab, n = ndimage.label(free)
        s = lab[self.start] if 0 <= self.start[0] < lab.shape[0] and 0 <= self.start[1] < lab.shape[1] else 0
        if s: free = lab == s
        d = ndimage.distance_transform_edt(free | self.unknown) * self.res
        r = self.P["ring_m"]
        cells = np.argwhere(free & (np.abs(d - r) < 0.06))
        if len(cells) < 8: cells = np.argwhere(free & (d > r * 0.6))
        if not len(cells): return []
        c = cells.mean(0)
        ang = np.arctan2(cells[:, 0] - c[0], cells[:, 1] - c[1])
        pts = []
        for a in np.linspace(-math.pi, math.pi, 16, endpoint=False):  # по точке на сектор 22.5°
            sel = cells[np.abs(np.angle(np.exp(1j * (ang - a)))) < 0.2]
            if len(sel): pts.append(self.to_xy(*sel[np.argmin(np.abs(np.hypot(*(sel - c).T) - np.hypot(*(sel - c).T).max()))]))
        return pts

class Autopilot:
    def __init__(self, a, P):
        self.a, self.P = a, P
        self.live = Live(argparse.Namespace(voxel=0.04, depth_max=2.5, key_m=0.08, key_deg=8, pitch=25, cam_height=0.10,
                                            z_min=0.04, z_max=0.35, car_radius=0.15, res=0.05, grid_m=30, max_points=1_500_000),
                         freeze_up=True)
        self.mem = Memory(); self.pose = None; self.pose_t = 0; self.pose_hist = []; self.home = None
        self.stats = None; self.stats_t = 0; self.tele = {}
        self.mode = "idle"; self.state = "жду"; self.state_t = time.time()
        self.path, self.target, self.goal, self.ring, self.ring_i, self.frontiers = [], None, None, [], 0, 0
        self.plan_t = 0; self.man = None; self.escapes = []; self.last_move = time.time()
        self.sent = (0.0, 0.0); self.override_since = None; self.ep = None; self.planning = False; self.loop = None; self.need_look = False

    # --- поза и геометрия ---
    def pose2d(self, T):
        g = self.live.grid
        q = g.R @ T[:3, 3]; f = g.R @ (T[:3, :3] @ [0, 0, 1.0])
        return float(q[0]), float(q[1]), math.atan2(f[1], f[0])

    def arc(self, s, length, step=0.05):
        x, y, yaw = self.pose
        k = -math.tan(math.radians(self.P["steer_max_deg"]) * s) / self.P["wheelbase"]
        for i in range(1, int(length / step) + 1):
            l = i * step
            if abs(k) < 1e-4: yield x + l * math.cos(yaw), y + l * math.sin(yaw), yaw, l
            else:
                pyaw = yaw + k * l
                yield x + (math.sin(pyaw) - math.sin(yaw)) / k, y - (math.cos(pyaw) - math.cos(yaw)) / k, pyaw, l

    def clearance(self, s, length=1.0):
        hw = self.P["half_w"]
        for px, py, pyaw, l in self.arc(s, length):
            for off in (-hw, 0, hw):
                if self.mem.occ(px - off * math.sin(pyaw), py + off * math.cos(pyaw)): return l
        return length

    def front_now(self):
        p = (self.stats or {}).get("prof") or []
        mid = [r for r in p[4:12] if r is not None]
        return (min(mid) if mid else 9.0), (self.stats or {}).get("hole_lo", 0)

    def open_side(self):
        p = (self.stats or {}).get("prof") or []
        side_m = lambda b: np.mean([r if r is not None else 0.3 for r in b]) if b else 0
        return -1 if self.clearance(-1) + 0.5 * side_m(p[:6]) > self.clearance(1) + 0.5 * side_m(p[10:]) else 1

    # --- план ---
    def replan(self):
        pl = Planner(self.P, self.live.grid, self.mem); pl.build(self.pose)
        self.target = None
        if self.mode == "explore":
            path, tgt, n = pl.frontier_goal(); self.frontiers = n
            if path is None:
                area = float(pl.free.sum()) * pl.res ** 2
                if area < 8.0:                                   # карта ещё маленькая — не «готово», а надо осмотреться
                    self.path, self.need_look = [], True; return
                print(f"фронтиров не осталось ({area:.1f} м²) — карта готова, еду домой"); self.mode, self.goal = "home", self.home[:2]
            else: self.path, self.target = path, tgt; return
        if self.mode == "lap":
            if not self.ring:
                self.ring = pl.ring(); self.ring_i = int(np.argmin([math.hypot(p[0] - self.pose[0], p[1] - self.pose[1]) for p in self.ring])) if self.ring else 0
                self.ring_start = self.ring_i; self.ring_done = 0
                print(f"круг: {len(self.ring)} точек")
                if not self.ring: print("мало карты для круга — сначала исследую"); self.mode = "explore"; return self.replan()
            self.goal = self.ring[self.ring_i]
        if self.mode in ("goto", "home", "lap") and self.goal:
            self.path = pl.goal_path(*self.goal) or []; self.target = self.goal
            if not self.path: print("до цели пути нет"); self.state = "нет пути"

    def _replan(self):
        try: self.replan()
        except Exception as e: print("план не построился:", e)
        finally: self.planning = False

    def follow(self):
        """Pure pursuit: руль на точку пути в lookahead впереди; ±1, вправо +."""
        x, y, yaw = self.pose; L = self.P["lookahead"]
        while len(self.path) > 1 and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < L * 0.6: self.path.pop(0)
        tx, ty = next(((px, py) for px, py in self.path if math.hypot(px - x, py - y) >= L), self.path[-1])
        alpha = math.atan2(ty - y, tx - x) - yaw; alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        if abs(alpha) > math.radians(100): return None, alpha            # цель сзади — нужен разворот
        delta = math.atan2(2 * self.P["wheelbase"] * math.sin(alpha), max(math.hypot(tx - x, ty - y), 0.2))
        return float(np.clip(-delta / math.radians(self.P["steer_max_deg"]), -1, 1)), alpha

    def escape(self, now, why, side=None):
        side = side or self.open_side()
        near = [e for e in self.escapes if now - e[0] < 15 and math.hypot(e[1] - self.pose[0], e[2] - self.pose[1]) < 0.4]
        self.escapes.append((now, self.pose[0], self.pose[1]))
        if self.ep: self.ep["reverses"] += 1
        back = self.P["reverse_s"] * (1.6 if len(near) >= 2 else 1)
        self.man = {"phase": "назад", "t0": now, "until": now + 0.3 + back, "side": side, "p0": self.pose}
        return -1.0, -side, why

    def tick(self, now):
        """→ (газ, руль, состояние). Газ 1.0 = потолок ESP."""
        P = self.P
        if self.pose is None or self.stats is None: return 0.0, 0.0, "жду SLAM"
        if now - self.stats_t > 0.5: return 0.0, 0.0, "нет глубины"
        if self.live.tr and self.live.tr.miss >= 3: return 0.0, 0.0, "SLAM потерян"
        if self.man:
            m = self.man
            if m["phase"] == "назад":
                el = now - m["t0"]
                # автомобильный регулятор: первое «назад» после хода вперёд — тормоз; нейтраль и снова «назад» — задний ход
                if el < 0.15: return -1.0, -m["side"], "назад"
                if el < 0.3: return 0.0, -m["side"], "назад"
                moved = math.hypot(self.pose[0] - m["p0"][0], self.pose[1] - m["p0"][1])
                if now > m["until"] or (el > 1.3 and moved < 0.03):       # сзади упор — разворачиваемся вперёд
                    if moved < 0.03 and self.ep: self.ep["rear_blocked"] = self.ep.get("rear_blocked", 0) + 1
                    self.man = {"phase": "разворот", "until": now + 1.5, "side": m["side"]}
                return -1.0, -m["side"], "назад"
            if now > m["until"] or self.clearance(m["side"], 0.7) < P["stop_m"] - 0.1:
                self.man = None; self.plan_t = 0
            elif self.front_now()[0] < P["stop_m"] - 0.1: return self.escape(now, "разворот упёрся")
            else: return 1.0, float(m["side"]), "разворот"
        if (now - self.plan_t > 1.0 or not self.path) and not self.planning:   # план — в отдельном потоке,
            self.planning, self.plan_t = True, now                             # цикл управления не замирает
            self.loop.run_in_executor(None, self._replan)
        if self.mode == "idle": return 0.0, 0.0, "жду"
        if not self.path and self.need_look:
            self.need_look = False; return self.escape(now, "осмотр")
        if not self.path:
            if self.state in ("план", "нет пути", "фронтир") and now - self.state_t > 3:   # плана нет — сменить точку обзора
                self.state_t = now; return self.escape(now, "смена обзора")
            return 0.0, 0.0, self.state if self.state == "нет пути" else "план"
        end = self.path[-1]
        if math.hypot(end[0] - self.pose[0], end[1] - self.pose[1]) < 0.25:   # дошли
            if self.mode == "lap":
                self.ring_i = (self.ring_i + 1) % len(self.ring); self.ring_done += 1; self.path = []
                if self.ring_done >= len(self.ring):
                    print("КРУГ ЗАМКНУТ");
                    if self.ep: self.ep["lap_closed"] = True
                    self.set_mode("idle")
                return 0.0, 0.0, "точка круга"
            if self.mode in ("goto", "home"):
                print("на месте"); self.set_mode("idle"); return 0.0, 0.0, "на месте"
            if self.target: self.mem.mark_visited(*self.target, r=0.3)   # этот фронтир больше не цель
            self.path = []; return 0.0, 0.0, "фронтир"
        s, alpha = self.follow()
        if s is None: return self.escape(now, "цель сзади", side=-1 if alpha > 0 else 1)
        front, hole = self.front_now()
        stop_at = P["stop_m"] + P["coast_m"] * 0.5
        if front < P["stop_m"] - 0.05 or (hole > 0.55 and front < 0.6) or self.clearance(s, stop_at + 0.05) < stop_at:
            if self.state != "тупик": self.state_t = now; self.ep and self.ep.__setitem__("stops", self.ep["stops"] + 1)
            if now - self.state_t > 0.4: self.plan_t = 0; return self.escape(now, "тупик")
            return 0.0, s, "тупик"
        return (0.6 if self.clearance(s, 0.8) < 0.8 else 1.0), s, "еду"

    def set_mode(self, m, goal=None):
        if self.ep and m != self.mode: self.finish_episode()
        self.mode, self.goal, self.path, self.man, self.plan_t = m, goal, [], None, 0
        if m == "lap": self.ring = []
        if m != "idle": self.ep = {"mode": m, "t0": time.time(), "path0": self.dist_total, "bumps": 0, "stops": 0, "reverses": 0, "coast": []}
        print("режим:", m, goal or "")

    dist_total = 0.0

    def finish_episode(self):
        e = self.ep; self.ep = None
        g = self.live.grid
        rep = {"mode": e["mode"], "seconds": round(time.time() - e["t0"], 1), "path_m": round(self.dist_total - e["path0"], 2),
               "free_m2": round(float((g.L < -0.5).sum()) * g.a.res ** 2, 2), "bumps": e["bumps"], "stops": e["stops"],
               "reverses": e["reverses"], "lap_closed": e.get("lap_closed", False), "coast_m": e["coast"], "params": dict(self.P)}
        if e["coast"]: self.P["coast_m"] = round(0.5 * self.P["coast_m"] + 0.5 * float(np.median(e["coast"])), 3)
        if e["bumps"]: self.P["stop_m"] = round(min(self.P["stop_m"] + 0.05 * e["bumps"], 0.8), 2)
        elif e["stops"] >= 2: self.P["stop_m"] = round(max(self.P["stop_m"] - 0.05, 0.30), 2)   # без упоров — смелее, до 30 см
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"report-{time.strftime('%Y%m%d-%H%M%S')}-{e['mode']}.json").write_text(json.dumps({**rep, "new_params": self.P}, ensure_ascii=False, indent=1))
        (OUT / "params.json").write_text(json.dumps(self.P, indent=1))
        print("разбор:", json.dumps(rep, ensure_ascii=False))

    def plan_msg(self):
        g = self.live.grid
        return {"type": "plan", "mode": self.mode, "state": self.state, "path": [[round(x, 3), round(y, 3)] for x, y in self.path[:200]],
                "target": [round(v, 3) for v in self.target] if self.target else None, "frontiers": self.frontiers,
                "ring": [[round(x, 3), round(y, 3)] for x, y in self.ring], "home": self.home[:2] if self.home else None,
                "R": g.R.round(5).tolist() if g.R is not None else None, "floor": -g.a.cam_height,
                "stop_m": self.P["stop_m"], "pose": [round(v, 3) for v in self.pose] if self.pose else None}

async def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / "params.json"; P = {**DEF, **(json.loads(pf.read_text()) if pf.exists() else {})}
    print("параметры:", P)
    ap = Autopilot(a, P); loop = asyncio.get_running_loop(); ap.loop = loop
    async with aiohttp.ClientSession(trust_env=False) as s:
        while True:
            try:
                cam = await s.ws_connect(a.pi4, max_msg_size=8 << 20)
                pult = await s.ws_connect(f"{a.relay}/ws/pult")
                out = await s.ws_connect(f"{a.relay}/ws/map3d", max_msg_size=32 << 20)
                await cam.send_str(json.dumps({"type": "raw"}))
                if ap.pose is None: await out.send_str(json.dumps({"type": "reset", "rec": None}))
                print("на связи: камера, пульт, карта")
                latest = {"b": None}; ev = asyncio.Event()

                async def rx_cam():
                    async for m in cam:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            j = json.loads(m.data)
                            if j.get("type") == "depth" and "prof" in j:
                                ap.stats, ap.stats_t = j, time.time()
                                if ap.pose: ap.mem.add(*ap.pose, j["prof"])
                        elif m.type == aiohttp.WSMsgType.BINARY and m.data[:4] == b"RGBD": latest["b"] = m.data; ev.set()

                async def rx_pult():
                    async for m in pult:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            j = json.loads(m.data)
                            if j.get("type") == "tele": ap.tele = j

                async def rx_cmd():                                  # кнопки из окна карты
                    async for m in out:
                        if m.type != aiohttp.WSMsgType.TEXT: continue
                        j = json.loads(m.data)
                        if j.get("type") != "auto": continue
                        c = j.get("cmd")
                        if c == "stop": ap.set_mode("idle")
                        elif c == "home" and ap.home: ap.set_mode("home", ap.home[:2])
                        elif c == "goto": ap.set_mode("goto", (float(j["x"]), float(j["y"])))
                        elif c in ("explore", "lap"): ap.set_mode(c)

                async def slam():
                    while True:
                        await ev.wait(); ev.clear()
                        meta, zpng, cjpg = decode(latest["b"])
                        chunk, grid, pose = await loop.run_in_executor(None, ap.live.process, meta, zpng, cjpg)
                        T = np.array(pose["T"]).reshape(4, 4); x, y, yaw = ap.pose2d(T)
                        if ap.pose: ap.dist_total += math.hypot(x - ap.pose[0], y - ap.pose[1])
                        ap.pose, ap.pose_t = (x, y, yaw), time.time(); ap.pose_hist = (ap.pose_hist + [(time.time(), x, y)])[-100:]
                        ap.mem.mark_visited(x, y)
                        if ap.home is None: ap.home = (x, y, yaw)
                        if chunk: await out.send_bytes(chunk)
                        if grid: await out.send_bytes(grid)
                        await out.send_str(json.dumps(pose))

                tasks = [asyncio.create_task(c()) for c in (rx_cam, rx_pult, rx_cmd, slam)]
                if a.start and ap.mode == "idle": ap.set_mode(a.start, ap.home[:2] if a.start == "home" and ap.home else None); a.start = None
                i, driving, coast_from, plan_sent, decay_t = 0, False, None, 0, time.time()
                while not any(t.done() for t in tasks):
                    now = time.time()
                    thr, st, why = ap.tick(now) if ap.mode != "idle" or ap.man else (0.0, 0.0, "жду")
                    # упор: газ вперёд, а поза за 1.5 с почти не сдвинулась
                    if thr > 0 and ap.state in ("еду", "разворот"):
                        old = [h for h in ap.pose_hist if now - h[0] > 1.5]
                        if old and math.hypot(ap.pose[0] - old[-1][1], ap.pose[1] - old[-1][2]) > 0.05: ap.last_move = now
                        elif old and now - ap.last_move > 1.5:
                            if ap.ep: ap.ep["bumps"] += 1
                            ap.last_move = now; thr, st, why = ap.escape(now, "упор")
                    else: ap.last_move = now
                    # накат после «стоп»
                    if ap.state == "еду" and why == "тупик": coast_from = (ap.pose, now)
                    if thr < 0: coast_from = None                   # уже сдаём назад — это не накат, замер выбрасываем
                    if coast_from and now - coast_from[1] > 1.0:
                        if ap.ep: ap.ep["coast"].append(round(math.hypot(ap.pose[0] - coast_from[0][0], ap.pose[1] - coast_from[0][1]), 3))
                        coast_from = None
                    if why != ap.state: ap.state, ap.state_t = why, now
                    active = ap.mode != "idle" or ap.man is not None
                    # человек взял управление: машинка делает не то, что мы шлём
                    if driving and ap.tele and abs(float(ap.tele.get("thr", 0)) - ap.sent[0] * 1.0) > 0.3 and abs(ap.sent[0]) > 0:
                        ap.override_since = ap.override_since or now
                        if now - ap.override_since > 0.6: print("человек взял управление"); ap.set_mode("idle"); ap.man = None; active = False
                    else: ap.override_since = None
                    if active:
                        if not driving: await pult.send_str(json.dumps({"type": "esp", "cmd": "mode,2"})); driving = True
                        await pult.send_str(json.dumps({"type": "cmd", "id": i, "steer": round(st, 2), "throttle": round(thr, 2)})); i += 1
                        ap.sent = (thr, st)
                    elif driving:                                     # закончили — нейтраль и ручной руль человеку
                        for _ in range(6):
                            await pult.send_str(json.dumps({"type": "cmd", "id": i, "steer": 0, "throttle": 0})); i += 1; await asyncio.sleep(0.05)
                        await pult.send_str(json.dumps({"type": "esp", "cmd": "mode,1"})); driving = False; ap.sent = (0, 0)
                    if now - decay_t > 1.0: ap.mem.decay(now - decay_t); decay_t = now
                    if now - plan_sent > 0.5:
                        await out.send_str(json.dumps(ap.plan_msg())); plan_sent = now
                    if active and i % 10 == 0:
                        print(f"{ap.mode:7s} {why:12s} руль {st:+.2f} газ {thr:+.1f} впереди {ap.front_now()[0]:.2f} поза {[round(v, 2) for v in ap.pose] if ap.pose else None}")
                    await asyncio.sleep(0.1)
                for t in tasks: t.cancel()
                raise ConnectionError("связь оборвалась")
            except Exception as e:
                print(f"нет связи ({e.__class__.__name__}: {e}) — повтор через 2 с"); ap.man = None
                if ap.mode != "idle": ap.set_mode("idle")
                await asyncio.sleep(2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", choices=["explore", "lap", "home"], help="сразу начать")
    p.add_argument("--relay", default="http://192.168.1.137:8080")
    p.add_argument("--pi4", default="http://192.168.1.109:8090/")
    asyncio.run(main(p.parse_args()))
