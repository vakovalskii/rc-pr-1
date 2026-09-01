"""
Обучение политики поведенческим клонированием на записанных заездах.

    .venv/bin/python train_bc.py datasets/zal datasets/kuhnya --epochs 20
    .venv/bin/python train_bc.py datasets/*

Вход — кадр 160x120, выход — руль и газ в [-1, 1]. Модель — маленькая
свёртка как у DonkeyCar; на M1 эпоха на 10 тысячах кадров занимает секунды.
Сохраняется в models/bc.pt, метрики рядом в models/bc.json.

Валидация — хвост каждого заезда (последние 10%), а не случайные кадры:
соседние кадры почти одинаковы, случайный сплит показал бы дутую точность.
"""
import argparse, json, pathlib, time
import numpy as np
import torch, torch.nn as nn
from PIL import Image

IMG_W, IMG_H = 160, 120

def block(i, o, k, s):
    return [nn.Conv2d(i, o, k, s), nn.BatchNorm2d(o), nn.ReLU()]

class BCNet(nn.Module):
    """Свёртка как у DonkeyCar, плюс BatchNorm: без него сеть на голых ReLU
    временами стартует мёртвой и до конца обучения выдаёт константу."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(*block(3, 24, 5, 2), *block(24, 32, 5, 2), *block(32, 64, 5, 2),
                                  *block(64, 64, 3, 1), *block(64, 64, 3, 1), nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, IMG_H, IMG_W)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, 100), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 2), nn.Tanh())
    def forward(self, x):
        return self.head(self.conv(x * 2 - 1))                   # вход [0,1] -> [-1,1]

def preprocess(img):
    """PIL -> float tensor (3,H,W) в [0,1]. Одно и то же в обучении и в автопилоте."""
    arr = np.asarray(img.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR), np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def load_dataset(d, drop_idle):
    d = pathlib.Path(d); rows = []
    for line in (d / "log.jsonl").read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("failsafe"): continue                            # команд не было — учить нечему
        if drop_idle and abs(r["steer"]) < 1e-3 and abs(r["throttle"]) < 1e-3: continue
        f = d / "frames" / f"{r['i']:06d}.jpg"
        if f.exists(): rows.append((f, r.get("steer_label", r["steer"]), r["throttle"]))   # чистая метка эксперта, если есть
    return rows

def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="models/bc.pt")
    p.add_argument("--keep-idle", action="store_true", help="не выбрасывать кадры с нулевыми газом и рулём")
    p.add_argument("--val", type=float, default=0.1, help="доля хвоста каждого заезда на валидацию")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)

    tr, va = [], []
    for d in a.datasets:
        rows = load_dataset(d, not a.keep_idle)
        k = int(len(rows) * (1 - a.val))
        tr += rows[:k]; va += rows[k:]
        print(f"{d}: {len(rows)} кадров")
    if len(tr) < 50: raise SystemExit("слишком мало кадров — покатайся подольше")
    print(f"обучение {len(tr)}, валидация {len(va)}; загружаю кадры ...")
    def stack(rows):
        X = torch.stack([preprocess(Image.open(f)) for f, _, _ in rows])
        Y = torch.tensor([[s, t] for _, s, t in rows], dtype=torch.float32)
        return X, Y
    Xtr, Ytr = stack(tr); Xva, Yva = stack(va)
    dev = device(); print("устройство:", dev)
    model = BCNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    w = torch.tensor([2.0, 1.0], device=dev)                     # руль важнее газа
    base = (Yva - Ytr.mean(0)).abs().mean(0)                     # что даёт "всегда среднее"
    print(f"базовая MAE (всегда среднее): руль {base[0]:.3f}, газ {base[1]:.3f}")

    best, hist = 1e9, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            x = Xtr[idx].clone(); y = Ytr[idx].clone()
            # аугментации: яркость/контраст, зеркало (руль меняет знак), сдвиг по горизонтали
            x = x * (0.7 + 0.6 * torch.rand(len(x), 1, 1, 1)) + 0.1 * (torch.rand(len(x), 1, 1, 1) - 0.5)
            flip = torch.rand(len(x)) < 0.5
            x[flip] = x[flip].flip(-1); y[flip, 0] = -y[flip, 0]
            sh = int(torch.randint(-8, 9, (1,)))
            x = torch.roll(x, sh, dims=-1)
            x, y = x.clamp(0, 1).to(dev), y.to(dev)
            loss = ((model(x) - y) ** 2 * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xva[i:i + 256].to(dev)).cpu() for i in range(0, len(Xva), 256)]) if len(Xva) else Yva
            mae = (pred - Yva).abs().mean(0) if len(Xva) else torch.zeros(2)
        hist.append({"epoch": ep + 1, "loss": tot / len(Xtr), "val_mae_steer": mae[0].item(), "val_mae_throttle": mae[1].item()})
        print(f"эпоха {ep + 1:2d}  loss {tot / len(Xtr):.4f}  val MAE руль {mae[0]:.3f} газ {mae[1]:.3f}  {time.time() - t0:.1f} с")
        score = mae[0].item() + 0.5 * mae[1].item()
        if score < best:
            best = score
            out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "img": [IMG_W, IMG_H]}, out)
    (pathlib.Path(a.out).with_suffix(".json")).write_text(json.dumps({
        "datasets": a.datasets, "train": len(tr), "val": len(va), "epochs": a.epochs,
        "baseline_mae": base.tolist(), "history": hist}, ensure_ascii=False, indent=1))
    print(f"лучшая модель -> {a.out}")

if __name__ == "__main__":
    main()
