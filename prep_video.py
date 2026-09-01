"""
Готовит съёмку к использованию: режет видео на кадры и выбрасывает смазанные.

Смазанные кадры вредят дважды — в симуляторе дают кашу, а в COLMAP ломают
восстановление поз. Дешевле выкинуть их сразу, чем разбираться потом.

    python3 prep_video.py дом.mov data/dom --fps 4
"""
import argparse, json, pathlib, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageFilter

def extract(video, outdir, fps):
    frames = outdir / "raw"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"fps={fps}", "-q:v", "2", str(frames / "%05d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(frames.glob("*.jpg"))

def sharpness(path):
    """Дисперсия карты краёв: чем меньше, тем более смазан кадр."""
    im = Image.open(path).convert("L")
    im.thumbnail((320, 320))
    return float(np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("outdir")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--drop", type=float, default=0.15,
                   help="доля самых смазанных кадров на выброс")
    a = p.parse_args()

    outdir = pathlib.Path(a.outdir)
    print(f"режу {a.video} на {a.fps} кадров/с ...")
    raw = extract(pathlib.Path(a.video), outdir, a.fps)
    if not raw:
        sys.exit("ffmpeg не дал ни одного кадра — проверь путь к видео")

    print(f"получено {len(raw)}, считаю резкость ...")
    scored = sorted(((sharpness(f), f) for f in raw), key=lambda x: x[0])
    cut = int(len(scored) * a.drop)
    keep = sorted(f for _, f in scored[cut:])

    good = outdir / "frames"
    if good.exists():
        shutil.rmtree(good)
    good.mkdir()
    for i, f in enumerate(keep):
        shutil.copy(f, good / f"{i:05d}.jpg")

    w, h = Image.open(keep[0]).size
    (outdir / "manifest.json").write_text(json.dumps(
        {"frames": len(keep), "width": w, "height": h, "fps": a.fps,
         "dropped": cut, "source": str(a.video)}, ensure_ascii=False, indent=1))

    lo, hi = scored[cut][0], scored[-1][0]
    print(f"оставил {len(keep)}, выбросил {cut} смазанных")
    print(f"резкость от {lo:.0f} до {hi:.0f}  |  кадр {w}x{h}")
    print(f"готово: {good}")

if __name__ == "__main__":
    main()
