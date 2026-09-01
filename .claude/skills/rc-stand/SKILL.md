---
name: rc-stand
description: Поднять, проверить или остановить стенд RC-багги (релей + машинка + веб-пульт) в этом репозитории. Использовать, когда просят запустить стенд, симулятор, полигон, езду по съёмке, показать статус или логи, начать/остановить запись заезда, дать адрес пульта для телефона.
---

# Стенд RC-багги: запуск одной командой

Всё через `python3 stand.py` из корня репозитория. Он сам берёт `.venv/bin/python`,
если venv есть, кладёт PID и логи в `.stand/`, при старте гасит прошлые процессы.

| что просят | команда |
|---|---|
| поднять стенд с синтетикой | `python3 stand.py start` |
| полигон по карте из съёмки | `python3 stand.py start --car polygon --data data/<имя>` |
| езда по кадрам съёмки | `python3 stand.py start --car video --data data/<имя>` |
| только релей (ждём настоящую машинку) | `python3 stand.py start --car none` |
| статус, адреса для телефона | `python3 stand.py status` |
| логи машинки / релея | `python3 stand.py logs car` / `logs relay` |
| записать заезд | `python3 stand.py rec start <имя>` … `rec stop` → `datasets/<имя>/` |
| автопилот фоном (машинка ездит сама) | `python3 stand.py pilot --seconds 3600` |
| остановить всё | `python3 stand.py stop` |

После `start`/`status` обязательно сообщить человеку адрес пульта
(`http://<ip-мака>:8080`) и карты (`/map.html`) — он открывает их с телефона.

## Конвейер до полигона (если `data/<имя>/map` ещё нет)

```bash
.venv/bin/python prep_video.py дом.mov data/dom --fps 4      # кадры
.venv/bin/python recon.py data/dom --cam-height 0.12         # COLMAP → позы, вердикт
.venv/bin/python map_build.py data/dom                        # карта занятости
python3 stand.py start --car polygon --data data/dom
```

Синтетическая комната для проверки конвейера: `synth_room.py data/synth --frames 150`,
дальше `recon.py data/synth --camera SIMPLE_PINHOLE`.

## Обучение и автопилот

```bash
.venv/bin/python train_bc.py datasets/<заезд> ...            # → models/bc.pt
.venv/bin/python pilot.py --seconds 60 --reset               # едет сам, печатает пробег и удары
```
Пока едет автопилот, релей игнорирует нули с открытого пульта; стрелки или палец
перехватывают управление, R / «↺ старт» возвращает на старт.

## Грабли
- Порт 8080 занят → `python3 stand.py stop`, потом снова `start`.
- `websockets` лезет через системный `HTTPS_PROXY` даже на localhost — в коде уже `proxy=None`.
- В COLMAP брать только модели камеры с одним фокусом (SIMPLE_RADIAL, SIMPLE_PINHOLE):
  при съёмке на постоянной высоте без наклонов PINHOLE/OPENCV сплющивают мир по вертикали.
