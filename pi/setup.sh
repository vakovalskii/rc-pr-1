#!/usr/bin/env bash
# Готовит Raspberry Pi 4 (Raspberry Pi OS Bookworm 64-bit) к роли головы машинки.
#   sudo ./pi/setup.sh ws://<ip-мака>:8080/ws/car
set -euo pipefail
RELAY=${1:?укажи релей: sudo ./pi/setup.sh ws://<ip-мака>:8080/ws/car}
DIR=$(cd "$(dirname "$0")/.." && pwd)
CFG=/boot/firmware/config.txt

apt-get update
apt-get install -y --no-install-recommends python3-picamera2 python3-websockets python3-pil python3-serial

reboot_needed=0
if ! grep -q '^dtoverlay=pwm-2chan' "$CFG"; then
  echo 'dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4' >> "$CFG"; reboot_needed=1
  echo "добавил аппаратный ШИМ на GPIO12/13"
fi

printf 'RELAY=%s\nARGS=\n' "$RELAY" > /etc/default/rc-car
sed "s|__DIR__|$DIR|g" "$DIR/pi/rc-car.service" > /etc/systemd/system/rc-car.service
systemctl daemon-reload
systemctl enable rc-car.service

echo "камера:"; rpicam-hello --list-cameras 2>&1 | head -5 || true
if [ $reboot_needed = 1 ]; then
  echo "перезагрузи Pi (sudo reboot) — сервис стартует сам"
else
  systemctl restart rc-car.service; sleep 3; systemctl --no-pager status rc-car.service | tail -5
fi
echo "лог: journalctl -u rc-car -f    настройки: /etc/default/rc-car (ARGS=--max-throttle 0.3 --steer-trim 0 ...)"
