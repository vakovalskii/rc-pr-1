#!/bin/bash
# Собрать прошивку ESP8266 на маке и залить её через Pi Zero (без USB и кнопок).
#   firmware/flash_via_zero.sh            — собрать rc_esp и прошить
#   firmware/flash_via_zero.sh probe      — только проверить, что малина видит загрузчик ESP
set -e
ZERO=${ZERO:-pi@192.168.1.137}
cd "$(dirname "$0")"
if [ "$1" = "probe" ]; then ssh "$ZERO" '~/rc/esp_flash.sh probe'; exit; fi
OUT=$(mktemp -d)
arduino-cli compile --fqbn esp8266:esp8266:nodemcuv2 --output-dir "$OUT" rc_esp >/dev/null
scp -q "$OUT/rc_esp.ino.bin" "$ZERO:/tmp/rc_esp.bin"
ssh "$ZERO" '~/rc/esp_flash.sh write /tmp/rc_esp.bin'
