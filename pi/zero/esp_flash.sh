#!/bin/bash
# Прошивка ESP8266 прямо с Pi Zero по UART (/dev/serial0).
# Проводка: пин 11 (GPIO17) -> RST ESP, пин 13 (GPIO27) -> GPIO0/D3 ESP, плюс UART и GND.
#
#   esp_flash.sh probe            — перевести ESP в загрузчик и прочитать чип, ничего не записывая
#   esp_flash.sh write fw.bin     — прошить и запустить
#   esp_flash.sh reset            — просто перезагрузить ESP в обычный режим
#
# На время прошивки останавливает rc-car (он держит /dev/serial0), потом запускает обратно.
set -e
RST=17; BOOT=27; PORT=/dev/serial0

release() { pinctrl set $BOOT ip pn; pinctrl set $RST ip pn; }   # отпустить: на ESP свои подтяжки к 3.3 В
run_mode() { pinctrl set $BOOT ip pn; pinctrl set $RST op dl; sleep 0.1; pinctrl set $RST ip pn; }
boot_mode() {                                                     # GPIO0 в ноль во время сброса = загрузчик
  pinctrl set $BOOT op dl; pinctrl set $RST op dl; sleep 0.1
  pinctrl set $RST ip pn; sleep 0.3
}
car_stop()  { sudo systemctl stop rc-car 2>/dev/null || true; }
car_start() { sudo systemctl start rc-car 2>/dev/null || true; }
trap 'release' EXIT

case "$1" in
  probe)
    car_stop; boot_mode
    esptool --chip esp8266 -p $PORT -b 115200 --before no_reset --after no_reset chip_id || { run_mode; car_start; exit 1; }
    run_mode; car_start ;;
  write)
    [ -f "$2" ] || { echo "нет файла прошивки: $2"; exit 1; }
    car_stop; boot_mode
    esptool --chip esp8266 -p $PORT -b 460800 --before no_reset --after no_reset write_flash 0x0 "$2" \
      || esptool --chip esp8266 -p $PORT -b 115200 --before no_reset --after no_reset write_flash 0x0 "$2"
    run_mode; sleep 1; car_start; echo "ESP прошита и запущена" ;;
  reset)
    run_mode; echo "ESP перезагружена" ;;
  *) echo "использование: $0 probe | write fw.bin | reset"; exit 1 ;;
esac
