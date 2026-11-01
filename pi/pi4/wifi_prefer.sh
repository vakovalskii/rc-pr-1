#!/bin/bash
# Pi 4 держится за Wi-Fi модема (NZT2): если загрузилась раньше, чем модем поднял точку, NetworkManager
# цепляется за домашний Keenetic и сам на лучшую сеть не переходит. Таймер раз в 30 с это исправляет.
WANT=NZT2
cur=$(nmcli -t -f NAME,DEVICE con show --active | awk -F: '$2=="wlan0"{print $1}')
[ "$cur" = "$WANT" ] && exit 0
nmcli dev wifi rescan >/dev/null 2>&1; sleep 3
nmcli -t -f SSID dev wifi list | grep -qx "$WANT" || exit 0
echo "Wi-Fi: $cur → $WANT"
nmcli con up "$WANT" >/dev/null 2>&1 || echo "не вышло подключиться к $WANT"
