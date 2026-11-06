#!/bin/sh
# Журнал работы от аккумулятора: строка раз в 30 с, сразу на диск (sync).
# Последняя строка после того, как аккумулятор сел, — это время отключения.
LOG=/home/pi/battery.log
echo "# загрузка $(date -Is)" >> "$LOG"
while true; do
  T=$(vcgencmd get_throttled | cut -d= -f2)
  C=$(vcgencmd measure_temp | cut -d= -f2)
  L=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date -Is) up=$(cut -d. -f1 /proc/uptime)s throttled=$T temp=$C load=$L" >> "$LOG"
  sync "$LOG"
  sleep 30
done
