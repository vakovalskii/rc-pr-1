#!/bin/bash
# Pi Zero 2 W: релей с пультом на :8080 + борт (камера, ESP8266 по UART).
# Пульт: http://<адрес малины>:8080
cd "$(dirname "$0")"
pkill -f "python3 -u relay.py" ; pkill -f "python3 -u pi_car.py" ; sleep 1
setsid nohup python3 -u relay.py --port 8080 > relay.log 2>&1 < /dev/null &
sleep 3
setsid nohup python3 -u pi_car.py --esp /dev/serial0 --rotate180 --url ws://127.0.0.1:8080/ws/car > car.log 2>&1 < /dev/null &
echo "запущено: релей и борт"
