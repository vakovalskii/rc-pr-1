#!/bin/bash
# Zero как сетевая карта по USB-кабелю к Pi 4 (CDC ECM через configfs): связь между платами без Wi-Fi.
# Кабель — в порт «USB» на Zero (не PWR), он же её и питает. Адреса: Pi 4 = 10.55.0.1, Zero = 10.55.0.2.
# MAC фиксированы, чтобы у Pi 4 интерфейс всегда назывался одинаково (enx025500000001).
set -e
G=/sys/kernel/config/usb_gadget/rcnet
modprobe libcomposite
if [ ! -d $G ]; then
mkdir -p $G && cd $G
# full-speed (12 Мбит/с, на деле ~6): на high-speed связь рвалась дважды в секунду — кабель/питание не тянут.
# Видео 3 Мбит/с и команды влезают. С коротким хорошим кабелем можно попробовать убрать эту строку.
echo full-speed > max_speed
echo 0x1d6b > idVendor; echo 0x0104 > idProduct            # Linux Foundation, Multifunction Composite Gadget
mkdir -p strings/0x409; echo rc-buggy > strings/0x409/manufacturer; echo "Pi Zero USB net" > strings/0x409/product
mkdir -p configs/c.1 functions/ecm.usb0
echo 02:55:00:00:00:02 > functions/ecm.usb0/dev_addr      # Zero
echo 02:55:00:00:00:01 > functions/ecm.usb0/host_addr     # Pi 4
ln -s functions/ecm.usb0 configs/c.1/
ls /sys/class/udc | head -1 > UDC
fi
# Адрес ставим сами: NetworkManager гаджет-интерфейсы считает unmanaged и профиль rc-usb не применял —
# после перезагрузки кабель молчал. Маршрут в интернет через Pi 4 — запасной (metric 700, Wi-Fi главнее).
for i in $(seq 1 50); do [ -e /sys/class/net/usb0 ] && break; sleep 0.1; done
ip link set usb0 up
ip addr replace 10.55.0.2/24 dev usb0
ip route replace default via 10.55.0.1 dev usb0 metric 700
