// RC-багги: реалтайм-контроллер на ESP8266 (NodeMCU).
//
// Своя Wi-Fi точка RC-BUGGY (пароль buggy1234), пульт на http://192.168.4.1.
// Те же команды принимаются по USB (115200) — так ими будет управлять Pi.
// FAILSAFE живёт здесь: 300 мс без команд -> газ в нейтраль, руль в центр.
//
//   D1 (GPIO5)  -> сигнал регулятора хода (ESC)
//   D5 (GPIO14) -> IN1 драйвера моторчика руля
//   D6 (GPIO12) -> IN2 драйвера моторчика руля
//   A0          <- движок потенциометра сервы (крайние на 3V3 и GND, не 5 В!)
//
// Руль 5-проводной сервы: режим 0 выкл, 1 ручной (стик = ШИМ моторчика,
// чтобы проверить провода), 2 по потенциометру (после калибровки).
//
// Протокол (текстом, строка на команду; одинаково по WebSocket и USB):
//   c,<id>,<руль -1000..1000>,<газ -1000..1000>    команда, 20 Гц
//   mode,<0|1|2>  cal,<l|c|r>  inv  max,<0..1000>  калибровка и настройки
// Ответ: t,<ack>,<pot>,<esc_us>,<мотор>,<failsafe>,<mode>,<potL>,<potC>,<potR>,<inv>,<max>,<источник>

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <WebSocketsServer.h>
#include <Servo.h>
#include <EEPROM.h>
#include "page.h"

const char* AP_SSID = "RC-BUGGY";
const char* AP_PASS = "buggy1234";

const int PIN_ESC = 5, PIN_IN1 = 14, PIN_IN2 = 12, PIN_LED = 2;
const uint32_t FAILSAFE_MS = 300;

struct Cfg { uint32_t magic; int16_t potL, potC, potR; uint8_t inv, mode; int16_t maxThr, kp, db, maxPwm, minPwm; };
const uint32_t MAGIC = 0xB0661E02;
Cfg cfg;

void cfgDefaults() { cfg = {MAGIC, -1, -1, -1, 0, 0, 300, 4, 12, 700, 250}; }
void cfgLoad() { EEPROM.begin(64); EEPROM.get(0, cfg); if (cfg.magic != MAGIC) cfgDefaults(); }
void cfgSave() { EEPROM.put(0, cfg); EEPROM.commit(); }
bool calibrated() { return cfg.potL >= 0 && cfg.potC >= 0 && cfg.potR >= 0 && abs(cfg.potR - cfg.potL) > 60; }

ESP8266WebServer http(80);
WebSocketsServer ws(81);
Servo esc;

int16_t cmdSteer = 0, cmdThr = 0;
int32_t lastId = -1;
uint32_t lastCmdMs = 0, stallSince = 0, stallBlockUntil = 0;
bool failsafe = true;
int pot = 0, motorOut = 0, escUs = 1500;
const char* src = "-";

int readPot() { return (analogRead(A0) + analogRead(A0)) / 2; }

void setMotor(int out) {
  motorOut = out;
  if (out > 0)      { analogWrite(PIN_IN1, out);  analogWrite(PIN_IN2, 0); }
  else if (out < 0) { analogWrite(PIN_IN1, 0);    analogWrite(PIN_IN2, -out); }
  else              { analogWrite(PIN_IN1, 0);    analogWrite(PIN_IN2, 0); }
}

void control() {
  uint32_t now = millis();
  failsafe = now - lastCmdMs > FAILSAFE_MS;
  int thr = failsafe ? 0 : constrain((int)cmdThr, -(int)cfg.maxThr, (int)cfg.maxThr);
  escUs = 1500 + thr / 2;                                   // ±1000 -> ±500 мкс
  esc.writeMicroseconds(escUs);

  pot = readPot();
  int steer = failsafe ? 0 : cmdSteer, out = 0;
  if (cfg.mode == 1 && !failsafe) {
    out = (long)steer * cfg.maxPwm / 1000;                  // ручной: стик прямо в моторчик
  } else if (cfg.mode == 2 && calibrated()) {
    // цель на потенциометре; работает при любом направлении шкалы (L>R или L<R)
    int span = steer > 0 ? cfg.potR - cfg.potC : cfg.potC - cfg.potL;
    int target = cfg.potC + (long)span * steer / 1000;
    int err = target - pot;
    int lo = min(cfg.potL, cfg.potR) - 30, hi = max(cfg.potL, cfg.potR) + 30;
    if (pot < lo || pot > hi) out = 0;                      // вылет за упоры: провода или знак не те
    else if (abs(err) > cfg.db) {
      out = constrain(err * cfg.kp, -(int)cfg.maxPwm, (int)cfg.maxPwm);
      if (abs(out) < cfg.minPwm) out = out > 0 ? cfg.minPwm : -cfg.minPwm;
    }
    if (out != 0) {                                          // крутим долго и не доезжаем — бережём моторчик
      if (!stallSince) stallSince = now;
      if (now - stallSince > 1500) { stallBlockUntil = now + 1000; stallSince = 0; }
    } else stallSince = 0;
    if (now < stallBlockUntil) out = 0;
  }
  if (cfg.inv) out = -out;
  setMotor(out);
  digitalWrite(PIN_LED, failsafe ? HIGH : LOW);             // горит = есть команды
}

void handle(char* s, const char* from) {
  if (s[0] == 'c' && s[1] == ',') {
    long id, st, th;
    if (sscanf(s + 2, "%ld,%ld,%ld", &id, &st, &th) == 3) {
      lastId = id; cmdSteer = constrain(st, -1000, 1000); cmdThr = constrain(th, -1000, 1000);
      lastCmdMs = millis(); src = from;
    }
  } else if (!strncmp(s, "mode,", 5)) { cfg.mode = constrain(atoi(s + 5), 0, 2); cfgSave(); }
  else if (!strncmp(s, "cal,", 4)) {
    int p = readPot();
    if (s[4] == 'l') cfg.potL = p; else if (s[4] == 'c') cfg.potC = p; else if (s[4] == 'r') cfg.potR = p;
    cfgSave();
  }
  else if (!strcmp(s, "inv")) { cfg.inv = !cfg.inv; cfgSave(); }
  else if (!strncmp(s, "max,", 4)) { cfg.maxThr = constrain(atoi(s + 4), 0, 1000); cfgSave(); }
  else if (!strcmp(s, "reset")) { cfgDefaults(); cfgSave(); }
}

String tele() {
  char b[128];
  snprintf(b, sizeof b, "t,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s", (long)lastId, pot, escUs, motorOut, failsafe,
           cfg.mode, cfg.potL, cfg.potC, cfg.potR, cfg.inv, cfg.maxThr, src);
  return String(b);
}

void onWs(uint8_t num, WStype_t type, uint8_t* payload, size_t len) {
  if (type == WStype_TEXT) { char buf[64]; size_t n = min(len, sizeof buf - 1); memcpy(buf, payload, n); buf[n] = 0; handle(buf, "wifi"); }
}

void setup() {
  pinMode(PIN_LED, OUTPUT); digitalWrite(PIN_LED, HIGH);
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT);
  analogWriteRange(1000); analogWriteFreq(10000);
  setMotor(0);
  esc.attach(PIN_ESC, 1000, 2000); esc.writeMicroseconds(1500);   // нейтраль сразу: регулятор взводится
  Serial.begin(115200);
  cfgLoad();

  WiFi.mode(WIFI_AP);
  WiFi.setSleepMode(WIFI_NONE_SLEEP);
  WiFi.softAP(AP_SSID, AP_PASS, 6, false, 4);
  http.on("/", [] { http.send_P(200, "text/html; charset=utf-8", PAGE); });
  http.onNotFound([] { http.sendHeader("Location", "/"); http.send(302); });
  http.begin();
  ws.begin(); ws.onEvent(onWs);
  Serial.printf("\nrc_esp готов: точка %s, http://%s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
}

char line[64]; size_t lineLen = 0;
uint32_t lastCtl = 0, lastTele = 0;

void loop() {
  http.handleClient();
  ws.loop();
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') { if (lineLen) { line[lineLen] = 0; handle(line, "usb"); lineLen = 0; } }
    else if (lineLen < sizeof line - 1) line[lineLen++] = ch;
  }
  uint32_t now = millis();
  if (now - lastCtl >= 10) { lastCtl = now; control(); }
  if (now - lastTele >= 66) {
    lastTele = now;
    String t = tele();
    ws.broadcastTXT(t);
    if (!strcmp(src, "usb")) Serial.println(t);             // Pi получает телеметрию, только когда сам командует
  }
}
