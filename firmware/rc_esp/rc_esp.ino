// RC-багги: реалтайм-контроллер на ESP8266 (NodeMCU).
//
// Своя Wi-Fi точка RC-BUGGY (пароль 12345678), пульт на http://192.168.4.1 —
// и сам выскакивает при подключении (captive portal: DNS отвечает ESP на любой адрес).
// Те же команды принимаются по USB (115200) — так ими будет управлять Pi.
// FAILSAFE живёт здесь: 300 мс без команд -> газ в нейтраль, руль в центр.
//
//   D1 (GPIO5)  -> сигнал регулятора хода (ESC)
//   D5 (GPIO14) -> IN1 драйвера моторчика руля
//   D6 (GPIO12) -> IN2 драйвера моторчика руля
//   A0          <- движок потенциометра сервы (крайние на 3V3 и GND, не 5 В!)
//
// Руль 5-проводной сервы: режим 0 выкл, 1 ручной (стик = ШИМ моторчика; отпустил стик —
// после калибровки колёса сами доезжают в центр по потенциометру), 2 по потенциометру
// (стик = угол колёс).
//
// Протокол (текстом, строка на команду; одинаково по WebSocket и USB):
//   c,<id>,<руль -1000..1000>,<газ -1000..1000>    команда, 20 Гц
//   mode,<0|1|2>  cal,<l|c|r>  inv  max,<0..1000>  acc,<мс>  brk,<мс>  калибровка и настройки
//   pwm,<200..1000> сила руля (потолок ШИМ моторчика)   kp,<1..30> жёсткость удержания   minpwm,<0..800>
//   brake,<мкс>,<мс>  активный тормоз при отпускании газа (импульс «назад» силой мкс на мс; 0 — выкл)
// Ответ: t,<ack>,<pot>,<esc_us>,<мотор>,<failsafe>,<mode>,<potL>,<potC>,<potR>,<inv>,<max>,<источник>,<разгон мс>,<торможение мс>,<pwm>,<kp>,<minpwm>,<тормоз мкс>,<тормоз мс>

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <DNSServer.h>
#include <WebSocketsServer.h>
#include <Servo.h>
#include <EEPROM.h>
#include "page.h"

const char* AP_SSID = "RC-BUGGY";
const char* AP_PASS = "12345678";

const int PIN_ESC = 5, PIN_IN1 = 14, PIN_IN2 = 12, PIN_LED = 2;
const uint32_t FAILSAFE_MS = 300;

struct Cfg { uint32_t magic; int16_t potL, potC, potR; uint8_t inv, mode; int16_t maxThr, kp, db, maxPwm, minPwm, accMs, brkMs;
             int16_t brakeUs, brakeMs; };                  // активный тормоз: дописан в конец — калибровка в EEPROM сохраняется
const uint32_t MAGIC = 0xB0661E03;
Cfg cfg;

// accMs/brkMs: за сколько мс газ прошёл бы всю шкалу 0..100% при разгоне и при торможении
void cfgDefaults() { cfg = {MAGIC, -1, -1, -1, 0, 0, 300, 4, 12, 700, 250, 600, 1500, 150, 300}; }
void cfgLoad() {
  EEPROM.begin(64); EEPROM.get(0, cfg);
  if (cfg.magic != MAGIC) cfgDefaults();
  if (cfg.brakeUs < 0 || cfg.brakeUs > 400) cfg.brakeUs = 150;   // EEPROM от прошлой прошивки: полей тормоза ещё не было
  if (cfg.brakeMs < 0 || cfg.brakeMs > 1500) cfg.brakeMs = 300;
}
void cfgSave() { EEPROM.put(0, cfg); EEPROM.commit(); }
bool calibrated() { return cfg.potL >= 0 && cfg.potC >= 0 && cfg.potR >= 0 && abs(cfg.potR - cfg.potL) > 60; }

ESP8266WebServer http(80);
DNSServer dns;                                   // любой адрес -> ESP: телефон сам открывает пульт
WebSocketsServer ws(81);
Servo esc;

int16_t cmdSteer = 0, cmdThr = 0;
int32_t lastId = -1;
uint32_t lastCmdMs = 0, stallSince = 0, stallBlockUntil = 0;
int stallPot = 0;
const int CENTER_DB = 30;                                    // |стик| < 3% — «отпустил руль»
bool failsafe = true;
int pot = 0, motorOut = 0, escUs = 1500;
float thrOut = 0;                                          // газ после рампы, -1000..1000
int lastSign = 0; uint32_t zeroSince = 0, brakeUntil = 0;
const uint32_t FLIP_HOLD_MS = 150, FAILSAFE_STOP_MS = 150;
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
  int target = failsafe ? 0 : constrain((int)cmdThr, -(int)cfg.maxThr, (int)cfg.maxThr);
  // смена направления: сначала до нуля, нейтраль FLIP_HOLD_MS, потом в другую сторону
  int tsign = (target > 0) - (target < 0);
  if (tsign && lastSign && tsign != lastSign) {
    if (thrOut != 0 || now - zeroSince < FLIP_HOLD_MS) target = 0;
  }
  // активный тормоз: газ вперёд отпустили (или потянули назад, или failsafe) — регулятору на brakeMs импульс
  // «назад» силой brakeUs: автомобильный регулятор после хода вперёд понимает его как тормоз, а не задний ход.
  // Без него машинка просто катится (накат ~20 см на 18%).
  if (cfg.brakeUs > 0 && cfg.brakeMs > 0 && target <= 0 && thrOut > 40 && !brakeUntil) {
    brakeUntil = now + cfg.brakeMs; thrOut = 0;
  }
  bool braked = false;
  if (brakeUntil) {
    if (target > 0 || now >= brakeUntil) { brakeUntil = 0; zeroSince = now; }   // снова газ — тормоз снят; отжали — нейтраль
    else {
      escUs = 1500 - cfg.brakeUs; esc.writeMicroseconds(escUs);
      lastSign = 1; zeroSince = now; braked = true;        // после тормоза задний ход — как обычно, через нейтраль
    }
  }
  if (!braked) {
  // рампа: разгон и торможение с разной скоростью; failsafe гасит быстро
  bool braking = abs(target) < fabsf(thrOut) || (target > 0) != (thrOut > 0);
  float ms = failsafe ? FAILSAFE_STOP_MS : (braking ? cfg.brkMs : cfg.accMs);
  float step = 1000.0f * 10 / max(ms, 10.0f);              // за такт control() = 10 мс
  if (thrOut < target) thrOut = min(thrOut + step, (float)target);
  else if (thrOut > target) thrOut = max(thrOut - step, (float)target);
  if (thrOut == 0) { if (!zeroSince) zeroSince = now; } else { zeroSince = 0; lastSign = thrOut > 0 ? 1 : -1; }
  if (thrOut == 0 && zeroSince && now - zeroSince >= FLIP_HOLD_MS) lastSign = 0;
  escUs = 1500 + (int)(thrOut / 2);                          // ±1000 -> ±500 мкс
  esc.writeMicroseconds(escUs);
  }

  pot = readPot();
  int steer = failsafe ? 0 : cmdSteer, out = 0;
  bool hold = false; int aim = 0;
  if (cfg.mode == 1 && !failsafe && abs(steer) >= CENTER_DB) {
    out = (long)steer * cfg.maxPwm / 1000;                  // ручной: стик прямо в моторчик
  } else if (cfg.mode == 1 && calibrated()) {
    hold = true; aim = cfg.potC;                          // ручной, стик отпущен (или failsafe) — колёса сами прямо
  } else if (cfg.mode == 2 && calibrated()) {
    // цель на потенциометре; работает при любом направлении шкалы (L>R или L<R)
    int span = steer > 0 ? cfg.potR - cfg.potC : cfg.potC - cfg.potL;
    hold = true; aim = cfg.potC + (long)span * steer / 1000;
  }
  if (hold) {
    // знак: «+» у моторчика (до inv) — это вправо, как у стика; вправо потенциометр идёт к potR
    int dir = cfg.potR > cfg.potL ? 1 : -1;
    int err = aim - pot;
    int lo = min(cfg.potL, cfg.potR) - 30, hi = max(cfg.potL, cfg.potR) + 30;
    if (pot < lo || pot > hi) out = 0;                      // вылет за упоры: провода или знак не те
    else if (abs(err) > cfg.db) {
      out = constrain(err * cfg.kp, -(int)cfg.maxPwm, (int)cfg.maxPwm) * dir;
      if (abs(out) < cfg.minPwm) out = out > 0 ? cfg.minPwm : -cfg.minPwm;
    }
    // заклинило: моторчик крутит, а потенциометр 0.8 с стоит на месте — пауза, бережём моторчик.
    // (по одному только времени работы нельзя: при езде к далёкой цели это нормальная долгая работа)
    if (out != 0) {
      if (!stallSince || abs(pot - stallPot) > 15) { stallSince = now; stallPot = pot; }
      else if (now - stallSince > 800) { stallBlockUntil = now + 600; stallSince = 0; }
    } else stallSince = 0;
    if (now < stallBlockUntil) out = 0;
  } else stallSince = 0;
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
  else if (!strncmp(s, "acc,", 4)) { cfg.accMs = constrain(atoi(s + 4), 50, 5000); cfgSave(); }
  else if (!strncmp(s, "brk,", 4)) { cfg.brkMs = constrain(atoi(s + 4), 50, 5000); cfgSave(); }
  else if (!strncmp(s, "brake,", 6)) {                     // brake,<мкс 0..400>,<мс 0..1500>  (0 — без тормоза)
    const char* c = strchr(s + 6, ',');
    cfg.brakeUs = constrain(atoi(s + 6), 0, 400); if (c) cfg.brakeMs = constrain(atoi(c + 1), 0, 1500); cfgSave();
  }
  else if (!strncmp(s, "pwm,", 4)) { cfg.maxPwm = constrain(atoi(s + 4), 200, 1000); cfgSave(); }
  else if (!strncmp(s, "kp,", 3)) { cfg.kp = constrain(atoi(s + 3), 1, 30); cfgSave(); }
  else if (!strncmp(s, "minpwm,", 7)) { cfg.minPwm = constrain(atoi(s + 7), 0, 800); cfgSave(); }
  else if (!strcmp(s, "reset")) { cfgDefaults(); cfgSave(); }
}

String tele() {
  char b[160];
  snprintf(b, sizeof b, "t,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d", (long)lastId, pot, escUs, motorOut, failsafe,
           cfg.mode, cfg.potL, cfg.potC, cfg.potR, cfg.inv, cfg.maxThr, src, cfg.accMs, cfg.brkMs, cfg.maxPwm, cfg.kp, cfg.minPwm, cfg.brakeUs, cfg.brakeMs);
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
  http.onNotFound([] { http.sendHeader("Location", "http://192.168.4.1/"); http.send(302); });   // проверки captive portal тоже сюда
  dns.start(53, "*", WiFi.softAPIP());
  http.begin();
  ws.begin(); ws.onEvent(onWs);
  Serial.printf("\nrc_esp готов: точка %s, http://%s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
}

char line[64]; size_t lineLen = 0;
uint32_t lastCtl = 0, lastTele = 0;

void loop() {
  dns.processNextRequest();
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
