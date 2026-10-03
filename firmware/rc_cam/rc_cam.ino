// RC-багги: всё на одной ESP32-CAM (AI-Thinker) — видео, ход, руль.
//
// Своя Wi-Fi точка RC-BUGGY (пароль 12345678), пульт на http://192.168.4.1 и
// выскакивает сам (captive portal). Видео MJPEG на :81/stream — фоном пульта.
// Управление по WebSocket :82 и тем же текстом по USB (115200) — для Pi.
//
//   GPIO13 -> сигнал регулятора хода (ESC), 50 Гц
//   GPIO14 -> IN1 драйвера моторчика руля (DRV8837)
//   GPIO15 -> IN2 драйвера
//   GPIO4  -> вспышка-светодиод (фара)
//
// Видео: пульт тянет /jpg по одному кадру (задержка = передача одного кадра);
// /stream — обычный MJPEG-поток для Pi и сторонних плееров.
//
// Руль: у ESP32-CAM все свободные пины с АЦП сидят на ADC2, а ADC2 занят Wi-Fi,
// поэтому потенциометр 5-проводной сервы здесь не прочитать. Руль работает
// без обратной связи: стик = ШИМ моторчика. Чтобы не жечь моторчик в упоре,
// полный ток идёт только STEER_PUSH_MS, дальше — удержание STEER_HOLD.
//
// Протокол: c,<id>,<руль -1000..1000>,<газ -1000..1000>   max,<0..1000>  acc,<мс>  brk,<мс>  inv  light,<0|1>  res,<0|1|2>  q,<8..40>
// Ответ:    t,<ack>,<esc_us>,<мотор>,<failsafe>,<inv>,<max>,<источник>,<acc>,<brk>,<fps>,<клиентов>,<res>,<q>,<RSSI телефона, дБм>

#include <WiFi.h>
#include <DNSServer.h>
#include <WebSocketsServer.h>
#include <Preferences.h>
#include "esp_camera.h"
#include "esp_wifi.h"
#include "esp_http_server.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "page.h"

const char* AP_SSID = "RC-BUGGY";
const char* AP_PASS = "12345678";

const int PIN_ESC = 13, PIN_IN1 = 14, PIN_IN2 = 15, PIN_FLASH = 4;
const uint32_t FAILSAFE_MS = 300, FLIP_HOLD_MS = 150, FAILSAFE_STOP_MS = 150;
const uint32_t STEER_PUSH_MS = 400;           // полный ток на моторчик руля
const int STEER_HOLD = 350;                   // из 1000 — удерживать упор, не сжигая
const int ESC_RES = 14;                       // бит ШИМ регулятора
// каналы LEDC (ядро 2.x): у пары каналов общий таймер, поэтому 50 Гц хода и 10 кГц руля разводим по разным парам
const int CH_ESC = 0, CH_IN1 = 2, CH_IN2 = 3;     // камера сидит на канале 7 / таймере 3

// AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

Preferences prefs;
struct { int maxThr = 300, accMs = 600, brkMs = 1500; bool inv = false; int res = 1, q = 12; } cfg;
const framesize_t RES[] = {FRAMESIZE_QVGA, FRAMESIZE_HVGA, FRAMESIZE_VGA};   // 320x240, 480x320, 640x480

DNSServer dns;
WebSocketsServer ws(82);
httpd_handle_t httpPage = nullptr, httpStream = nullptr;

volatile int16_t cmdSteer = 0, cmdThr = 0;
volatile int32_t lastId = -1;
volatile uint32_t lastCmdMs = 0;
bool failsafe = true;
float thrOut = 0; int lastSign = 0; uint32_t zeroSince = 0;
int escUs = 1500, motorOut = 0, steerDir = 0; uint32_t steerSince = 0;
const char* src = "-";
volatile uint32_t frames = 0; float fps = 0; volatile int streamClients = 0;

void cfgLoad() {
  prefs.begin("rc", false);
  cfg.maxThr = prefs.getInt("max", 300); cfg.accMs = prefs.getInt("acc", 600);
  cfg.brkMs = prefs.getInt("brk", 1500); cfg.inv = prefs.getBool("inv", false);
  cfg.res = prefs.getInt("res", 1); cfg.q = prefs.getInt("q", 12);
}

void escWrite(int us) { ledcWrite(CH_ESC, (uint32_t)us * ((1 << ESC_RES) - 1) / 20000); }

void setMotor(int out) {
  motorOut = out;
  if (cfg.inv) out = -out;
  ledcWrite(CH_IN1, out > 0 ? out : 0);
  ledcWrite(CH_IN2, out < 0 ? -out : 0);
}

void control() {
  uint32_t now = millis();
  failsafe = now - lastCmdMs > FAILSAFE_MS;
  // --- ход: рампа и смена направления через нейтраль (как на ESP8266)
  int target = failsafe ? 0 : constrain((int)cmdThr, -cfg.maxThr, cfg.maxThr);
  int tsign = (target > 0) - (target < 0);
  if (tsign && lastSign && tsign != lastSign && (thrOut != 0 || now - zeroSince < FLIP_HOLD_MS)) target = 0;
  bool braking = abs(target) < fabsf(thrOut) || (target > 0) != (thrOut > 0);
  float ms = failsafe ? FAILSAFE_STOP_MS : (braking ? cfg.brkMs : cfg.accMs);
  float step = 1000.0f * 10 / max(ms, 10.0f);
  if (thrOut < target) thrOut = min(thrOut + step, (float)target);
  else if (thrOut > target) thrOut = max(thrOut - step, (float)target);
  if (thrOut == 0) { if (!zeroSince) zeroSince = now; } else { zeroSince = 0; lastSign = thrOut > 0 ? 1 : -1; }
  if (thrOut == 0 && zeroSince && now - zeroSince >= FLIP_HOLD_MS) lastSign = 0;
  escUs = 1500 + (int)(thrOut / 2);
  escWrite(escUs);
  // --- руль без обратной связи: толчок полным током, потом удержание
  int s = failsafe ? 0 : cmdSteer;
  int dir = (s > 150) - (s < -150);
  if (dir != steerDir) { steerDir = dir; steerSince = now; }
  int out = 0;
  if (dir) {
    int full = map(abs(s), 150, 1000, 500, 1000);
    out = dir * (now - steerSince < STEER_PUSH_MS ? full : min(full, STEER_HOLD));
  }
  setMotor(out);
}

void applyCam() {
  sensor_t* sn = esp_camera_sensor_get();
  if (sn) { sn->set_framesize(sn, RES[cfg.res]); sn->set_quality(sn, cfg.q); }
}

void handle(const char* s, const char* from) {
  if (s[0] == 'c' && s[1] == ',') {
    long id, st, th;
    if (sscanf(s + 2, "%ld,%ld,%ld", &id, &st, &th) == 3) {
      lastId = id; cmdSteer = constrain(st, -1000, 1000); cmdThr = constrain(th, -1000, 1000);
      lastCmdMs = millis(); src = from;
    }
  } else if (!strncmp(s, "max,", 4)) { cfg.maxThr = constrain(atoi(s + 4), 0, 1000); prefs.putInt("max", cfg.maxThr); }
  else if (!strncmp(s, "acc,", 4)) { cfg.accMs = constrain(atoi(s + 4), 50, 5000); prefs.putInt("acc", cfg.accMs); }
  else if (!strncmp(s, "brk,", 4)) { cfg.brkMs = constrain(atoi(s + 4), 50, 5000); prefs.putInt("brk", cfg.brkMs); }
  else if (!strcmp(s, "inv")) { cfg.inv = !cfg.inv; prefs.putBool("inv", cfg.inv); }
  else if (!strncmp(s, "light,", 6)) { digitalWrite(PIN_FLASH, atoi(s + 6) ? HIGH : LOW); }
  else if (!strncmp(s, "res,", 4)) { cfg.res = constrain(atoi(s + 4), 0, 2); prefs.putInt("res", cfg.res); applyCam(); }
  else if (!strncmp(s, "q,", 2)) { cfg.q = constrain(atoi(s + 2), 8, 40); prefs.putInt("q", cfg.q); applyCam(); }
}

int staRssi() {                                   // как ESP слышит телефон: для замера антенны и дальности
  wifi_sta_list_t l;
  if (esp_wifi_ap_get_sta_list(&l) != ESP_OK || l.num == 0) return 0;
  int best = -127;
  for (int i = 0; i < l.num; i++) best = max(best, (int)l.sta[i].rssi);
  return best;
}

String tele() {
  char b[160];
  snprintf(b, sizeof b, "t,%ld,%d,%d,%d,%d,%d,%s,%d,%d,%.1f,%d,%d,%d,%d", (long)lastId, escUs, motorOut, failsafe,
           cfg.inv, cfg.maxThr, src, cfg.accMs, cfg.brkMs, fps, (int)streamClients, cfg.res, cfg.q, staRssi());
  return String(b);
}

void onWs(uint8_t num, WStype_t type, uint8_t* payload, size_t len) {
  if (type == WStype_TEXT) { char buf[64]; size_t n = min(len, sizeof buf - 1); memcpy(buf, payload, n); buf[n] = 0; handle(buf, "wifi"); }
}

// ---------------------------------------------------------------- HTTP ---
esp_err_t pageHandler(httpd_req_t* req) {
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  return httpd_resp_send(req, PAGE, HTTPD_RESP_USE_STRLEN);
}
esp_err_t redirect404(httpd_req_t* req, httpd_err_code_t) {   // captive portal: любые адреса -> пульт
  httpd_resp_set_status(req, "302 Found");
  httpd_resp_set_hdr(req, "Location", "http://192.168.4.1/");
  return httpd_resp_send(req, nullptr, 0);
}

// Один свежий кадр на запрос. Страница просит следующий, только когда показала
// предыдущий — очередь из кадров не копится ни в сокете, ни в браузере.
esp_err_t jpgHandler(httpd_req_t* req) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { httpd_resp_send_500(req); return ESP_FAIL; }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  esp_err_t r = httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  frames++;
  return r;
}

#define BOUNDARY "rcframe"
esp_err_t streamHandler(httpd_req_t* req) {
  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" BOUNDARY);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  streamClients++;
  char head[96]; esp_err_t res = ESP_OK;
  while (res == ESP_OK) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(5); continue; }
    int n = snprintf(head, sizeof head, "\r\n--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    res = httpd_resp_send_chunk(req, head, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    frames++;
  }
  streamClients--;
  return res;
}

void startHttp() {
  httpd_config_t c = HTTPD_DEFAULT_CONFIG();
  c.server_port = 80; c.ctrl_port = 32768; c.lru_purge_enable = true;
  if (httpd_start(&httpPage, &c) == ESP_OK) {
    httpd_uri_t root = {"/", HTTP_GET, pageHandler, nullptr};
    httpd_register_uri_handler(httpPage, &root);
    httpd_register_err_handler(httpPage, HTTPD_404_NOT_FOUND, redirect404);
  }
  httpd_config_t s = HTTPD_DEFAULT_CONFIG();
  s.server_port = 81; s.ctrl_port = 32769; s.max_open_sockets = 3;
  if (httpd_start(&httpStream, &s) == ESP_OK) {
    httpd_uri_t st = {"/stream", HTTP_GET, streamHandler, nullptr};      // поток для Pi и сторонних плееров
    httpd_register_uri_handler(httpStream, &st);
    httpd_uri_t jpg = {"/jpg", HTTP_GET, jpgHandler, nullptr};            // кадр по запросу — для пульта
    httpd_register_uri_handler(httpStream, &jpg);
  }
}

bool startCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_7; c.ledc_timer = LEDC_TIMER_3;     // не пересекаемся с ШИМ хода и руля
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM; c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000; c.pixel_format = PIXFORMAT_JPEG;
  c.grab_mode = CAMERA_GRAB_LATEST;                                 // всегда свежий кадр, без очереди
  if (psramFound()) { c.frame_size = FRAMESIZE_VGA; c.jpeg_quality = 12; c.fb_count = 2; c.fb_location = CAMERA_FB_IN_PSRAM; }
  else              { c.frame_size = FRAMESIZE_QVGA; c.jpeg_quality = 14; c.fb_count = 1; c.fb_location = CAMERA_FB_IN_DRAM; }
  esp_err_t e = esp_camera_init(&c);
  if (e == ESP_OK) applyCam();       // буферы под VGA, а реальный размер — из настроек
  Serial.printf("камера: %s, PSRAM %s, %s\n", e == ESP_OK ? "ok" : "ОШИБКА", psramFound() ? "есть" : "нет",
                psramFound() ? "VGA 640x480" : "QVGA 320x240");
  return e == ESP_OK;
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);      // просадка от Wi-Fi при старте не должна ребутить плату
  Serial.begin(115200);
  pinMode(PIN_FLASH, OUTPUT); digitalWrite(PIN_FLASH, LOW);
  ledcSetup(CH_ESC, 50, ESC_RES); ledcAttachPin(PIN_ESC, CH_ESC); escWrite(1500);   // нейтраль сразу: регулятор взводится
  ledcSetup(CH_IN1, 10000, 10); ledcAttachPin(PIN_IN1, CH_IN1);
  ledcSetup(CH_IN2, 10000, 10); ledcAttachPin(PIN_IN2, CH_IN2); setMotor(0);
  cfgLoad();
  startCamera();
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 4);
  WiFi.setSleep(false);
  dns.start(53, "*", WiFi.softAPIP());
  startHttp();
  ws.begin(); ws.onEvent(onWs);
  Serial.printf("rc_cam готов: точка %s, http://%s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
}

char line[64]; size_t lineLen = 0;
uint32_t lastCtl = 0, lastTele = 0, lastFps = 0, framesAtFps = 0;

void loop() {
  dns.processNextRequest();
  ws.loop();
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') { if (lineLen) { line[lineLen] = 0; handle(line, "usb"); lineLen = 0; } }
    else if (lineLen < sizeof line - 1) line[lineLen++] = ch;
  }
  uint32_t now = millis();
  if (now - lastCtl >= 10) { lastCtl = now; control(); }
  if (now - lastFps >= 1000) { fps = (frames - framesAtFps) * 1000.0f / (now - lastFps); framesAtFps = frames; lastFps = now; }
  if (now - lastTele >= 66) {
    lastTele = now;
    String t = tele();
    ws.broadcastTXT(t);
    if (!strcmp(src, "usb")) Serial.println(t);
  }
  delay(1);
}
