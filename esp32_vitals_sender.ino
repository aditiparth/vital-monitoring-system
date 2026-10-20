/*
 * ESP32 Vitals Sender Firmware
 * AI-Powered Post-Surgery Patient Monitoring System
 * EPICS Project | VIT Bhopal University
 *
 * Hardware:
 *   - MAX30105   → GPIO 21 (SDA), 22 (SCL)  — Heart Rate + SpO2
 *   - MPU6050    → GPIO 21 (SDA), 22 (SCL)  — Fall Detection (accel + gyro)
 *   - DS18B20    → GPIO 4                   — Temperature
 *   - LCD 16x2   → GPIO 21 (SDA), 22 (SCL)  — Display
 *   - Buzzer     → GPIO 25                  — Alarm
 *
 * Output (Serial @ 115200 baud):
 *   CSV line every 2 s:
 *   HR,SpO2,RespRate,Temp_C,AccelMag,FallDetected
 *
 * Dependencies (install via Arduino Library Manager):
 *   - SparkFun MAX3010x Sensor Library  (SparkFun_MAX3010x_Pulse_And_Proximity_Sensor_Library)
 *   - MPU6050 by Electronic Cats        (or Adafruit MPU6050)
 *   - DallasTemperature + OneWire
 *   - LiquidCrystal_I2C
 */

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <MPU6050.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ── Pin definitions ──────────────────────────────────────────────────────────
#define ONE_WIRE_BUS  4
#define BUZZER_PIN   25

// ── Hardware objects ─────────────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);
MAX30105 particleSensor;
MPU6050 mpu;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

// ── MAX30105 SpO2 / HR buffers ────────────────────────────────────────────────
#define BUFFER_LENGTH  100
uint32_t irBuffer[BUFFER_LENGTH];
uint32_t redBuffer[BUFFER_LENGTH];
int32_t  spo2Value;
int8_t   validSPO2;
int32_t  heartRateValue;
int8_t   validHeartRate;

// ── Respiratory rate estimation ───────────────────────────────────────────────
// We derive RespRate from IR signal peak-to-peak variation over 30 s window
#define RR_WINDOW    30000UL   // 30 seconds in ms
float    rrPeakBuf[60];
uint8_t  rrPeakIdx   = 0;
uint32_t rrLastPeak  = 0;
float    lastIR      = 0;
bool     irRising    = false;
float    respRate    = 16.0;   // initialise to normal default

// ── Fall detection ────────────────────────────────────────────────────────────
// Using MPU6050 total acceleration magnitude threshold
// Clinical standard: sudden spike > 2.5 g (24.5 m/s²) → fall event
#define FALL_THRESH_MPS2  24.5f   // m/s² (2.5 g)
#define FALL_HOLD_MS       5000   // latch alarm for 5 s after fall event

bool     fallDetected    = false;
uint32_t fallTimestamp   = 0;

// ── Alert thresholds (POST-SURGERY RECOVERY — clinical EWS/NEWS2 guidelines) ─
// Source: NEWS2 (Royal College of Physicians, UK) + AAMI/ANSI vital sign ranges
struct Thresholds {
  // Heart Rate (bpm)
  float hr_low_warn  = 50;   // Warn below 50 bpm  (NEWS2: ≤50 = score 1)
  float hr_high_warn = 100;  // Warn above 100 bpm (NEWS2: 91-110 = score 1)
  float hr_low_crit  = 40;   // Critical <40 bpm   (NEWS2: ≤40 = score 3)
  float hr_high_crit = 120;  // Critical >120 bpm  (NEWS2: ≥111 = score 2-3)

  // SpO2 (%)
  float spo2_warn    = 94;   // Warn <94%   (NEWS2: 93-94 = score 1)
  float spo2_crit    = 90;   // Critical <90% (NEWS2: ≤91 = score 3)

  // Respiratory Rate (breaths/min)
  float rr_low_warn  = 10;   // Warn <10    (NEWS2: 9 = score 1)
  float rr_high_warn = 20;   // Warn >20    (NEWS2: 21-24 = score 2)
  float rr_low_crit  =  8;   // Critical <8 (NEWS2: ≤8 = score 3)
  float rr_high_crit = 25;   // Critical >25 (NEWS2: ≥25 = score 3)

  // Temperature (°C)
  float temp_low_warn  = 36.0; // Warn <36°C  (NEWS2: 35.1-36.0 = score 1)
  float temp_high_warn = 38.0; // Warn >38°C  (NEWS2: 38.1-39.0 = score 1)
  float temp_low_crit  = 35.0; // Critical <35°C (NEWS2: ≤35 = score 3)
  float temp_high_crit = 39.0; // Critical >39°C (NEWS2: ≥39.1 = score 2)

  // Acceleration magnitude (m/s²) — fall indicator
  float accel_fall = 24.5;   // 2.5 g threshold for fall detection
} THRESH;

// ── LCD helpers ───────────────────────────────────────────────────────────────
void lcdShowVitals(float hr, float spo2, float rr, float temp, bool fall) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("HR:");  lcd.print((int)hr);
  lcd.print(" SpO2:"); lcd.print((int)spo2); lcd.print("%");

  lcd.setCursor(0, 1);
  lcd.print("T:"); lcd.print(temp, 1);
  lcd.print(" RR:"); lcd.print((int)rr);
  if (fall) {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("!! FALL DETECTED");
    lcd.setCursor(0, 1); lcd.print("!! ALERT DOCTOR");
  }
}

void lcdAlert(const char* line1, const char* line2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2);
}

// ── Buzzer helpers ────────────────────────────────────────────────────────────
void buzzCritical() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(100);
    digitalWrite(BUZZER_PIN, LOW);  delay(100);
  }
}
void buzzWarning() {
  digitalWrite(BUZZER_PIN, HIGH); delay(400);
  digitalWrite(BUZZER_PIN, LOW);
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("EPICS VIT Bhopal");
  lcd.setCursor(0, 1); lcd.print("Initialising...");
  delay(1500);

  // MAX30105
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    lcdAlert("MAX30105 Error", "Check wiring!");
    while (1) { buzzCritical(); delay(2000); }
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);

  // MPU6050
  mpu.initialize();
  if (!mpu.testConnection()) {
    lcdAlert("MPU6050 Error", "Check wiring!");
    while (1) { buzzCritical(); delay(2000); }
  }

  // DS18B20
  tempSensor.begin();
  if (tempSensor.getDeviceCount() == 0) {
    lcdAlert("DS18B20 Error", "Check wiring!");
    while (1) { buzzCritical(); delay(2000); }
  }

  // Prefill MAX30105 buffer (100 samples to get stable reading)
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Place finger on");
  lcd.setCursor(0, 1); lcd.print("MAX30105 sensor");
  for (int i = 0; i < BUFFER_LENGTH; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, BUFFER_LENGTH, redBuffer,
    &spo2Value, &validSPO2, &heartRateValue, &validHeartRate
  );

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("System Ready");
  lcd.setCursor(0, 1); lcd.print("Monitoring...");
  delay(1000);

  Serial.println("HR,SpO2,RespRate,Temp_C,AccelMag,FallDetected");
}

// ── Respiratory Rate — sliding window peak counter from IR AC component ───────
void updateRespRate(float irVal) {
  if (irVal > lastIR + 200 && !irRising) {
    irRising = true;
  } else if (irVal < lastIR - 200 && irRising) {
    irRising = false;
    uint32_t now = millis();
    if (rrLastPeak > 0 && (now - rrLastPeak) < 5000UL) {
      rrPeakBuf[rrPeakIdx % 60] = (float)(now - rrLastPeak) / 1000.0f;
      rrPeakIdx++;
    }
    rrLastPeak = now;
  }
  lastIR = irVal;

  // Average last up-to-8 inter-peak intervals → breaths per minute
  uint8_t n = min((uint8_t)8, rrPeakIdx);
  if (n >= 2) {
    float sum = 0;
    for (uint8_t i = 0; i < n; i++) sum += rrPeakBuf[(rrPeakIdx - 1 - i) % 60];
    float avgInterval = sum / n;            // seconds per breath
    if (avgInterval > 0.5f && avgInterval < 12.0f) {
      respRate = 60.0f / avgInterval;       // breaths per minute
      respRate = constrain(respRate, 4, 40);
    }
  }
}

// ── Fall detection from MPU6050 ────────────────────────────────────────────────
void checkFall() {
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);
  // Convert raw to m/s²: range ±2g, scale = 16384 LSB/g, g = 9.81 m/s²
  float ax_ms2 = (ax / 16384.0f) * 9.81f;
  float ay_ms2 = (ay / 16384.0f) * 9.81f;
  float az_ms2 = (az / 16384.0f) * 9.81f;
  float mag = sqrt(ax_ms2*ax_ms2 + ay_ms2*ay_ms2 + az_ms2*az_ms2);

  if (mag > FALL_THRESH_MPS2) {
    fallDetected  = true;
    fallTimestamp = millis();
  }
  // Auto-clear fall flag after hold period
  if (fallDetected && (millis() - fallTimestamp > FALL_HOLD_MS)) {
    fallDetected = false;
  }
}

// ── Get current acceleration magnitude (for logging) ──────────────────────────
float getAccelMag() {
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);
  float ax_ms2 = (ax / 16384.0f) * 9.81f;
  float ay_ms2 = (ay / 16384.0f) * 9.81f;
  float az_ms2 = (az / 16384.0f) * 9.81f;
  return sqrt(ax_ms2*ax_ms2 + ay_ms2*ay_ms2 + az_ms2*az_ms2);
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
  static uint32_t lastSend = 0;

  // Continuously update SpO2/HR buffer (shift window by 25 samples each cycle)
  for (int i = 25; i < BUFFER_LENGTH; i++) {
    redBuffer[i - 25] = redBuffer[i];
    irBuffer[i - 25]  = irBuffer[i];
  }
  for (int i = 75; i < BUFFER_LENGTH; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    updateRespRate((float)irBuffer[i]);
    particleSensor.nextSample();
  }
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, BUFFER_LENGTH, redBuffer,
    &spo2Value, &validSPO2, &heartRateValue, &validHeartRate
  );

  // Fall detection (runs every loop iteration for responsiveness)
  checkFall();

  // Send readings every 2 seconds
  if (millis() - lastSend >= 2000UL) {
    lastSend = millis();

    float hr   = (validHeartRate && heartRateValue > 20 && heartRateValue < 220)
                   ? (float)heartRateValue : -1.0f;
    float spo2 = (validSPO2 && spo2Value > 70 && spo2Value <= 100)
                   ? (float)spo2Value : -1.0f;
    float rr   = respRate;

    tempSensor.requestTemperatures();
    float temp = tempSensor.getTempCByIndex(0);

    float accelMag = getAccelMag();
    int   fall     = fallDetected ? 1 : 0;

    // ── Alarm logic ───────────────────────────────────────────────────────────
    bool critical = false, warning = false;
    if (fall) { critical = true; }
    if (hr > 0   && (hr < THRESH.hr_low_crit  || hr > THRESH.hr_high_crit))  critical = true;
    if (spo2 > 0 && spo2 < THRESH.spo2_crit)                                  critical = true;
    if (rr > 0   && (rr < THRESH.rr_low_crit  || rr > THRESH.rr_high_crit))  critical = true;
    if (temp > 0 && (temp < THRESH.temp_low_crit || temp > THRESH.temp_high_crit)) critical = true;

    if (!critical) {
      if (hr > 0   && (hr < THRESH.hr_low_warn   || hr > THRESH.hr_high_warn))  warning = true;
      if (spo2 > 0 && spo2 < THRESH.spo2_warn)                                   warning = true;
      if (rr > 0   && (rr < THRESH.rr_low_warn   || rr > THRESH.rr_high_warn))  warning = true;
      if (temp > 0 && (temp < THRESH.temp_low_warn || temp > THRESH.temp_high_warn)) warning = true;
    }

    if (critical)      buzzCritical();
    else if (warning)  buzzWarning();

    // ── LCD display ───────────────────────────────────────────────────────────
    lcdShowVitals(hr > 0 ? hr : 0, spo2 > 0 ? spo2 : 0, rr, temp, fall);

    // ── Serial output → Python reads this ────────────────────────────────────
    // Format: HR,SpO2,RespRate,Temp_C,AccelMag,FallDetected
    Serial.print(hr, 1);    Serial.print(",");
    Serial.print(spo2, 1);  Serial.print(",");
    Serial.print(rr, 1);    Serial.print(",");
    Serial.print(temp, 2);  Serial.print(",");
    Serial.print(accelMag, 3); Serial.print(",");
    Serial.println(fall);
  }
}
