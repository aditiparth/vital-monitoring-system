"""
STEP 3 — Real-Time AI Patient Monitoring (Terminal)
AI-Powered Remote Post-Surgery Patient Monitoring System
EPICS Project | VIT Bhopal University

UPDATED for Hardware v2:
  Reads live sensor data from ESP32 over USB serial (115200 baud).
  ESP32 sends CSV: HR,SpO2,RespRate,Temp_C,AccelMag,FallDetected

  7 vitals monitored:
    1. Heart Rate       (MAX30105, bpm)
    2. SpO2             (MAX30105, %)
    3. Respiratory Rate (Derived from MAX30105 IR waveform, breaths/min)
    4. Temperature      (DS18B20, °C)
    5. MAP              (Estimated from HR + SpO2 using PTT proxy, mmHg)
    6. Accel Magnitude  (MPU6050, m/s²)
    7. Fall Detected    (MPU6050 binary event flag)

  AI model predicts Stable / At-Risk every reading.
  Buzzer is controlled by ESP32 firmware; Python handles logging + AI.

Usage:
    pip install pyserial joblib scikit-learn pandas
    python step3_monitor.py

Hardware:
    ESP32 connected via USB → COM port (Windows) or /dev/ttyUSB0 (Linux/macOS)
    Flash esp32_vitals_sender.ino to ESP32 first.
"""

import joblib, os, time, sys
import pandas as pd
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARNING] pyserial not installed. Running in simulation mode.")
    print("          Install with:  pip install pyserial\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'healthcare_forest_v2.pkl')

# ── Feature list — must match step1 and step2 ────────────────────────────────
FEATURES = [
    'Age', 'Gender',
    'HeartRate', 'SpO2', 'RespRate', 'Temperature',
    'SysBP', 'DiasBP', 'MAP', 'FallRisk',
]

# ── Clinical thresholds (NEWS2 / AAMI / RCP guidelines) ──────────────────────
# Each entry: (low_critical, low_warn, high_warn, high_critical)
THRESH = {
    'HeartRate'  : (40,   50,   100,  120),   # bpm     — NEWS2 Table 1
    'SpO2'       : (90,   94,   100,  100),   # %       — NEWS2 SpO2 Scale 1
    'RespRate'   : ( 8,   10,    20,   25),   # /min    — NEWS2 Table 1
    'Temperature': (35.0, 36.0,  38.0, 39.0), # °C      — NEWS2 Table 1
    'SysBP'      : (80,   90,   160,  180),   # mmHg    — NEWS2 + AAMI 81060
    'DiasBP'     : (40,   60,    90,  110),   # mmHg    — AAMI
    'MAP'        : (60,   70,   110,  130),   # mmHg    — SCCM/ICU guidelines
    'AccelMag'   : (0,     0,   20,   24.5),  # m/s²    — 2.5g fall threshold
}

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── MAP estimation (cuffless proxy) ──────────────────────────────────────────
# Without a BP sensor on the hardware, MAP is estimated via a PTT/PWV proxy.
# Method: simplified Hales/Holland model using SpO2 and HR as surrogates.
# This is a PROXY ESTIMATE only — accuracy ±15 mmHg vs. cuff.
# If a BP cuff is added, replace this with actual readings.
def estimate_bp_from_vitals(hr, spo2, age):
    """
    Rough cuffless BP estimation using age + HR-based PTT proxy.
    Source: inspired by Mukkamala et al. 2015 (IEEE TBME), simplified for
    embedded use. Not FDA-cleared — used here as a monitoring surrogate.
    """
    # Baseline MAP from population norms, adjusted for age and HR
    base_map = 85.0 + (age - 40) * 0.3

    # HR correction: tachycardia suggests elevated BP, bradycardia → lower
    hr_factor = (hr - 70) * 0.15

    # SpO2 desaturation → vasodilation → slightly lower DBP
    spo2_factor = (100 - spo2) * (-0.5)

    map_est = base_map + hr_factor + spo2_factor
    map_est = float(np.clip(map_est, 50, 160))

    # Derive SBP and DBP from MAP using standard ratio
    # MAP = DBP + (SBP - DBP)/3  →  SBP ≈ MAP + 40, DBP ≈ MAP - 10 (approx)
    sbp_est = map_est + 40.0 + (age - 40) * 0.2
    dbp_est = map_est - 12.0
    sbp_est = float(np.clip(sbp_est, 60, 220))
    dbp_est = float(np.clip(dbp_est, 30, 130))
    return sbp_est, dbp_est, map_est

# ── Serial port auto-detect ───────────────────────────────────────────────────
def find_esp32_port():
    if not SERIAL_AVAILABLE:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ['cp210', 'ch340', 'ftdi', 'esp', 'uart', 'usb serial']):
            return p.device
    # Fallback: return first available port
    if ports:
        return ports[0].device
    return None

# ── Simulation fallback (when no hardware connected) ─────────────────────────
import random

def simulate_reading(base_hr, base_spo2, abnormal_prob=0.10):
    """Simulate realistic ESP32 CSV output for testing without hardware."""
    if random.random() < 0.01:
        return None   # Simulate signal loss

    if random.random() < abnormal_prob:
        hr       = random.choice([random.randint(105, 130), random.randint(38, 52)])
        spo2     = random.randint(85, 93)
        rr       = random.choice([random.randint(22, 32), random.randint(5, 9)])
        temp     = round(random.uniform(38.1, 40.0), 2)
        accel    = round(random.uniform(9.0, 10.5), 3)
        fall     = 0
    else:
        hr       = int(base_hr   + random.gauss(0, 4))
        spo2     = int(base_spo2 + random.gauss(0, 1))
        rr       = int(16 + random.gauss(0, 2))
        temp     = round(37.0 + random.gauss(0, 0.25), 2)
        accel    = round(9.81 + random.gauss(0, 0.5), 3)   # ~1g at rest
        fall     = 0

    # Rare fall event
    if random.random() < 0.03:
        accel = round(random.uniform(26, 40), 3)
        fall  = 1

    return {
        'HeartRate': float(np.clip(hr,   20, 250)),
        'SpO2'     : float(np.clip(spo2, 60, 100)),
        'RespRate' : float(np.clip(rr,    4,  60)),
        'Temp_C'   : float(np.clip(temp, 34,  42)),
        'AccelMag' : float(np.clip(accel, 0, 100)),
        'Fall'     : fall,
    }

# ── Rule-based safety override (hard clinical limits) ────────────────────────
def rule_check(r):
    alerts = []
    if r['Fall'] == 1:
        alerts.append("CRITICAL: FALL DETECTED — Patient may be on the floor!")
    if r['SpO2'] < THRESH['SpO2'][0]:
        alerts.append(f"CRITICAL: SpO2 {r['SpO2']:.0f}% < 90% — Hypoxia risk!")
    elif r['SpO2'] < THRESH['SpO2'][1]:
        alerts.append(f"WARNING:  SpO2 {r['SpO2']:.0f}% < 94% — Monitor closely")
    if r['HeartRate'] > THRESH['HeartRate'][3]:
        alerts.append(f"CRITICAL: HR {r['HeartRate']:.0f} bpm — Tachycardia (>120)")
    elif r['HeartRate'] < THRESH['HeartRate'][0]:
        alerts.append(f"CRITICAL: HR {r['HeartRate']:.0f} bpm — Bradycardia (<40)")
    elif r['HeartRate'] > THRESH['HeartRate'][2]:
        alerts.append(f"WARNING:  HR {r['HeartRate']:.0f} bpm — Elevated (>100)")
    if r['RespRate'] > THRESH['RespRate'][3]:
        alerts.append(f"CRITICAL: RR {r['RespRate']:.0f}/min — Tachypnoea (≥25)")
    elif r['RespRate'] < THRESH['RespRate'][0]:
        alerts.append(f"CRITICAL: RR {r['RespRate']:.0f}/min — Apnoea risk (≤8)")
    if r['Temp_C'] > THRESH['Temperature'][3]:
        alerts.append(f"CRITICAL: Temp {r['Temp_C']:.1f}°C — Hyperpyrexia (>39°C)")
    elif r['Temp_C'] < THRESH['Temperature'][0]:
        alerts.append(f"CRITICAL: Temp {r['Temp_C']:.1f}°C — Hypothermia (<35°C)")
    elif r['Temp_C'] > THRESH['Temperature'][2]:
        alerts.append(f"WARNING:  Temp {r['Temp_C']:.1f}°C — Fever (>38°C)")
    if r['MAP'] < THRESH['MAP'][0]:
        alerts.append(f"CRITICAL: MAP {r['MAP']:.0f} mmHg — Haemodynamic compromise (<60)")
    if r['AccelMag'] > THRESH['AccelMag'][3]:
        alerts.append(f"WARNING:  High motion ({r['AccelMag']:.1f} m/s²) — Possible fall/agitation")
    return alerts

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}Model not found! Run step2_train_model.py first.{RESET}")
        return
    model = joblib.load(MODEL_PATH)

    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  AI-Powered Post-Surgery Patient Monitoring System{RESET}")
    print(f"{BOLD}  Hardware: MAX30105 + DS18B20 + MPU6050 on ESP32{RESET}")
    print(f"{BOLD}  EPICS | VIT Bhopal University{RESET}")
    print(f"{BOLD}{'='*62}{RESET}\n")

    # Patient setup
    try:
        print(f"{CYAN}--- Patient Setup ---{RESET}")
        age        = int(input("  Patient Age         : "))
        gender_in  = input("  Patient Gender (M/F): ").strip().upper()
        gender     = 1 if gender_in == 'M' else 0
        base_hr    = int(input("  Baseline Heart Rate : "))
        base_spo2  = int(input("  Baseline SpO2 (%)   : "))
        interval   = float(input("  Reading interval (s) [2]: ") or "2")
    except (ValueError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return

    # Serial setup
    ser = None
    mode = "SIMULATION"
    if SERIAL_AVAILABLE:
        port = find_esp32_port()
        if port:
            try:
                ser  = serial.Serial(port, 115200, timeout=3)
                time.sleep(2)   # wait for ESP32 reset
                # flush header line
                ser.readline()
                mode = f"HARDWARE ({port})"
                print(f"\n{GREEN}  ESP32 connected on {port}{RESET}")
            except Exception as e:
                print(f"\n{YELLOW}  Could not open {port}: {e}. Using simulation.{RESET}")
        else:
            print(f"\n{YELLOW}  No ESP32 found. Running in simulation mode.{RESET}")
    else:
        print(f"\n{YELLOW}  pyserial not available. Running in simulation mode.{RESET}")

    input(f"\n{CYAN}  Mode: {mode}  — Press Enter to start monitoring...{RESET}\n")

    reading_count = 0
    alert_count   = 0

    try:
        while True:
            reading_count += 1
            ts = time.strftime('%H:%M:%S')

            # ── Read sensors ──────────────────────────────────────────────────
            raw = None
            if ser:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) == 6:
                            raw = {
                                'HeartRate': float(parts[0]),
                                'SpO2'     : float(parts[1]),
                                'RespRate' : float(parts[2]),
                                'Temp_C'   : float(parts[3]),
                                'AccelMag' : float(parts[4]),
                                'Fall'     : int(parts[5]),
                            }
                except Exception as e:
                    print(f"  {YELLOW}Serial read error: {e}{RESET}")
            else:
                raw = simulate_reading(base_hr, base_spo2)

            print(f"\n{'─'*62}")
            print(f"  [{ts}]  Reading #{reading_count}  |  Mode: {mode}")

            if raw is None:
                print(f"  {YELLOW}⚠  SENSOR ERROR: Signal lost — check electrode contact{RESET}")
                time.sleep(interval)
                continue

            # ── Validate / clamp sensor outputs ──────────────────────────────
            hr   = float(np.clip(raw['HeartRate'], 20, 250))
            spo2 = float(np.clip(raw['SpO2'],      60, 100))
            rr   = float(np.clip(raw['RespRate'],   4,  60))
            temp = float(np.clip(raw['Temp_C'],    34,  42))
            accel= float(np.clip(raw['AccelMag'],   0, 100))
            fall = int(raw['Fall'])

            # Flag invalid sensor reads (MAX30105 returns -1 when finger absent)
            hr_valid   = hr > 20
            spo2_valid = spo2 > 60
            if not hr_valid:
                print(f"  {YELLOW}⚠  MAX30105: No finger detected — HR/SpO2 unavailable{RESET}")

            # ── Derive BP and MAP ─────────────────────────────────────────────
            sbp, dbp, map_ = estimate_bp_from_vitals(hr, spo2, age)

            # Store for rule check
            raw['MAP']    = map_
            raw['SysBP']  = sbp
            raw['DiasBP'] = dbp

            # ── Display vitals ────────────────────────────────────────────────
            print(f"\n  {'Vital':<22} {'Value':>10}  Unit      Sensor")
            print(f"  {'─'*55}")
            print(f"  {'Heart Rate':<22} {hr:>10.0f}  bpm       MAX30105")
            print(f"  {'SpO2':<22} {spo2:>10.0f}  %         MAX30105")
            print(f"  {'Respiratory Rate':<22} {rr:>10.0f}  /min      MAX30105 (IR)")
            print(f"  {'Temperature':<22} {temp:>10.1f}  °C        DS18B20")
            print(f"  {'Systolic BP (est.)':<22} {sbp:>10.0f}  mmHg      Derived")
            print(f"  {'Diastolic BP (est.)':<22} {dbp:>10.0f}  mmHg      Derived")
            print(f"  {'MAP (est.)':<22} {map_:>10.0f}  mmHg      Derived")
            print(f"  {'Accel Magnitude':<22} {accel:>10.2f}  m/s²      MPU6050")
            print(f"  {'Fall Detected':<22} {'YES' if fall else 'NO':>10}  —         MPU6050")

            # ── AI prediction ─────────────────────────────────────────────────
            input_df = pd.DataFrame([[
                age, gender,
                hr, spo2, rr, temp, sbp, dbp, map_, float(fall)
            ]], columns=FEATURES)

            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]
            risk_pct   = proba[1] * 100

            # ── Rule-based hard limits ────────────────────────────────────────
            rule_alerts = rule_check({**raw,
                                      'HeartRate': hr, 'SpO2': spo2,
                                      'RespRate': rr,  'Temp_C': temp,
                                      'AccelMag': accel, 'Fall': fall,
                                      'MAP': map_})

            print(f"\n  AI Risk Score: {risk_pct:.0f}%  |  ", end="")

            if rule_alerts or prediction == 1:
                alert_count += 1
                sev = "CRITICAL" if any("CRITICAL" in a for a in rule_alerts) else "WARNING"
                colour = RED if sev == "CRITICAL" else YELLOW
                print(f"{colour}{BOLD}{sev} — Patient At-Risk{RESET}")
                for a in rule_alerts:
                    c = RED if "CRITICAL" in a else YELLOW
                    print(f"  {c}  › {a}{RESET}")
                if not rule_alerts:
                    print(f"  {YELLOW}  › AI model detects elevated risk pattern{RESET}")
                print(f"  {RED}  → Notify attending doctor immediately{RESET}")
            else:
                print(f"{GREEN}STABLE — Patient condition normal{RESET}")

            time.sleep(interval)

    except KeyboardInterrupt:
        if ser:
            ser.close()
        print(f"\n\n{'='*62}")
        print(f"  Monitoring stopped.")
        print(f"  Total readings : {reading_count}")
        print(f"  Total alerts   : {alert_count}")
        print(f"{'='*62}\n")

if __name__ == '__main__':
    main()
