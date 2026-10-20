"""
STEP 1 — Data Extraction from MIMIC-III chartevents
AI-Powered Remote Post-Surgery Patient Monitoring System
EPICS Project | VIT Bhopal University

UPDATED for Hardware v2:
  Sensors: MAX30105 (HR + SpO2), DS18B20 (Temp), MPU6050 (Fall/Accel),
           LCD 16×2, Buzzer @ GPIO 25
  Derives 7 clinical features that match the hardware sensor outputs:
    HeartRate, SpO2, RespRate, Temperature, MAP (derived), SBP, DBP
  Plus FallRisk (binary, derived from MPU6050 accel — not in MIMIC, added
  as rule-based feature from age+vitals so the same model column exists).

Run this first to regenerate the training dataset.

Usage:
    python step1_extract_data.py

Output:
    data/mimic_7feature_dataset.csv
"""

import pandas as pd
import numpy as np
import os

# ── File paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
CHART_CSV   = os.path.join(BASE_DIR, 'chartevents.csv')
PATIENT_CSV = os.path.join(BASE_DIR, 'patients.csv')
OUT_CSV     = os.path.join(DATA_DIR, 'mimic_7feature_dataset.csv')
os.makedirs(DATA_DIR, exist_ok=True)

# ── MIMIC-III MetaVision item IDs ────────────────────────────────────────────
# Ref: MIMIC-III documentation, Johnson et al. 2016 (Scientific Data)
VITAL_ITEMS = {
    220045: 'HeartRate',    # Heart Rate (bpm)
    220210: 'RespRate',     # Respiratory Rate (breaths/min)
    220277: 'SpO2',         # Peripheral O2 Saturation (%)
    223762: 'Temp_C',       # Temperature Celsius
    223761: 'Temp_F',       # Temperature Fahrenheit (will convert)
    220179: 'SysBP',        # Non-invasive Systolic BP (mmHg)
    220180: 'DiasBP',       # Non-invasive Diastolic BP (mmHg)
    220050: 'SysBP',        # Arterial Systolic BP (mmHg)
    220051: 'DiasBP',       # Arterial Diastolic BP (mmHg)
}

# ── Clinical thresholds (NEWS2 / EWS standard) ───────────────────────────────
# Source: Royal College of Physicians NEWS2 (2017), NICE guideline NG51
THRESHOLDS = {
    # Vital         : (low_critical, low_warn, high_warn, high_critical)
    'HeartRate'     : (40,   50,  100, 120),   # bpm
    'SpO2'          : (90,   94,  100, 100),   # %  (only lower bounds matter)
    'RespRate'      : ( 8,   10,   20,  25),   # breaths/min
    'Temperature'   : (35.0, 36.0, 38.0, 39.0), # °C
    'SysBP'         : (80,   90,  160, 180),   # mmHg
    'DiasBP'        : (40,   60,   90, 110),   # mmHg
    'MAP'           : (60,   70,  110, 130),   # mmHg  (derived)
}

print("=" * 62)
print("  EPICS — MIMIC-III Data Extraction (Hardware v2 — 7 features)")
print("=" * 62)

# ── Load chartevents ─────────────────────────────────────────────────────────
print("\n[1/5] Loading chartevents.csv ...")
df = pd.read_csv(CHART_CSV)
print(f"      Total rows: {len(df):,}")

# ── Filter to vital rows ──────────────────────────────────────────────────────
vitals = df[df['itemid'].isin(VITAL_ITEMS.keys())].copy()
vitals['vital_name'] = vitals['itemid'].map(VITAL_ITEMS)

# Convert °F → °C
f_mask = vitals['vital_name'] == 'Temp_F'
vitals.loc[f_mask, 'valuenum']   = (vitals.loc[f_mask, 'valuenum'] - 32) * 5 / 9
vitals.loc[f_mask, 'vital_name'] = 'Temp_C'
vitals.loc[vitals['vital_name'] == 'Temp_C', 'vital_name'] = 'Temperature'

print(f"\n[2/5] Vital readings extracted ({len(vitals):,}):")
counts = vitals.groupby('vital_name')['valuenum'].count().sort_values(ascending=False)
for name, cnt in counts.items():
    print(f"      {name:<15} {cnt:>7,} readings")

# ── Plausibility filter (remove sensor artefacts) ────────────────────────────
filters = {
    'HeartRate'  : (20,   250),
    'SpO2'       : (60,   100),
    'RespRate'   : ( 4,    60),
    'Temperature': (34.0, 42.0),
    'SysBP'      : (50,   250),
    'DiasBP'     : (20,   150),
}
for vital, (lo, hi) in filters.items():
    mask = (vitals['vital_name'] == vital)
    vitals = vitals[~(mask & ((vitals['valuenum'] < lo) | (vitals['valuenum'] > hi)))]

# ── Aggregate per hospital admission ─────────────────────────────────────────
print("\n[3/5] Aggregating per patient admission ...")
pivot = vitals.pivot_table(
    index=['subject_id', 'hadm_id'],
    columns='vital_name',
    values='valuenum',
    aggfunc='mean'
).reset_index()

# ── Derive MAP (Mean Arterial Pressure) from SBP + DBP ───────────────────────
# MAP = DBP + (SBP - DBP) / 3   — standard clinical formula
if 'SysBP' in pivot.columns and 'DiasBP' in pivot.columns:
    pivot['MAP'] = pivot['DiasBP'] + (pivot['SysBP'] - pivot['DiasBP']) / 3.0

# ── Merge patient demographics ────────────────────────────────────────────────
patients = pd.read_csv(PATIENT_CSV)
merged = pivot.merge(
    patients[['subject_id', 'gender', 'anchor_age']],
    on='subject_id', how='left'
)
merged['Age']    = merged['anchor_age']
merged['Gender'] = (merged['gender'] == 'M').astype(int)  # 1=Male, 0=Female

# ── Derive FallRisk feature ───────────────────────────────────────────────────
# MPU6050 measures actual falls in hardware. In the MIMIC training set we don't
# have accelerometer data, so we proxy FallRisk using established clinical risk
# factors for post-surgery falls (Morse Fall Scale components available in MIMIC):
#   - Age ≥ 70 → elevated fall risk
#   - SysBP < 100 (orthostatic hypotension risk)
#   - Temperature > 38.5 (delirium/fever increases fall risk)
#   Ref: Morse (1997), JM Inouye et al. JAMA 2014
def compute_fall_risk(row):
    risk = 0
    age  = row.get('Age', np.nan)
    sbp  = row.get('SysBP', np.nan)
    temp = row.get('Temperature', np.nan)
    hr   = row.get('HeartRate', np.nan)

    if pd.notna(age)  and age >= 70:          risk += 1
    if pd.notna(sbp)  and sbp < 100:          risk += 1
    if pd.notna(temp) and temp > 38.5:        risk += 1
    if pd.notna(hr)   and hr > 100:           risk += 1   # fever/pain tachycardia
    return int(risk >= 2)   # ≥2 factors → FallRisk=1

merged['FallRisk'] = merged.apply(compute_fall_risk, axis=1)

print(f"\n[4/5] FallRisk distribution:")
print(f"      FallRisk=0 (Low)  : {(merged['FallRisk']==0).sum()}")
print(f"      FallRisk=1 (High) : {(merged['FallRisk']==1).sum()}")

# ── Assign clinical risk label (NEWS2-based EWS scoring) ─────────────────────
# Reference: Royal College of Physicians NEWS2 (2017) scoring table
def assign_risk(row):
    score = 0

    hr   = row.get('HeartRate');     rr   = row.get('RespRate')
    spo  = row.get('SpO2');          temp = row.get('Temperature')
    sbp  = row.get('SysBP');         map_ = row.get('MAP')
    fall = row.get('FallRisk', 0)

    # Heart Rate (NEWS2 Table 1)
    if pd.notna(hr):
        if hr <= 40:                  score += 3
        elif hr <= 50:                score += 1
        elif hr <= 90:                pass        # normal
        elif hr <= 110:               score += 1
        elif hr <= 130:               score += 2
        else:                         score += 3

    # SpO2 — Scale 1 (NEWS2 default for non-hypercapnic)
    if pd.notna(spo):
        if spo <= 91:                 score += 3
        elif spo <= 93:               score += 2
        elif spo <= 94:               score += 1

    # Respiratory Rate (NEWS2)
    if pd.notna(rr):
        if rr <= 8:                   score += 3
        elif rr <= 11:                score += 1
        elif rr <= 20:                pass        # normal
        elif rr <= 24:                score += 2
        else:                         score += 3

    # Temperature (NEWS2)
    if pd.notna(temp):
        if temp <= 35.0:              score += 3
        elif temp <= 36.0:            score += 1
        elif temp <= 38.0:            pass        # normal
        elif temp <= 39.0:            score += 1
        else:                         score += 2

    # Systolic BP (NEWS2)
    if pd.notna(sbp):
        if sbp <= 90:                 score += 3
        elif sbp <= 100:              score += 2
        elif sbp <= 110:              score += 1
        elif sbp <= 219:              pass        # normal
        else:                         score += 3

    # MAP add-on (not in standard NEWS2, added for ICU context)
    if pd.notna(map_):
        if map_ < 65:                 score += 2

    # Fall risk adds urgency
    if fall == 1:                     score += 2

    return 1 if score >= 3 else 0    # 0=Stable, 1=At-Risk

merged['Target'] = merged.apply(assign_risk, axis=1)

# ── Final feature set (matches hardware sensor outputs exactly) ──────────────
# Hardware sensors:  MAX30105 → HR, SpO2
#                    DS18B20  → Temperature
#                    MPU6050  → FallRisk (real-time) / AccelMag
#                    Derived  → RespRate (from IR waveform), MAP, SBP, DBP
FEATURES = [
    'Age', 'Gender',        # patient demographics
    'HeartRate',            # MAX30105
    'SpO2',                 # MAX30105
    'RespRate',             # Derived from MAX30105 IR waveform
    'Temperature',          # DS18B20
    'SysBP',                # Estimated / not directly measured — see step3 note
    'DiasBP',               # Estimated / not directly measured
    'MAP',                  # Derived from SBP+DBP
    'FallRisk',             # MPU6050 (real-time) / proxy in training set
]
final = merged[FEATURES + ['Target']].dropna()

print(f"\n[5/5] Dataset summary:")
print(f"      Patients         : {len(final)}")
print(f"      Features (10)    : {', '.join(FEATURES)}")
print(f"      Stable (0)       : {(final.Target==0).sum()}")
print(f"      At-Risk (1)      : {(final.Target==1).sum()}")
print(f"      Fall-Risk pts    : {(final.FallRisk==1).sum()}")

final.to_csv(OUT_CSV, index=False)
print(f"\n  Saved → {OUT_CSV}")
print("=" * 62)
print("  Run step2_train_model.py next")
print("=" * 62)
