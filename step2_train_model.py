"""
STEP 2 — Model Training & Evaluation
AI-Powered Remote Post-Surgery Patient Monitoring System
EPICS Project | VIT Bhopal University

UPDATED for Hardware v2:
  Trains on 10 features derived from: MAX30105, DS18B20, MPU6050 sensors.
  Uses NEWS2-scored labelling from step1.

Usage:
    python step2_train_model.py

Output:
    models/healthcare_forest_v2.pkl
    outputs/evaluation_report.txt
"""

import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      accuracy_score, precision_score,
                                      recall_score, f1_score)
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(BASE_DIR, 'data',    'mimic_7feature_dataset.csv')
MODEL_OUT  = os.path.join(BASE_DIR, 'models',  'healthcare_forest_v2.pkl')
REPORT_OUT = os.path.join(BASE_DIR, 'outputs', 'evaluation_report.txt')
os.makedirs(os.path.join(BASE_DIR, 'models'),  exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)

# ── Feature list — must match hardware sensor outputs in step3/step4 ─────────
FEATURES = [
    'Age', 'Gender',
    'HeartRate',    # MAX30105
    'SpO2',         # MAX30105
    'RespRate',     # Derived from MAX30105 IR waveform (on ESP32)
    'Temperature',  # DS18B20
    'SysBP',        # Estimated cuffless (PTT/PWV approximation)
    'DiasBP',       # Estimated cuffless
    'MAP',          # Derived: DBP + (SBP-DBP)/3
    'FallRisk',     # MPU6050 fall detection flag
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  EPICS — Model Training & Evaluation (Hardware v2)")
print("=" * 62)

df = pd.read_csv(DATA_CSV)
X  = df[FEATURES]
y  = df['FallRisk']

print(f"\n  Dataset: {len(df)} patients | {len(FEATURES)} features")
print(f"  Stable : {(y==0).sum()}  |  At-Risk: {(y==1).sum()}")
print(f"  Fall-Risk patients: {(df.FallRisk==1).sum()}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ── Model pipeline ────────────────────────────────────────────────────────────
# RandomForest is inherently scale-invariant, but we include StandardScaler
# so the same pipeline can be swapped for an SVM/LR in the future.
print("\n  Training Random Forest (300 trees) ...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',   # handles imbalanced At-Risk vs Stable
        n_jobs=-1
    ))
])
model.fit(X_train, y_train)
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)
print(f"\n  Model  → {MODEL_OUT}")
print(f"  Report → {REPORT_OUT}")


# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
prec  = precision_score(y_test, preds, zero_division=0)
rec   = recall_score(y_test, preds, zero_division=0)
f1    = f1_score(y_test, preds, zero_division=0)
cm    = confusion_matrix(y_test, preds)

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_sc = cross_val_score(model, X, y, cv=cv, scoring='f1')

# Feature importances from the RF inside the pipeline
fi = sorted(
    zip(FEATURES, model.named_steps['rf'].feature_importances_),
    key=lambda x: -x[1]
)

# ── Report ────────────────────────────────────────────────────────────────────
sep = "-" * 62

report_lines = [
    "=" * 62,
    "  EPICS — Model Evaluation Report (Hardware v2 — 10 features)",
    "  AI-Powered Post-Surgery Patient Monitoring System",
    "  Dataset: MIMIC-III Clinical Database",
    "  Hardware: MAX30105 + DS18B20 + MPU6050 on ESP32",
    "=" * 62,
    "",
    "DATASET SUMMARY",
    sep,
    f"  Total patients     : {len(df)}",
    f"  Features used      : {len(FEATURES)}",
    f"  Feature list       : {', '.join(FEATURES)}",
    f"  Stable patients    : {(y==0).sum()}",
    f"  At-Risk patients   : {(y==1).sum()}",
    f"  Train / Test split : 80% / 20%",
    "",
    "MODEL CONFIGURATION",
    sep,
    "  Algorithm          : Random Forest Classifier (in Pipeline)",
    "  Number of trees    : 300",
    "  Max depth          : 12",
    "  Min samples/leaf   : 2",
    "  Class weighting    : Balanced (handles class imbalance)",
    "  Preprocessing      : StandardScaler",
    "",
    "CLINICAL THRESHOLD REFERENCE (NEWS2 / AAMI)",
    sep,
    "  Vital           Low-Crit  Low-Warn  High-Warn  High-Crit  Unit",
    "  Heart Rate      ≤40       ≤50       >100       >120       bpm",
    "  SpO2            <90       <94       —          —          %",
    "  Resp Rate       ≤8        ≤10       >20        ≥25        breaths/min",
    "  Temperature     ≤35.0     ≤36.0     >38.0      >39.0      °C",
    "  Systolic BP     ≤90       ≤100      >160       >180       mmHg",
    "  Diastolic BP    ≤40       ≤60       >90        >110       mmHg",
    "  MAP             <60       <70       >110       >130       mmHg",
    "  Fall Detected   —         —         —          YES        binary",
    "  Source: RCP NEWS2 (2017), AAMI/ISO 81060, Morse Fall Scale",
    "",
    "PERFORMANCE METRICS (Test Set)",
    sep,
    f"  Accuracy           : {acc*100:.1f}%",
    f"  Precision          : {prec*100:.1f}%",
    f"  Recall             : {rec*100:.1f}%",
    f"  F1 Score           : {f1*100:.1f}%",
    "",
    "CROSS-VALIDATION (5-Fold Stratified — F1 score)",
    sep,
    f"  CV F1              : {cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.1f}%",
    f"  Per-fold F1        : {[f'{s*100:.1f}%' for s in cv_sc]}",
    "",
    "CONFUSION MATRIX",
    sep,
    "                 Predicted",
    "                 Stable   At-Risk",
    f"  Actual Stable : {cm[0,0]:>6}   {cm[0,1]:>6}",
    f"  Actual At-Risk: {cm[1,0]:>6}   {cm[1,1]:>6}",
    "",
    "DETAILED CLASSIFICATION REPORT",
    sep,
    classification_report(y_test, preds, target_names=['Stable', 'At-Risk'],
                          zero_division=0),
    "FEATURE IMPORTANCE (which sensor/vital matters most)",
    sep,
]

bar_max = fi[0][1] if fi[0][1] > 0 else 1
for feat, imp in fi:
    bar_len = int((imp / bar_max) * 30)
    bar     = "█" * bar_len
    report_lines.append(f"  {feat:<15} {imp*100:>5.1f}%  {bar}")

report_lines += [
    "",
    "=" * 62,
    "  Model saved → models/healthcare_forest_v2.pkl",
    "=" * 62,
]

full_report = "\n".join(report_lines)
print(full_report)

with open(REPORT_OUT, 'w', encoding='utf-8') as f:
    f.write(full_report)

try:
    print(full_report)
except UnicodeEncodeError:
    print("\n[Note] Model trained and saved, but terminal could not display special characters.")
    print("Check your 'models' folder for the .pkl file.")