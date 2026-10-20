"""
VitalDB Explorer — Run this FIRST on your laptop
EPICS Project | VIT Bhopal University
 
This script tells us:
  1. What surgery types are available
  2. What vital sign track names VitalDB uses
  3. How many cardiac + orthopedic cases exist
 
Run this and paste the output back so the training script can be written.
"""
 
import vitaldb
import pandas as pd
 
print("=" * 60)
print("  VitalDB Explorer")
print("=" * 60)
 
# ── Step 1: Load clinical data (patient + surgery info) ───────────────────────
print("\n[1/3] Loading clinical data...")
cdf = vitaldb.load_clinical_data()
print(f"      Total cases: {len(cdf)}")
print(f"\n      Columns available:")
for col in cdf.columns:
    print(f"        - {col}")
 
# ── Step 2: Show surgery types ────────────────────────────────────────────────
print("\n[2/3] Surgery types (opname) — top 30:")
if 'opname' in cdf.columns:
    counts = cdf['opname'].value_counts().head(30)
    for name, count in counts.items():
        print(f"      {count:>5}  {name}")
else:
    print("      'opname' column not found. Showing all unique values in likely columns:")
    for col in ['op_name', 'surgery', 'operation', 'procedure']:
        if col in cdf.columns:
            print(f"\n      Column '{col}':")
            print(cdf[col].value_counts().head(20))
 
# ── Step 3: Show available vital sign tracks ──────────────────────────────────
print("\n[3/3] Available vital sign tracks (first case sample):")
try:
    tracks = vitaldb.get_track_names()
    print(f"      Total tracks available: {len(tracks)}")
    print("\n      Tracks containing key vitals:")
    keywords = ['HR', 'SpO2', 'SPO2', 'RESP', 'RR', 'TEMP', 'BT',
                'SBP', 'DBP', 'MBP', 'MAP', 'ABP', 'NIBP']
    for t in tracks:
        if any(k.lower() in t.lower() for k in keywords):
            print(f"        {t}")
except Exception as e:
    print(f"      get_track_names() failed: {e}")
    print("      Trying load_case(1) to inspect tracks...")
    try:
        data = vitaldb.load_case(1, ['Solar8000/HR', 'Solar8000/SpO2',
                                      'Primus/RR', 'BIS/BT',
                                      'Solar8000/ABP_MBP'], interval=1)
        print(f"      Sample data shape: {data.shape}")
        print(f"      Columns: {list(data.columns)}")
    except Exception as e2:
        print(f"      load_case also failed: {e2}")
 
print("\n" + "=" * 60)
print("  Paste this entire output back to Claude!")
print("=" * 60)