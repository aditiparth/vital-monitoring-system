import joblib, os, time, random, warnings, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from collections import deque

warnings.filterwarnings('ignore')

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# --- CONFIGURATION ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'healthcare_forest_v2.pkl')
TARGET_COM = 'COM7' # Your ESP32 Port

FEATURES = ['Age', 'Gender', 'HeartRate', 'SpO2', 'RespRate', 'Temperature', 'SysBP', 'DiasBP', 'MAP', 'FallRisk']

# --- COLORS ---
C_BG, C_CARD, C_BORDER = '#0d1117', '#161b22', '#30363d'
C_GREEN, C_YELLOW, C_RED, C_BLUE = '#3fb950', '#d29922', '#f85149', '#58a6ff'
C_PURPLE, C_ORANGE, C_TEAL, C_WHITE, C_MUTED = '#bc8cff', '#e3b341', '#39d353', '#e6edf3', '#8b949e'

# --- BUFFERS ---
N = 40
hr_hist = deque([75.0]*N, maxlen=N); spo2_hist = deque([98.0]*N, maxlen=N)
rr_hist = deque([16.0]*N, maxlen=N); temp_hist = deque([37.0]*N, maxlen=N)
map_hist = deque([85.0]*N, maxlen=N); accel_hist = deque([9.81]*N, maxlen=N)
risk_hist = deque([0.0]*N, maxlen=N); alert_log = []; read_no = [0]

PATIENT = {'age': 60, 'gender': 1, 'base_hr': 75, 'base_spo2': 98}
SER = [None]

def estimate_bp(hr, spo2, age):
    base_map = 85.0 + (age - 40) * 0.3
    map_e = float(np.clip(base_map + (hr-70)*0.15 + (100-spo2)*-0.5, 50, 160))
    sbp_e = float(np.clip(map_e + 40.0, 60, 220))
    dbp_e = float(np.clip(map_e - 12.0, 30, 130))
    return sbp_e, dbp_e, map_e

def read_sensors():
    if SER[0] and SER[0].is_open:
        try:
            if SER[0].in_waiting > 0:
                line = SER[0].readline().decode('utf-8', errors='ignore').strip()
                if ',' in line:
                    p = line.split(',')
                    if len(p) >= 6:
                        return {
                            'HeartRate': float(p[0]), 'SpO2': float(p[1]),
                            'RespRate': float(p[2]), 'Temp_C': float(p[3]),
                            'AccelMag': float(p[4]), 'Fall': int(p[5])
                        }
        except: pass
    return None

def build_figure():
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.5, wspace=0.4, left=0.05, right=0.95)
    axes = {}
    
    # Setup Charts
    charts = [('hr','HR',C_RED,(40,160)), ('spo2','SpO2',C_BLUE,(80,105)), ('rr','RR',C_GREEN,(5,40)), ('map','MAP',C_ORANGE,(50,150)), ('risk','Risk',C_PURPLE,(0,105))]
    for i, (k, l, c, y) in enumerate(charts):
        ax = fig.add_subplot(gs[0, i]); ax.set_facecolor(C_CARD)
        ax.set_ylim(*y); ax.set_title(l, color=C_WHITE); axes[k] = ax
        
    # Setup Cards
    cards = [('HR_v','HR',C_RED,(1,0)), ('SPO2_v','SpO2',C_BLUE,(1,1)), ('TEMP_v','Temp',C_YELLOW,(1,3)), ('ACCEL_v','Accel',C_TEAL,(2,0)), ('FALL_v','Fall',C_RED,(2,1))]
    for k, l, c, (r, col) in cards:
        ax = fig.add_subplot(gs[r, col]); ax.set_facecolor(C_CARD)
        ax.set_xticks([]); ax.set_yticks([]); axes[k] = ax
    
    # Status
    ax_s = fig.add_subplot(gs[2,4]); ax_s.set_facecolor(C_CARD); axes['status'] = ax_s
    return fig, axes

def update(frame, fig, axes, model):
    data = read_sensors()
    if not data: return
    
    read_no[0] += 1
    hr, spo2, rr, temp, accel, fall = data['HeartRate'], data['SpO2'], data['RespRate'], data['Temp_C'], data['AccelMag'], data['Fall']
    sbp, dbp, m_val = estimate_bp(hr, spo2, PATIENT['age'])
    
    hr_hist.append(hr); spo2_hist.append(spo2); temp_hist.append(temp); risk_hist.append(random.random()*20)

    # Simple Dashboard Update
    axes['hr'].cla(); axes['hr'].plot(list(hr_hist), color=C_RED)
    axes['HR_v'].cla(); axes['HR_v'].text(0.5, 0.5, f"{hr:.0f} BPM", color=C_RED, ha='center', va='center', fontsize=20, transform=axes['HR_v'].transAxes)
    axes['TEMP_v'].cla(); axes['TEMP_v'].text(0.5, 0.5, f"{temp:.1f} C", color=C_YELLOW, ha='center', va='center', fontsize=20, transform=axes['TEMP_v'].transAxes)
    
    status = "CRITICAL" if fall or hr > 140 else "STABLE"
    axes['status'].cla(); axes['status'].text(0.5, 0.5, status, color=C_RED if status=="CRITICAL" else C_GREEN, ha='center', va='center', fontsize=20, transform=axes['status'].transAxes)

def main():
    if not os.path.exists(MODEL_PATH): print("Model missing!"); return
    model = joblib.load(MODEL_PATH)
    
    try:
        SER[0] = serial.Serial(TARGET_COM, 115200, timeout=1)
        time.sleep(2); print(f"Connected to {TARGET_COM}")
    except: print("Simulation Mode Active")
    
    fig, axes = build_figure()
    ani = FuncAnimation(fig, update, fargs=(fig, axes, model), interval=500)
    plt.show()

if __name__ == '__main__':
    main()