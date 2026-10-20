================================================================
 AI-POWERED POST-SURGERY PATIENT MONITORING SYSTEM — Hardware v2
 EPICS Project | VIT Bhopal University
================================================================

HARDWARE SENSORS
----------------
Sensor        GPIO          Measures
-----------   -----------   ------------------------------------------
MAX30105      SDA→21,       Heart Rate (bpm)
              SCL→22        SpO₂ (%)
                            Respiratory Rate (derived from IR waveform)

DS18B20       DATA→GPIO 4   Temperature (°C)
              + 4.7kΩ       (4.7kΩ pull-up between DATA and 3.3V)
              resistor

MPU6050       SDA→21,       Acceleration magnitude (m/s²)
              SCL→22        Fall Detection (binary flag)

LCD 16×2      SDA→21,       Displays vitals + alert messages
(I2C, 0x27)   SCL→22

Buzzer        GPIO 25       Critical/warning alarm
              → GND

All I2C devices share the same SDA/SCL bus (GPIO 21/22).
VCC: MAX30105 → 3.3V or 5V | MPU6050 → 3.3V | DS18B20 → 3.3V | LCD → 5V

7 VITALS MONITORED
-------------------
#   Vital              Sensor/Method              Unit
1   Heart Rate         MAX30105 (red + IR LED)    bpm
2   SpO₂               MAX30105 (red/IR ratio)    %
3   Respiratory Rate   MAX30105 IR waveform       breaths/min
                       (peak-to-peak interval)
4   Temperature        DS18B20 (1-Wire)            °C
5   MAP (estimated)    HR + SpO₂ + age proxy      mmHg
6   Accel Magnitude    MPU6050 (3-axis)           m/s²
7   Fall Detected      MPU6050 (>2.5g spike)      binary (0/1)

  Note: SBP and DBP are derived from MAP via standard formula
        MAP = DBP + (SBP-DBP)/3. They are estimated surrogates,
        not direct cuff measurements.

CLINICAL THRESHOLDS (NEWS2 / AAMI / RCP guidelines)
------------------------------------------------------
Vital         Low-Critical  Low-Warn  High-Warn  High-Crit   Source
Heart Rate    ≤40 bpm       ≤50       >100       >120        NEWS2
SpO₂          <90%          <94%      —          —           NEWS2
Resp Rate     ≤8 /min       ≤10       >20        ≥25         NEWS2
Temperature   ≤35.0°C       ≤36.0     >38.0      >39.0       NEWS2
MAP (est.)    <60 mmHg      <70       >110       >130        SCCM/ICU
Accel Mag.    —             —         >20 m/s²   >24.5 m/s²  (2.5g fall)
Fall          —             —         —          YES         MPU6050 event

FILES
------
esp32_firmware/
  esp32_vitals_sender.ino   Arduino firmware for ESP32
                             Sends:  HR,SpO2,RespRate,Temp_C,AccelMag,FallDetected
                             Baud:   115200
                             Rate:   every 2 seconds

step1_extract_data.py       Extract 10 features from MIMIC-III
                             Output: data/mimic_7feature_dataset.csv

step2_train_model.py        Train Random Forest (300 trees)
                             Output: models/healthcare_forest_v2.pkl
                                     outputs/evaluation_report.txt

step3_monitor.py            Terminal real-time monitor
                             Reads from ESP32 serial (auto-detects port)
                             Falls back to simulation if no hardware

step4_dashboard.py          Live Matplotlib dashboard
                             Shows all 7 vitals + AI risk + fall indicator
                             Falls back to simulation if no hardware

HOW TO RUN (first time)
------------------------
1. Flash ESP32:
     Open Arduino IDE
     Install libraries (Library Manager):
       - SparkFun MAX3010x Sensor Library
       - MPU6050 by Electronic Cats
       - DallasTemperature + OneWire
       - LiquidCrystal_I2C
     Open esp32_firmware/esp32_vitals_sender.ino
     Select board: ESP32 Dev Module, Port: COMX / /dev/ttyUSB0
     Upload

2. Install Python dependencies:
     pip install scikit-learn pandas numpy joblib matplotlib pyserial

3. Place MIMIC-III files in project root:
     chartevents.csv  (from MIMIC-III PhysioNet)
     patients.csv

4. Run pipeline:
     python step1_extract_data.py     (generates dataset)
     python step2_train_model.py      (trains & saves model)
     python step3_monitor.py          (terminal monitor)
   OR
     python step4_dashboard.py        (visual dashboard)

IF MODEL IS ALREADY TRAINED
------------------------------
Skip steps 3 & 4. The included healthcare_forest_v2.pkl works directly.
Just run step3_monitor.py or step4_dashboard.py.

SERIAL PORT
-----------
The Python scripts auto-detect the ESP32 port by looking for CP210x/
CH340/FTDI descriptors. If not auto-detected, open step3_monitor.py or
step4_dashboard.py and set SER[0] = serial.Serial('/dev/ttyUSB0', 115200)
manually at the top.

================================================================
