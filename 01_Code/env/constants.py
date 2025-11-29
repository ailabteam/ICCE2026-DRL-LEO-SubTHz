# 01_Code/env/constants.py

import numpy as np

# --- 1. Tham số Hệ thống LEO ---
ORBIT_HEIGHT = 600 * 10**3  # Độ cao quỹ đạo LEO (600 km)
EARTH_RADIUS = 6371 * 10**3 # Bán kính Trái đất (meters)
NUM_UE = 5                 # Số lượng người dùng
VELOCITY_M_S = 7500         # Tốc độ vệ tinh LEO (7.5 km/s)

# --- 2. Tham số Sub-THz ---
FREQUENCY = 150 * 10**9     # Tần số hoạt động (150 GHz)
BANDWIDTH = 10 * 10**6      # Băng thông hệ thống (10 MHz)
SPEED_OF_LIGHT = 3e8

# --- 3. Tham số Công suất và Nhiễu ---
NOISE_TEMP = 290
BOLTZMANN_CONST = 1.38e-23
NOISE_POWER = BOLTZMANN_CONST * NOISE_TEMP * BANDWIDTH

MAX_SAT_TX_POWER_DBM = 50.0
MAX_SAT_TX_POWER_LINEAR = 10**((MAX_SAT_TX_POWER_DBM - 30) / 10)

# --- 4. Tham số Mảng Ăng-ten (Tăng Beamwidth) ---
ANTENNA_SIZE_TX = 100
G_MAX_DB = 50.0 # 50 dBi
G_MAX_LINEAR = 10**(G_MAX_DB / 10)

BEAMWIDTH_3DB_DEG = 3.0     # TĂNG ĐỘ RỘNG CHÙM TIA: 3.0 độ
BEAMWIDTH_3DB_RAD = np.radians(BEAMWIDTH_3DB_DEG)

# --- 5. Tham số DRL ---
SINR_MIN_DB = 5.0           
SINR_MIN_LINEAR = 10**(SINR_MIN_DB / 10)
TIME_SLOT_DURATION = 0.1
MAX_STEPS_PER_EPISODE = 500
