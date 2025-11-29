# 01_Code/env/constants.py

import numpy as np

# --- 1. Tham số Hệ thống LEO ---
ORBIT_HEIGHT = 600 * 10**3  # Độ cao quỹ đạo LEO (600 km)
EARTH_RADIUS = 6371 * 10**3 # Bán kính Trái đất (meters)
NUM_UE = 5                 # Số lượng người dùng (giảm xuống 5 để đơn giản hóa State/Action space ban đầu)
VELOCITY_M_S = 7500         # Tốc độ vệ tinh LEO (7.5 km/s)

# --- 2. Tham số Sub-THz ---
FREQUENCY = 150 * 10**9     # Tần số hoạt động (150 GHz)
BANDWIDTH = 10 * 10**6      # Băng thông hệ thống (10 MHz)
SPEED_OF_LIGHT = 3e8

# --- 3. Tham số Công suất và Nhiễu ---
NOISE_TEMP = 290            # Nhiệt độ nhiễu (Kelvin)
BOLTZMANN_CONST = 1.38e-23  # Hằng số Boltzmann
NOISE_POWER = BOLTZMANN_CONST * NOISE_TEMP * BANDWIDTH # Công suất nhiễu tạp âm

# Tổng công suất truyền tối đa của vệ tinh (dBm)
MAX_SAT_TX_POWER_DBM = 50.0 # Tương đương 100W
MAX_SAT_TX_POWER_LINEAR = 10**((MAX_SAT_TX_POWER_DBM - 30) / 10)

# --- 4. Tham số Mảng Ăng-ten ---
ANTENNA_SIZE_TX = 64        # Số phần tử ăng-ten
# Độ lợi tối đa ăng-ten: G_max_linear = N^2 (rất lớn, 4096)
# Tuy nhiên, chúng ta chỉ cần ước tính độ lợi
G_MAX_DB = 40.0 # Độ lợi tối đa (40 dBi)
G_MAX_LINEAR = 10**(G_MAX_DB / 10)

# Beamwidth (Độ rộng chùm tia) cho mô hình Gaussian
BEAMWIDTH_3DB_DEG = 1.0     # 1 độ
BEAMWIDTH_3DB_RAD = np.radians(BEAMWIDTH_3DB_DEG)

# --- 5. Tham số DRL ---
SINR_MIN_DB = 5.0           # Yêu cầu SINR tối thiểu (5 dB)
SINR_MIN_LINEAR = 10**(SINR_MIN_DB / 10)
TIME_SLOT_DURATION = 0.1    # Thời gian của một bước (0.1 giây)
MAX_STEPS_PER_EPISODE = 500
