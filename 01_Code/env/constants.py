# 01_Code/env/constants.py

import numpy as np

# --- 1. Tham số Hệ thống LEO ---
ORBIT_HEIGHT = 600 * 10**3  # Độ cao quỹ đạo LEO (600 km)
EARTH_RADIUS = 6371 * 10**3 # Bán kính Trái đất (meters)
COVERAGE_ANGLE_MAX = np.radians(60) # Góc bao phủ tối đa của vệ tinh (dễ dàng đơn giản hóa phạm vi)
NUM_UE = 10                 # Số lượng người dùng (Users)

# --- 2. Tham số Sub-THz ---
FREQUENCY = 150 * 10**9     # Tần số hoạt động (150 GHz - dải Sub-THz)
WAVELENGTH = 3e8 / FREQUENCY # Bước sóng
BANDWIDTH = 10 * 10**6      # Băng thông hệ thống (10 MHz)

# --- 3. Tham số Công suất và Nhiễu ---
NOISE_TEMP = 290            # Nhiệt độ nhiễu (Kelvin)
BOLTZMANN_CONST = 1.38e-23  # Hằng số Boltzmann
NOISE_POWER_DENSITY = BOLTZMANN_CONST * NOISE_TEMP
NOISE_POWER = NOISE_POWER_DENSITY * BANDWIDTH # Công suất nhiễu tạp âm (N0 * B)

# Tổng công suất truyền tối đa của vệ tinh (đơn vị W)
# Giả sử công suất truyền lớn do sử dụng Sub-THz và mảng ăng-ten lớn
MAX_SAT_TX_POWER = 100 # 100 Watts

# --- 4. Tham số Mảng Ăng-ten (Massive Array) ---
ANTENNA_SIZE_TX = 64        # Số phần tử ăng-ten trên vệ tinh (Ví dụ: 8x8 = 64)
BEAMWIDTH_3DB = np.radians(2 * np.pi / ANTENNA_SIZE_TX) # Ước tính độ rộng chùm tia hẹp

# --- 5. Tham số DRL ---
SINR_MIN_DB = 5             # Yêu cầu SINR tối thiểu (QoS) cho mỗi UE (5 dB)
SINR_MIN_LINEAR = 10**(SINR_MIN_DB / 10)
