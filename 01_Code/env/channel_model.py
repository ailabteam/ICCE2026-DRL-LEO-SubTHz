# 01_Code/env/channel_model.py

import numpy as np
from .constants import *

def calculate_path_loss(distance):
    """
    Tính Path Loss cơ bản trong Sub-THz (FSPL + Atmospheric Absorption).
    """
    if distance == 0:
        return 1e100 # Suy hao vô hạn
    
    # 1. Free Space Path Loss (FSPL)
    fspl = (4 * np.pi * distance * FREQUENCY / SPEED_OF_LIGHT)**2
    
    # 2. Atmospheric Absorption (Suy hao Khí quyển)
    # Giảm GAMMA_DB_PER_KM xuống cực thấp (0.001 dB/km) để khắc phục lỗi 300 dB suy hao.
    # Trong môi trường LEO, hấp thụ chủ yếu xảy ra ở lớp thấp (dưới 10km).
    # Tuy nhiên, để mô hình hóa đơn giản, ta chỉ giảm hệ số này.
    GAMMA_DB_PER_KM = 0.001 
    distance_km = distance / 1000
    
    absorption_loss_db = GAMMA_DB_PER_KM * distance_km
    absorption_loss_linear = 10**(absorption_loss_db / 10)
    
    # Tổng suy hao tuyến tính
    total_loss_linear = fspl * absorption_loss_linear
    return total_loss_linear

def calculate_beam_gain(los_vector_norm, beam_direction):
    """
    Tính độ lợi ăng-ten (Tx) bằng mô hình Gaussian, phụ thuộc vào góc lệch.
    """
    
    # 1. Tính Góc Lệch (Offset Angle)
    offset_angle = np.arccos(np.clip(np.dot(los_vector_norm, beam_direction), -1.0, 1.0))

    # 2. Độ lợi Chùm tia (Gaussian Model)
    normalized_gain = np.exp(- (offset_angle / BEAMWIDTH_3DB_RAD)**2 )
    
    # Total Gain (G_sat)
    G_sat_linear = G_MAX_LINEAR * normalized_gain
    
    # Độ lợi ăng-ten Rx của UE
    G_ue_linear = 10**(10/10) # 10 dBi
    
    return G_sat_linear * G_ue_linear

def calculate_channel_gain(sat_pos, ue_pos, beam_direction):
    """Tổng hợp tất cả suy hao và độ lợi H^2."""
    
    los_vector = ue_pos - sat_pos
    distance = np.linalg.norm(los_vector)
    los_vector_norm = los_vector / (distance + 1e-9)

    path_loss = calculate_path_loss(distance)
    
    antenna_gain = calculate_beam_gain(los_vector_norm, beam_direction)
    
    # Channel Gain H^2 = Antenna Gain / Path Loss
    H_sq = antenna_gain / path_loss
    
    return H_sq
