# 01_Code/env/system_utils.py

import numpy as np
from .constants import *

def initialize_ue_positions():
    """ Khởi tạo ngẫu nhiên vị trí người dùng (UEs) trên mặt đất (2D). """
    AREA_SIZE = 50 * 10**3 # Khu vực 50 km x 50 km
    
    # [x, y, z=0]
    ue_positions = np.random.uniform(-AREA_SIZE / 2, AREA_SIZE / 2, size=(NUM_UE, 2))
    ue_positions_3d = np.hstack((ue_positions, np.zeros((NUM_UE, 1))))
    
    return ue_positions_3d

def calculate_sat_position(time_step):
    """ Tính vị trí 3D của vệ tinh LEO tại thời điểm t. """
    t = time_step * TIME_SLOT_DURATION
    
    # Giả định bắt đầu từ -100km (x) và bay thẳng qua khu vực trung tâm
    SAT_START_X = -100 * 10**3
    
    sat_x = SAT_START_X + VELOCITY_M_S * t
    sat_y = 0.0
    sat_z = ORBIT_HEIGHT 
    
    return np.array([sat_x, sat_y, sat_z])

def get_los_vectors(sat_pos, ue_positions):
    """Tính vector hướng Line-of-Sight từ vệ tinh đến tất cả UEs."""
    los_vectors = ue_positions - sat_pos
    
    # Chuẩn hóa vector và tính khoảng cách
    norms = np.linalg.norm(los_vectors, axis=1, keepdims=True)
    los_normalized = los_vectors / (norms + 1e-9) # Thêm 1e-9 để tránh chia cho 0
    
    return los_normalized, norms.flatten()

def cartesian_to_spherical(vector):
    """Chuyển đổi vector [x, y, z] sang [radius, azimuth (phi), elevation (theta)]."""
    x, y, z = vector
    radius = np.linalg.norm(vector)
    
    # Góc Azimuth (Phi): từ trục x dương
    phi = np.arctan2(y, x) 
    
    # Góc Elevation/Zenith (Theta): từ trục z dương (90 - góc hạ)
    # LEO thường dùng góc Zenith/Nadir. Nếu ta dùng góc từ mặt phẳng ngang:
    theta = np.arccos(z / radius) # Angle from Z-axis (Zenith angle)
    
    return radius, phi, theta
