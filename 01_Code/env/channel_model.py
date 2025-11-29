# 01_Code/env/channel_model.py
import numpy as np
from .constants import * # Import các hằng số

def calculate_distance(sat_pos, ue_pos):
    """Tính khoảng cách 3D giữa vệ tinh và người dùng."""
    return np.linalg.norm(sat_pos - ue_pos)

def calculate_path_loss(distance, frequency_hz):
    """
    Tính Path Loss cơ bản trong Sub-THz (bao gồm suy hao do khoảng cách).
    Giả sử Line-of-Sight (LoS) cho môi trường LEO.
    """
    if distance == 0:
        return 0 
    
    # 1. Free Space Path Loss (FSPL)
    c = 3e8 # Tốc độ ánh sáng
    fspl = (4 * np.pi * distance * frequency_hz / c)**2
    
    # 2. Atmospheric Absorption (Suy hao Khí quyển)
    # Đây là phần đặc trưng của Sub-THz/mmWave. 
    # Chúng ta sử dụng một mô hình đơn giản: L_abs = gamma * d (dB)
    # gamma là hệ số hấp thụ (absorption coefficient)
    
    # Ở 150 GHz, gamma (dB/km) có thể xấp xỉ 0.1 - 1.0 (tùy thuộc vào khí hậu)
    gamma_db_per_km = 0.5  # 0.5 dB/km
    distance_km = distance / 1000
    
    absorption_loss_db = gamma_db_per_km * distance_km
    absorption_loss_linear = 10**(absorption_loss_db / 10)
    
    # Tổng suy hao
    total_loss_linear = fspl * absorption_loss_linear
    return total_loss_linear

def calculate_beam_gain(sat_pos, ue_pos, beam_angle_rad):
    """
    Tính độ lợi ăng-ten từ vệ tinh đến UE, dựa trên góc lệch của UE so với tâm chùm tia.
    Sử dụng mô hình độ lợi chùm tia đơn giản (vd: Gaussian-like).
    
    sat_pos, ue_pos: Vị trí 3D
    beam_angle_rad: Hướng chùm tia (vector đơn vị)
    """
    
    # 1. Tính vector hướng từ vệ tinh đến UE
    LOS_vector = ue_pos - sat_pos
    LOS_vector_norm = LOS_vector / np.linalg.norm(LOS_vector)
    
    # 2. Tính góc lệch (offset angle) so với tâm chùm tia
    # Chùm tia được điều chỉnh theo beam_angle_rad (vector hướng của chùm tia)
    
    # Giả sử beam_angle_rad là vector hướng của chùm tia (direction vector)
    # Góc lệch = arccos(dot product giữa LOS vector và Beam vector)
    # Với mục đích đơn giản, ta cần hàm này nhận vector hướng của chùm tia.
    
    # Nếu beam_angle_rad là vector [phi, theta] (góc phương vị, góc độ cao):
    # Cần chuyển đổi góc beam thành vector hướng.
    
    # Tạm thời đơn giản hóa: giả sử `beam_direction` là vector hướng 3D của chùm tia
    beam_direction = beam_angle_rad # Thay đổi tên biến để dễ hiểu
    offset_angle = np.arccos(np.clip(np.dot(LOS_vector_norm, beam_direction), -1.0, 1.0))

    # 3. Tính độ lợi theo mô hình chùm tia (ví dụ: mô hình Sinc đơn giản)
    # G(theta) = G_max * (J1(u) / u)^2, hoặc đơn giản hơn là mô hình cosine.
    
    # Độ lợi tối đa (Max Gain)
    # Đối với mảng ăng-ten lớn, G_max có thể ước tính là (4 * pi * A / lambda^2) hoặc N*G_element
    G_max_linear = ANTENNA_SIZE_TX * 10 # Giả sử 10 là độ lợi của 1 phần tử
    
    # Sử dụng mô hình độ lợi dựa trên góc lệch và độ rộng chùm tia 3dB
    # Giả sử độ lợi giảm theo hàm mũ/Gaussian khi lệch khỏi tâm chùm
    
    # DRL Agent sẽ điều chỉnh beam_direction.
    
    # Normalized Gain (giảm dần từ 1)
    normalized_gain = np.exp(- (offset_angle / BEAMWIDTH_3DB)**2 )
    
    # Total Gain
    G_sat = G_max_linear * normalized_gain
    
    # Giả sử UE cũng có ăng-ten có độ lợi nhỏ (G_ue)
    G_ue = 10 # 10 dBi (linear)
    
    return G_sat * G_ue

def calculate_channel_gain(sat_pos, ue_pos, beam_direction, frequency_hz):
    """Tổng hợp tất cả suy hao và độ lợi."""
    distance = calculate_distance(sat_pos, ue_pos)
    if distance == 0:
        return 0
    
    path_loss = calculate_path_loss(distance, frequency_hz)
    
    # Độ lợi ăng-ten (tính cả Tx và Rx)
    antenna_gain = calculate_beam_gain(sat_pos, ue_pos, beam_direction)
    
    # Channel Gain (H^2) = (Antenna Gain) / Path Loss
    H_sq = antenna_gain / path_loss
    
    return H_sq
