# 01_Code/env/leo_drl_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .constants import *
from .channel_model import calculate_channel_gain
from .system_utils import initialize_ue_positions, calculate_sat_position, get_los_vectors

class LEODRL150GHzEnv(gym.Env):
    """
    Môi trường DRL cho Quản lý Beam và Tài nguyên trong Mạng LEO Sub-THz.
    Action Space: [Power Fraction, Beam Dir X, Beam Dir Y, Beam Dir Z]
    """
    
    def __init__(self):
        super(LEODRL150GHzEnv, self).__init__()

        # ACTION SPACE: [Power Fraction (0-1), Beam Dir Vector X (-1-1), Y, Z]
        # Kích thước: 1 (Power) + 3 (Beam Vector) = 4
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: [LOS vectors (NUM_UE * 3), Estimated Path Losses (NUM_UE)]
        # Kích thước: (NUM_UE * 3) + NUM_UE
        obs_size = NUM_UE * 3 + NUM_UE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.time_step = 0
        self.ue_positions = initialize_ue_positions() 
        self.sat_pos = calculate_sat_position(0)

    def _get_obs(self, sat_pos):
        """Tính toán trạng thái quan sát (Observation)"""
        
        los_normalized, distances = get_los_vectors(sat_pos, self.ue_positions)
        
        # Để đơn giản, Estimated Path Losses chỉ là giá trị Path Loss (linear)
        estimated_pl = np.array([calculate_path_loss(d) for d in distances])
        
        # Trạng thái: [LOS vectors, Path Losses]
        obs = np.concatenate([
            los_normalized.flatten(), 
            estimated_pl 
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.time_step = 0
        # Reset UE positions và tính vị trí vệ tinh ban đầu
        self.ue_positions = initialize_ue_positions() 
        self.sat_pos = calculate_sat_position(self.time_step)
        
        observation = self._get_obs(self.sat_pos)
        info = {}
        return observation, info

    def step(self, action):
        
        # 1. Giải mã Hành động
        power_fraction = np.clip(action[0], 0.0, 1.0)
        total_tx_power = power_fraction * MAX_SAT_TX_POWER_LINEAR 
        
        # Chuẩn hóa Beam Direction (đảm bảo là vector đơn vị)
        beam_direction = action[1:4]
        norm = np.linalg.norm(beam_direction)
        beam_direction_norm = beam_direction / (norm + 1e-9)

        # 2. Cập nhật vị trí (Mobility)
        self.time_step += 1 
        self.sat_pos = calculate_sat_position(self.time_step)
        
        # 3. Tính toán SINR và Throughput
        
        # Giả định phân bổ công suất đồng đều cho tất cả NUM_UE
        power_per_ue = total_tx_power / NUM_UE 

        total_rate = 0.0
        penalty = 0.0
        
        for i in range(NUM_UE):
            ue_pos = self.ue_positions[i]
            
            # Channel Gain (H^2)
            H_sq = calculate_channel_gain(self.sat_pos, ue_pos, beam_direction_norm)

            # SINR calculation (Giả định nhiễu xuyên kênh = 0 cho bản đầu tiên)
            # SINR_i = (Power * H^2) / Noise Power
            SINR_i = (power_per_ue * H_sq) / NOISE_POWER 

            # Tốc độ truyền tải (Shannon) (bits/sec)
            rate_i = BANDWIDTH * np.log2(1 + SINR_i)
            
            total_rate += rate_i
            
            # Ràng buộc QoS
            if SINR_i < SINR_MIN_LINEAR:
                penalty += 1.0 # Phạt cho mỗi UE vi phạm QoS

        # 4. Reward: Tối đa hóa Rate (Mbps) và Giảm Penalty
        # Chia 1e6 để reward nằm trong phạm vi dễ xử lý (Mbps)
        # Hệ số phạt w_penalty = 1000 (Điều chỉnh để Agent quan tâm đến QoS)
        w_penalty = 1000 
        reward = (total_rate / 1e6) - w_penalty * penalty 
        
        # 5. Kết thúc Episode
        terminated = False 
        truncated = (self.time_step >= MAX_STEPS_PER_EPISODE)
        
        observation = self._get_obs(self.sat_pos)
        info = {"sum_rate_mbps": total_rate / 1e6, "qos_violations": penalty}

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
