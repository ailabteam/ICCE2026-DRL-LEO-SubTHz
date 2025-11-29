# 01_Code/env/leo_drl_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .constants import *
from .channel_model import calculate_channel_gain, calculate_path_loss 
from .system_utils import initialize_ue_positions, calculate_sat_position, get_los_vectors

class LEODRL150GHzEnv(gym.Env):
# ... (Phần __init__, _get_obs, reset giữ nguyên) ...

    def __init__(self):
        super(LEODRL150GHzEnv, self).__init__()

        # --- ĐỊNH NGHĨA ACTION SPACE (1 chiều) ---
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # --- ĐỊNH NGHĨA OBSERVATION SPACE ---
        obs_size = NUM_UE * 3 + NUM_UE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.time_step = 0
        self.ue_positions = initialize_ue_positions() 
        self.sat_pos = calculate_sat_position(0)
        
        self.PL_MIN_DB = 170.0 
        self.PL_RANGE_DB = 25.0 

    def _get_obs(self, sat_pos):
        los_normalized, distances = get_los_vectors(sat_pos, self.ue_positions)
        raw_pl = np.array([calculate_path_loss(d) for d in distances])
        pl_db = 10 * np.log10(raw_pl)
        normalized_pl = np.clip((pl_db - self.PL_MIN_DB) / self.PL_RANGE_DB, 0.0, 1.0)
        
        obs = np.concatenate([
            los_normalized.flatten(), 
            normalized_pl 
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.time_step = 0
        self.ue_positions = initialize_ue_positions() 
        self.sat_pos = calculate_sat_position(self.time_step)
        
        observation = self._get_obs(self.sat_pos)
        info = {}
        return observation, info


    def step(self, action):
        
        power_fraction = np.clip(action[0], 0.0, 1.0)
        total_tx_power = power_fraction * MAX_SAT_TX_POWER_LINEAR 
        
        los_normalized, dists = get_los_vectors(self.sat_pos, self.ue_positions)
        
        # 2. Cố định Beam Direction (Hướng vào LOS trung bình)
        beam_direction_norm = np.mean(los_normalized, axis=0) 
        beam_direction_norm = beam_direction_norm / (np.linalg.norm(beam_direction_norm) + 1e-9)

        self.time_step += 1 
        self.sat_pos = calculate_sat_position(self.time_step)
        
        power_per_ue = total_tx_power / NUM_UE 

        total_rate = 0.0
        qos_violations = 0.0
        
        sinr_dbs = []
        
        for i in range(NUM_UE):
            ue_pos = self.ue_positions[i]
            
            # Tính Channel Gain (H^2)
            # H_sq = G_total / PL
            H_sq = calculate_channel_gain(self.sat_pos, ue_pos, beam_direction_norm)
            
            SINR_i = (power_per_ue * H_sq) / NOISE_POWER 
            
            SINR_i_clipped = np.clip(SINR_i, 1e-12, None)
            rate_i = BANDWIDTH * np.log2(1 + SINR_i_clipped)
            total_rate += rate_i
            
            if SINR_i < SINR_MIN_LINEAR:
                qos_violations += 1.0 

            sinr_dbs.append(10 * np.log10(SINR_i_clipped))
        
        # --- DEBUG THỰC TẾ (chỉ in khi bắt đầu học) ---
        if self.time_step > 10000 and self.time_step % 500 == 0:
            print(f"[DEBUG @ {self.time_step}]: SINR (dB) for UEs: {sinr_dbs}")
            print(f"[DEBUG @ {self.time_step}]: QOS Violations: {qos_violations}")

        # 4. Reward
        w_penalty = 100 
        reward = (total_rate / 1e6) - w_penalty * qos_violations 
        
        terminated = False 
        truncated = (self.time_step >= MAX_STEPS_PER_EPISODE)
        
        observation = self._get_obs(self.sat_pos)
        info = {"sum_rate_mbps": total_rate / 1e6, "qos_violations": qos_violations}

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
