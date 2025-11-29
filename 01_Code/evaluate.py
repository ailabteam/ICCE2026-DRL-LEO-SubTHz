# 01_Code/evaluate.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from env.leo_drl_env import LEODRL150GHzEnv
from env.constants import MAX_STEPS_PER_EPISODE, MAX_SAT_TX_POWER_LINEAR, NUM_UE, SINR_MIN_LINEAR

# --- Tham số Đánh giá ---
N_EVAL_EPISODES = 20
MODEL_PATH = "./02_Results/checkpoints/final_model.zip"

def evaluate_model(model, env, name):
    """ Đánh giá mô hình đã học """
    print(f"--- Đánh giá mô hình: {name} ---")
    
    env.reset()
    results = {'Sum_Rate_Mbps': [], 'QoS_Violation_Rate': []}
    
    for episode in range(N_EVAL_EPISODES):
        obs, info = env.reset()
        episode_rate = 0
        episode_qos_violations = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Lấy hành động từ mô hình DRL
            action, _ = model.predict(obs, deterministic=True) 
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rate += info['sum_rate_mbps']
            episode_qos_violations += info['qos_violations']
            
            if terminated or truncated:
                break
        
        avg_rate = episode_rate / MAX_STEPS_PER_EPISODE
        violation_rate = (episode_qos_violations / NUM_UE) / MAX_STEPS_PER_EPISODE
        
        results['Sum_Rate_Mbps'].append(avg_rate)
        results['QoS_Violation_Rate'].append(violation_rate)

    print(f"[{name}] Avg Rate: {np.mean(results['Sum_Rate_Mbps']):.2f} Mbps")
    print(f"[{name}] Avg QoS Violation Rate: {np.mean(results['QoS_Violation_Rate'])*100:.2f}%")
    
    return results

def evaluate_baseline(env, name, power_strategy):
    """ Đánh giá các phương pháp cơ sở (Baselines) """
    print(f"--- Đánh giá Baseline: {name} ---")
    
    env.reset()
    results = {'Sum_Rate_Mbps': [], 'QoS_Violation_Rate': []}
    
    for episode in range(N_EVAL_EPISODES):
        obs, info = env.reset()
        episode_rate = 0
        episode_qos_violations = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            
            # 1. Hành động Baseline (Action Space 1 chiều: Power Fraction)
            if power_strategy == 'MAX_POWER':
                action = np.array([1.0], dtype=np.float32) # Luôn dùng 100% công suất
            elif power_strategy == 'HALF_POWER':
                action = np.array([0.5], dtype=np.float32) # Luôn dùng 50% công suất
            else: # Random Power
                action = np.array([np.random.rand()], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rate += info['sum_rate_mbps']
            episode_qos_violations += info['qos_violations']
            
            if terminated or truncated:
                break

        avg_rate = episode_rate / MAX_STEPS_PER_EPISODE
        violation_rate = (episode_qos_violations / NUM_UE) / MAX_STEPS_PER_EPISODE
        
        results['Sum_Rate_Mbps'].append(avg_rate)
        results['QoS_Violation_Rate'].append(violation_rate)
    
    print(f"[{name}] Avg Rate: {np.mean(results['Sum_Rate_Mbps']):.2f} Mbps")
    print(f"[{name}] Avg QoS Violation Rate: {np.mean(results['QoS_Violation_Rate'])*100:.2f}%")
    
    return results

if __name__ == "__main__":
    
    # 1. Chuẩn bị môi trường đánh giá (chỉ cần 1 môi trường)
    eval_env = LEODRL150GHzEnv()
    
    all_results = {}
    
    # 2. Đánh giá Baselines
    all_results['Baseline_MaxPower'] = evaluate_baseline(eval_env, 'Max Power (100%)', 'MAX_POWER')
    all_results['Baseline_HalfPower'] = evaluate_baseline(eval_env, 'Half Power (50%)', 'HALF_POWER')
    
    # 3. Đánh giá DRL Agent (Đảm bảo mô hình đã được lưu sau huấn luyện)
    if os.path.exists(MODEL_PATH):
        # Tải mô hình SAC
        # Phải dùng MlpPolicy và device phù hợp
        model = SAC.load(MODEL_PATH, env=eval_env, device="cpu") 
        all_results['DRL_SAC'] = evaluate_model(model, eval_env, 'DRL SAC Agent')
    else:
        print(f"Lỗi: Không tìm thấy mô hình DRL tại {MODEL_PATH}. Vui lòng huấn luyện trước.")

    # 4. Lưu kết quả
    df = pd.DataFrame({
        (method, metric): results[metric]
        for method, results in all_results.items()
        for metric in results
    })
    df.to_csv('./02_Results/evaluation_results.csv')
    print("\nKết quả đã được lưu vào 02_Results/evaluation_results.csv")
