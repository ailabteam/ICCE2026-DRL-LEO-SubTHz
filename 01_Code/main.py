# 01_Code/main.py

import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from env.leo_drl_env import LEODRL150GHzEnv

# --- Thiết lập Tham số Huấn luyện ---
LOG_DIR = "./02_Results/logs/"
CHECKPOINT_DIR = "./02_Results/checkpoints/"
TOTAL_TIMESTEPS = 500000  # Tổng số bước học (có thể cần tăng sau này)
N_ENVS = 4 # Số môi trường song song (Parallel Environments)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Đảm bảo các thư mục tồn tại
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_sac_agent():
    """
    Khởi tạo và huấn luyện Agent SAC.
    """
    print(f"Bắt đầu huấn luyện SAC trên thiết bị: {DEVICE}")
    
    # 1. Tạo môi trường Vectorized (song song)
    # Sử dụng SubprocVecEnv để tăng tốc độ thu thập dữ liệu trên CPU đa lõi
    env = make_vec_env(LEODRL150GHzEnv, n_envs=N_ENVS, seed=0)
    
    # Thiết lập Logger SB3
    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

    # 2. Định nghĩa Kiến trúc Mạng Nơ-ron (Policy)
    # SAC sử dụng MLP (Mô hình Mạng Đa Tầng)
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], qf=[256, 256])] # Hai lớp ẩn 256 cho Actor (pi) và Critic (qf)
    )

    # 3. Khởi tạo Agent SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000, # Bắt đầu học sau 10k bước khám phá
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=0,
        device=DEVICE
    )
    
    model.set_logger(new_logger)

    # 4. Thiết lập Callbacks (Lưu trữ mô hình tốt nhất)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # Lưu trữ mỗi 10,000 bước
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_leosubthz_model",
        save_replay_buffer=False,
        save_vec_normalize=False,
    )

    # 5. Bắt đầu Huấn luyện
    print(f"Bắt đầu huấn luyện với {TOTAL_TIMESTEPS} Timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback
        )
    except Exception as e:
        print(f"Lỗi xảy ra trong quá trình huấn luyện: {e}")
    finally:
        # Lưu mô hình cuối cùng
        model.save(os.path.join(CHECKPOINT_DIR, "final_model.zip"))
        end_time = time.time()
        print(f"Huấn luyện hoàn thành. Thời gian: {end_time - start_time:.2f} giây.")


if __name__ == "__main__":
    import torch
    train_sac_agent()
