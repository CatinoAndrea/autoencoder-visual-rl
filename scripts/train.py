import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


train_env = make_vec_env(
    "CarRacing-v3",
    n_envs=2,
    env_kwargs={
        "render_mode": "rgb_array",
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
    },
)

#model = PPO("CnnPolicy", train_env, verbose=1)

model = PPO.load("ppo_CarRacing", env=train_env)
model.learn(total_timesteps=200000, reset_num_timesteps=False)
model.save("ppo_CarRacing")

del model

train_env.close()

