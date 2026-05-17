import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

model = PPO.load("ppo_CarRacing")

eval_env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)

obs, info = eval_env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

eval_env.close()