from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID = "CarRacing-v3"
FINAL_MODEL_PATH = Path("results/checkpoints/raw/last_ppo_CarRacing_v2.zip")
BEST_MODEL_PATH = Path("results/checkpoints/raw/best_raw/v2/best_model.zip")
N_EVAL_EPISODES = 3
DEVICE = "cuda"

ENV_KWARGS = {
    "render_mode": "human",
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
}


def main():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {BEST_MODEL_PATH}")

    model = PPO.load(BEST_MODEL_PATH, device=DEVICE)
    eval_env = gym.make(ENV_ID, **ENV_KWARGS)

    try:
        for episode in range(N_EVAL_EPISODES):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                episode_reward += float(reward)
                episode_length += 1
                done = terminated or truncated

            print(
                f"Episode {episode + 1}: "
                f"reward={episode_reward:.2f}, length={episode_length}"
            )
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
