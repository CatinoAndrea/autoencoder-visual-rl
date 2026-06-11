from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from src.envs.latent_obs_wrapper import LatentObservationWrapper


ENV_ID = "CarRacing-v3"
AGENT_TYPE = "raw"
TRAIN_SEED = 42
EVAL_SEED = 10_000
USE_BEST_MODEL = True
N_EVAL_EPISODES = 3

RAW_DEVICE = "cuda"
LATENT_DEVICE = "cpu"

RUN_DIR = Path("results/main") / AGENT_TYPE / f"seed_{TRAIN_SEED}"
CHECKPOINT_PATH = (
    RUN_DIR / "checkpoints" / "best" / "best_model.zip"
    if USE_BEST_MODEL
    else RUN_DIR / "checkpoints" / "final_model.zip"
)
AUTOENCODER_CHECKPOINT_PATH = Path(
    "results/shared/autoencoder/autoencoder_latent128.pt"
)

ENV_KWARGS = {
    "render_mode": "human",
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
}


def make_eval_env():
    """Create the raw or latent environment used for visual evaluation."""
    env = gym.make(ENV_ID, **ENV_KWARGS)

    if AGENT_TYPE == "latent":
        env = LatentObservationWrapper(
            env,
            checkpoint_path=AUTOENCODER_CHECKPOINT_PATH,
            device=LATENT_DEVICE,
        )
    elif AGENT_TYPE != "raw":
        env.close()
        raise ValueError("AGENT_TYPE must be 'raw' or 'latent'")

    return env


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    device = RAW_DEVICE if AGENT_TYPE == "raw" else LATENT_DEVICE
    model = PPO.load(CHECKPOINT_PATH, device=device)
    eval_env = make_eval_env()

    try:
        for episode in range(N_EVAL_EPISODES):
            obs, info = eval_env.reset(seed=EVAL_SEED + episode)
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
