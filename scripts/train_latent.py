from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from src.envs.latent_obs_wrapper import LatentObservationWrapper


ENV_ID = "CarRacing-v3"
RUN_NAME = "v3"
N_ENVS = 2
TRAIN_TIMESTEPS = 2_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
DEVICE = "cpu"
LEARNING_RATE = 3e-4

AUTOENCODER_CHECKPOINT_PATH = Path("results/checkpoints/autoencoder/autoencoder_latent128.pt")
LOG_DIR = Path("results/logs/latent")
MODEL_DIR = Path("results/checkpoints/latent")

START_MODEL_PATH = None

# Final checkpoint from this run. The best evaluated checkpoint is saved separately.
FINAL_MODEL_PATH = MODEL_DIR / f"last_ppo_CarRacing_{RUN_NAME}.zip"
BEST_MODEL_DIR = MODEL_DIR / f"best_model_{RUN_NAME}"

ENV_KWARGS = {
    "render_mode": "rgb_array",
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
}

def make_latent_env():
    env = gym.make(ENV_ID, **ENV_KWARGS)
    env = LatentObservationWrapper(
        env, 
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        device=DEVICE,
    )
    return env

def load_or_create_model(train_env):
    """Load an existing latent PPO checkpoint, or create a new latent PPO model."""
    if START_MODEL_PATH is not None and START_MODEL_PATH.exists():
        print(f"Loading fine-tuning start model from {START_MODEL_PATH}")
        model = PPO.load(START_MODEL_PATH, env=train_env, device=DEVICE)
        model.learning_rate = LEARNING_RATE
        model.lr_schedule = lambda _: LEARNING_RATE
        return model, True

    print("Creating new PPO latent model")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
    )
    return model, False


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        make_latent_env,
        n_envs=N_ENVS,
        monitor_dir=LOG_DIR / "train" / RUN_NAME,
    )

    eval_env = make_vec_env(
        make_latent_env,
        n_envs=1,
        monitor_dir=LOG_DIR / "eval" / RUN_NAME,
    )

    # The evaluation environment is separate from training so the saved best model
    # is selected using deterministic evaluation, not noisy rollout rewards.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR / "eval" / RUN_NAME,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    model, model_exists = load_or_create_model(train_env)

    try:
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            reset_num_timesteps=not model_exists,
            callback=eval_callback,
        )

        model.save(FINAL_MODEL_PATH)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
