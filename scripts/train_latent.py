import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from src.envs.latent_obs_wrapper import LatentObservationWrapper


ENV_ID = "CarRacing-v3"
N_ENVS = 2
TRAIN_TIMESTEPS = 2_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
DEVICE = "cpu"
LEARNING_RATE = 1e-4
EVAL_SEED = 10_000


AUTOENCODER_CHECKPOINT_PATH = Path(
    "results/shared/autoencoder/autoencoder_latent128.pt"
)

RESULTS_DIR = Path("results/main")

# Fine-tuning starts from an existing latent PPO checkpoint.
START_MODEL_PATH = None

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train latent-observation PPO.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed and run identifier.",
    )
    return parser.parse_args()


def build_run_paths(train_seed):
    run_dir = RESULTS_DIR / "latent" / f"seed_{train_seed}"
    return {
        "run_dir": run_dir,
        "train_log_dir": run_dir / "train_logs",
        "eval_log_dir": run_dir / "eval_logs",
        "checkpoint_dir": run_dir / "checkpoints",
        "final_model_path": run_dir / "checkpoints" / "final_model.zip",
        "best_model_dir": run_dir / "checkpoints" / "best",
    }


def ensure_new_run(run_dir):
    if run_dir.exists() and any(run_dir.rglob("*")):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. "
            "Use a new seed or archive the existing run."
        )


def save_config(run_dir, train_seed):
    config = {
        "agent_type": "latent",
        "run_name": f"seed_{train_seed}",
        "train_seed": train_seed,
        "eval_seed": EVAL_SEED,
        "environment": ENV_ID,
        "environment_kwargs": ENV_KWARGS,
        "n_envs": N_ENVS,
        "train_timesteps": TRAIN_TIMESTEPS,
        "eval_freq_callback_calls": EVAL_FREQ,
        "effective_eval_interval_timesteps": EVAL_FREQ * N_ENVS,
        "n_eval_episodes": N_EVAL_EPISODES,
        "learning_rate": LEARNING_RATE,
        "device": DEVICE,
        "policy": "MlpPolicy",
        "autoencoder_checkpoint": str(AUTOENCODER_CHECKPOINT_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def load_or_create_model(train_env, train_seed):
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
        seed=train_seed,
    )
    return model, False


def main(train_seed):
    paths = build_run_paths(train_seed)
    ensure_new_run(paths["run_dir"])

    paths["train_log_dir"].mkdir(parents=True, exist_ok=True)
    paths["eval_log_dir"].mkdir(parents=True, exist_ok=True)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    paths["best_model_dir"].mkdir(parents=True, exist_ok=True)
    save_config(paths["run_dir"], train_seed)

    train_env = make_vec_env(
        make_latent_env,
        n_envs=N_ENVS,
        seed=train_seed,
        monitor_dir=paths["train_log_dir"],
    )

    eval_env = make_vec_env(
        make_latent_env,
        n_envs=1,
        seed=EVAL_SEED,
        monitor_dir=paths["eval_log_dir"],
    )

    # The evaluation environment is separate from training so the saved best model
    # is selected using deterministic evaluation, not noisy rollout rewards.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=paths["best_model_dir"],
        log_path=paths["eval_log_dir"],
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    model, model_exists = load_or_create_model(train_env, train_seed)

    try:
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            reset_num_timesteps=not model_exists,
            callback=eval_callback,
        )

        model.save(paths["final_model_path"])
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.seed)
