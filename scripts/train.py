import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from src.callbacks.frame_saving_callback import FrameSavingCallback
from src.envs.carracing_env import make_carracing_eval_env, make_carracing_train_env


ENV_ID = "CarRacing-v3"
N_ENVS = 2
TRAIN_TIMESTEPS = 2_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
DEVICE = "cuda"
LEARNING_RATE = 1e-4
EVAL_SEED = 10_000

# When enabled, frames are sampled during PPO training to build the
# unsupervised dataset used later for autoencoder training.
SAVE_FRAMES = False
TARGET_FRAMES = 50_000
SAVE_INTERVAL_MIN = 80
SAVE_INTERVAL_MAX = 180
MIN_STEPS_AFTER_RESET = 50
FRAME_SAMPLING_SEED = 42
FRAME_DIR = Path("data/frames")


RESULTS_DIR = Path("results/main")

# Fine-tuning starts from the best model found in the previous raw PPO run.
START_MODEL_PATH = None


ENV_KWARGS = {
    "render_mode": "rgb_array",
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train raw-pixel PPO.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed and run identifier.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Collect frames for the autoencoder dataset during training.",
    )
    return parser.parse_args()


def build_run_paths(train_seed):
    run_dir = RESULTS_DIR / "raw" / f"seed_{train_seed}"
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


def save_config(run_dir, train_seed, save_frames):
    config = {
        "agent_type": "raw",
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
        "policy": "CnnPolicy",
        "save_frames": save_frames,
        "frame_dir": str(FRAME_DIR) if save_frames else None,
        "frame_sampling_seed": FRAME_SAMPLING_SEED if save_frames else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def load_or_create_model(train_env, train_seed):
    """Load an existing PPO checkpoint, or create the raw-pixel baseline model."""
    if START_MODEL_PATH is not None and START_MODEL_PATH.exists():
        print(f"Loading fine-tuning start model from {START_MODEL_PATH}")
        model = PPO.load(START_MODEL_PATH, env=train_env, device=DEVICE)
        model.learning_rate = LEARNING_RATE
        model.lr_schedule = lambda _: LEARNING_RATE
        return model, True

    print("Creating new PPO raw-pixel model")
    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        seed=train_seed,
    )
    return model, False


def main(train_seed, save_frames=SAVE_FRAMES):
    paths = build_run_paths(train_seed)
    ensure_new_run(paths["run_dir"])

    paths["train_log_dir"].mkdir(parents=True, exist_ok=True)
    paths["eval_log_dir"].mkdir(parents=True, exist_ok=True)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    paths["best_model_dir"].mkdir(parents=True, exist_ok=True)
    save_config(paths["run_dir"], train_seed, save_frames)

    train_env = make_carracing_train_env(
        ENV_ID,
        N_ENVS,
        paths["train_log_dir"],
        ENV_KWARGS,
        seed=train_seed,
    )
    eval_env = make_carracing_eval_env(
        ENV_ID,
        paths["eval_log_dir"],
        ENV_KWARGS,
        seed=EVAL_SEED,
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

    callbacks = [eval_callback]

    if save_frames:
        # These frames form the unsupervised dataset for the CNN autoencoder.
        frame_callback = FrameSavingCallback(
            frame_dir=FRAME_DIR,
            min_save_interval=SAVE_INTERVAL_MIN,
            max_save_interval=SAVE_INTERVAL_MAX,
            target_frames=TARGET_FRAMES,
            min_steps_after_reset=MIN_STEPS_AFTER_RESET,
            seed=FRAME_SAMPLING_SEED,
        )
        callbacks.append(frame_callback)

    callbacks = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            reset_num_timesteps=not model_exists,
            callback=callbacks,
        )

        model.save(paths["final_model_path"])
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.seed, save_frames=args.save_frames)
