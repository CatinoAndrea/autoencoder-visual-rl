from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
LOG_ROOT = RESULTS_DIR / "logs"
PLOT_ROOT = RESULTS_DIR / "plots"

WINDOW = 20

AGENT_LABELS = {
    "raw": "PPO Raw Pixels",
    "latent": "PPO Latent Representation",
}


def load_monitor_file(path: Path) -> pd.DataFrame:
    """Load one Stable-Baselines3 Monitor CSV file."""
    return pd.read_csv(path, comment="#")


def load_training_data(log_dir: Path) -> pd.DataFrame:
    """Load all Monitor CSV files from one training run."""
    monitor_files = sorted(log_dir.glob("*.monitor.csv"))

    if not monitor_files:
        raise FileNotFoundError(f"No monitor files found in {log_dir}")

    dfs = []
    for file in monitor_files:
        df = load_monitor_file(file)
        if df.empty:
            continue
        df["source"] = file.name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No completed episodes found in {log_dir}")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("t").reset_index(drop=True)
    data["episode"] = range(len(data))

    # Approximate cumulative environment steps from completed episode lengths.
    data["timesteps"] = data["l"].cumsum()

    data["reward_ma"] = data["r"].rolling(window=WINDOW, min_periods=1).mean()
    data["reward_std"] = data["r"].rolling(window=WINDOW, min_periods=2).std()
    data["length_ma"] = data["l"].rolling(window=WINDOW, min_periods=1).mean()
    data["best_reward_so_far"] = data["r"].cummax()

    return data


def title(agent_type: str, run_name: str, metric: str) -> str:
    return f"CarRacing {AGENT_LABELS[agent_type]} {run_name} - {metric}"


def save_reward_curve(data: pd.DataFrame, output_dir: Path, agent_type: str, run_name: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["r"], alpha=0.25, label="Episode reward")
    plt.plot(
        data["timesteps"],
        data["reward_ma"],
        linewidth=2,
        label=f"Moving average ({WINDOW})",
    )
    plt.xlabel("Environment timesteps")
    plt.ylabel("Episode reward")
    plt.title(title(agent_type, run_name, "Training Reward"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=160)
    plt.close()


def save_reward_stability(data: pd.DataFrame, output_dir: Path, agent_type: str, run_name: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["reward_std"], linewidth=2)
    plt.xlabel("Environment timesteps")
    plt.ylabel(f"Reward std, rolling window {WINDOW}")
    plt.title(title(agent_type, run_name, "Reward Variability"))
    plt.tight_layout()
    plt.savefig(output_dir / "reward_variability.png", dpi=160)
    plt.close()


def save_episode_length_curve(data: pd.DataFrame, output_dir: Path, agent_type: str, run_name: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["l"], alpha=0.25, label="Episode length")
    plt.plot(
        data["timesteps"],
        data["length_ma"],
        linewidth=2,
        label=f"Moving average ({WINDOW})",
    )
    plt.xlabel("Environment timesteps")
    plt.ylabel("Episode length")
    plt.title(title(agent_type, run_name, "Episode Length"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "episode_length.png", dpi=160)
    plt.close()


def save_best_reward_curve(data: pd.DataFrame, output_dir: Path, agent_type: str, run_name: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["best_reward_so_far"], linewidth=2)
    plt.xlabel("Environment timesteps")
    plt.ylabel("Best episode reward so far")
    plt.title(title(agent_type, run_name, "Best Training Reward Over Time"))
    plt.tight_layout()
    plt.savefig(output_dir / "best_reward.png", dpi=160)
    plt.close()


def save_reward_histogram(data: pd.DataFrame, output_dir: Path, agent_type: str, run_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(data["r"], bins=25, edgecolor="black", alpha=0.8)
    plt.xlabel("Episode reward")
    plt.ylabel("Number of episodes")
    plt.title(title(agent_type, run_name, "Training Reward Distribution"))
    plt.tight_layout()
    plt.savefig(output_dir / "reward_distribution.png", dpi=160)
    plt.close()


def save_summary(data: pd.DataFrame, output_dir: Path) -> None:
    last_window = data.tail(WINDOW)
    summary = {
        "episodes": len(data),
        "total_completed_timesteps": int(data["timesteps"].iloc[-1]),
        "mean_reward_all": data["r"].mean(),
        "std_reward_all": data["r"].std(),
        "mean_reward_last_window": last_window["r"].mean(),
        "std_reward_last_window": last_window["r"].std(),
        "best_reward": data["r"].max(),
        "mean_episode_length_all": data["l"].mean(),
        "mean_episode_length_last_window": last_window["l"].mean(),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "training_summary.csv", index=False)


def save_training_plots(agent_type: str, run_name: str) -> None:
    log_dir = LOG_ROOT / agent_type / "train" / run_name
    output_dir = PLOT_ROOT / agent_type / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = load_training_data(log_dir)
    except FileNotFoundError as error:
        print(f"Skipping {agent_type}/{run_name}: {error}")
        return

    save_reward_curve(data, output_dir, agent_type, run_name)
    save_reward_stability(data, output_dir, agent_type, run_name)
    save_episode_length_curve(data, output_dir, agent_type, run_name)
    save_best_reward_curve(data, output_dir, agent_type, run_name)
    save_reward_histogram(data, output_dir, agent_type, run_name)
    save_summary(data, output_dir)

    print(f"Saved training plots to {output_dir}")


def load_evaluation_data(agent_type: str, run_name: str) -> pd.DataFrame:
    eval_path = LOG_ROOT / agent_type / "eval" / run_name / "evaluations.npz"

    if not eval_path.exists():
        raise FileNotFoundError(f"No evaluation file found at {eval_path}")

    data = np.load(eval_path)
    rewards = data["results"]

    return pd.DataFrame(
        {
            "agent_type": agent_type,
            "run_name": run_name,
            "timesteps": data["timesteps"],
            "mean_reward": rewards.mean(axis=1),
            "std_reward": rewards.std(axis=1),
            "mean_episode_length": data["ep_lengths"].mean(axis=1),
        }
    )


def save_evaluation_curve(eval_data: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))

    for (agent_type, run_name), group in eval_data.groupby(["agent_type", "run_name"]):
        label = f"{AGENT_LABELS[agent_type]} {run_name}"
        plt.plot(group["timesteps"], group["mean_reward"], linewidth=2, label=label)
        plt.fill_between(
            group["timesteps"],
            group["mean_reward"] - group["std_reward"],
            group["mean_reward"] + group["std_reward"],
            alpha=0.12,
        )

    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.title("CarRacing PPO Evaluation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eval_reward_comparison.png", dpi=160)
    plt.close()


def save_evaluation_summary(eval_data: pd.DataFrame, output_dir: Path) -> None:
    rows = []

    for (agent_type, run_name), group in eval_data.groupby(["agent_type", "run_name"]):
        best_idx = group["mean_reward"].idxmax()
        best_row = group.loc[best_idx]
        final_row = group.iloc[-1]

        rows.append(
            {
                "agent_type": agent_type,
                "run_name": run_name,
                "num_evaluations": len(group),
                "last_timestep": int(final_row["timesteps"]),
                "best_mean_reward": best_row["mean_reward"],
                "best_std_reward": best_row["std_reward"],
                "best_timestep": int(best_row["timesteps"]),
                "final_mean_reward": final_row["mean_reward"],
                "final_std_reward": final_row["std_reward"],
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "eval_summary.csv", index=False)


def save_evaluation_plots() -> None:
    output_dir = PLOT_ROOT / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for agent_type in AGENT_LABELS:
        eval_root = LOG_ROOT / agent_type / "eval"
        for run_dir in sorted(eval_root.glob("v*")):
            try:
                dfs.append(load_evaluation_data(agent_type, run_dir.name))
            except FileNotFoundError:
                continue

    if not dfs:
        raise FileNotFoundError("No evaluation files found for comparison plots")

    eval_data = pd.concat(dfs, ignore_index=True)
    save_evaluation_curve(eval_data, output_dir)
    save_evaluation_summary(eval_data, output_dir)

    print(f"Saved evaluation comparison plots to {output_dir}")


def discover_training_runs() -> list[tuple[str, str]]:
    runs = []

    for agent_type in AGENT_LABELS:
        train_root = LOG_ROOT / agent_type / "train"
        if not train_root.exists():
            continue

        for run_dir in sorted(train_root.glob("v*")):
            monitor_files = list(run_dir.glob("*.monitor.csv"))
            has_data = any(file.stat().st_size > 80 for file in monitor_files)
            if has_data:
                runs.append((agent_type, run_dir.name))

    return runs


def main():
    for agent_type, run_name in discover_training_runs():
        save_training_plots(agent_type, run_name)

    save_evaluation_plots()


if __name__ == "__main__":
    main()
