from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


RESULTS_DIR = Path("results/main")
COMPARISON_DIR = RESULTS_DIR / "comparison"

WINDOW = 20
FINAL_EVAL_WINDOW = 5
REWARD_THRESHOLD = 700

AGENT_LABELS = {
    "raw": "PPO Raw Pixels",
    "latent": "PPO Latent Representation",
}

AGENT_COLORS = {
    "raw": "#E57200",
    "latent": "#0067B9",
}

SEED_COLORS = {
    "raw": {
        "seed_42": "#F6B26B",
        "seed_123": "#E67E22",
        "seed_456": "#A94700",
    },
    "latent": {
        "seed_42": "#8EC9F0",
        "seed_123": "#2E86C1",
        "seed_456": "#0B3C6F",
    },
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
        data = load_monitor_file(file)
        if data.empty:
            continue
        data["source"] = file.name
        dfs.append(data)

    if not dfs:
        raise FileNotFoundError(f"No completed episodes found in {log_dir}")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("t").reset_index(drop=True)
    data["episode"] = range(len(data))

    # This is approximate because vectorized environments finish episodes at
    # different times. Evaluation timesteps remain the authoritative x-axis.
    data["timesteps"] = data["l"].cumsum()
    data["reward_ma"] = data["r"].rolling(WINDOW, min_periods=1).mean()
    data["reward_std"] = data["r"].rolling(WINDOW, min_periods=2).std()
    data["length_ma"] = data["l"].rolling(WINDOW, min_periods=1).mean()
    data["best_reward_so_far"] = data["r"].cummax()

    return data


def load_evaluation_data(agent_type: str, run_dir: Path) -> pd.DataFrame:
    """Load periodic deterministic evaluations from one run."""
    eval_path = run_dir / "eval_logs" / "evaluations.npz"

    if not eval_path.exists():
        raise FileNotFoundError(f"No evaluation file found at {eval_path}")

    with np.load(eval_path) as data:
        rewards = data["results"]
        timesteps = data["timesteps"].copy()
        episode_lengths = data["ep_lengths"].copy()

    return pd.DataFrame(
        {
            "agent_type": agent_type,
            "run_name": run_dir.name,
            "timesteps": timesteps,
            "mean_reward": rewards.mean(axis=1),
            "within_eval_std": rewards.std(axis=1),
            "mean_episode_length": episode_lengths.mean(axis=1),
        }
    )


def discover_runs() -> list[tuple[str, Path]]:
    """Find main experiment runs containing training or evaluation logs."""
    runs = []

    for agent_type in AGENT_LABELS:
        agent_dir = RESULTS_DIR / agent_type
        if not agent_dir.exists():
            continue

        for run_dir in sorted(agent_dir.glob("seed_*")):
            has_training = any((run_dir / "train_logs").glob("*.monitor.csv"))
            has_evaluation = (run_dir / "eval_logs" / "evaluations.npz").exists()
            if has_training or has_evaluation:
                runs.append((agent_type, run_dir))

    return runs


def plot_title(agent_type: str, run_name: str, metric: str) -> str:
    return f"CarRacing {AGENT_LABELS[agent_type]} {run_name} - {metric}"


def save_training_plots(agent_type: str, run_dir: Path) -> None:
    """Generate training plots and a summary for one run."""
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data(run_dir / "train_logs")
    run_name = run_dir.name

    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["r"], alpha=0.25, label="Episode reward")
    plt.plot(
        data["timesteps"],
        data["reward_ma"],
        linewidth=2,
        label=f"Moving average ({WINDOW})",
    )
    plt.xlabel("Approximate environment timesteps")
    plt.ylabel("Episode reward")
    plt.title(plot_title(agent_type, run_name, "Training Reward"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["reward_std"], linewidth=2)
    plt.xlabel("Approximate environment timesteps")
    plt.ylabel(f"Reward std, rolling window {WINDOW}")
    plt.title(plot_title(agent_type, run_name, "Reward Variability"))
    plt.tight_layout()
    plt.savefig(output_dir / "reward_variability.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["l"], alpha=0.25, label="Episode length")
    plt.plot(
        data["timesteps"],
        data["length_ma"],
        linewidth=2,
        label=f"Moving average ({WINDOW})",
    )
    plt.xlabel("Approximate environment timesteps")
    plt.ylabel("Episode length")
    plt.title(plot_title(agent_type, run_name, "Episode Length"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "episode_length.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(data["timesteps"], data["best_reward_so_far"], linewidth=2)
    plt.xlabel("Approximate environment timesteps")
    plt.ylabel("Best episode reward so far")
    plt.title(plot_title(agent_type, run_name, "Best Training Reward"))
    plt.tight_layout()
    plt.savefig(output_dir / "best_reward.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(data["r"], bins=25, edgecolor="black", alpha=0.8)
    plt.xlabel("Episode reward")
    plt.ylabel("Number of episodes")
    plt.title(plot_title(agent_type, run_name, "Reward Distribution"))
    plt.tight_layout()
    plt.savefig(output_dir / "reward_distribution.png", dpi=160)
    plt.close()

    last_window = data.tail(WINDOW)
    summary = {
        "episodes": len(data),
        "approx_completed_timesteps": int(data["timesteps"].iloc[-1]),
        "mean_reward_all": data["r"].mean(),
        "std_reward_all": data["r"].std(),
        "mean_reward_last_window": last_window["r"].mean(),
        "std_reward_last_window": last_window["r"].std(),
        "best_training_reward": data["r"].max(),
        "mean_episode_length_all": data["l"].mean(),
        "mean_episode_length_last_window": last_window["l"].mean(),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "training_summary.csv",
        index=False,
    )

    print(f"Saved training plots to {output_dir}")


def first_threshold_timestep(group: pd.DataFrame) -> float:
    reached = group[group["mean_reward"] >= REWARD_THRESHOLD]
    if reached.empty:
        return np.nan
    return float(reached.iloc[0]["timesteps"])


def evaluation_run_summary(eval_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize each seed independently."""
    rows = []

    for (agent_type, run_name), group in eval_data.groupby(
        ["agent_type", "run_name"],
        sort=True,
    ):
        group = group.sort_values("timesteps")
        best_row = group.loc[group["mean_reward"].idxmax()]
        final_window = group.tail(FINAL_EVAL_WINDOW)
        first_timestep = float(group["timesteps"].iloc[0])
        last_timestep = float(group["timesteps"].iloc[-1])
        observed_interval = last_timestep - first_timestep
        if observed_interval > 0:
            if hasattr(np, "trapezoid"):
                auc = np.trapezoid(
                    group["mean_reward"],
                    group["timesteps"],
                )
            else:
                auc = np.trapz(
                    group["mean_reward"],
                    group["timesteps"],
                )
            normalized_auc = auc / observed_interval
        else:
            normalized_auc = float(group["mean_reward"].iloc[0])

        rows.append(
            {
                "agent_type": agent_type,
                "run_name": run_name,
                "num_evaluations": len(group),
                "last_timestep": int(last_timestep),
                "best_mean_reward": best_row["mean_reward"],
                "best_timestep": int(best_row["timesteps"]),
                "final_window_mean_reward": final_window["mean_reward"].mean(),
                "final_window_std_reward": final_window["mean_reward"].std(),
                "normalized_auc": normalized_auc,
                f"first_timestep_reward_{REWARD_THRESHOLD}": first_threshold_timestep(group),
            }
        )

    return pd.DataFrame(rows)


def evaluation_method_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scalar run metrics across seeds for each method."""
    metrics = [
        "best_mean_reward",
        "best_timestep",
        "final_window_mean_reward",
        "normalized_auc",
        f"first_timestep_reward_{REWARD_THRESHOLD}",
    ]
    rows = []

    for agent_type, group in run_summary.groupby("agent_type", sort=True):
        row = {
            "agent_type": agent_type,
            "num_seeds": group["run_name"].nunique(),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_evaluations(eval_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evaluation curves across seeds at common timesteps."""
    frames = []

    for agent_type, agent_data in eval_data.groupby("agent_type"):
        seed_counts = agent_data.groupby("timesteps")["run_name"].nunique()
        total_seeds = agent_data["run_name"].nunique()
        common_timesteps = seed_counts[seed_counts == total_seeds].index
        common_data = agent_data[agent_data["timesteps"].isin(common_timesteps)]

        aggregated = (
            common_data.groupby("timesteps")["mean_reward"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_reward",
                    "std": "between_seed_std",
                    "count": "num_seeds",
                }
            )
        )
        aggregated["between_seed_std"] = aggregated["between_seed_std"].fillna(0.0)
        aggregated["agent_type"] = agent_type
        frames.append(aggregated)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def save_individual_evaluation_curve(
    eval_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot seeds in separate method panels using consistent method colors."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

    for axis, agent_type in zip(axes, ("raw", "latent")):
        agent_data = eval_data[eval_data["agent_type"] == agent_type]

        for run_name, group in agent_data.groupby("run_name", sort=True):
            seed = run_name.removeprefix("seed_")
            axis.plot(
                group["timesteps"],
                group["mean_reward"],
                color=SEED_COLORS[agent_type].get(
                    run_name,
                    AGENT_COLORS[agent_type],
                ),
                linestyle="-",
                linewidth=1.9,
                alpha=0.95,
                label=f"Seed {seed}",
            )

        axis.axhline(
            REWARD_THRESHOLD,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        axis.set_title(AGENT_LABELS[agent_type])
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

    axes[0].set_ylabel("Mean evaluation reward")
    fig.suptitle("CarRacing PPO Evaluation by Method and Seed")
    fig.tight_layout()
    fig.savefig(output_dir / "eval_reward_by_seed.png", dpi=160)
    plt.close(fig)


def save_paired_seed_evaluation_curve(
    eval_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compare raw and latent agents directly within each matched seed."""
    run_names = sorted(
        eval_data["run_name"].unique(),
        key=lambda name: int(name.removeprefix("seed_")),
    )
    fig, axes = plt.subplots(
        1,
        len(run_names),
        figsize=(15, 4.8),
        sharex=True,
        sharey=True,
    )

    if len(run_names) == 1:
        axes = [axes]

    for axis, run_name in zip(axes, run_names):
        run_data = eval_data[eval_data["run_name"] == run_name]

        for agent_type, group in run_data.groupby("agent_type"):
            axis.plot(
                group["timesteps"],
                group["mean_reward"],
                color=AGENT_COLORS[agent_type],
                linewidth=1.8,
                alpha=0.9,
                label=AGENT_LABELS[agent_type],
            )

        axis.axhline(
            REWARD_THRESHOLD,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        axis.set_title(f"Seed {run_name.removeprefix('seed_')}")
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Mean evaluation reward")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Matched-Seed Comparison: Raw Pixels vs Latent Representation")
    fig.tight_layout()
    fig.savefig(output_dir / "eval_reward_paired_seeds.png", dpi=160)
    plt.close(fig)


def save_aggregate_evaluation_curve(
    aggregate_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot the mean curve and between-seed standard deviation."""
    plt.figure(figsize=(10, 5))

    for agent_type, group in aggregate_data.groupby("agent_type"):
        x = group["timesteps"].to_numpy()
        mean = group["mean_reward"].to_numpy()
        std = group["between_seed_std"].to_numpy()

        plt.plot(x, mean, linewidth=2, label=AGENT_LABELS[agent_type])
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    plt.axhline(
        REWARD_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Reward threshold ({REWARD_THRESHOLD})",
    )
    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean evaluation reward across seeds")
    plt.title("CarRacing PPO Multi-Seed Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eval_reward_multiseed.png", dpi=160)
    plt.close()


def save_evaluation_dashboard(
    eval_data: pd.DataFrame,
    aggregate_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Combine aggregate and matched-seed comparisons for slides."""
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.08, 1],
        hspace=0.34,
        wspace=0.18,
    )

    aggregate_axis = fig.add_subplot(grid[0, :])
    run_names = sorted(
        eval_data["run_name"].unique(),
        key=lambda name: int(name.removeprefix("seed_")),
    )
    seed_axes = [
        fig.add_subplot(
            grid[1, index],
            sharex=None if index == 0 else None,
            sharey=None if index == 0 else None,
        )
        for index in range(len(run_names))
    ]

    for agent_type, group in aggregate_data.groupby("agent_type"):
        x = group["timesteps"].to_numpy()
        mean = group["mean_reward"].to_numpy()
        std = group["between_seed_std"].to_numpy()
        color = AGENT_COLORS[agent_type]

        aggregate_axis.plot(
            x,
            mean,
            color=color,
            linewidth=2.3,
            label=AGENT_LABELS[agent_type],
        )
        aggregate_axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.14,
        )

    aggregate_axis.axhline(
        REWARD_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label=f"Reward threshold ({REWARD_THRESHOLD})",
    )
    aggregate_axis.set_title("A. Aggregate evaluation across three seeds")
    aggregate_axis.set_ylabel("Mean evaluation reward")
    aggregate_axis.set_xlabel("Environment timesteps")
    aggregate_axis.grid(alpha=0.2)
    aggregate_axis.legend(frameon=False, ncol=3, loc="lower right")

    for axis, run_name in zip(seed_axes, run_names):
        run_data = eval_data[eval_data["run_name"] == run_name]

        for agent_type, group in run_data.groupby("agent_type"):
            axis.plot(
                group["timesteps"],
                group["mean_reward"],
                color=AGENT_COLORS[agent_type],
                linewidth=1.8,
                alpha=0.95,
                label=AGENT_LABELS[agent_type],
            )

        axis.axhline(
            REWARD_THRESHOLD,
            color="gray",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
        )
        axis.set_title(f"Seed {run_name.removeprefix('seed_')}")
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=0.2)

    seed_axes[0].set_ylabel("Mean evaluation reward")
    seed_axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    shared_y_min = min(axis.get_ylim()[0] for axis in seed_axes)
    shared_y_max = max(axis.get_ylim()[1] for axis in seed_axes)
    for axis in seed_axes:
        axis.set_ylim(shared_y_min, shared_y_max)

    fig.suptitle(
        "PPO Evaluation Results Across Three Training Seeds",
        fontsize=18,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.07, right=0.98)
    fig.savefig(
        output_dir / "eval_results_dashboard.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)


def save_evaluation_composite(output_dir: Path) -> None:
    """Arrange the three evaluation figures into one slide-ready image."""
    paths = [
        output_dir / "eval_reward_multiseed.png",
        output_dir / "eval_reward_by_seed.png",
        output_dir / "eval_reward_paired_seeds.png",
    ]
    images = [Image.open(path).convert("RGB") for path in paths]

    canvas_width = 3200
    canvas_height = 1800
    margin = 55
    gap = 35
    background = Image.new("RGB", (canvas_width, canvas_height), "white")

    def paste_contained(image, box):
        left, top, right, bottom = box
        max_width = right - left
        max_height = bottom - top
        scale = min(max_width / image.width, max_height / image.height)
        resized = image.resize(
            (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        x = left + (max_width - resized.width) // 2
        y = top + (max_height - resized.height) // 2
        background.paste(resized, (x, y))

    top_bottom = 890
    paste_contained(
        images[0],
        (margin, margin, canvas_width - margin, top_bottom),
    )

    middle = canvas_width // 2
    paste_contained(
        images[1],
        (
            margin,
            top_bottom + gap,
            middle - gap // 2,
            canvas_height - margin,
        ),
    )
    paste_contained(
        images[2],
        (
            middle + gap // 2,
            top_bottom + gap,
            canvas_width - margin,
            canvas_height - margin,
        ),
    )

    background.save(
        output_dir / "eval_results_all_views.png",
        quality=95,
    )

    for image in images:
        image.close()


def save_evaluation_outputs(eval_data: pd.DataFrame) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    run_summary = evaluation_run_summary(eval_data)
    method_summary = evaluation_method_summary(run_summary)
    aggregate_data = aggregate_evaluations(eval_data)

    run_summary.to_csv(COMPARISON_DIR / "eval_summary_by_seed.csv", index=False)
    method_summary.to_csv(
        COMPARISON_DIR / "eval_summary_by_method.csv",
        index=False,
    )
    aggregate_data.to_csv(
        COMPARISON_DIR / "eval_curve_aggregated.csv",
        index=False,
    )
    save_individual_evaluation_curve(eval_data, COMPARISON_DIR)
    save_paired_seed_evaluation_curve(eval_data, COMPARISON_DIR)
    save_aggregate_evaluation_curve(aggregate_data, COMPARISON_DIR)
    save_evaluation_dashboard(eval_data, aggregate_data, COMPARISON_DIR)
    save_evaluation_composite(COMPARISON_DIR)

    print(f"Saved evaluation comparison to {COMPARISON_DIR}")


def main():
    runs = discover_runs()
    if not runs:
        raise FileNotFoundError(f"No main experiment runs found in {RESULTS_DIR}")

    evaluation_frames = []

    for agent_type, run_dir in runs:
        try:
            save_training_plots(agent_type, run_dir)
        except FileNotFoundError as error:
            print(f"Skipping training plots for {agent_type}/{run_dir.name}: {error}")

        try:
            evaluation_frames.append(load_evaluation_data(agent_type, run_dir))
        except FileNotFoundError as error:
            print(f"Skipping evaluation for {agent_type}/{run_dir.name}: {error}")

    if evaluation_frames:
        save_evaluation_outputs(pd.concat(evaluation_frames, ignore_index=True))
    else:
        print("No evaluation data found.")


if __name__ == "__main__":
    main()
