"""Plot PPO learning curves from a run's training_log.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PANELS = (
    ("Mean episode return", "Return", ("ep_rew_mean", "Rolling episode mean", 1)),
    ("Policy loss", "Loss", ("policy_loss", "Policy loss", 1)),
    ("Value loss", "Loss", ("value_loss", "Value loss", 1)),
    ("Policy entropy", "Entropy", ("entropy", "Entropy", 1)),
    ("Acceleration exploration", "Std. dev.", ("std", "Action std.", 1)),
    ("Lane change probability", "%", ("p_lane_change", "Lane change probability", 100)),
    ("KL divergence", "Approx. KL", ("approx_kl", "Mean", 1), ("max_update_kl", "Max minibatch", 1)),
    ("Clipped probability ratios", "%", ("clip_frac", "Clip fraction", 100)),
    ("Optimizer steps per iteration", "Steps", ("optimizer_steps", "Optimizer steps", 1)),
)


def plot_training(log_path, output):
    with log_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty training log: {log_path}")

    x = "iteration" if "iteration" in rows[0] else "steps"
    iterations = np.array([float(row[x]) for row in rows])
    if not np.all(np.isfinite(iterations)) or np.any(np.diff(iterations) <= 0):
        raise ValueError(f"Training log requires strictly increasing finite {x} values")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    fig.suptitle(f"PPO training  |  {log_path.parent.name}", fontsize=16)
    try:
        for ax, (title, ylabel, *series) in zip(axes.flat, PANELS, strict=True):
            for column, label, scale in series:
                values = np.array([float(row.get(column) or "nan") for row in rows]) * scale
                if np.isfinite(values).any():
                    ax.plot(iterations, values, label=label, linewidth=1.4,
                            marker="o" if len(rows) == 1 else None)
            ax.set(title=title, ylabel=ylabel, xlabel="Iteration" if x == "iteration" else "Training steps")
            ax.grid(alpha=0.25)
            if not ax.lines:
                ax.text(0.5, 0.5, "Unavailable in training log", ha="center", transform=ax.transAxes)
            elif len(ax.lines) > 1:
                ax.legend(frameon=False, fontsize=8)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to learning_curve.png in the run directory")
    args = parser.parse_args()
    output = args.output or args.run_dir / "learning_curve.png"
    plot_training(args.run_dir / "training_log.csv", output)
    print(output)


if __name__ == "__main__":
    main()
