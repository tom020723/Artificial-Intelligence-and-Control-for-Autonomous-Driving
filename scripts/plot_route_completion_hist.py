"""Histogram of completion rate over the 94 route x start-lane types (75 test cases each).

    python scripts/plot_route_completion_hist.py --out results/figures/route_completion_hist \
        --method BC=results/final/BC/evaluation_diverse_start_lanes \
        --method "BC → PPO"=results/final/PPO/eval_runs/iter060/evaluation_diverse_start_lanes
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#257de0", "#f47230")            # categorical slots 1-2 (BC, BC -> PPO as in the report)
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
BINS = [(0, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100), (100, 100)]


def rates(evaluation):
    groups = defaultdict(list)
    for row in csv.DictReader((Path(evaluation) / "route_evaluation.csv").open()):
        groups[(row["route"], row["start_lane"])].append(row["status"] == "arrival")
    return np.array([100 * sum(v) / len(v) for v in groups.values()]), len(next(iter(groups.values())))


def counts(values):
    return [int(np.sum(values == 100)) if lo == hi == 100 else int(np.sum((values >= lo) & (values < hi)))
            for lo, hi in BINS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, help="LABEL=evaluation folder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = [m.split("=", 1) for m in args.method]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(BINS)); width = .38
    for i, ((label, path), color) in enumerate(zip(methods, COLORS)):
        values, per_type = rates(path)
        c = counts(values)
        pos = x + (i - (len(methods) - 1) / 2) * width
        ax.bar(pos, c, width * .92, color=color, label=f"{label}  (median {np.median(values):.1f}%, min {values.min():.1f}%)",
               edgecolor="white", linewidth=.8, zorder=2)
        for px, n in zip(pos, c):
            if n:
                ax.text(px, n + .6, str(n), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, ["<70%" if lo == 0 else "100%" if lo == 100 else f"{lo}–{hi}%" for lo, hi in BINS])
    ax.set_xlabel("Completion rate of a route × start-lane type")
    ax.set_ylabel(f"Number of types (of {len(values)})")
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"Completion rate over the {len(values)} route × start-lane types "
                 f"({per_type} test cases each, junction collisions counted)", fontsize=10, loc="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    print(f"Wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
