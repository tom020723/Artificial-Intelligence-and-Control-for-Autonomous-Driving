"""Paper-style figures comparing BC alone, BC → PPO, and PPO alone on the diverse start-lane test.

    python scripts/plot_report_figures.py --out docs/report/figures \
        --method bc=results/final/BC/evaluation_diverse_start_lanes \
        --method bc_ppo=results/final/PPO/eval_runs/iter080/evaluation_diverse_start_lanes \
        --training bc_ppo=results/final/PPO --validation bc_ppo=results/final/PPO

Methods are keyed bc, bc_ppo, ppo; each keeps its color whether or not the others are given.
Writes PDF and PNG figures and the same numbers as results_table.md / results_table.csv.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed identity per method (validated categorical slots 1-3, all-pairs).
METHODS = {
    "bc": ("BC", "#257de0"),
    "bc_ppo": ("BC → PPO", "#f47230"),
    "ppo": ("PPO only", "#00b894"),
}
FAILURES = (("collision", "Collision"), ("red_light", "Red light"),
            ("wrong_lane", "Wrong lane"), ("timeout", "Timeout"))
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def wilson(successes, n, z=1.96):
    """95% Wilson interval for a binomial proportion."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _checked(results):
    return "\njunction collisions counted" if all(r["junction"] for r in results.values()) else ""


def read_overall(directory):
    data = json.loads((Path(directory) / "evaluation_summary.json").read_text())
    overall, statuses = data["overall"], data["overall"]["statuses"]
    return dict(cases=overall["episodes"], arrival=statuses.get("arrival", 0),
                junction=bool(data.get("junction_collisions")),
                **{key: statuses.get(key, 0) for key, _ in FAILURES})


def save(fig, out, name):
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}")
    plt.close(fig)


def plot_success(results, out):
    """Point estimate and 95% Wilson interval; points need no zero baseline."""
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    keys = list(results)
    for i, key in enumerate(keys):
        r = results[key]
        rate = 100 * r["arrival"] / r["cases"]
        low, high = (100 * v for v in wilson(r["arrival"], r["cases"]))
        ax.plot([low, high], [i, i], color=INK, linewidth=1)
        ax.plot(rate, i, "o", markersize=7, color=METHODS[key][1], markeredgecolor="white", markeredgewidth=1.5)
        ax.text(high + .3, i, f"{rate:.2f}%  [{low:.1f}, {high:.1f}]", va="center", fontsize=7)
    ax.set_yticks(range(len(keys)), [METHODS[k][0] for k in keys])
    ax.set_ylim(len(keys) - .5, -.5)
    lowest = min(100 * wilson(r["arrival"], r["cases"])[0] for r in results.values())
    ax.set_xlim(5 * math.floor(lowest / 5 - .2), 100)
    ax.set_xlabel("Success rate (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Diverse start-lane test ({results[keys[0]]['cases']:,} cases, 95% CI){_checked(results)}", loc="left")
    save(fig, out, "fig_success")


def plot_failures(results, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    keys = list(results)
    height = .8 / len(keys)
    for j, key in enumerate(keys):
        r = results[key]
        y = np.arange(len(FAILURES)) + (j - (len(keys) - 1) / 2) * height
        values = [100 * r[f] / r["cases"] for f, _ in FAILURES]
        ax.barh(y, values, height=height * .9, color=METHODS[key][1], label=METHODS[key][0],
                edgecolor="white", linewidth=.8)
    ax.set_yticks(range(len(FAILURES)), [label for _, label in FAILURES])
    ax.invert_yaxis()
    ax.set_xlabel("Share of test cases (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Failure outcomes{_checked(results)}", loc="left")
    save(fig, out, "fig_failures")


def plot_training(training, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for key, run in training.items():
        rows = list(csv.DictReader((Path(run) / "training_log.csv").open()))
        steps = np.array([float(r["steps"]) for r in rows]) / 1e6
        arrival = np.array([float(r["arrival_rate"]) for r in rows]) * 100
        smooth = np.convolve(arrival, np.ones(5) / 5, mode="valid")
        color = METHODS[key][1]
        ax.plot(steps, arrival, color=color, alpha=.25, linewidth=.8)
        ax.plot(steps[4:], smooth, color=color, linewidth=1.6, label=METHODS[key][0])
        if len(training) > 1:
            ax.text(steps[-1], smooth[-1], f" {METHODS[key][0]}", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Arrival rate in rollouts (%)")
    ax.set_ylim(0, 100)
    if len(training) > 1:
        ax.legend(frameon=False, loc="lower right")
    ax.set_title("PPO training (stochastic policy; 5-iteration mean, raw faint)", loc="left")
    save(fig, out, "fig_training")


def plot_validation(validation, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    suffix = ""
    for key, run in validation.items():
        points = []
        for directory in sorted((Path(run) / "eval_runs").glob("iter*")):
            summary = directory / "evaluation_validation" / "evaluation_summary.json"
            if summary.is_file():
                data = json.loads(summary.read_text())
                overall = data["overall"]
                suffix = "\njunction collisions counted" if data.get("junction_collisions") else ""
                points.append((int(directory.name[4:]), 100 * overall["success_rate"]))
        if points:
            x, y = zip(*points)
            ax.plot(x, y, color=METHODS[key][1], linewidth=1.6, marker="o", markersize=4,
                    markeredgecolor="white", label=METHODS[key][0])
            best = max(points, key=lambda p: (p[1], p[0]))   # same rule as scripts/ppo_checkpoints.py
            ax.plot(*best, "o", markersize=10, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1)
            ax.annotate(f"selected: iter {best[0]} ({best[1]:.2f}%)", best, xytext=(0, 9),
                        textcoords="offset points", fontsize=7, ha="center")
            ax.margins(y=.18)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Validation success (%)")
    if len(validation) > 1:
        ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Checkpoint selection (2,820 validation cases){suffix}", loc="left")
    save(fig, out, "fig_validation")


def write_table(results, out):
    header = ["method", "cases", "success_%", "ci95_low_%", "ci95_high_%",
              *[key for key, _ in FAILURES]]
    rows = []
    for key, r in results.items():
        low, high = wilson(r["arrival"], r["cases"])
        rows.append([METHODS[key][0], r["cases"], f"{100 * r['arrival'] / r['cases']:.2f}",
                     f"{100 * low:.2f}", f"{100 * high:.2f}", *[r[f] for f, _ in FAILURES]])
    with (out / "results_table.csv").open("w", newline="") as file:
        csv.writer(file).writerows([header, *rows])
    lines = ["| Method | Cases | Success (%) | 95% CI | Collision | Red light | Wrong lane | Timeout |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]:,} | {r[2]} | {r[3]}–{r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} |" for r in rows]
    (out / "results_table.md").write_text("\n".join(lines) + "\n")


def pairs(values, parser, flag):
    result = {}
    for item in values or []:
        key, _, path = item.partition("=")
        if key not in METHODS or not path:
            parser.error(f"{flag} expects one of {', '.join(METHODS)}=PATH, got {item!r}")
        result[key] = Path(path)
    return {key: result[key] for key in METHODS if key in result}  # fixed method order


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", action="append", help="KEY=evaluation directory (test results)")
    parser.add_argument("--training", action="append", help="KEY=PPO run directory with training_log.csv")
    parser.add_argument("--validation", action="append", help="KEY=PPO run directory with eval_runs/iterNNN")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = pairs(args.method, parser, "--method")
    if not methods:
        parser.error("Give at least one --method")
    args.out.mkdir(parents=True, exist_ok=True)
    results = {key: read_overall(path) for key, path in methods.items()}
    if len({r["cases"] for r in results.values()}) != 1:
        parser.error("All methods must be evaluated on the same test cases")
    plot_success(results, args.out)
    plot_failures(results, args.out)
    write_table(results, args.out)
    training = pairs(args.training, parser, "--training")
    if training:
        plot_training(training, args.out)
    validation = pairs(args.validation, parser, "--validation")
    if validation:
        plot_validation(validation, args.out)
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
