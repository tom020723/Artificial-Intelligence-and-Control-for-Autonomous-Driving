"""Where each driver stops while waiting at a red light: distance to the stop line, or to the stopped
leader when queued. One value per test case (the last red-light wait), split by what it stopped behind.

    python scripts/plot_red_wait_clearance.py --out results/figures/red_wait_clearance \
        --method IDM=results/final/IDM/evaluation_diverse_start_lanes \
        --method BC=results/final/BC/evaluation_diverse_start_lanes \
        --method "BC → PPO"=results/final/PPO/eval_runs/iter060/evaluation_diverse_start_lanes
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"BC": "#257de0", "BC → PPO": "#f47230", "IDM": "#00b894", "BC → PPO (|e| bonus)": "#6a4ce2", "BC → PPO (Gaussian bonus)": "#f47230"}   # slots 1-3 as in the report, 7 (violet) for the new run
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
REFERENCES = (("stop_line", "Stopped at the stop line", "Distance to the stop line (m)", (0, 3)),
              ("queued_vehicle", "Stopped behind a queued vehicle", "Gap to the stopped leader (m)", (0, 4)))


def clearances(evaluation):
    out = {ref: [] for ref, *_ in REFERENCES}
    for row in csv.DictReader((Path(evaluation) / "route_evaluation.csv").open()):
        ref, value = row.get("red_wait_reference"), row.get("red_wait_clearance_m")
        if ref in out and value not in (None, ""):
            out[ref].append(float(value))
    return {k: np.array(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, help="LABEL=evaluation folder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = [m.split("=", 1) for m in args.method]
    data = {label: clearances(path) for label, path in methods}
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), gridspec_kw=dict(height_ratios=(2.2, 1)))
    rows = []
    for col, (ref, title, xlabel, xlim) in enumerate(REFERENCES):
        hist, box = axes[0, col], axes[1, col]
        bins = np.linspace(*xlim, 41)
        for label, _ in methods:
            v = data[label][ref]
            med = np.median(v); q1, q3 = np.percentile(v, [25, 75])
            rows.append((label, ref, len(v), med, q1, q3, np.mean(v)))
            inside = v[v <= xlim[1]]
            hist.hist(inside, bins=bins, histtype="step", linewidth=1.8, color=COLORS[label],
                      weights=np.full(len(inside), 100 / len(v)),
                      label=f"{label}: median {med:.2f} m\nn = {len(v):,}, {100 - 100 * len(inside) / len(v):.1f}% beyond {xlim[1]} m")
            hist.axvline(med, color=COLORS[label], linewidth=1, linestyle=":")
        hist.set(title=title, ylabel=f"Share of stops (%) per {(xlim[1] - xlim[0]) / 40:.3g} m", xlim=xlim)
        right = max(np.median(data[l][ref]) for l, _ in methods)   # legend sits right of every median line
        hist.legend(frameon=False, fontsize=8, loc="upper left", labelspacing=.9,
                    bbox_to_anchor=((right - xlim[0]) / (xlim[1] - xlim[0]) + .04, .97))
        hist.grid(axis="y", color=GRID)
        labels = [label for label, _ in methods]
        parts = box.boxplot([data[l][ref] for l in labels], vert=False, widths=.55,
                            showfliers=False, patch_artist=True, medianprops=dict(color=INK, linewidth=1.5))
        for patch, l in zip(parts["boxes"], labels):
            patch.set(facecolor=COLORS[l], alpha=.35, edgecolor=COLORS[l])
        box.set_yticks(range(1, len(labels) + 1), labels)
        box.invert_yaxis()
        box.set(xlabel=xlabel + "  (box: 25–75%, whiskers: 1.5 IQR)", xlim=xlim)
        box.grid(axis="x", color=GRID)
    fig.suptitle("Where the car stops while waiting at a red light (diverse start-lane test, last red-light wait per case)",
                 fontsize=11, x=.01, ha="left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    with args.out.with_suffix(".csv").open("w", newline="") as file:
        csv.writer(file).writerows([("method", "reference", "n", "median_m", "q1_m", "q3_m", "mean_m"),
                                    *[(a, b, n, *(round(x, 3) for x in rest)) for a, b, n, *rest in rows]])
    for r in rows:
        print(f"{r[0]:9s} {r[1]:15s} n={r[2]:5d} median {r[3]:.2f}  IQR {r[4]:.2f}-{r[5]:.2f}  mean {r[6]:.2f}")


if __name__ == "__main__":
    main()
