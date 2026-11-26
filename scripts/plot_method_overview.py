"""One-figure overview of the BC -> Anchored PPO method (Algorithms 1 and 2 in docs/algorithm_bc_ppo.tex).

    python scripts/plot_method_overview.py --out results/figures

Writes method_overview.pdf and method_overview.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE = "#191919", "#555555", "#9a9a9a"
BC, PPO = "#257de0", "#f47230"            # same identities as the result figures
BC_FILL, PPO_FILL, DATA_FILL = "#eaf2fc", "#fdeee6", "#f2f2f2"
TITLE, BODY = 6.6, 5.8

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})


def box(ax, x, y, w, h, title, lines, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=0.9, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=TITLE, fontweight="bold")
    if lines:
        ax.text(x + w / 2, y + h - 4.4, "\n".join(lines), ha="center", va="top", fontsize=BODY,
                color=MUTED, linespacing=1.5)


def path(ax, points, color=LINE, dashed=False):
    """Polyline through `points` with an arrowhead at the last point."""
    style = (0, (3, 2)) if dashed else "-"
    xs, ys = zip(*points[:-1])
    if len(points) > 2:
        ax.plot(xs, ys, color=color, linewidth=1, linestyle=style, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1, color=color, linestyle=style, shrinkA=0, shrinkB=0))


def stage(ax, x, number, name, color):
    ax.text(x, 46.8, number, fontsize=8, fontweight="bold", color=color, va="center")
    ax.text(x + 2.2, 46.8, name, fontsize=7.6, fontweight="bold", va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 120); ax.set_ylim(0, 49); ax.axis("off")

    # 1. Expert data (x 0.5-26.5) -------------------------------------------------
    stage(ax, 0.5, "1", "Expert data", MUTED)
    box(ax, 0.5, 31, 26, 12, "SUMO traffic", ["5-arm junction, 24 routes", "4 car-following models"],
        LINE, DATA_FILL)
    box(ax, 0.5, 15.5, 26, 12, "IDM experts",
        [r"observed as ego: $o\in\mathbb{R}^{52}$", "acceleration + lane labels"], LINE, DATA_FILL)
    box(ax, 0.5, 1, 26, 11, "Dataset", ["480k trajectories", "98.8M state-action pairs"], LINE, DATA_FILL)
    path(ax, [(13.5, 31), (13.5, 27.5)])
    path(ax, [(13.5, 15.5), (13.5, 12)])

    # 2. Behavior cloning (x 31-56) -------------------------------------------------
    stage(ax, 31, "2", "Behavior cloning", BC)
    box(ax, 31, 19, 25, 24, "Actor network",
        [r"MLP $52\!-\!256\!-\!256$", r"$\mu_\theta=\tanh(\cdot)$, lane logits $\ell_\theta$", "",
         "MSE (acceleration)", "+ class-weighted CE (lane)", "best validation epoch"], BC, BC_FILL)
    box(ax, 31, 3.5, 25, 11.5, r"BC policy $\pi_{\mathrm{BC}}$",
        ["normalization folded", "into the first layer"], BC, "white")
    path(ax, [(26.5, 6.5), (28.7, 6.5), (28.7, 31), (31, 31)])          # dataset -> actor
    path(ax, [(43.5, 19), (43.5, 15)], color=BC)

    # 3. Anchored PPO (x 61-119) ----------------------------------------------------
    stage(ax, 61, "3", "Anchored PPO", PPO)
    box(ax, 61, 30, 26.5, 13, r"$N=512$ parallel envs",
        ["random route / lane / phase", r"$a^{\mathrm{acc}}\sim\mathcal{N}(\mu_\theta,\sigma^2)$",
         r"lane $\sim\mathrm{Cat}(\kappa\,\ell_\theta)$"],
        PPO, PPO_FILL)
    box(ax, 92.5, 30, 26.5, 13, r"Event reward $r_t$",
        [r"$+1$ arrival", r"$-1$ wrong lane, timeout", r"$-2$ collision, red light"], PPO, PPO_FILL)
    box(ax, 92.5, 11, 26.5, 13, r"Critic $V_\phi$ + GAE",
        [r"$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$", r"$\gamma=0.99,\ \lambda=0.95$"], PPO, PPO_FILL)
    box(ax, 61, 11, 26.5, 13, "Actor update",
        [r"$\mathcal{L}_{\mathrm{clip}}+\beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{BC}})$",
         r"$\beta=0.03$, KL early stop"], PPO, PPO_FILL)
    path(ax, [(87.5, 36.5), (92.5, 36.5)], color=PPO)                   # rollouts -> reward
    path(ax, [(105.75, 30), (105.75, 24)], color=PPO)                   # reward -> critic
    path(ax, [(92.5, 17.5), (87.5, 17.5)], color=PPO)                   # advantages -> actor
    path(ax, [(74.25, 24), (74.25, 30)], color=PPO)                     # actor -> next rollout
    ax.text(90, 27, r"$\times\,100$ iterations", ha="center", va="center", fontsize=BODY, color=MUTED)
    # From BC: initialization (solid) and KL anchor (dashed)
    path(ax, [(56, 14), (58.7, 14), (58.7, 36.5), (61, 36.5)], color=BC)
    ax.text(57.9, 25, r"init $\theta$", rotation=90, ha="center", va="center", fontsize=BODY, color=BC,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    path(ax, [(56, 12.0), (61, 12.0)], color=BC, dashed=True)
    ax.text(58.2, 10.4, "anchor", ha="center", va="top", fontsize=BODY - .4, color=BC)
    # Selection
    box(ax, 61, 1, 58, 6.5, "", [], LINE, "white")
    ax.text(90, 4.25, r"best validation checkpoint $\rightarrow\ \pi^\star$  $\cdot$  deterministic test",
            ha="center", va="center", fontsize=BODY)
    path(ax, [(74.25, 11), (74.25, 7.5)])

    for suffix in ("pdf", "png"):
        fig.savefig(args.out / f"method_overview.{suffix}")
    print(f"Wrote {args.out}/method_overview.pdf and .png")


if __name__ == "__main__":
    main()
