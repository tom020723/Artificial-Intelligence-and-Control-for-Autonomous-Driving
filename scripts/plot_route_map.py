"""Small-multiple map of the 24 ego routes and their start lanes (94 route x start-lane cases).

    python scripts/plot_route_map.py results/final/suite_diverse_test --out results/figures/route_map
"""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sumolib

CENTRAL, RING = "#257de0", "#f47230"          # categorical slots 1-2
ROAD, INK, MUTED = "#d4d4d4", "#191919", "#666666"


def label(route):
    if route.startswith("ego_r_RING_"):
        a, b = route.split("_")[3:5]
        return f"Ring {a} → {b}"
    a, b = route.removeprefix("ego_d_").split("_")
    return f"{a} → {b}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    net = sumolib.net.readNet(str(args.suite / "sumo/highway.net.xml"), withInternal=True)
    edges = {r.get("id"): r.get("edges").split() for r in ET.parse(args.suite / "sumo/highway.ego.rou.xml").iter("route")}
    lanes = defaultdict(set)
    for case in json.loads((args.suite / "cases.json").read_text()):
        lanes[case["route"]].add(case["start_lane"])
    routes = sorted(lanes, key=lambda r: (r.startswith("ego_r_"), r))

    road = [lane.getShape() for edge in net.getEdges(withInternal=True) for lane in edge.getLanes()]
    xs = [x for s in road for x, _ in s]; ys = [y for s in road for _, y in s]
    pad = 70

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK})
    cols, rows = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.75))
    for ax, route in zip(axes.flat, routes):
        color = RING if route.startswith("ego_r_") else CENTRAL
        for shape in road:
            ax.plot(*zip(*shape), color=ROAD, linewidth=.6, solid_capstyle="round", zorder=1)
        path = edges[route]
        for i, edge_id in enumerate(path):
            edge = net.getEdge(edge_id)
            for lane in edge.getLanes():
                ax.plot(*zip(*lane.getShape()), color=color, linewidth=1.1, solid_capstyle="round", zorder=2)
            if i + 1 < len(path):   # internal junction lanes joining this edge to the next
                for conn in edge.getOutgoing().get(net.getEdge(path[i + 1]), []):
                    via = conn.getViaLaneID()
                    if via:
                        ax.plot(*zip(*net.getLane(via).getShape()), color=color, linewidth=1.1, zorder=2)
        first = net.getEdge(path[0])
        start = first.getLane(first.getLaneNumber() // 2).getShape()[0]
        ax.plot(*start, "o", markersize=8, color=color, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        last = net.getEdge(path[-1])
        shape = last.getLane(last.getLaneNumber() // 2).getShape()
        (x0, y0), (x1, y1) = shape[-2], shape[-1]
        length = max(1e-6, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** .5)
        back = min(70, length)
        ax.annotate("", xy=(x1, y1), xytext=(x1 - (x1 - x0) / length * back, y1 - (y1 - y0) / length * back),
                    arrowprops=dict(arrowstyle="-|>,head_length=.7,head_width=.4", color=INK, linewidth=1.4),
                    zorder=5)
        for arm in ("A1", "A2", "A3", "A4", "A5"):   # arm names at the outer ends
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            ox, oy = max((pt for e in net.getEdges() if e.getID().startswith(f"{arm}_")
                          for lane in e.getLanes() for pt in lane.getShape()),
                         key=lambda pt: (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)   # outer end of the arm
            d = max(1e-6, ((ox - cx) ** 2 + (oy - cy) ** 2) ** .5)
            ax.text(ox + (ox - cx) / d * 45, oy + (oy - cy) / d * 45, arm, fontsize=7, color=MUTED,
                    ha="center", va="center", zorder=1)
        n = len(lanes[route])
        ax.set_title(f"{label(route)}  ·  {n} start lanes", fontsize=9, color=INK)
        ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal"); ax.axis("off")
    for ax in axes.flat[len(routes):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=CENTRAL, linewidth=2.5, label="Through the central junction (19 routes)"),
               plt.Line2D([], [], color=RING, linewidth=2.5, label="Right-turn ring road (5 routes)"),
               plt.Line2D([], [], marker="o", linestyle="", color=MUTED, markeredgecolor="white", markersize=8,
                          label="Start (every lane of the entry road is a test case)"),
               plt.Line2D([], [], marker=r"$\rightarrow$", linestyle="", color=INK, markersize=12,
                          label="Destination")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, -.005))
    total = sum(len(v) for v in lanes.values())
    fig.suptitle(f"{len(routes)} ego routes × their start lanes = {total} route–lane combinations", fontsize=13, y=.995)
    fig.tight_layout(rect=(0, .03, 1, .975))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}.png/.pdf ({len(routes)} routes, {total} combinations)")


if __name__ == "__main__":
    main()
