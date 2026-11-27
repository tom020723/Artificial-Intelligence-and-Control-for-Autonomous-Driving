"""All 24 ego routes overlaid on one road map, coloured by entry arm.

    python scripts/plot_route_overlay.py results/final/suite_diverse_test --out results/figures/route_overlay
"""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sumolib

ARMS = {"A1": "#257de0", "A2": "#f47230", "A3": "#00b894", "A4": "#f2a600", "A5": "#ee7fac"}   # slots 1-5
ROAD, INK, MUTED = "#e4e4e4", "#191919", "#666666"


def offset(points, distance):
    """Shift a polyline sideways by `distance` metres (positive = left)."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return p
    d = np.gradient(p, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return p + n * distance


def route_line(net, path):
    """One continuous lane-level polyline: follow the junction connections lane by lane.

    Starts on the middle lane of the first edge; at each edge end it takes the connection to the
    next edge whose source lane is nearest the current one (a lane change when it is not the same).
    """
    first = net.getEdge(path[0])
    lane = first.getLane(first.getLaneNumber() // 2)
    points = list(lane.getShape())
    for nxt in path[1:]:
        conns = lane.getEdge().getOutgoing().get(net.getEdge(nxt), [])
        if not conns:
            lane = net.getEdge(nxt).getLane(net.getEdge(nxt).getLaneNumber() // 2)
            points += lane.getShape()
            continue
        conn = min(conns, key=lambda c: abs(c.getFromLane().getIndex() - lane.getIndex()))
        if conn.getFromLane() is not lane:
            points += conn.getFromLane().getShape()[-1:]
        if conn.getViaLaneID():
            points += net.getLane(conn.getViaLaneID()).getShape()
        lane = conn.getToLane()
        points += lane.getShape()
    clean = [points[0]]
    for pt in points[1:]:
        if np.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1]) > .5:
            clean.append(pt)
    return clean


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
    routes = sorted(lanes)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9, 8.2))
    road = [(lane.getShape(), lane.getWidth()) for edge in net.getEdges(withInternal=True) for lane in edge.getLanes()]
    xs = [x for s, _ in road for x, _ in s]; ys = [y for s, _ in road for _, y in s]
    pad = 70
    ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal"); ax.axis("off")
    fig.canvas.draw()
    points_per_m = ax.transData.transform([(1, 0)])[0][0] - ax.transData.transform([(0, 0)])[0][0]
    points_per_m *= 72 / fig.dpi
    for junction in net.getNodes():  # junction areas filled like the pavement
        shape = junction.getShape()
        if len(shape) >= 3:
            ax.fill(*zip(*shape), color=ROAD, linewidth=0, zorder=1)
    for shape, width in road:       # each lane drawn at its real width, so the road reads as pavement
        ax.plot(*zip(*shape), color=ROAD, linewidth=width * points_per_m * 1.15,
                solid_capstyle="butt", solid_joinstyle="round", zorder=1)
    by_arm = defaultdict(list)
    for route in routes:
        arm = route.split("_")[3] if route.startswith("ego_r_") else route.split("_")[2]
        by_arm[arm].append(route)
    for arm, members in by_arm.items():
        for k, route in enumerate(members):
            line = offset(route_line(net, edges[route]), (k - (len(members) - 1) / 2) * 1.6)
            ax.plot(line[:, 0], line[:, 1], color=ARMS[arm], linewidth=.9, alpha=.95,
                    solid_capstyle="round", solid_joinstyle="round", zorder=2)
            (x0, y0), (x1, y1) = line[-2], line[-1]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>,head_length=.45,head_width=.22", color=ARMS[arm], linewidth=.9),
                        zorder=3)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for arm in ARMS:
        ox, oy = max((pt for e in net.getEdges() if e.getID().startswith(f"{arm}_")
                      for lane in e.getLanes() for pt in lane.getShape()),
                     key=lambda pt: (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
        d = np.hypot(ox - cx, oy - cy)
        ax.text(ox + (ox - cx) / d * 40, oy + (oy - cy) / d * 40, arm, fontsize=12, fontweight="bold",
                color=INK, ha="center", va="center")
    handles = [plt.Line2D([], [], color=c, linewidth=3, label=f"From {a}  ({len(by_arm[a])} routes, "
                          f"{sum(len(lanes[r]) for r in by_arm[a])} route–lane cases)") for a, c in ARMS.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9,
              title="Entry arm", title_fontsize=10)
    total = sum(len(v) for v in lanes.values())
    ax.set_title(f"All {len(routes)} ego routes on the Nohyeong junction ({total} route × start-lane cases)",
                 fontsize=12, loc="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    print(f"Wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
