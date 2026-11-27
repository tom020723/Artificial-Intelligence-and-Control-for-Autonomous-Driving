"""Outcome labels and per-group summaries of evaluation episodes."""

from collections import Counter
import csv

import numpy as np

# status: (label, short label, color)
OUTCOMES = {
    "arrival": ("Arrival", "OK", "#78c679"),
    "wrong_lane": ("Wrong lane", "LANE", "#fdae6b"),
    "red_light": ("Red light", "RED", "#ef3b2c"),
    "timeout": ("Timeout", "TIME", "#9e9ac8"),
    "collision": ("Collision", "CRASH", "#de6d69"),
    "unknown": ("Unknown", "?", "#cccccc"),
}

CLEARANCE_COLUMNS = (
    ("min_m", "Min"), ("q10_m", "P10"), ("q1_m", "Q1"),
    ("median_m", "Median"), ("q3_m", "Q3"), ("q90_m", "P90"),
    ("max_m", "Max"), ("mean_m", "Mean"), ("n", "n"),
)


def read_csv(path):
    """Rows of a CSV file as dicts; [] when the file does not exist."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def episode_rates(rows):
    return {
        "episodes": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        **{key: float(np.mean([row[key] for row in rows])) for key in (
            "success_rate", "collision_rate", "wrong_lane_rate", "timeout_rate",
            "red_light_violation_rate", "ep_return", "lane_changes", "mean_speed",
            "ep_length", "green_passes",
        )},
    }


def clearance_stats(rows, field="red_wait_clearance_m", reference=None):
    """Distance quantiles of `field`, optionally only for stops behind `reference`."""
    distances = [row[field] for row in rows if row[field] is not None
                 and (reference is None or row.get("red_wait_reference") == reference)]
    stats = {key: None for key, _ in CLEARANCE_COLUMNS}
    stats["n"] = len(distances)
    if distances:
        quantile_keys = [key for key, _ in CLEARANCE_COLUMNS[:7]]
        stats.update(zip(quantile_keys, np.percentile(distances, [0, 10, 25, 50, 75, 90, 100]).tolist()))
        stats["mean_m"] = float(np.mean(distances))
    return stats


def summarize(group):
    green = [row["green_restart_delay_s"] for row in group if row["green_restart_observed"]]
    restarted = [delay for delay in green if delay is not None]
    return {**episode_rates(group), "red_wait_clearance": clearance_stats(group),
            "red_wait_clearance_by_reference": {reference: clearance_stats(group, reference=reference)
                                                for reference in ("stop_line", "queued_vehicle")},
            "first_signal_stop_clearance": clearance_stats(group, "first_signal_stop_clearance_m"),
            "green_restart": {
                "observed": len(green), "no_restart": len(green) - len(restarted),
                "median_delay_s": float(np.median(restarted)) if restarted else None,
                "p90_delay_s": float(np.percentile(restarted, 90)) if restarted else None,
            }}
