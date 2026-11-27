"""Event rewards shared by BC-to-PPO fine-tuning and IDM calibration."""

from math import exp

STEP_REWARD = -0.0001
TERMINAL_REWARDS = {
    "arrival": 1.0,
    "wrong_lane": -1.0,
    "timeout": -1.0,
    "collision": -2.0,
    "red_light": -2.0,
}
# Speeding is penalized only above the lane limit plus this relative margin. It
# matches the IDM expert's speed spread (speed_dev 0.1), so the expert's own
# speeds are not penalized.
OVERSPEED_TOLERANCE = 0.1
STOP_AMPLITUDE = 0.01
STOP_SIGMA_M = 1.0


def overspeed_penalty(speed_mps, limit_mps, coef, tolerance=OVERSPEED_TOLERANCE):
    """Per-step penalty proportional to the relative excess over a lane speed limit.

    Returns ``coef * max(0, v - limit * (1 + tolerance)) / limit``: zero at or
    below the (tolerated) limit and linear above it. This is a reward term,
    not a cap; the ego's commanded speed is never altered.
    """
    if not (coef >= 0 and tolerance >= 0 and limit_mps > 0):
        raise ValueError("Use a nonnegative coefficient and tolerance and a positive limit")
    return coef * max(0.0, speed_mps - limit_mps * (1.0 + tolerance)) / limit_mps


def stop_reward(error_m, amplitude=STOP_AMPLITUDE, sigma_m=STOP_SIGMA_M):
    """Bonus for a red-light stop that ends ``error_m`` metres from its target.

    The Laplace shape ``exp(-|e| / sigma)`` is peaked at the target, so the last tens of
    centimetres still change the bonus (a Gaussian is flat there: 97% at 0.25 m). With the
    same amplitude it put 68% instead of 50% of stop-line stops within 0.25 m of the target
    (seed 0, diverse start-lane test).
    """
    return amplitude * exp(-abs(error_m) / sigma_m)


class StopRewardProbe:
    """Record the first qualified stop per visit to a signal approach."""

    def __init__(self):
        self.paid = set()
        self.events = []

    def observe(self, ego, info):
        """Return the bonus for this step and record any qualifying stop."""
        if not info["signal_red_visible"] or info["lane_change"]:
            return 0.0
        target = ego.get_qualified_red_stop(lane_change_pending=False)
        if target is None:
            return 0.0
        vehicle = ego._api.vehicle
        key = (vehicle.getRouteIndex(ego.ego_id), target["tls_id"])
        if key in self.paid:
            return 0.0
        self.paid.add(key)
        position = vehicle.getLanePosition(ego.ego_id)
        error = position - target["lane_position"]
        self.events.append(dict(step=ego.step_count, reference=target["reference"],
                                error_m=error, reward=stop_reward(error)))
        return self.events[-1]["reward"]

    def result(self, status, steps):
        terminal = TERMINAL_REWARDS[status]
        time_cost = STEP_REWARD * steps
        stop_bonus = sum(event["reward"] for event in self.events)
        return dict(probe_terminal_reward=terminal, probe_step_penalty=time_cost,
                    probe_stop_reward=stop_bonus, probe_total_reward=terminal + time_cost + stop_bonus,
                    probe_stops=self.events)
