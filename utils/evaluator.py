"""정책 평가 유틸 — 주행 지표 측정.

알고리즘과 분리해둔 이유: 평가는 "정책이 무엇으로 학습됐는지"와 무관하다.
`predict(obs, deterministic=True) -> action` 메서드만 있으면
PPO든 SAC든, 심지어 규칙 기반 컨트롤러든 똑같이 평가할 수 있다.

왜 별도 평가가 필요한가:
    학습 중 에피소드는 탐험 노이즈(σ)가 섞인 확률적 행동의 결과라
    "지금 정책의 실제 운전 실력"을 정확히 보여주지 못한다.
    그래서 주기적으로 노이즈 없는(deterministic) 정책으로 주행시켜
    충돌률 / 완주율 / 평균속도 / 차간거리를 따로 측정한다.
"""
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class DrivingMetrics:
    """평가 1회(n_episodes 평균)의 주행 지표."""
    collision_rate: float   # 충돌로 끝난 에피소드 비율 (0~1, 낮을수록 좋음)
    success_rate: float     # 도로 끝까지 완주한 비율 (0~1, 높을수록 좋음)
    timeout_rate: float     # 실제 시간 제한으로 끝난 비율
    wrong_lane_rate: float  # 경로에 맞지 않는 차선으로 종료된 비율
    red_light_violation_rate: float  # 신호 위반으로 종료된 비율
    mean_speed: float       # 전 스텝 평균 주행속도 (m/s)
    mean_gap: float         # 선행차 감지 스텝에서의 평균 차간거리 (m)
    min_gap: float          # 관측된 최소 차간거리 (m) — 위험 운전 지표
    ep_return: float        # 에피소드당 평균 누적 보상
    ep_length: float        # 에피소드당 평균 스텝 수
    lane_changes: float     # 에피소드당 평균 "실행된" 차선변경 횟수
                            # (추월 시나리오에서 정책이 실제로 차선을
                            #  활용하는지 보는 핵심 지표)
    green_passes: float     # 에피소드당 청신호 통과 보상을 받은 횟수
    red_stop_cost: float    # 에피소드당 적신호 정차 감점의 합

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """콘솔 한 줄 출력용 문자열."""
        return (f"충돌률={self.collision_rate:.0%}  "
                f"완주율={self.success_rate:.0%}  "
                f"신호위반={self.red_light_violation_rate:.0%}  "
                f"차선실패={self.wrong_lane_rate:.0%}  "
                f"시간초과={self.timeout_rate:.0%}  "
                f"청신호통과={self.green_passes:.1f}회  "
                f"평균속도={self.mean_speed:.1f}m/s  "
                f"평균차간={self.mean_gap:.1f}m  "
                f"최소차간={self.min_gap:.1f}m  "
                f"차선변경={self.lane_changes:.1f}회  "
                f"보상={self.ep_return:.1f}")


@dataclass
class _EpisodeStats:
    reward: float = 0.0
    length: int = 0
    lane_changes: int = 0
    green_passes: int = 0
    red_stop_cost: float = 0.0
    outcome: str | None = None
    last_stop_clearance_m: float | None = None
    last_stop_reference: str | None = None
    red_wait_clearance_m: float | None = None
    red_wait_reference: str | None = None
    start_lane: int | None = None
    signal_offset_s: float | None = None
    traffic_vph: int | None = None
    speeds: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)


class _DrivingStats:
    def __init__(self, n_episodes):
        self.episodes = [_EpisodeStats() for _ in range(n_episodes)]

    def record(self, episode, reward, truncated, info):
        stats = self.episodes[episode]
        stats.reward += reward
        stats.length += 1
        stats.lane_changes += int(info.get("lane_change", 0) != 0 and bool(info.get("lane_change_applied")))
        stats.green_passes += int(info.get("green_pass_count", info.get("green_pass_reward", 0.0) > 0))
        stats.red_stop_cost += float(info.get("red_stop_penalty", 0.0))
        if "stop_clearance_m" in info:
            stats.last_stop_clearance_m = info["stop_clearance_m"]
            stats.last_stop_reference = info["stop_reference"]
            signals = info.get("observed_signals", ())
            if (info["stop_clearance_m"] is not None and signals
                    and signals[0]["state"].lower() == "r"):
                stats.red_wait_clearance_m = info["stop_clearance_m"]
                stats.red_wait_reference = info["stop_reference"]
        if "start_lane" in info:
            stats.start_lane = info["start_lane"]
            stats.signal_offset_s = info["signal_offset_s"]
            stats.traffic_vph = info["traffic_vph"]

        for outcome in ("red_light_violation", "collided", "arrived", "wrong_lane"):
            if info.get(outcome):
                stats.outcome = outcome
                break
        else:
            if truncated:
                stats.outcome = "timeout"

        if info.get("speed") is not None:
            stats.speeds.append(float(info["speed"]))
        if info.get("gap") is not None:
            stats.gaps.append(float(info["gap"]))

    def metrics(self, visibility, episodes=None):
        episodes = self.episodes if episodes is None else episodes
        outcomes = Counter(episode.outcome for episode in episodes)
        speeds = [speed for episode in episodes for speed in episode.speeds]
        gaps = [gap for episode in episodes for gap in episode.gaps]
        return DrivingMetrics(
            collision_rate=outcomes["collided"] / len(episodes),
            success_rate=outcomes["arrived"] / len(episodes),
            timeout_rate=outcomes["timeout"] / len(episodes),
            wrong_lane_rate=outcomes["wrong_lane"] / len(episodes),
            red_light_violation_rate=outcomes["red_light_violation"] / len(episodes),
            mean_speed=float(np.mean(speeds)) if speeds else 0.0,
            mean_gap=float(np.mean(gaps)) if gaps else float(visibility),
            min_gap=float(np.min(gaps)) if gaps else float(visibility),
            ep_return=float(np.mean([episode.reward for episode in episodes])),
            ep_length=float(np.mean([episode.length for episode in episodes])),
            lane_changes=float(np.mean([episode.lane_changes for episode in episodes])),
            green_passes=float(np.mean([episode.green_passes for episode in episodes])),
            red_stop_cost=float(np.mean([episode.red_stop_cost for episode in episodes])),
        )


def evaluate_policy(policy, env, n_episodes: int = 5,
                    deterministic: bool = True,
                    seed: int | None = None) -> DrivingMetrics:
    """정책을 n_episodes 주행시키고 주행 지표 평균을 반환.

    Args:
        policy: `predict(obs, deterministic) -> action` 을 가진 객체
                (PPOAgent, 또는 같은 인터페이스의 무엇이든)
        env:    SumoHighwayEnv (mean_gap 기본값용 visibility 속성만 사용)
        n_episodes: 평가 에피소드 수. 많을수록 지표가 안정적이지만 느리다
        deterministic: True면 탐험 노이즈 없이 평균 행동만 사용
        seed: 지정하면 에피소드마다 seed, seed+1, ...로 환경과 SUMO 난수 초기화

    Note:
        모든 지표는 env.step()의 info(collided/arrived/speed/gap)에서 읽는다.
        관측(obs) 벡터의 구성과 무관하므로, 관측을 바꿔도 이 코드는 안 깨진다.
    """
    stats = _DrivingStats(n_episodes)
    for episode in range(n_episodes):
        if seed is None:
            obs, _ = env.reset()
        else:
            episode_seed = seed + episode
            # SUMO 시작 시 전역 NumPy에서 뽑는 seed도 에피소드별로 고정한다.
            np.random.seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)

        done = False
        while not done:
            action = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            stats.record(episode, reward, truncated, info)
            done = terminated or truncated
    return stats.metrics(env.visibility)


def evaluate_parallel_policy(policy, env, device):
    """Run one deterministic episode per road and summarize the batch."""
    return _run_parallel_episodes(policy, env, device).metrics(env.prototype.visibility)


def evaluate_parallel_routes(policy, env, device, routes, **reset_options):
    """Run one episode for each explicit route and return results in route order."""
    stats = _run_parallel_episodes(policy, env, device, routes, **reset_options)
    statuses = {"arrived": "arrival", "collided": "collision", "red_light_violation": "red_light"}
    return [
        {"status": statuses.get(episode.outcome, episode.outcome or "unknown"),
         "start_lane": episode.start_lane, "signal_offset_s": episode.signal_offset_s,
         "traffic_vph": episode.traffic_vph,
         "last_stop_clearance_m": episode.last_stop_clearance_m,
         "last_stop_reference": episode.last_stop_reference,
         "red_wait_clearance_m": episode.red_wait_clearance_m,
         "red_wait_reference": episode.red_wait_reference,
         **stats.metrics(env.prototype.visibility, [episode]).as_dict()}
        for episode in stats.episodes
    ]


def _run_parallel_episodes(policy, env, device, routes=None, **reset_options):
    import torch

    policy.to(device).eval()
    stats = _DrivingStats(env.num_envs)
    active = list(range(env.num_envs))
    try:
        obs = env.reset(**reset_options) if routes is None else env.reset(routes=routes, **reset_options)
        with torch.inference_mode():
            while active:
                obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = policy.det_action(obs_batch) if hasattr(policy, "det_action") else policy(obs_batch)[0]
                pending, next_observations = [], []
                for index, transition in zip(
                        active, env.step_evaluation(active, actions.cpu().numpy()), strict=True):
                    next_obs, reward, terminated, truncated, info = transition
                    stats.record(index, reward, truncated, info)
                    if not (terminated or truncated):
                        pending.append(index)
                        next_observations.append(next_obs)
                active = pending
                if active:
                    obs = np.stack(next_observations)
    finally:
        env.close()
    return stats
