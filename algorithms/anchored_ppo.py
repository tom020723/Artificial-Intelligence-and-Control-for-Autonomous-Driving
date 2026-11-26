"""PPO fine-tuning that starts from a BC actor and stays close to it.

Differences from ``ParallelPPO``:

* The critic is a separate network. The BC checkpoint has no value head, so
  a shared trunk would let the untrained value loss rewrite the BC features.
* The first ``critic_warmup_iterations`` update only the critic, so the first
  actor update uses advantages from a critic that has seen the BC policy.
* The actor loss adds ``anchor_coef * KL(pi || pi_BC)`` on rollout states.
  The BC policy is frozen; the term keeps states without a clear advantage
  signal at the BC behaviour.
* Episode outcomes are counted from the reward of each episode's last step.

Saved checkpoints contain only the actor in the ``HybridActorCritic`` format,
so every existing evaluator loads them unchanged.
"""

from collections import Counter, deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from algorithms.parallel_ppo import ParallelPPO
from env.bc_ppo_reward import TERMINAL_REWARDS
from utils.networks import build_mlp


class Critic(nn.Module):
    def __init__(self, mean, std, hidden_sizes=(256, 256)):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(std, 1e-3), dtype=torch.float32))
        self.net = build_mlp(len(mean), hidden_sizes, 1, "relu")

    def forward(self, obs):
        return self.net((obs - self.mean) / self.std).squeeze(-1)


OUTCOMES = {"arrival": TERMINAL_REWARDS["arrival"],
            # These pairs share a terminal reward, so the return cannot tell them apart.
            "collision_or_red_light": TERMINAL_REWARDS["collision"],
            "wrong_lane_or_timeout": TERMINAL_REWARDS["wrong_lane"]}
assert TERMINAL_REWARDS["red_light"] == TERMINAL_REWARDS["collision"]
assert TERMINAL_REWARDS["timeout"] == TERMINAL_REWARDS["wrong_lane"]


def outcome(terminal_step_reward):
    """Name the terminal event from the reward of an episode's last step.

    That step carries the terminal reward plus the step cost and at most one
    step of speeding penalty, so it stays close to its terminal value even when
    speeding penalties accumulated over the episode move the return.
    """
    name, value = min(OUTCOMES.items(), key=lambda item: abs(terminal_step_reward - item[1]))
    return name if abs(terminal_step_reward - value) < 0.5 else "other"


class TerminalRewardRecorder:
    """Wrap a vectorized env and keep each finished episode's last-step reward.

    ``ParallelPPO._collect`` appends finished episodes in the order of the
    transitions, which is the order recorded here.
    """

    def __init__(self, env):
        self.env = env
        self.terminal = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        transitions = self.env.step(actions)
        self.terminal.extend(t[1] for t in transitions if t[2] or t[3])
        return transitions

    def drain(self):
        rewards, self.terminal = self.terminal, []
        return rewards


class AnchoredPPO(ParallelPPO):
    def __init__(self, *args, normalization, critic_lr=3e-4, anchor_coef=0.1,
                 critic_warmup_iterations=5, critic_epochs=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic = Critic(normalization["mean"], normalization["std"],
                             kwargs.get("hidden_sizes", (256, 256))).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.anchor_coef = anchor_coef
        self.critic_warmup_iterations = critic_warmup_iterations
        self.critic_epochs = critic_epochs
        self.anchor = None
        self.anchor_std = None
        self.outcomes = Counter()

    def init_from_bc(self, path, init_std, lane_sharpness):
        """Load the BC actor, set the exploration std and freeze a copy as the anchor.

        ``lane_sharpness`` scales the lane logits. The argmax is unchanged, so
        the deterministic policy is still exactly the BC policy; only the
        sampled lane changes during rollouts become rarer.
        """
        self.load(path)
        with torch.no_grad():
            self.policy.log_std.fill_(float(np.log(init_std)))
            self.policy.lane_head.weight.mul_(lane_sharpness)
            self.policy.lane_head.bias.mul_(lane_sharpness)
            self.policy.v_head.weight.zero_()
            self.policy.v_head.bias.zero_()
        # The actor optimizer must not touch the unused value head.
        self.optimizer = torch.optim.Adam(
            [p for name, p in self.policy.named_parameters() if not name.startswith("v_head")],
            lr=self.optimizer.param_groups[0]["lr"])
        self.anchor = deepcopy(self.policy).eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor_std = float(init_std)

    def init_from_scratch(self, normalization, init_std, lane_keep_prob):
        """Start from a randomly initialized actor instead of BC; no anchor.

        The dataset mean and std are folded into the first layer, so the actor
        takes raw observations like the BC export (input scaling only, no expert
        behavior). The lane head starts at ``lane_keep_prob`` for keep and an
        equal share for left and right; its small random weights barely move that.
        """
        if not 0 < lane_keep_prob < 1:
            raise ValueError("lane_keep_prob must be in (0, 1)")
        mean = torch.as_tensor(normalization["mean"], dtype=torch.float32, device=self.device)
        std = torch.as_tensor(normalization["std"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            first = self.policy.trunk[0]
            first.weight.div_(std.unsqueeze(0))
            first.bias.sub_(first.weight @ mean)
            self.policy.log_std.fill_(float(np.log(init_std)))
            keep = float(np.log(2 * lane_keep_prob / (1 - lane_keep_prob)))
            self.policy.lane_head.bias.copy_(torch.tensor([0.0, keep, 0.0], device=self.device))
            self.policy.v_head.weight.zero_()
            self.policy.v_head.bias.zero_()
        self.optimizer = torch.optim.Adam(
            [p for name, p in self.policy.named_parameters() if not name.startswith("v_head")],
            lr=self.optimizer.param_groups[0]["lr"])
        self.anchor = None
        self.anchor_coef = 0.0

    @torch.no_grad()
    def _act_batch(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions, log_probs, _ = self.policy.sample(obs)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), self.critic(obs).cpu().numpy()

    @torch.no_grad()
    def _values(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.critic(obs).cpu().numpy()

    def _anchor_kl(self, obs, mu, logits):
        if self.anchor is None:                       # trained from scratch: no anchor term
            zero = mu.new_zeros(mu.shape[0])
            return zero, zero
        with torch.no_grad():
            anchor_mu, _, anchor_logits, _ = self.anchor(obs)
        accel = ((mu - anchor_mu) ** 2).sum(-1) / (2 * self.anchor_std ** 2)
        log_p = torch.log_softmax(logits, -1)
        lane = (log_p.exp() * (log_p - torch.log_softmax(anchor_logits, -1))).sum(-1)
        return accel, lane

    def update(self, advantages, returns, *, actor=True):
        obs_t, act_t, old_log_prob_t = self.buffer.to_tensors(self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        indices = np.arange(len(self.buffer))
        sums = Counter()
        n_batches = n_critic = 0
        stopped = False
        max_update_kl = 0.0

        for _ in range(self.critic_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
                value_loss = ((self.critic(obs_t[mb]) - ret_t[mb]) ** 2).mean()
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                sums["value_loss"] += value_loss.item()
                n_critic += 1

        for _ in range(self.n_epochs if actor else 0):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
                mu, std, logits, _ = self.policy(obs_t[mb])
                normal = torch.distributions.Normal(mu, std)
                lanes = Categorical(logits=logits)
                lane_idx = (1.0 - torch.round(act_t[mb, 1])).long().clamp(0, 2)
                log_prob = normal.log_prob(act_t[mb, :1]).sum(-1) + lanes.log_prob(lane_idx)
                log_ratio = log_prob - old_log_prob_t[mb]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    update_kl = (ratio - 1 - log_ratio).mean().item()
                max_update_kl = max(max_update_kl, update_kl)
                if self.target_kl is not None and update_kl > 1.5 * self.target_kl:
                    stopped = True
                    break
                surrogate = torch.min(ratio * adv_t[mb],
                                      torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb])
                policy_loss = -surrogate.mean()
                accel_kl, lane_kl = self._anchor_kl(obs_t[mb], mu, logits)
                anchor_loss = (accel_kl + lane_kl).mean()
                loss = policy_loss + self.anchor_coef * anchor_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                sums["policy_loss"] += policy_loss.item()
                sums["anchor_accel_kl"] += accel_kl.mean().item()
                sums["anchor_lane_kl"] += lane_kl.mean().item()
                sums["approx_kl"] += update_kl
                sums["clip_frac"] += ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                n_batches += 1
            if stopped:
                break

        out = {key: value / max(n_batches, 1) for key, value in sums.items() if key != "value_loss"}
        out.update(value_loss=sums["value_loss"] / max(n_critic, 1), optimizer_steps=n_batches,
                   kl_early_stop=int(stopped), max_update_kl=max_update_kl, actor_updated=int(actor))
        with torch.no_grad():
            out["p_lane_change"] = float(self.policy.lane_change_prob(obs_t).mean().item())
            # How often the deterministic lane choice differs from BC on the visited states.
            if self.anchor is not None:
                _, _, logits, _ = self.policy(obs_t)
                _, _, anchor_logits, _ = self.anchor(obs_t)
                out["lane_argmax_changed"] = float((logits.argmax(-1) != anchor_logits.argmax(-1)).float().mean())
            else:
                out["lane_argmax_changed"] = float("nan")
        return out

    def learn_parallel(self, env, iterations, logger=None, on_iteration=None):
        env = TerminalRewardRecorder(env)
        obs = env.reset()
        ep_returns, ep_lengths = [0.0] * env.num_envs, [0] * env.num_envs
        recent = deque(maxlen=2000)
        for iteration in range(1, iterations + 1):
            obs, advantages, returns, finished = self._collect(env, obs, ep_returns, ep_lengths, logger)
            terminal = env.drain()
            if len(terminal) != len(finished):
                raise RuntimeError("Terminal rewards and finished episodes are out of step")
            names = [outcome(r) for r in terminal]
            self.outcomes.update(names)
            recent.extend(names)
            stats = self.update(advantages, returns, actor=iteration > self.critic_warmup_iterations)
            window = Counter(names)
            total = max(len(names), 1)
            metrics = dict(iteration=iteration, episodes=len(names),
                           ep_rew_mean=float(np.mean(finished)) if finished else float("nan"),
                           **{f"{name}_rate": window[name] / total for name in (*OUTCOMES, "other")},
                           arrival_rate_recent=sum(n == "arrival" for n in recent) / max(len(recent), 1),
                           std=self.policy.current_std, **stats)
            if logger is not None:
                logger.log_train(self.num_timesteps, metrics, verbose=True)
            if on_iteration is not None:
                on_iteration(iteration)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def save_critic(self, path):
        torch.save(self.critic.state_dict(), path)
