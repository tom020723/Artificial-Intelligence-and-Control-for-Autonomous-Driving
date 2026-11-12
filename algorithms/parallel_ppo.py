"""Collect synchronous rollouts from parallel SUMO environments."""

from collections import deque

import numpy as np
import torch

from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer


class ParallelPPO(PPO):
    @torch.no_grad()
    def _act_batch(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if hasattr(self.policy, "sample"):
            actions, log_probs, values = self.policy.sample(obs)
        else:
            mu, std, values = self.policy(obs)
            distribution = torch.distributions.Normal(mu, std)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions).sum(-1)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def _values(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.policy(obs)[-1].cpu().numpy()

    def _collect(self, env, obs, ep_returns, ep_lengths, logger):
        buffers = [RolloutBuffer() for _ in range(env.num_envs)]
        finished = []
        for _ in range(self.n_steps // env.num_envs):
            actions, log_probs, values = self._act_batch(obs)
            transitions = env.step(actions)
            timeout_indices = [i for i, step in enumerate(transitions) if step[4] is not None]
            timeout_values = {}
            if timeout_indices:
                timeout_obs = np.stack([transitions[i][4] for i in timeout_indices])
                timeout_values = dict(zip(timeout_indices, self._values(timeout_obs)))

            for i, (next_obs, reward, terminated, truncated, _) in enumerate(transitions):
                self.num_timesteps += 1
                ep_returns[i] += reward
                ep_lengths[i] += 1
                done = terminated or truncated
                stored_reward = reward + self.gamma * timeout_values.get(i, 0.0)
                buffers[i].add(obs[i], actions[i], log_probs[i], stored_reward, values[i], done)
                if done:
                    finished.append(ep_returns[i])
                    if logger is not None:
                        logger.log_episode(self.num_timesteps, ep_returns[i], ep_lengths[i])
                    ep_returns[i], ep_lengths[i] = 0.0, 0
            obs = np.stack([step[0] for step in transitions])

        # Finish each trajectory before flattening the samples for PPO.update().
        self.buffer.clear()
        advantages, returns = [], []
        for buffer, last_value in zip(buffers, self._values(obs), strict=True):
            adv, ret = buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
            advantages.append(adv)
            returns.append(ret)
            for field in ("obs", "actions", "log_probs", "rewards", "values", "dones"):
                getattr(self.buffer, field).extend(getattr(buffer, field))
        return obs, np.concatenate(advantages), np.concatenate(returns), finished

    def learn_parallel(self, env, total_timesteps, logger=None, log_interval=1,
                       evaluate=None, eval_interval=0,
                       checkpoint=None, checkpoint_interval=0):
        if self.n_steps < env.num_envs or self.n_steps % env.num_envs:
            raise ValueError("n_steps must be a positive multiple of num_envs")
        if eval_interval > 0 and evaluate is None:
            raise ValueError("Periodic evaluation requires an evaluator")

        obs = env.reset()
        ep_returns, ep_lengths = [0.0] * env.num_envs, [0] * env.num_envs
        recent_returns = deque(maxlen=20)
        iteration = 0
        while self.num_timesteps < total_timesteps:
            obs, advantages, returns, finished = self._collect(
                env, obs, ep_returns, ep_lengths, logger)
            recent_returns.extend(finished)
            stats = self.update(advantages, returns)
            iteration += 1
            if logger is not None:
                metrics = {
                    "iteration": iteration,
                    "ep_rew_mean": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                    "std": self.policy.current_std,
                    **stats,
                }
                logger.log_train(self.num_timesteps, metrics, verbose=iteration % log_interval == 0)
            if checkpoint is not None and checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                checkpoint(iteration)
            if eval_interval > 0 and iteration % eval_interval == 0:
                metrics = evaluate()
                if logger is not None:
                    logger.log_eval(self.num_timesteps, metrics.as_dict(), metrics.summary(), env.num_envs)
