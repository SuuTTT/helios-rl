"""Ephemeral rollout buffer for on-policy algorithms like PPO.

The buffer collects a fixed-length trajectory from multiple parallel
environments and is cleared after each update.  All data is stored as JAX
arrays to avoid expensive host–device transfers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RolloutBatch(NamedTuple):
    """A batch of experience collected from parallel environments.

    All arrays have shape ``(num_steps, num_envs, ...)``.
    """

    obs: jax.Array           # (T, N, *obs_shape)
    actions: jax.Array       # (T, N, *act_shape)
    rewards: jax.Array       # (T, N)
    dones: jax.Array         # (T, N)  – True when episode ended
    values: jax.Array        # (T, N)  – critic estimates
    log_probs: jax.Array     # (T, N)  – action log-probabilities
    advantages: jax.Array    # (T, N)  – GAE advantages (filled by compute_gae)
    returns: jax.Array       # (T, N)  – GAE returns  (filled by compute_gae)


class RolloutBuffer:
    """Fixed-capacity on-policy buffer for PPO.

    Data is stored in pre-allocated Python lists and converted to JAX arrays
    when :meth:`get` is called.  The buffer is automatically reset after
    :meth:`get`.

    Args:
        num_steps: Rollout horizon (number of env steps per update).
        num_envs: Number of parallel environments.
        obs_shape: Shape of a single observation.
        action_shape: Shape of a single action.
        gamma: Discount factor used for GAE.
        gae_lambda: GAE lambda parameter.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._obs: list[jax.Array] = []
        self._actions: list[jax.Array] = []
        self._rewards: list[jax.Array] = []
        self._dones: list[jax.Array] = []
        self._values: list[jax.Array] = []
        self._log_probs: list[jax.Array] = []
        self._ptr = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def full(self) -> bool:
        """True once ``num_steps`` transitions have been stored."""
        return self._ptr >= self.num_steps

    def add(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        value: jax.Array,
        log_prob: jax.Array,
    ) -> None:
        """Store a single step of transitions.

        Args:
            obs: Observations, shape (num_envs, *obs_shape).
            action: Actions, shape (num_envs, *action_shape).
            reward: Rewards, shape (num_envs,).
            done: Episode-end flags, shape (num_envs,).
            value: Critic value estimates, shape (num_envs,).
            log_prob: Action log-probabilities, shape (num_envs,).
        """
        if self.full:
            raise RuntimeError("RolloutBuffer is full. Call get() to consume the buffer first.")
        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(value)
        self._log_probs.append(log_prob)
        self._ptr += 1

    def get(
        self,
        last_value: jax.Array,
        last_done: jax.Array,
    ) -> RolloutBatch:
        """Compute GAE advantages, assemble the batch, and reset the buffer.

        Args:
            last_value: Critic estimate for the step *after* the rollout,
                shape ``(num_envs,)``.
            last_done: Done flag for that last step, shape ``(num_envs,)``.

        Returns:
            A :class:`RolloutBatch` with ``advantages`` and ``returns`` filled.
        """
        obs = jnp.stack(self._obs, axis=0)           # (T, N, *obs_shape)
        actions = jnp.stack(self._actions, axis=0)   # (T, N, *act_shape)
        rewards = jnp.stack(self._rewards, axis=0)   # (T, N)
        dones = jnp.stack(self._dones, axis=0)       # (T, N)
        values = jnp.stack(self._values, axis=0)     # (T, N)
        log_probs = jnp.stack(self._log_probs, axis=0)  # (T, N)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            last_done=last_done,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self._reset()

        return RolloutBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            log_probs=log_probs,
            advantages=advantages,
            returns=returns,
        )


# ---------------------------------------------------------------------------
# GAE computation (pure JAX, JIT-compilable)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    last_done: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalised Advantage Estimation (GAE).

    Args:
        rewards: Shape (T, N).
        values: Shape (T, N).
        dones: Shape (T, N).  1.0 when the episode ended.
        last_value: Shape (N,). Critic estimate after the last step.
        last_done: Shape (N,). Done flags after the last step.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple ``(advantages, returns)`` each of shape (T, N).
    """

    def _step(
        carry: jax.Array,
        x: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        gae = carry
        reward, value, next_value, done = x
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return gae, gae

    # Build next_values: shift values by one step, append last_value
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)  # (T, N)
    next_dones = jnp.concatenate([dones[1:], last_done[None]], axis=0)      # (T, N)

    # Scan backwards through time
    _, advantages = jax.lax.scan(
        _step,
        jnp.zeros_like(last_value),
        (rewards[::-1], values[::-1], next_values[::-1], next_dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns
