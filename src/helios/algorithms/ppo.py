"""Proximal Policy Optimisation (PPO) with Generalised Advantage Estimation.

Supports both discrete and continuous action spaces via the distributions in
``helios.core.distributions``.

Reference: Schulman et al. (2017) - https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from helios.algorithms.base import BaseAgent
from helios.core.distributions import Gaussian, TanhNormal
from helios.core.networks import MLP
from helios.memory.rollout import RolloutBatch, compute_gae


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------


class ActorCriticNetwork(nn.Module):
    """Shared-trunk actor-critic architecture.

    Args:
        action_dim: Dimensionality of the action space.
        hidden_dims: Hidden layer widths for the shared trunk and heads.
        continuous: True for continuous actions (outputs mean/log_std),
                    False for discrete (outputs logits).
        activation: Name of the activation function.
    """

    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    continuous: bool = True
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs: jax.Array) -> dict[str, jax.Array]:
        trunk = MLP(hidden_dims=self.hidden_dims, activation=self.activation)
        features = trunk(obs)

        # Critic head
        value = nn.Dense(1)(features).squeeze(-1)  # (batch,)

        # Actor head
        if self.continuous:
            action_mean = nn.Dense(self.action_dim)(features)
            log_std = self.param(
                "log_std",
                nn.initializers.zeros,
                (self.action_dim,),
            )
            log_std = jnp.broadcast_to(log_std, action_mean.shape)
            return {"value": value, "action_mean": action_mean, "log_std": log_std}
        else:
            logits = nn.Dense(self.action_dim)(features)
            return {"value": value, "logits": logits}


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------


class PPOAgent(BaseAgent):
    """Standard PPO agent with GAE and optional learning-rate annealing.

    Args:
        config: Hydra DictConfig with PPO hyperparameters.
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.
    """

    def initial_state(self, key: jax.Array) -> dict[str, Any]:
        """Initialise network parameters and Adam optimizer.

        Args:
            key: JAX PRNG key.

        Returns:
            Dict with ``train_state`` (Flax TrainState) and ``step`` counter.
        """
        import gymnasium as gym

        continuous = isinstance(self.action_space, gym.spaces.Box)
        if continuous:
            action_dim = int(self.action_space.shape[0])
        else:
            action_dim = int(self.action_space.n)

        obs_dim = int(jnp.prod(jnp.array(self.observation_space.shape)))
        hidden_dims = tuple(self.config.hidden_dims)

        network = ActorCriticNetwork(
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            continuous=continuous,
            activation=self.config.activation,
        )

        dummy_obs = jnp.zeros((1, obs_dim))
        params = network.init(key, dummy_obs)

        total_steps = int(
            getattr(self.config, "total_timesteps", 1_000_000)
        )
        num_updates = total_steps // (
            int(self.config.num_envs) * int(self.config.num_steps)
        )

        if self.config.anneal_lr:
            lr_schedule = optax.linear_schedule(
                init_value=float(self.config.lr),
                end_value=0.0,
                transition_steps=num_updates,
            )
        else:
            lr_schedule = float(self.config.lr)

        tx = optax.chain(
            optax.clip_by_global_norm(float(self.config.max_grad_norm)),
            optax.adam(lr_schedule, eps=1e-5),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        return {
            "train_state": train_state,
            "network": network,
            "continuous": continuous,
            "step": 0,
        }

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        obs: jax.Array,
        state: dict[str, Any],
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Select actions.

        Args:
            obs: Observations, shape (num_envs, obs_dim).
            state: Agent state.
            key: PRNG key.
            deterministic: If True, return mode (no noise).

        Returns:
            Tuple of ``(actions, hidden_state)`` where ``hidden_state`` is
            just ``{}`` for a stateless agent like PPO.
        """
        ts = state["train_state"]
        outputs = ts.apply_fn(ts.params, obs)

        if state["continuous"]:
            dist = TanhNormal(outputs["action_mean"], outputs["log_std"])
            if deterministic:
                action = dist.mode()
                log_prob = dist.log_prob(action)
            else:
                action, pre_squash = dist.sample(key)
                log_prob = dist.log_prob(action, pre_squash)
        else:
            if deterministic:
                action = jnp.argmax(outputs["logits"], axis=-1)
                log_prob = jax.nn.log_softmax(outputs["logits"])[
                    jnp.arange(len(action)), action
                ]
            else:
                action = jax.random.categorical(key, outputs["logits"])
                log_prob = jax.nn.log_softmax(outputs["logits"])[
                    jnp.arange(len(action)), action
                ]

        return action, {
            "value": outputs["value"],
            "log_prob": log_prob,
        }

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        batch: RolloutBatch,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Run PPO update epochs on the collected rollout.

        Args:
            batch: :class:`~helios.memory.rollout.RolloutBatch`.
            state: Agent state.

        Returns:
            ``(new_state, metrics)``.
        """
        ts = state["train_state"]
        continuous = state["continuous"]
        key = jax.random.PRNGKey(state["step"])

        # Flatten time and env dims: (T, N, ...) -> (T*N, ...)
        def _flat(x: jax.Array) -> jax.Array:
            return x.reshape(-1, *x.shape[2:])

        obs = _flat(batch.obs)
        actions = _flat(batch.actions)
        old_log_probs = _flat(batch.log_probs)
        advantages = _flat(batch.advantages)
        returns = _flat(batch.returns)
        old_values = _flat(batch.values)

        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.shape[0]
        minibatch_size = batch_size // int(self.config.num_minibatches)

        all_metrics: list[dict[str, float]] = []

        for _epoch in range(int(self.config.update_epochs)):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]
                ts, metrics = _ppo_minibatch_update(
                    ts,
                    obs[idx],
                    actions[idx],
                    old_log_probs[idx],
                    advantages[idx],
                    returns[idx],
                    old_values[idx],
                    continuous=continuous,
                    clip_coef=float(self.config.clip_coef),
                    vf_coef=float(self.config.vf_coef),
                    ent_coef=float(self.config.ent_coef),
                    clip_vloss=bool(self.config.clip_vloss),
                )
                all_metrics.append(metrics)

        # Average metrics across all minibatches
        avg_metrics = {
            k: float(jnp.mean(jnp.array([m[k] for m in all_metrics])))
            for k in all_metrics[0]
        }

        new_state = {**state, "train_state": ts, "step": state["step"] + 1}
        return new_state, avg_metrics


# ---------------------------------------------------------------------------
# JIT-compiled minibatch update
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("continuous", "clip_vloss"))
def _ppo_minibatch_update(
    ts: TrainState,
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    old_values: jax.Array,
    *,
    continuous: bool,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    clip_vloss: bool,
) -> tuple[TrainState, dict[str, float]]:
    """Compute PPO loss and apply one gradient step."""

    def loss_fn(params):
        outputs = ts.apply_fn(params, obs)
        new_values = outputs["value"]

        if continuous:
            dist = TanhNormal(outputs["action_mean"], outputs["log_std"])
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            log_probs_all = jax.nn.log_softmax(outputs["logits"])
            new_log_probs = log_probs_all[jnp.arange(len(actions)), actions.astype(jnp.int32)]
            probs = jax.nn.softmax(outputs["logits"])
            entropy = -jnp.sum(probs * log_probs_all, axis=-1)

        # Policy loss
        ratio = jnp.exp(new_log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        # Value loss
        if clip_vloss:
            v_clipped = old_values + jnp.clip(new_values - old_values, -clip_coef, clip_coef)
            vf_loss1 = (new_values - returns) ** 2
            vf_loss2 = (v_clipped - returns) ** 2
            vf_loss = 0.5 * jnp.mean(jnp.maximum(vf_loss1, vf_loss2))
        else:
            vf_loss = 0.5 * jnp.mean((new_values - returns) ** 2)

        # Entropy bonus
        ent_loss = jnp.mean(entropy)

        total_loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

        metrics = {
            "loss/total": total_loss,
            "loss/policy": pg_loss,
            "loss/value": vf_loss,
            "loss/entropy": ent_loss,
            "policy/approx_kl": jnp.mean((ratio - 1.0) - jnp.log(ratio)),
            "policy/clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_coef),
        }
        return total_loss, metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
    ts = ts.apply_gradients(grads=grads)
    return ts, metrics
