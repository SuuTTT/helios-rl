"""TD-MPC2: Temporal Difference Learning for Model Predictive Control.

TD-MPC2 learns a latent world model jointly with a Q-function and uses
MPPI planning at inference time.  Key innovations over the original TD-MPC:
- Siamese latent consistency loss (predicting future latents without decoder).
- No target-encoder exponential moving average for the encoder (only for Q).
- Scale-invariant reward normalisation via symlog.

Reference:
    Hansen et al. (2023) - "TD-MPC2: Scalable, Robust World Models for
    Continuous Control" - https://arxiv.org/abs/2310.16828
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from helios.algorithms.base import BaseAgent
from helios.core.networks import MLP, NormedLinear
from helios.memory.trajectory import SequenceBatch
from helios.planners.mppi import mppi_plan_jit


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class LatentEncoder(nn.Module):
    """Encodes raw observations into a latent vector."""

    latent_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.latent_dim,
            activation=self.activation,
        )(obs)


class LatentDynamics(nn.Module):
    """Predicts next latent state from current state and action."""

    latent_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, z: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([z, action], axis=-1)
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.latent_dim,
            activation=self.activation,
        )(x)



class Pi(nn.Module):
    """Policy prior network."""
    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = z
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = getattr(nn, self.activation)(x)
            x = nn.LayerNorm()(x)
        
        # output mean and log_std
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -10, 2)
        return mean, log_std

class RewardModel(nn.Module):
    """Predicts reward from (z, action) pair."""

    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, z: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([z, action], axis=-1)
        x = MLP(hidden_dims=self.hidden_dims, output_dim=1, activation=self.activation)(x)
        return x.squeeze(-1)


class QFunction(nn.Module):
    """Ensemble of two Q-networks for double-Q learning."""

    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, z: jax.Array, action: jax.Array) -> jax.Array:
        """Returns Q-values from both ensemble members, shape (2,) per sample."""
        x = jnp.concatenate([z, action], axis=-1)
        q1 = MLP(hidden_dims=self.hidden_dims, output_dim=1, activation=self.activation)(x).squeeze(-1)
        q2 = MLP(hidden_dims=self.hidden_dims, output_dim=1, activation=self.activation)(x).squeeze(-1)
        return jnp.stack([q1, q2], axis=-1)  # (..., 2)


# ---------------------------------------------------------------------------
# symlog / symexp utilities (TD-MPC2 reward normalisation)
# ---------------------------------------------------------------------------


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric log: sign(x) * log(|x| + 1)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


# ---------------------------------------------------------------------------
# TD-MPC2 Agent
# ---------------------------------------------------------------------------


class TDMPCAgent(BaseAgent):
    """TD-MPC2 planning-based latent model agent.

    Args:
        config: Hydra DictConfig with tdmpc2 hyperparameters.
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.
    """

    def initial_state(self, key: jax.Array) -> dict[str, Any]:
        """Initialise all networks, optimizers, and target parameters."""
        import gymnasium as gym

        cfg = self.config
        obs_shape = self.observation_space.shape
        obs_dim = int(jnp.prod(jnp.array(obs_shape)))
        if isinstance(self.action_space, gym.spaces.Box):
            action_dim = int(self.action_space.shape[0])
        else:
            action_dim = int(self.action_space.n)

        latent_dim = int(cfg.latent_dim)
        hidden_dims = tuple(cfg.mlp_dims)

        encoder = LatentEncoder(latent_dim=latent_dim, hidden_dims=hidden_dims)
        dynamics = LatentDynamics(latent_dim=latent_dim, hidden_dims=hidden_dims)
        reward_model = RewardModel(hidden_dims=hidden_dims)
        q_fn = QFunction(hidden_dims=hidden_dims)
        pi_net = Pi(action_dim=action_dim, hidden_dims=hidden_dims)

        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_z = jnp.zeros((1, latent_dim))
        dummy_act = jnp.zeros((1, action_dim))

        enc_params = encoder.init(k1, dummy_obs)
        dyn_params = dynamics.init(k2, dummy_z, dummy_act)
        rew_params = reward_model.init(k3, dummy_z, dummy_act)
        q_params = q_fn.init(k4, dummy_z, dummy_act)
        target_q_params = q_params  # initialise equal
        pi_params = pi_net.init(k4, dummy_z)

        # Single optimizer for all world model components
        pi_net = Pi(action_dim=action_dim, hidden_dims=hidden_dims)
        pi_params = pi_net.init(k4, dummy_z)
        
        all_params = {
            "encoder": enc_params,
            "dynamics": dyn_params,
            "reward": rew_params,
            "q": q_params,
            "pi": pi_params,
        }


        tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.lr)),
        )
        opt_state = tx.init(all_params)

        # MPPI action-sequence mean (warm-started across steps)
        mppi_mu = jnp.zeros((int(cfg.mppi.horizon), action_dim))

        return {
            # Modules
            "encoder": encoder,
            "dynamics": dynamics,
            "reward_model": reward_model,
            "q_fn": q_fn,
            "pi": pi_net,
            # Params
            "params": all_params,
            "target_q_params": target_q_params,
            # Optimizer
            "tx": tx,
            "opt_state": opt_state,
            # Planning state
            "mppi_mu": mppi_mu,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
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
        """Encode observation and run MPPI planning.

        Args:
            obs: Shape (num_envs, obs_dim).
            state: Agent state.
            key: PRNG key.
            deterministic: If True, use the argmax / mean action.

        Returns:
            ``(action, new_state)``.
        """
        cfg = self.config
        enc = state["encoder"]
        dyn = state["dynamics"]
        rew_mod = state["reward_model"]
        q_fn = state["q_fn"]
        params = state["params"]

        z = enc.apply(params["encoder"], obs)  # (N_envs, latent_dim)

        # For simplicity, plan for the first environment only
        z0 = z[0:1]  # (1, latent_dim)

        def _imagine(s, a, k, p):
            return {"z": dyn.apply(p["dynamics"], s["z"], a)}, {}

        def _reward(s, a):
            return rew_mod.apply(params["reward"], s["z"], a)

        def _value(s, a):
            q_vals = q_fn.apply(state["target_q_params"], s["z"], a)
            return jnp.min(q_vals, axis=-1)

        horizon = int(cfg.mppi.horizon)
        num_samples = int(cfg.mppi.num_samples)
        action_dim = state["action_dim"]

        best_action_seq = state["mppi_mu"]
        best_return = jnp.array(-jnp.inf)

        for _ in range(int(cfg.mppi.num_iterations)):
            key, sub_key = jax.random.split(key)
            # Broadcast state for vmap
            batched_state = {"z": jnp.broadcast_to(z0, (num_samples,) + z0.shape[1:])}
            new_mu, ret, best_seq = mppi_plan_jit(
                sub_key,
                batched_state,
                _imagine,
                _reward,
                _value,
                best_action_seq,
                horizon=horizon,
                num_samples=num_samples,
                action_dim=action_dim,
                action_low=float(self.action_space.low.min()),
                action_high=float(self.action_space.high.max()),
                temperature=float(cfg.mppi.temperature),
                noise_beta=float(cfg.mppi.noise_beta),
                dynamics_params=params,
            )
            best_action_seq = new_mu
            best_return = ret

        # Take the first action; shift the plan for warm-starting
        action = best_action_seq[0:1]  # (1, action_dim)
        if obs.shape[0] > 1:
            action = jnp.broadcast_to(action, (obs.shape[0], action_dim))

        # Warm-start: shift plan one step and zero-pad
        new_mu = jnp.concatenate([best_action_seq[1:], jnp.zeros((1, action_dim))], axis=0)
        new_state = {**state, "mppi_mu": new_mu, "step": state["step"] + 1}
        return action, new_state

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        batch: SequenceBatch,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Single TD-MPC2 gradient update step.

        Args:
            batch: :class:`~helios.memory.trajectory.SequenceBatch`.
            state: Agent state.

        Returns:
            ``(new_state, metrics)``.
        """
        cfg = self.config

        params, opt_state, metrics = _tdmpc_update(
            params=state["params"],
            target_q_params=state["target_q_params"],
            opt_state=state["opt_state"],
            tx=state["tx"],
            encoder=state["encoder"],
            dynamics=state["dynamics"],
            reward_model=state["reward_model"],
            q_fn=state["q_fn"],
        pi=state["pi"],
            batch=batch,
            gamma=float(cfg.gamma),
            consistency_weight=float(cfg.consistency_loss_weight),
            reward_weight=float(cfg.reward_loss_weight),
            value_weight=float(cfg.value_loss_weight),
        )

        # Soft target update for Q-function
        new_target_q = jax.tree_util.tree_map(
            lambda t, f: (1.0 - float(cfg.tau)) * t + float(cfg.tau) * f,
            state["target_q_params"],
            params,
        )

        new_state = {
            **state,
            "params": params,
            "target_q_params": new_target_q,
            "opt_state": opt_state,
            "step": state["step"] + 1,
        }
        return new_state, metrics


# ---------------------------------------------------------------------------
# Functional update helper
# ---------------------------------------------------------------------------


def _tdmpc_update(
    params,
    target_q_params,
    opt_state,
    tx,
    encoder,
    dynamics,
    reward_model,
    q_fn,
    batch: SequenceBatch,
    gamma: float,
    consistency_weight: float,
    reward_weight: float,
    value_weight: float,
):
    """Compute all TD-MPC2 losses and apply one optimizer step."""

    def loss_fn(p):
        B, T = batch.obs.shape[:2]

        # Encode observations
        obs_flat = batch.obs.reshape(B * T, -1)
        z_all = encoder.apply(p["encoder"], obs_flat).reshape(B, T, -1)

        z0 = z_all[:, 0]  # Starting latents

        # Unroll latent dynamics for T-1 steps
        latent_preds = [z0]
        z = z0
        for t in range(T - 1):
            z = dynamics.apply(p["dynamics"], z, batch.actions[:, t])
            latent_preds.append(z)
        latent_preds = jnp.stack(latent_preds, axis=1)  # (B, T, latent_dim)

        # Consistency loss: predicted latents vs encoded targets
        consistency_loss = jnp.mean(
            jnp.sum(
                (latent_preds[:, 1:] - jax.lax.stop_gradient(z_all[:, 1:])) ** 2,
                axis=-1,
            )
        )

        # Reward loss
        z_flat = latent_preds[:, :-1].reshape(B * (T - 1), -1)
        a_flat = batch.actions[:, :-1].reshape(B * (T - 1), -1)
        pred_rew = reward_model.apply(p["reward"], z_flat, a_flat)
        target_rew = symlog(batch.rewards[:, :-1].reshape(B * (T - 1)))
        reward_loss = jnp.mean((pred_rew - target_rew) ** 2)

        # Value (TD) loss using target Q-network
        z_next_flat = jax.lax.stop_gradient(z_all[:, 1:].reshape(B * (T - 1), -1))
        # Use a zero action as the "greedy" action for value estimation
        dummy_a = jnp.zeros_like(a_flat)
        target_q = q_fn.apply(target_q_params["q"], z_next_flat, dummy_a)
        target_v = jnp.min(target_q, axis=-1)  # double-Q pessimism
        target_td = symlog(batch.rewards[:, :-1].reshape(B * (T - 1))) + gamma * target_v

        q_vals = q_fn.apply(p["q"], z_flat, a_flat)  # (B*(T-1), 2)
        q_loss = jnp.mean(
            jnp.sum((q_vals - jax.lax.stop_gradient(target_td[:, None])) ** 2, axis=-1)
        )

        total_loss = (
            consistency_weight * consistency_loss
            + reward_weight * reward_loss
            + value_weight * q_loss
        )
        metrics = {
            "tdmpc/consistency": consistency_loss,
            "tdmpc/reward": reward_loss,
            "tdmpc/q_loss": q_loss,
            "tdmpc/total": total_loss,
        }
        return total_loss, metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics
