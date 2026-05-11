"""Soft Actor-Critic (SAC) — custom v1 milestone implementation.

Milestone: best=653 @ 9.1M steps on MuJoCo Playground HopperStand (seed=1).

Architecture:
    Actor:       MLP(512, 512) + relu → separate mu / log_std heads
                 log_std clipped to [-5, 2], TanhNormal distribution
                 lecun_uniform init on all Dense layers
    TwinCritic:  concat(obs, action) → two independent MLP(512, 512) + relu + LayerNorm
                 lecun_uniform init, separate Dense(1) head for each Q

Key design choices vs baseline SAC:
    GPU replay buffer:   brax.training.replay_buffers.UniformSamplingQueue
                        zero CPU↔GPU transfers during training
    lax.scan updates:   k_updates gradient steps per collect cycle stay fully on GPU
    lax.scan collect:   collect_steps env steps via scan — eliminates Python overhead
    Three @jax.jit fns: update_critic / update_actor / update_alpha compiled separately
                        avoids retracing on each outer call
    Auto-tune alpha:     log_alpha starts at 0, target_entropy = -0.5 * action_size
    Obs normalisation:  Running Welford mean/var updated once per collect cycle
                        (one GPU→CPU transfer per cycle, not per step)

Reference: Haarnoja et al. (2018) - https://arxiv.org/abs/1812.05905
Source:    /workspace/helios-rl/scripts/run_sac_custom.py
"""
from __future__ import annotations

from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import lax, random


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

_lecu = jax.nn.initializers.lecun_uniform()


class MLP(nn.Module):
    """Feedforward MLP with relu activations and optional LayerNorm.

    Args:
        features:    Hidden layer widths.
        layer_norm:  If True, apply LayerNorm after each activation.
    """

    features: Tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=_lecu)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return x


class Actor(nn.Module):
    """TanhNormal actor with separate mu / log_std heads.

    Args:
        action_size:  Number of action dimensions.
        hidden:       Hidden layer widths (default: (512, 512)).
        log_std_min:  Lower clip for log_std (default: -5.0).
        log_std_max:  Upper clip for log_std (default:  2.0).

    Returns (from ``__call__``):
        mu:      Gaussian mean, shape (..., action_size).
        log_std: Clipped log standard deviation, shape (..., action_size).
    """

    action_size: int
    hidden: Tuple[int, ...] = (512, 512)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = MLP(self.hidden)(obs)
        mu      = nn.Dense(self.action_size, kernel_init=_lecu)(x)
        log_std = nn.Dense(self.action_size, kernel_init=_lecu)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class TwinCritic(nn.Module):
    """Twin Q-network: two independent MLPs computing Q1, Q2.

    Input is ``concat(obs, action)``.

    Args:
        hidden:     Hidden layer widths (default: (512, 512)).
        layer_norm: Apply LayerNorm after each activation (default: True).
    """

    hidden: Tuple[int, ...] = (512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        x  = jnp.concatenate([obs, action], axis=-1)
        q1 = MLP(self.hidden, layer_norm=self.layer_norm)(x)
        q1 = nn.Dense(1, kernel_init=_lecu)(q1)[..., 0]
        q2 = MLP(self.hidden, layer_norm=self.layer_norm)(x)
        q2 = nn.Dense(1, kernel_init=_lecu)(q2)[..., 0]
        return q1, q2


# ---------------------------------------------------------------------------
# TanhNormal distribution helpers
# ---------------------------------------------------------------------------


def tanh_normal_sample(
    mu: jax.Array, log_std: jax.Array, key: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Sample from TanhNormal(mu, exp(log_std)).

    Args:
        mu:      Gaussian mean.
        log_std: Log standard deviation (NOT softplus — direct exp).
        key:     JAX PRNG key.

    Returns:
        (action, raw_action) where action = tanh(raw_action).
    """
    std = jnp.exp(log_std)
    u   = mu + std * random.normal(key, mu.shape)
    return jnp.tanh(u), u


def tanh_normal_log_prob(
    mu: jax.Array, log_std: jax.Array, u: jax.Array
) -> jax.Array:
    """Log-prob of a TanhNormal sample.

    Uses the pre-squash value ``u`` for numerical stability.

    Args:
        mu:      Gaussian mean.
        log_std: Log standard deviation.
        u:       Pre-squash sample (stored from tanh_normal_sample).

    Returns:
        Per-sample scalar log probability, shape (batch,).
    """
    std = jnp.exp(log_std)
    log_p_u = -0.5 * jnp.sum(
        ((u - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )
    log_det = jnp.sum(jnp.log(1.0 - jnp.tanh(u) ** 2 + 1e-7), axis=-1)
    return log_p_u - log_det


# ---------------------------------------------------------------------------
# SAC update functions factory
# ---------------------------------------------------------------------------


def make_sac_fns(
    actor_apply: Callable,
    critic_apply: Callable,
    actor_opt: optax.GradientTransformation,
    critic_opt: optax.GradientTransformation,
    alpha_opt_inst: optax.GradientTransformation,
    gamma: float,
    reward_scaling: float,
    target_entropy: float,
    tau: float,
) -> Callable:
    """Build and return the ``one_step`` SAC update function.

    Compiles three @jax.jit sub-functions (critic, actor, alpha) that are
    called sequentially within ``one_step``. Keeping them separate avoids
    retracing when wrapped in ``make_scan_update``.

    Args:
        actor_apply, critic_apply:  Module apply functions (e.g. net.apply).
        actor_opt, critic_opt:      Optax optimizers for actor/critic.
        alpha_opt_inst:             Optax optimizer for log_alpha.
        gamma:                      Discount factor.
        reward_scaling:             Multiply rewards before Bellman target.
        target_entropy:             Entropy target for alpha auto-tuning.
                                    Typical: -0.5 * action_size.
        tau:                        Soft target update coefficient (e.g. 0.005).

    Returns:
        one_step(actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
                 log_alpha, alpha_opt_s, obs_mean, obs_var,
                 obs, action, reward, next_obs, done, key)
            → (new_actor_p, new_actor_opt_s, new_critic_p, new_critic_opt_s,
               new_target_p, new_log_alpha, new_alpha_opt_s)
    """

    @jax.jit
    def update_critic(
        critic_p, critic_opt_s, actor_p, target_p, log_alpha,
        obs_n, next_obs_n, actions, rewards, dones, key,
    ):
        alpha = jnp.exp(log_alpha)
        mu_n, ls_n = actor_apply(actor_p, next_obs_n)
        na, un = tanh_normal_sample(mu_n, ls_n, key)
        nlp = tanh_normal_log_prob(mu_n, ls_n, un)
        tq1, tq2 = critic_apply(target_p, next_obs_n, na)
        next_v   = jnp.minimum(tq1, tq2) - alpha * nlp
        target_q = lax.stop_gradient(
            rewards * reward_scaling + gamma * (1.0 - dones) * next_v
        )

        def loss_fn(cp):
            q1, q2 = critic_apply(cp, obs_n, actions)
            return 0.5 * (
                jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
            )

        c_grads = jax.grad(loss_fn)(critic_p)
        c_upd, new_opt_s = critic_opt.update(c_grads, critic_opt_s)
        return optax.apply_updates(critic_p, c_upd), new_opt_s

    @jax.jit
    def update_actor(actor_p, actor_opt_s, critic_p, log_alpha, obs_n, key):
        alpha = jnp.exp(log_alpha)

        def loss_fn(ap):
            mu, ls = actor_apply(ap, obs_n)
            a, u   = tanh_normal_sample(mu, ls, key)
            lp     = tanh_normal_log_prob(mu, ls, u)
            q1, q2 = critic_apply(critic_p, obs_n, a)
            return jnp.mean(alpha * lp - jnp.minimum(q1, q2))

        a_grads = jax.grad(loss_fn)(actor_p)
        a_upd, new_opt_s = actor_opt.update(a_grads, actor_opt_s)
        return optax.apply_updates(actor_p, a_upd), new_opt_s

    @jax.jit
    def update_alpha(log_alpha, alpha_opt_s, actor_p, obs_n, key):
        mu, ls = actor_apply(actor_p, obs_n)
        _, u   = tanh_normal_sample(mu, ls, key)
        lp     = lax.stop_gradient(tanh_normal_log_prob(mu, ls, u))

        def loss_fn(la):
            return jnp.mean(jnp.exp(la) * (-lp - target_entropy))

        al_grads = jax.grad(loss_fn)(log_alpha)
        al_upd, new_opt_s = alpha_opt_inst.update(al_grads, alpha_opt_s)
        return optax.apply_updates(log_alpha, al_upd), new_opt_s

    def one_step(
        actor_p, actor_opt_s,
        critic_p, critic_opt_s,
        target_p, log_alpha, alpha_opt_s,
        obs_mean, obs_var,
        obs, action, reward, next_obs, done,
        key,
    ):
        """One SAC gradient step: critic → actor → alpha → soft target update."""
        obs_n      = (obs      - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        next_obs_n = (next_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        key, k1, k2, k3 = random.split(key, 4)

        new_critic_p, new_critic_opt_s = update_critic(
            critic_p, critic_opt_s, actor_p, target_p, log_alpha,
            obs_n, next_obs_n, action, reward, done, k1,
        )
        new_actor_p, new_actor_opt_s = update_actor(
            actor_p, actor_opt_s, new_critic_p, log_alpha, obs_n, k2,
        )
        new_log_alpha, new_alpha_opt_s = update_alpha(
            log_alpha, alpha_opt_s, new_actor_p, obs_n, k3,
        )
        new_target_p = jax.tree_util.tree_map(
            lambda tp, qp: (1.0 - tau) * tp + tau * qp,
            target_p, new_critic_p,
        )
        return (
            new_actor_p, new_actor_opt_s,
            new_critic_p, new_critic_opt_s,
            new_target_p, new_log_alpha, new_alpha_opt_s,
        )

    return one_step


# ---------------------------------------------------------------------------
# Scan-based update (k_updates gradient steps, fully on GPU)
# ---------------------------------------------------------------------------


def make_scan_update(one_step_fn: Callable, buf, k_updates: int) -> Callable:
    """Build a JIT-compiled ``lax.scan`` over ``k_updates`` gradient steps.

    Each scan iteration samples a fresh batch from ``buf`` (GPU replay buffer).
    All computation stays on GPU — no CPU transfers.

    Args:
        one_step_fn:  The function returned by ``make_sac_fns``.
        buf:          brax.training.replay_buffers.UniformSamplingQueue instance.
        k_updates:    Number of gradient steps to scan over.

    Returns:
        scan_update(actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
                    log_alpha, alpha_opt_s, obs_mean, obs_var, buf_state, rng)
            → (actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
               log_alpha, alpha_opt_s, new_buf_state)
    """

    @jax.jit
    def scan_update(
        actor_p, actor_opt_s,
        critic_p, critic_opt_s,
        target_p, log_alpha, alpha_opt_s,
        obs_mean, obs_var,
        buf_state, rng,
    ):
        def body(carry, _):
            ap, ao, cp, co, tp, la, alo, bs, k = carry
            k, uk = random.split(k)
            new_bs, batch = buf.sample(bs)
            ap2, ao2, cp2, co2, tp2, la2, alo2 = one_step_fn(
                ap, ao, cp, co, tp, la, alo,
                obs_mean, obs_var,
                batch["obs"], batch["action"], batch["reward"],
                batch["next_obs"], batch["done"],
                uk,
            )
            return (ap2, ao2, cp2, co2, tp2, la2, alo2, new_bs, k), None

        carry0 = (
            actor_p, actor_opt_s,
            critic_p, critic_opt_s,
            target_p, log_alpha, alpha_opt_s,
            buf_state, rng,
        )
        (ap, ao, cp, co, tp, la, alo, new_bs, _), _ = lax.scan(
            body, carry0, None, length=k_updates
        )
        return ap, ao, cp, co, tp, la, alo, new_bs

    return scan_update


# ---------------------------------------------------------------------------
# Scan-based env collection (collect_steps steps, fully on GPU)
# ---------------------------------------------------------------------------


def make_collect_fn(
    env_step_fn: Callable, actor_apply: Callable, collect_steps: int
) -> Callable:
    """Build a JIT-compiled ``lax.scan`` over ``collect_steps`` env steps.

    Actions are sampled stochastically from the actor (TanhNormal).
    Transitions are returned flattened: shape (collect_steps * num_envs, ...).

    Args:
        env_step_fn:   JIT-compiled env.step(state, action) → EnvState.
        actor_apply:   actor_net.apply.
        collect_steps: Steps to scan over per call.

    Returns:
        collect(env_state, actor_p, obs_mean, obs_var, rng)
            → (new_env_state, new_rng,
               (flat_obs, flat_action, flat_reward, flat_next_obs, flat_done))
        Each flat array has shape (collect_steps * num_envs, ...).
    """

    @jax.jit
    def collect(env_state, actor_p, obs_mean, obs_var, rng):
        def step_body(carry, _):
            es, k = carry
            k, ak = random.split(k)
            obs_n  = (es.obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
            mu, ls = actor_apply(actor_p, obs_n)
            action, _ = tanh_normal_sample(mu, ls, ak)
            ns = env_step_fn(es, action)
            return (ns, k), (es.obs, action, ns.reward, ns.obs, ns.done)

        (new_es, new_rng), traj = lax.scan(
            step_body, (env_state, rng), None, length=collect_steps
        )
        # (collect_steps, num_envs, ...) → (collect_steps * num_envs, ...)
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), traj
        )
        return new_es, new_rng, flat

    return collect


# ---------------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------------


def evaluate(
    env_reset_fn: Callable,
    env_step_fn: Callable,
    actor_apply: Callable,
    actor_p,
    obs_mean,
    obs_var,
    episode_length: int = 1000,
    seed: int = 0,
    num_envs: int = 10,
) -> float:
    """Run ``num_envs`` episodes deterministically (action = tanh(mu)).

    Uses NumPy arrays for the accumulation loop to avoid re-tracing.

    Args:
        env_reset_fn, env_step_fn: JIT-compiled environment functions.
        actor_apply:               actor_net.apply.
        actor_p:                   Current actor parameters.
        obs_mean, obs_var:         Running obs normalisation stats (JAX arrays).
        episode_length:            Maximum episode steps.
        seed:                      Seed for eval env reset (offset from train seed).
        num_envs:                  Number of parallel eval environments.

    Returns:
        Mean undiscounted episode return across ``num_envs`` environments.
    """
    import numpy as np

    key  = random.PRNGKey(seed + 10000)
    keys = random.split(key, num_envs)
    es   = env_reset_fn(keys)
    total = np.zeros(num_envs)
    done  = np.zeros(num_envs, bool)

    for _ in range(episode_length):
        obs_n = (np.array(es.obs) - np.array(obs_mean)) / np.sqrt(
            np.array(obs_var) + 1e-8
        )
        mu, _ = actor_apply(actor_p, jnp.array(obs_n))
        action = jnp.tanh(mu)
        es     = env_step_fn(es, action)
        r = np.array(es.reward)
        d = np.array(es.done).astype(bool)
        total += r * (~done)
        done  |= d
        if done.all():
            break

    return float(np.mean(total))


# ---------------------------------------------------------------------------
# Default hyperparameters (custom v1 milestone)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # Environment
    env_id          = "HopperStand",
    num_envs        = 32,
    episode_length  = 1000,
    # Networks
    hidden          = (512, 512),
    log_std_min     = -5.0,
    log_std_max     = 2.0,
    q_layer_norm    = True,
    # Training
    total_timesteps = 10_000_000,
    learning_rate   = 1e-3,
    alpha_lr        = 3e-4,
    gamma           = 0.99,
    tau             = 0.005,
    reward_scaling  = 1.0,         # env-specific; see brax sac config
    normalize_obs   = True,
    # Buffer
    min_replay_size = 10_000,
    max_replay_size = 1_000_000,
    batch_size      = 512,
    # Collect / update schedule
    collect_steps   = 64,          # env steps per scan collect call
    grad_updates_per_step = 2,     # k_updates = collect_steps * grad_updates_per_step
    # Evaluation
    num_evals       = 20,
    # Auto-tune alpha
    target_entropy_override = None,  # None → -0.5 * action_size
)
