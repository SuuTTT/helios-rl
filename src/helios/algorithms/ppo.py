"""Proximal Policy Optimisation (PPO) — v34s3 milestone implementation.

Milestone: 904.5 @ 74M steps on MuJoCo Playground CheetahRun, matching Brax PPO reference.

Architecture (Brax-exact separate networks):
    PolicyNet: 4 × Dense(32) + swish → Dense(2*action_dim)  [mean | raw_scale]
               lecun_uniform init on all layers
    ValueNet:  5 × Dense(256) + swish → Dense(1)
               lecun_uniform init on all layers
    Distribution: NormalTanh (squashed Gaussian, Jacobian-corrected log-prob)
    Optimizer: Adam(lr=1e-3, eps=1e-5) + clip_by_global_norm(1.0)

Training structure (Brax-exact):
    Per outer iteration:
      Phase 1 — collect update_epochs=16 independent rollouts (FIXED policy)
                → merge 16 × 2048 × 30 = 983040 steps into 32768 traj × 30 steps
      Phase 2 — update_epochs=16 SGD rounds, each:
                  shuffle 32768 traj → num_minibatches=32 groups of 1024 traj × 30 steps
                  per-minibatch GAE with fresh critic
                  one gradient step per minibatch

Key fixes in v34s3 vs earlier versions:
    v15  NormalTanh with Jacobian correction (not Gaussian + clip)
    v17  Per-epoch GAE with fresh critic values per minibatch
    v25  16 merged rollouts (Brax-exact data volume)
    v19  max_grad_norm=1.0 (not 0.5)
    v34  Crash recovery WITHOUT optimizer reset (reset causes instability)

Reference: Schulman et al. (2017) - https://arxiv.org/abs/1707.06347
"""
from __future__ import annotations

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


# ---------------------------------------------------------------------------
# Networks (Brax-exact separate architecture)
# ---------------------------------------------------------------------------

_lecu = jax.nn.initializers.lecun_uniform()


class PolicyNet(nn.Module):
    """Policy network: 4 × Dense(32) + swish → Dense(2*action_dim).

    Outputs [mean | raw_scale] concatenated. std = softplus(raw_scale) + 0.001.

    Args:
        action_dim: Number of action dimensions.
    """

    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(4):
            x = nn.Dense(32, kernel_init=_lecu)(x)
            x = nn.swish(x)
        return nn.Dense(2 * self.action_dim, kernel_init=_lecu)(x)


class ValueNet(nn.Module):
    """Value network: 5 × Dense(256) + swish → Dense(1)."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(5):
            x = nn.Dense(256, kernel_init=_lecu)(x)
            x = nn.swish(x)
        return nn.Dense(1, kernel_init=_lecu)(x)


# ---------------------------------------------------------------------------
# NormalTanh distribution helpers (Brax-exact numerically stable form)
# ---------------------------------------------------------------------------


def _tanh_log_det_jac(x: jax.Array) -> jax.Array:
    """log|d/dx tanh(x)| per element — numerically stable Brax form."""
    return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


def tanh_normal_logprob(
    raw_action: jax.Array, mean: jax.Array, log_scale: jax.Array
) -> jax.Array:
    """Log-prob under NormalTanh: Gaussian log-prob minus tanh Jacobian.

    Args:
        raw_action: Pre-squash action stored during rollout, shape (..., act_dim).
        mean:       Policy mean output.
        log_scale:  Raw scale output (std = softplus(log_scale) + 0.001).

    Returns:
        Per-sample scalar log probability, shape (...,).
    """
    std = jax.nn.softplus(log_scale) + 0.001
    log_prob_gauss = (
        -0.5 * ((raw_action - mean) / std) ** 2
        - jnp.log(std)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    return (log_prob_gauss - _tanh_log_det_jac(raw_action)).sum(axis=-1)


def tanh_normal_entropy(
    mean: jax.Array, log_scale: jax.Array, fresh_raw: jax.Array
) -> jax.Array:
    """Entropy estimate via a fresh sample (Brax-exact).

    Uses a fresh sample (not the stored rollout action) as the Jacobian point
    to avoid correlation between entropy gradient and policy gradient.

    Args:
        mean:       Policy mean output.
        log_scale:  Raw scale output.
        fresh_raw:  Freshly sampled pre-squash action.

    Returns:
        Per-sample scalar entropy, shape (...,).
    """
    std = jax.nn.softplus(log_scale) + 0.001
    gauss_entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + jnp.log(std)
    return (gauss_entropy + _tanh_log_det_jac(fresh_raw)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Running observation normalisation (Welford online algorithm)
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class ObsNormState:
    """Welford running mean/variance for observation normalisation."""

    mean: jax.Array   # (obs_dim,)
    var:  jax.Array   # (obs_dim,)
    count: jax.Array  # scalar int32


def obs_norm_init(obs_dim: int) -> ObsNormState:
    return ObsNormState(
        mean=jnp.zeros(obs_dim),
        var=jnp.ones(obs_dim),
        count=jnp.zeros((), dtype=jnp.int32),
    )


def obs_norm_update(state: ObsNormState, batch: jax.Array) -> ObsNormState:
    """Welford online update over a (N, obs_dim) batch."""
    n = batch.shape[0]
    batch_mean = batch.mean(axis=0)
    batch_var  = batch.var(axis=0)
    new_count  = state.count + n
    delta      = batch_mean - state.mean
    new_mean   = state.mean + delta * n / new_count
    m_a = state.var * state.count
    m_b = batch_var * n
    new_var = (m_a + m_b + delta ** 2 * state.count * n / new_count) / new_count
    return ObsNormState(mean=new_mean, var=new_var, count=new_count)


def obs_norm_apply(
    state: ObsNormState, obs: jax.Array, eps: float = 1e-6
) -> jax.Array:
    """Normalise: (obs - mean) / clip(sqrt(var + eps), 1e-6, 1e6)."""
    std = jnp.clip(jnp.sqrt(state.var + eps), 1e-6, 1e6)
    return (obs - state.mean) / std


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class Storage:
    """Per-step rollout data for all envs."""

    obs: jax.Array          # (T, N, obs_dim) or (N, T, ...) after merge
    actions: jax.Array      # pre-squash raw actions
    logprobs: jax.Array     # (T, N)
    dones: jax.Array        # float32, (T, N)
    values: jax.Array       # (T, N)
    rewards: jax.Array      # scaled, (T, N)
    returns: jax.Array      # filled by GAE
    advantages: jax.Array   # filled by GAE
    truncations: jax.Array  # float32, (T, N)


# ---------------------------------------------------------------------------
# Per-minibatch GAE (fresh critic values)
# ---------------------------------------------------------------------------


def compute_gae_mb(
    value_apply,
    value_params: dict,
    next_obs_mb: jax.Array,
    storage_mb: Storage,
    gamma: float,
    gae_lambda: float,
) -> Storage:
    """Brax-exact 2-pass GAE on a (T, mb_size) trajectory block.

    Recomputes critic values from current params — fresh estimates each minibatch.

    Args:
        value_apply:   value_net.apply.
        value_params:  Current value network parameters.
        next_obs_mb:   Bootstrap observations, shape (mb_size, obs_dim).
        storage_mb:    Storage with obs shape (T, mb_size, obs_dim).
        gamma:         Discount factor.
        gae_lambda:    GAE lambda.

    Returns:
        Storage with advantages and returns filled in.
    """
    T, mb_size = storage_mb.obs.shape[:2]
    flat_v = value_apply(
        value_params, storage_mb.obs.reshape(T * mb_size, -1)
    ).squeeze(-1).reshape(T, mb_size)
    bootstrap_v = value_apply(value_params, next_obs_mb).squeeze(-1)

    termination = storage_mb.dones * (1.0 - storage_mb.truncations)
    trunc_mask  = 1.0 - storage_mb.truncations
    v_t1        = jnp.concatenate([flat_v[1:], bootstrap_v[None, :]], axis=0)
    deltas      = (
        storage_mb.rewards + gamma * (1.0 - termination) * v_t1 - flat_v
    ) * trunc_mask

    def vs_step(acc, t):
        tm, delta, term = t
        new_acc = delta + gamma * (1.0 - term) * tm * gae_lambda * acc
        return new_acc, new_acc  # carry + stacked output (T, mb_size)

    _, vs_minus_v = jax.lax.scan(
        vs_step, jnp.zeros(mb_size), (trunc_mask, deltas, termination), reverse=True
    )
    vs    = vs_minus_v + flat_v
    vs_t1 = jnp.concatenate([vs[1:], bootstrap_v[None, :]], axis=0)
    adv   = (
        storage_mb.rewards + gamma * (1.0 - termination) * vs_t1 - flat_v
    ) * trunc_mask
    return storage_mb.replace(advantages=adv, returns=vs)


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_loss_fn(
    params: dict,
    policy_apply,
    value_apply,
    obs: jax.Array,
    raw_actions: jax.Array,
    old_logprobs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    key: jax.Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    norm_adv: bool = True,
) -> tuple[jax.Array, tuple]:
    """PPO clipped surrogate loss.

    Returns:
        (total_loss, (pg_loss, v_loss, entropy_loss, approx_kl))
    """
    logits   = policy_apply(params["policy_params"], obs)
    mean, log_scale = jnp.split(logits, 2, axis=-1)
    new_logprobs = tanh_normal_logprob(raw_actions, mean, log_scale)

    # Fresh sample for entropy — avoids gradient correlation with policy loss
    std      = jax.nn.softplus(log_scale) + 0.001
    fresh    = mean + std * jax.random.normal(key, mean.shape)
    entropy  = tanh_normal_entropy(mean, log_scale, fresh)

    new_values = value_apply(params["value_params"], obs).squeeze(-1)

    logratio   = new_logprobs - old_logprobs
    ratio      = jnp.exp(logratio)
    approx_kl  = ((ratio - 1) - logratio).mean()

    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss  = jnp.maximum(
        -advantages * ratio,
        -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef),
    ).mean()
    v_loss   = 0.5 * ((new_values - returns) ** 2).mean()
    ent_loss = entropy.mean()
    total    = pg_loss - ent_coef * ent_loss + vf_coef * v_loss
    return total, (pg_loss, v_loss, ent_loss, jax.lax.stop_gradient(approx_kl))


_ppo_loss_grad = jax.value_and_grad(ppo_loss_fn, argnums=0, has_aux=True)


# ---------------------------------------------------------------------------
# Outer update function factory (merged-rollout Brax-exact)
# ---------------------------------------------------------------------------


def make_update_fn(
    policy_net: PolicyNet,
    value_net: ValueNet,
    env_step_fn,
    num_envs: int,
    num_steps: int,
    update_epochs: int,
    num_minibatches: int,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.3,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    norm_adv: bool = True,
    normalize_obs: bool = True,
    reward_scaling: float = 10.0,
    target_kl: float = 0.0,
):
    """Build the JIT-compiled outer update function (v34s3 Brax-exact structure).

    Outer iteration structure:
      Phase 1: Collect ``update_epochs`` rollouts with FIXED policy.
               Total = update_epochs × num_envs × num_steps env steps.
               Merge into (update_epochs × num_envs) trajectory-first format.
      Phase 2: ``update_epochs`` SGD rounds, each shuffling all trajectories
               into ``num_minibatches`` groups, computing per-minibatch GAE with
               the current critic, and taking one gradient step per minibatch.

    Args:
        policy_net, value_net:  Initialised or uninitialised Flax modules.
        env_step_fn:            JIT-compiled env.step(state, action) → EnvState.
        num_envs:               Parallel environments (e.g. 2048).
        num_steps:              Steps per rollout (e.g. 30).
        update_epochs:          Rollout count / SGD rounds (e.g. 16).
        num_minibatches:        Gradient steps per SGD round (e.g. 32).
        gamma, gae_lambda:      Discount and GAE lambda.
        clip_coef:              PPO clip epsilon (e.g. 0.3).
        vf_coef, ent_coef:      Value and entropy loss coefficients.
        norm_adv:               Normalise advantages within each minibatch.
        normalize_obs:          Apply ObsNormState Welford normalisation.
        reward_scaling:         Multiply raw rewards before GAE (e.g. 10.0).
        target_kl:              Early-stop SGD round if mean KL > this (0 = off).

    Returns:
        rollout_and_update(agent_state, env_state, next_obs, next_done,
                           key, ep_return, ep_len, obs_ns)
            → (agent_state, env_state, next_obs, next_done,
               key, ep_return, ep_len, obs_ns, mean_ep_return)

        ``agent_state`` must be a Flax TrainState with
        ``params = {"policy_params": ..., "value_params": ...}``.
    """
    policy_apply = jax.jit(policy_net.apply)
    value_apply  = jax.jit(value_net.apply)
    total_traj   = update_epochs * num_envs
    eff_bs       = total_traj // num_minibatches

    @jax.jit
    def get_action_and_value(params, obs, key):
        logits = policy_apply(params["policy_params"], obs)
        mean, log_scale = jnp.split(logits, 2, axis=-1)
        std = jax.nn.softplus(log_scale) + 0.001
        key, sk = jax.random.split(key)
        raw = mean + std * jax.random.normal(sk, mean.shape)
        lp  = tanh_normal_logprob(raw, mean, log_scale)
        val = value_apply(params["value_params"], obs).squeeze(-1)
        return raw, lp, val, key

    def step_once(carry, _):
        ag, es, obs, done, key, ep_ret, ep_len, obs_ns = carry
        norm_obs = obs_norm_apply(obs_ns, obs) if normalize_obs else obs
        raw, lp, val, key = get_action_and_value(ag.params, norm_obs, key)
        env_action = jnp.tanh(raw)
        nes = env_step_fn(es, env_action)
        new_done = nes.done > 0.5
        trunc    = nes.info["truncation"] > 0.5
        ep_ret   = ep_ret + nes.reward
        ep_len   = ep_len + 1
        completed = jnp.where(new_done, ep_ret, 0.0)
        ep_ret    = jnp.where(new_done, 0.0, ep_ret)
        ep_len    = jnp.where(new_done, 0,   ep_len)
        s = Storage(
            obs=obs, actions=raw, logprobs=lp,
            dones=new_done.astype(jnp.float32), values=val,
            rewards=nes.reward * reward_scaling,
            returns=jnp.zeros_like(nes.reward),
            advantages=jnp.zeros_like(nes.reward),
            truncations=trunc.astype(jnp.float32),
        )
        return (ag, nes, nes.obs, new_done, key, ep_ret, ep_len, obs_ns), \
               (s, completed, new_done)

    @jax.jit
    def rollout_and_update(
        agent_state: TrainState,
        env_state,
        next_obs: jax.Array,
        next_done: jax.Array,
        key: jax.Array,
        ep_return: jax.Array,
        ep_len: jax.Array,
        obs_ns: ObsNormState,
    ):
        # ── Phase 1: Collect update_epochs rollouts with FIXED policy ───────
        def collect_one(carry, _):
            es, obs, done, key, ep_ret, ep_len = carry
            (_, es2, obs2, done2, key2, ep_ret2, ep_len2, _), \
                (storage, completed, dones) = jax.lax.scan(
                    step_once,
                    (agent_state, es, obs, done, key, ep_ret, ep_len, obs_ns),
                    (), length=num_steps,
                )
            n_done   = dones.sum()
            mean_ret = jnp.where(n_done > 0, completed.sum() / n_done, -1.0)
            return (es2, obs2, done2, key2, ep_ret2, ep_len2), (storage, obs2, mean_ret)

        (env_state, next_obs, next_done, key, ep_return, ep_len), \
            (all_storages, all_next_obs, all_ep_rets) = jax.lax.scan(
                collect_one,
                (env_state, next_obs, next_done, key, ep_return, ep_len),
                (), length=update_epochs,
            )

        # ── Merge (R, T, N, ...) → (R*N, T, ...) trajectory-first ──────────
        def merge(x):
            return x.swapaxes(1, 2).reshape((-1,) + x.shape[2:])

        merged          = jax.tree_util.tree_map(merge, all_storages)
        merged_next_obs = all_next_obs.reshape(-1, all_next_obs.shape[-1])

        # ── Obs-norm update (once with all raw merged obs) ───────────────────
        if normalize_obs:
            flat_raw       = merged.obs.reshape(-1, merged.obs.shape[-1])
            norm_merged    = merged.replace(obs=obs_norm_apply(obs_ns, merged.obs))
            norm_next_obs  = obs_norm_apply(obs_ns, merged_next_obs)
            obs_ns         = obs_norm_update(obs_ns, flat_raw)
        else:
            norm_merged   = merged
            norm_next_obs = merged_next_obs

        # ── Phase 2: SGD rounds ──────────────────────────────────────────────
        def sgd_round(carry, _):
            ag, key, kl_exceeded = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, total_traj)
            shuffled      = jax.tree_util.tree_map(lambda x: x[perm], norm_merged)
            shuffled_nobs = norm_next_obs[perm]

            def split_mb(x):
                return x.reshape(num_minibatches, eff_bs, *x.shape[1:])

            mb_s    = jax.tree_util.tree_map(split_mb, shuffled)
            mb_nobs = shuffled_nobs.reshape(num_minibatches, eff_bs, -1)

            def update_minibatch(carry, inp):
                ag_m, key_m = carry
                mb, mn = inp  # (eff_bs, T, ...), (eff_bs, obs_dim)
                # Transpose (eff_bs, T, ...) → (T, eff_bs, ...) for compute_gae_mb
                mb_T = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), mb)
                gae  = compute_gae_mb(
                    value_apply, ag_m.params["value_params"],
                    mn, mb_T, gamma, gae_lambda,
                )
                fo  = gae.obs.reshape(-1, gae.obs.shape[-1])
                fa  = gae.actions.reshape(-1, gae.actions.shape[-1])
                flp = gae.logprobs.reshape(-1)
                fad = gae.advantages.reshape(-1)
                frt = gae.returns.reshape(-1)
                key_m, lk = jax.random.split(key_m)
                (loss, aux), grads = _ppo_loss_grad(
                    ag_m.params, policy_apply, value_apply,
                    fo, fa, flp, fad, frt, lk,
                    clip_coef, vf_coef, ent_coef, norm_adv,
                )
                return (ag_m.apply_gradients(grads=grads), key_m), aux

            def do_updates(args_):
                ag_, key_ = args_
                (ag_out, key_out), aux = jax.lax.scan(
                    update_minibatch, (ag_, key_), (mb_s, mb_nobs)
                )
                return ag_out, key_out, aux[3].mean()

            def skip_updates(args_):
                ag_, key_ = args_
                return ag_, key_, jnp.float32(0.0)

            if target_kl > 0.0:
                ag, key, mean_kl = jax.lax.cond(
                    kl_exceeded, skip_updates, do_updates, (ag, key)
                )
                new_kl_exceeded = kl_exceeded | (mean_kl > target_kl)
            else:
                ag, key, mean_kl = do_updates((ag, key))
                new_kl_exceeded = jnp.bool_(False)

            return (ag, key, new_kl_exceeded), mean_kl

        (agent_state, key, _), _ = jax.lax.scan(
            sgd_round, (agent_state, key, jnp.bool_(False)), (), length=update_epochs
        )

        valid          = all_ep_rets > 0
        mean_ep_return = jnp.where(
            valid.sum() > 0,
            (all_ep_rets * valid).sum() / valid.sum().clip(1),
            -1.0,
        )
        return (
            agent_state, env_state, next_obs, next_done,
            key, ep_return, ep_len, obs_ns, mean_ep_return,
        )

    return rollout_and_update


# ---------------------------------------------------------------------------
# Default hyperparameters (v34s3 milestone)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # Environment
    env_id              = "CheetahRun",
    num_envs            = 2048,
    num_steps           = 30,
    episode_length      = 1000,
    reward_scaling      = 10.0,
    normalize_obs       = True,
    # Training schedule
    total_timesteps     = 180_000_000,
    learning_rate       = 1e-3,
    adam_eps            = 1e-5,
    anneal_lr           = False,
    update_epochs       = 16,
    num_minibatches     = 32,
    gamma               = 0.995,
    gae_lambda          = 0.95,
    # PPO loss
    clip_coef           = 0.3,
    vf_coef             = 0.5,
    ent_coef            = 0.01,
    max_grad_norm       = 1.0,
    norm_adv            = True,
    target_kl           = 0.0,
    # Crash recovery (v34 critical fix)
    crash_threshold     = 150.0,  # drop from best before recovery
    optimizer_reset_on_recovery = False,  # False = stable (v34s3); True = fragile (v32)
    # Evaluation
    eval_freq           = 30,     # iterations between evals
)
