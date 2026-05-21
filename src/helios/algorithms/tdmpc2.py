"""TD-MPC2 JAX implementation — v24 milestone.

Milestone: MPPI=357@3M, stable 340–357 from 1.25M onwards on HopperHop.
Key innovations vs prior versions:
- SimNorm(V=8) encoder and dynamics (bounded latents)
- Two-hot distributional Q/reward with symlog/symexp
- MPPI planner with elite-based selection, pi-trajectory seeding
- RunningScale (IQR-based, EMA tau=0.01) capped at [1.0, 4.0] — prevents
  gradient crushing while maintaining stability (v24 key fix over v18)
- lax.scan fused multi-update loop (K gradient steps in one JIT dispatch)

Architecture:
    latent_dim=512, hidden=(512,512), num_bins=101, vmin/vmax=±20
    MPPI: H=3, NS=512, num_elites=64, num_pi_trajs=24, n_iter=6
    std range: [0.05, 2.0]

Ref: Hansen et al. (2023) TD-MPC2 — https://arxiv.org/abs/2310.16828
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


# ---------------------------------------------------------------------------
# SimNorm and math utilities
# ---------------------------------------------------------------------------


def simnorm(x: jax.Array, V: int = 8) -> jax.Array:
    """Simplex normalisation: partition into V groups, softmax each group."""
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)


def log_std_fn(x: jax.Array, low: float = -10.0, dif: float = 12.0) -> jax.Array:
    """Map raw network output to log_std in [low, low+dif]."""
    return low + 0.5 * dif * (jnp.tanh(x) + 1.0)


def gaussian_logprob(eps: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log probability of eps under N(0,1) reparameterized with log_std."""
    residual = -0.5 * (eps ** 2) - log_std
    log_prob = residual - 0.9189385175704956  # -0.5 * log(2π)
    return jnp.sum(log_prob, axis=-1, keepdims=True)


def squash(
    mu: jax.Array, pi: jax.Array, log_pi: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply tanh squashing with Jacobian correction to log_pi."""
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    squashed_pi = jnp.log(jax.nn.relu(1 - pi ** 2) + 1e-6)
    log_pi = log_pi - jnp.sum(squashed_pi, axis=-1, keepdims=True)
    return mu, pi, log_pi


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric log: sign(x) * log(1 + |x|)."""
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot(
    x: jax.Array, vmin: float = -20.0, vmax: float = 20.0, num_bins: int = 101
) -> jax.Array:
    """Two-hot encoding in symlog space.

    Clips symlog(x) to [vmin, vmax] then distributes weight between the two
    nearest bin indices proportionally.
    """
    x = jnp.clip(symlog(x), vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    bin_index = (x - vmin) / bin_size
    lower = jnp.floor(bin_index).astype(jnp.int32)
    upper = jnp.ceil(bin_index).astype(jnp.int32)
    p_upper = bin_index - lower
    p_lower = 1.0 - p_upper
    lower_hot = jax.nn.one_hot(lower, num_bins) * p_lower[..., None]
    upper_hot = jax.nn.one_hot(upper, num_bins) * p_upper[..., None]
    return lower_hot + upper_hot


def soft_ce(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Cross-entropy loss for distributional targets."""
    return -jnp.sum(target * jax.nn.log_softmax(pred, axis=-1), axis=-1)


def two_hot_inv(
    logits: jax.Array, vmin: float = -20.0, vmax: float = 20.0, num_bins: int = 101
) -> jax.Array:
    """Decode distributional logits back to scalar in original space."""
    probs = jax.nn.softmax(logits, axis=-1)
    bins = jnp.linspace(vmin, vmax, num_bins)
    return symexp(jnp.sum(probs * bins, axis=-1))


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class NormMLP(nn.Module):
    """Dense → LayerNorm → SiLU stack with a final linear output."""

    dims: tuple[int, ...]
    out: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for d in self.dims:
            x = nn.Dense(d)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.out)(x)


class Encoder(nn.Module):
    """Encodes observations to SimNorm-bounded latent vectors."""

    latent_dim: int
    hidden: tuple[int, ...] = (512, 512)
    V: int = 8

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)


class Dynamics(nn.Module):
    """Predicts next latent from (z, a) using SimNorm-bounded output."""

    latent_dim: int
    hidden: tuple[int, ...] = (512, 512)
    V: int = 8

    @nn.compact
    def __call__(self, z: jax.Array, a: jax.Array) -> jax.Array:
        return simnorm(
            NormMLP(self.hidden, self.latent_dim)(jnp.concatenate([z, a], -1)), self.V
        )


class RewardHead(nn.Module):
    """Distributional reward predictor (101-bin two-hot)."""

    hidden: tuple[int, ...] = (512, 512)
    num_bins: int = 101

    @nn.compact
    def __call__(self, z: jax.Array, a: jax.Array) -> jax.Array:
        return NormMLP(self.hidden, self.num_bins)(jnp.concatenate([z, a], -1))


class QEnsemble(nn.Module):
    """Two-head Q ensemble with distributional outputs (101-bin two-hot).

    Returns shape (..., 2, num_bins). Take jnp.min over axis=-2 after
    two_hot_inv to get the min-Q value estimate.
    """

    hidden: tuple[int, ...] = (512, 512)
    num_bins: int = 101

    @nn.compact
    def __call__(self, z: jax.Array, a: jax.Array) -> jax.Array:
        x = jnp.concatenate([z, a], -1)
        return jnp.stack(
            [NormMLP(self.hidden, self.num_bins)(x),
             NormMLP(self.hidden, self.num_bins)(x)],
            axis=-2,
        )


class Pi(nn.Module):
    """Stochastic Gaussian policy with tanh squashing.

    Outputs (mean, log_std). Use sample_pi() to sample an action.
    For deterministic execution: action = tanh(mean).
    """

    action_dim: int
    hidden: tuple[int, ...] = (512, 512)
    log_std_min: float = -10.0
    log_std_dif: float = 12.0

    @nn.compact
    def __call__(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = NormMLP(self.hidden, self.action_dim * 2)(z)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = log_std_fn(log_std, self.log_std_min, self.log_std_dif)
        return mean, log_std


def sample_pi(
    mean_logstd: tuple[jax.Array, jax.Array], key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a tanh-squashed action from the policy distribution.

    Args:
        mean_logstd: (mean, log_std) from Pi.__call__
        key: PRNGKey

    Returns:
        (action, log_prob, scaled_entropy)
    """
    mean, log_std = mean_logstd
    eps = jax.random.normal(key, mean.shape)
    log_prob = gaussian_logprob(eps, log_std)
    size = eps.shape[-1]
    scaled_log_prob = log_prob * size
    action_pre = mean + eps * jnp.exp(log_std)
    _, action, log_prob = squash(mean, action_pre, log_prob)
    entropy_scale = scaled_log_prob / (log_prob + 1e-8)
    scaled_entropy = -log_prob * entropy_scale
    return action, log_prob, scaled_entropy


# ---------------------------------------------------------------------------
# Multi-environment replay buffer
# ---------------------------------------------------------------------------


class MultiEnvBuffer:
    """Per-environment ring buffer storing (obs, act, rew, done) transitions.

    Stores separate circular buffers for each environment to allow
    sampling contiguous sequences without episode boundary contamination.

    Args:
        cap:     Per-environment buffer capacity.
        n_envs:  Number of parallel environments.
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        seq_len: Sequence length T (horizon H+1) for each sampled batch.
    """

    def __init__(
        self,
        cap: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        seq_len: int,
    ) -> None:
        self.cap = cap
        self.N = n_envs
        self.T = seq_len
        self.obs  = np.zeros((n_envs, cap, obs_dim), np.float32)
        self.acts = np.zeros((n_envs, cap, act_dim), np.float32)
        self.rews = np.zeros((n_envs, cap), np.float32)
        self.done = np.zeros((n_envs, cap), np.float32)
        self.ptr  = np.zeros(n_envs, np.int64)
        self.size = np.zeros(n_envs, np.int64)

    def add_batch(
        self,
        obs_b: np.ndarray,
        acts_b: np.ndarray,
        rews_b: np.ndarray,
        done_b: np.ndarray,
    ) -> None:
        """Insert one transition per environment."""
        p = self.ptr
        self.obs[np.arange(self.N), p]  = obs_b
        self.acts[np.arange(self.N), p] = acts_b
        self.rews[np.arange(self.N), p] = rews_b
        self.done[np.arange(self.N), p] = done_b
        self.ptr  = (p + 1) % self.cap
        self.size = np.minimum(self.size + 1, self.cap)

    def total_size(self) -> int:
        return int(self.size.sum())

    def sample(self, B: int, rng: np.random.Generator) -> tuple | None:
        """Sample B independent sequences of length T."""
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        env_ids = rng.choice(valid, size=B, replace=True)
        sizes   = self.size[env_ids]
        starts  = (rng.random(B) * (sizes - self.T)).astype(np.int64)
        idx     = starts[:, None] + np.arange(self.T)[None, :]
        return (
            self.obs[env_ids[:, None], idx],
            self.acts[env_ids[:, None], idx],
            self.rews[env_ids[:, None], idx],
            self.done[env_ids[:, None], idx],
        )

    def sample_k(
        self, K: int, B: int, rng: np.random.Generator
    ) -> tuple | None:
        """Sample K×B sequences in one vectorised numpy call.

        Returns arrays shaped (K, B, T, dim) suitable for lax.scan.
        """
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        KB      = K * B
        env_ids = rng.choice(valid, size=KB, replace=True)
        sizes   = self.size[env_ids]
        starts  = (rng.random(KB) * (sizes - self.T)).astype(np.int64)
        idx     = starts[:, None] + np.arange(self.T)[None, :]
        obs_kb  = self.obs [env_ids[:, None], idx]
        acts_kb = self.acts[env_ids[:, None], idx]
        rews_kb = self.rews[env_ids[:, None], idx]
        done_kb = self.done[env_ids[:, None], idx]
        obs_dim = obs_kb.shape[-1]
        act_dim = acts_kb.shape[-1]
        return (
            obs_kb.reshape(K, B, self.T, obs_dim),
            acts_kb.reshape(K, B, self.T, act_dim),
            rews_kb.reshape(K, B, self.T),
            done_kb.reshape(K, B, self.T),
        )


# ---------------------------------------------------------------------------
# Update function factory
# ---------------------------------------------------------------------------


def make_update_fn(
    enc: Encoder,
    dyn: Dynamics,
    rew_net: RewardHead,
    q_net: QEnsemble,
    pi_net: Pi,
    tx: optax.GradientTransformation,
    gamma: float = 0.99,
    rho: float = 0.5,
    tau: float = 0.01,
    rew_scale: float = 10.0,
    scale_min: float = 1.0,
    scale_max: float = 4.0,
    latent_action_smooth_coef: float = 0.0,
    consistency_coef: float = 2.0,
    smoothing_enabled: bool = True,
    mpc_distill_enabled: bool = False,
) -> tuple:
    """Build (single_step, multi_step) JIT-compiled update functions.

    RunningScale (v24):
        Tracks IQR (5th–95th percentile) of Q_pi values via EMA (tau=0.01).
        Capped to [scale_min, scale_max] = [1.0, 4.0].
        Pi loss = -mean(min(Q/scale)) over the rollout horizon.

    Args:
        enc, dyn, rew_net, q_net, pi_net: Flax modules (uninitialised).
        tx:         Optax gradient transform (applied to all params).
        gamma:      Discount factor.
        rho:        Consistency loss horizon decay (weight = rho^t).
        tau:        EMA coefficient for target params and RunningScale.
        rew_scale:  Reward scaling factor (targets multiplied, MPPI divided).
        scale_min:  RunningScale lower bound (prevents gradient collapse).
        scale_max:  RunningScale upper bound (v24 fix — prevents over-normalisation).

    Returns:
        (single_step, multi_step) — both @jax.jit compiled.
    """

    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b, rng, scale,
                mpc_obs_anchor, mpc_action_target, mpc_distill_coef):
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B * T, -1)).reshape(B, T, -1)
        z0    = z_all[:, 0]
        acts_T = jnp.transpose(act_b[:, :-1], (1, 0, 2))

        def dyn_step(z, a):
            return dyn.apply(params["dyn"], z, a), z

        z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
        zs = jnp.concatenate(
            [jnp.transpose(zs_prefix, (1, 0, 2)), z_final[:, None, :]], 1
        )

        z_tgt   = jax.lax.stop_gradient(z_all)
        weights = jnp.array([rho ** t for t in range(T - 1)])
        z_t_T   = jnp.transpose(zs[:, :-1],  (1, 0, 2))
        a_T     = acts_T
        r_T     = jnp.transpose(rew_b[:, :-1],  (1, 0))
        d_T     = jnp.transpose(done_b[:, :-1], (1, 0))
        z_t1_T  = jnp.transpose(z_tgt[:, 1:],   (1, 0, 2))
        zs_t1_T = jnp.transpose(zs[:, 1:],      (1, 0, 2))

        def step_loss(carry, inp):
            k, s = carry
            w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp

            # Consistency loss
            cl = w * jnp.mean(jnp.sum((zs_t1 - z_tgt_t1) ** 2, -1))

            # Reward loss (distributional)
            pr = rew_net.apply(params["rew"], z_t, a_t)
            rl = w * jnp.mean(soft_ce(pr, two_hot(r_t)))

            # Value / Q loss
            z_n = jax.lax.stop_gradient(z_tgt_t1)
            k, _ = jax.random.split(k)
            tp_mean_std = pi_net.apply(tp["pi"], z_n)
            pi_a_mean = jnp.tanh(tp_mean_std[0])
            q_next_logits = q_net.apply(tp["q"], z_n, pi_a_mean)
            q_next_vals   = two_hot_inv(q_next_logits)
            v_n = jnp.maximum(jnp.min(q_next_vals, -1), 0.0)
            td  = r_t + gamma * (1 - d_t) * jax.lax.stop_gradient(v_n)
            qp  = q_net.apply(params["q"], z_t, a_t)
            vl  = w * jnp.mean(jnp.sum(soft_ce(qp, two_hot(td)[:, None, :]), -1))

            # Policy loss with RunningScale (v24)
            pi_mean_std = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_t))
            pi2_mean    = jnp.tanh(pi_mean_std[0])
            q_pi2_logits = q_net.apply(
                jax.lax.stop_gradient(params["q"]),
                jax.lax.stop_gradient(z_t),
                pi2_mean,
            )
            q_pi2_vals = two_hot_inv(q_pi2_logits)  # (B, 2)

            q_flat = q_pi2_vals.flatten()
            p5     = jnp.percentile(q_flat, 5)
            p95    = jnp.percentile(q_flat, 95)
            new_s  = jnp.clip(
                (1 - tau) * s + tau * jnp.maximum(p95 - p5, 1.0),
                scale_min,
                scale_max,
            )
            pl = -w * jnp.mean(jnp.min(q_pi2_vals / new_s, -1))

            return (k, new_s), (cl, rl, vl, pl)

        (_, final_scale), (cls, rls, vls, pls) = jax.lax.scan(
            step_loss,
            (rng, scale),
            (weights, z_t_T, a_T, r_T, d_T, z_t1_T, zs_t1_T),
        )
        n = T - 1
        total = (consistency_coef * jnp.sum(cls) + 2 * jnp.sum(rls) + jnp.sum(vls) + 0.1 * jnp.sum(pls)) / n

        # Latent action smoothing — Python-conditional so the graph
        # matches the pre-smoothing version exactly when disabled (Phase-m fix).
        if smoothing_enabled:
            def _pi_mean_at_z(z_step):
                m, _ = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_step))
                return jnp.tanh(m)
            all_pi_mean = jax.vmap(_pi_mean_at_z)(z_t_T)  # (T-1, B, act_dim)
            smooth_loss = jnp.mean(jnp.sum((all_pi_mean[1:] - all_pi_mean[:-1]) ** 2, axis=-1))
            total = total + latent_action_smooth_coef * smooth_loss
        if mpc_distill_enabled:
            z_anchor = enc.apply(params["enc"], mpc_obs_anchor)
            pi_anchor_mean, _ = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_anchor))
            pi_anchor_mean = jnp.tanh(pi_anchor_mean)
            mpc_loss = jnp.mean(jnp.sum((pi_anchor_mean - jax.lax.stop_gradient(mpc_action_target)) ** 2, axis=-1))
            total = total + mpc_distill_coef * mpc_loss
        else:
            mpc_loss = jnp.array(0.0)
        aux = {
            "c": jnp.sum(cls) / n,
            "r": jnp.sum(rls) / n,
            "v": jnp.sum(vls) / n,
            "p": jnp.sum(pls) / n,
            "mpc": mpc_loss,
            "scale": final_scale,
        }
        return total, aux

    @jax.jit
    def single_step(params, tp, opt, ob, ab, rb, db, rng, scale,
                    mpc_obs_anchor, mpc_action_target, mpc_distill_coef):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, tp, ob, ab, rb, db, rng, scale,
            mpc_obs_anchor, mpc_action_target, mpc_distill_coef,
        )
        upd, nopt = tx.update(grads, opt, params)
        new_params = optax.apply_updates(params, upd)
        new_tp = jax.tree_util.tree_map(
            lambda t, p: (1 - tau) * t + tau * p, tp, new_params
        )
        return new_params, new_tp, nopt, loss, aux

    @jax.jit
    def multi_step(params, tp, opt, all_obs, all_acts, all_rews, all_done, key, scale,
                   mpc_obs_anchor, mpc_action_target, mpc_distill_coef):
        """K gradient updates in one JIT dispatch via jax.lax.scan.

        Scale is carried across gradient steps so that the RunningScale
        accumulates over K updates before being returned.
        """

        def one_step(carry, batch):
            params, tp, opt, key, scale = carry
            ob, ab, rb, db = batch
            key, uk = jax.random.split(key)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, tp, ob, ab, rb, db, uk, scale,
                mpc_obs_anchor, mpc_action_target, mpc_distill_coef,
            )
            upds, nopt = tx.update(grads, opt, params)
            new_params = optax.apply_updates(params, upds)
            new_tp = jax.tree_util.tree_map(
                lambda t, p: (1 - tau) * t + tau * p, tp, new_params
            )
            return (new_params, new_tp, nopt, key, aux["scale"]), (loss, aux)

        (params, tp, opt, key, scale), (losses, auxs) = jax.lax.scan(
            one_step,
            (params, tp, opt, key, scale),
            (all_obs, all_acts, all_rews, all_done),
        )
        last_aux = jax.tree_util.tree_map(lambda x: x[-1], auxs)
        return params, tp, opt, key, scale, losses[-1], last_aux

    return single_step, multi_step


# ---------------------------------------------------------------------------
# MPPI planner factory
# ---------------------------------------------------------------------------


def make_mppi_fn(
    enc: Encoder,
    dyn: Dynamics,
    rew_net: RewardHead,
    q_net: QEnsemble,
    pi_net: Pi,
    horizon: int = 3,
    n_samples: int = 512,
    num_elites: int = 64,
    num_pi_trajs: int = 24,
    n_iter: int = 6,
    min_std: float = 0.05,
    max_std: float = 2.0,
    act_low: float = -1.0,
    act_high: float = 1.0,
    act_dim: int = 4,
    gamma: float = 0.99,
    rew_scale: float = 10.0,
):
    """Build the JIT-compiled MPPI planning function (v24 parity).

    Features:
    - num_pi_trajs stochastic pi-trajectory seeds per iteration
    - n_noise = n_samples - num_pi_trajs Gaussian noise trajectories
    - Elite-based update: mean/std from top num_elites by return
    - std clamped to [min_std, max_std]
    - t0 flag resets mu/std on episode start
    - gamma-weighted reward accumulation + terminal value from Q

    Returns:
        plan(params, obs, mu, std, key, t0)
            → (action, new_mu, new_std)
    """
    n_noise  = n_samples - num_pi_trajs
    _gammas  = jnp.array([gamma ** t for t in range(horizon)])
    _gamma_H = float(gamma ** horizon)

    @jax.jit
    def plan(
        params: dict,
        obs: jax.Array,
        mu: jax.Array,
        std: jax.Array,
        key: jax.Array,
        t0: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Plan one step.

        Args:
            params: model params dict with keys enc/dyn/rew/q/pi.
            obs:    (obs_dim,) current observation.
            mu:     (H, act_dim) warm-start mean (zeros on fresh episode).
            std:    (H, act_dim) warm-start std (max_std on fresh episode).
            key:    PRNGKey.
            t0:     bool scalar — True on episode start, resets mu/std.

        Returns:
            action:  (act_dim,) deterministic action (mu[0] after planning).
            new_mu:  (H, act_dim) shifted warm-start for next step.
            new_std: (H, act_dim) shifted std for next step.
        """
        mu  = jnp.where(t0, jnp.zeros_like(mu),  mu)
        std = jnp.where(t0, jnp.full_like(std, max_std), std)

        z0_single = enc.apply(params["enc"], obs[None])[0]

        def pi_rollout_stoch(key):
            def pi_step(z, k):
                mean_a, log_std_a = pi_net.apply(params["pi"], z[None])
                mean_a    = mean_a[0]
                log_std_a = log_std_a[0]
                eps = jax.random.normal(k, mean_a.shape)
                a   = jnp.tanh(mean_a + eps * jnp.exp(log_std_a))
                z2  = dyn.apply(params["dyn"], z[None], a[None])[0]
                return z2, a

            keys_h = jax.random.split(key, horizon)
            _, traj = jax.lax.scan(pi_step, z0_single, keys_h)
            return traj  # (H, act_dim)

        key, pk = jax.random.split(key)
        pi_keys  = jax.random.split(pk, num_pi_trajs)
        pi_trajs = jax.vmap(pi_rollout_stoch)(pi_keys)  # (num_pi_trajs, H, act_dim)

        def one_iter(carry, _):
            mu_i, std_i, k = carry
            k, sk = jax.random.split(k)

            noise      = jax.random.normal(sk, (n_noise, horizon, act_dim)) * std_i[None]
            noise_acts = jnp.clip(mu_i[None] + noise, act_low, act_high)
            acts       = jnp.concatenate([pi_trajs, noise_acts], axis=0)

            z0_batch = jnp.tile(z0_single[None], (n_samples, 1))

            def rollout_one(z_i, a_seq):
                def env_step(z, a):
                    r_logits = rew_net.apply(params["rew"], z[None], a[None])
                    r  = two_hot_inv(r_logits).squeeze()
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r

                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                pi_a_squashed = jnp.tanh(pi_a_mean)
                q_logits = q_net.apply(params["q"], zf[None], pi_a_squashed)
                vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                return jnp.sum(_gammas * rs) + _gamma_H * vt

            rets = jax.vmap(rollout_one)(z0_batch, acts)

            _, elite_idx = jax.lax.top_k(rets, num_elites)
            elite_acts = acts[elite_idx]
            new_mu  = jnp.mean(elite_acts, axis=0)
            new_std = jnp.clip(jnp.std(elite_acts, axis=0) + 1e-6, min_std, max_std)
            return (new_mu, new_std, k), None

        (muf, stdf, _), _ = jax.lax.scan(one_iter, (mu, std, key), None, length=n_iter)

        action  = jnp.clip(muf[0], act_low, act_high)
        new_mu  = jnp.concatenate([muf[1:],  jnp.zeros((1, act_dim))],        axis=0)
        new_std = jnp.concatenate([stdf[1:], jnp.full((1, act_dim), max_std)], axis=0)
        return action, new_mu, new_std

    return plan


# ---------------------------------------------------------------------------
# Default hyperparameters (v24 milestone)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # Architecture
    latent_dim    = 512,
    hidden        = (512, 512),
    num_bins      = 101,
    vmin          = -20.0,
    vmax          = 20.0,
    V             = 8,          # SimNorm groups
    # Training
    lr            = 3e-4,
    gamma         = 0.99,
    tau           = 0.01,       # EMA coefficient (target params + RunningScale)
    rho           = 0.5,        # Consistency loss horizon decay
    rew_scale     = 10.0,
    K_UPDATE      = 64,
    BS            = 256,
    N_ENVS        = 256,
    WARMUP_ENV    = 25_000,
    EXPL_NOISE    = 0.3,
    EXPL_UNTIL    = 25_000,
    # RunningScale (v24)
    scale_min     = 1.0,
    scale_max     = 4.0,
    # MPPI (v24 parity)
    H             = 3,
    NS            = 512,
    NUM_ELITES    = 64,
    NUM_PI_TRAJS  = 24,
    NI            = 6,
    MIN_STD       = 0.05,
    MAX_STD       = 2.0,
    # Loss coefficients (v13 stable)
    consistency_coef = 2.0,
    reward_coef      = 2.0,
    value_coef       = 1.0,
    pi_coef          = 0.1,
)
