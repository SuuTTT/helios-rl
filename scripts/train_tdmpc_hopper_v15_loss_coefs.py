"""Train TD-MPC2 on mujoco_playground HopperHop (JAX vectorised).

v15 = v14 + Phase 3: Official Loss Coefficients.

Changes vs v14:
  1. CONSISTENCY_COEF = 20 (was 2)  — official: dynamics model is primary objective
  2. REWARD_COEF = 0.1 (was 2)      — reward head is secondary
  3. VALUE_COEF = 0.1 (was 1)       — critic is secondary
  4. Value loss uses mean over Q heads (was sum) — matches official /num_q normalisation
  5. Pi term removed from WM loss; pi optimized fully separately (matches official 2-pass)

All Phase 1+2 preserved:
  - H=3, NS=512, NUM_ELITES=64, NUM_PI_TRAJS=24, adaptive std [0.05,2.0]
  - Stochastic pi with entropy, target-Q-only EMA, separate wm/pi optimizers

Rationale:
  - Official total_loss = 20*c + 0.1*r + 0.1*v  (consistency-dominant)
  - v14 had 2*c + 2*r + 1*v  (balanced, hurts dynamics quality)
  - Consistency-dominant training → better latent dynamics → better MPPI planning
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp, optax
import flax.linen as nn

sys.path.insert(0, "/workspace/wiki/learn_mujoco_playground/repo")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")


# ─── SimNorm and Distributional Math ────────────────────────────────────────

def simnorm(x, V=8):
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)

def log_std_fn(x, low=-10.0, dif=12.0):
    return low + 0.5 * dif * (jnp.tanh(x) + 1.0)

def gaussian_logprob(eps, log_std):
    residual = -0.5 * (eps ** 2) - log_std
    log_prob = residual - 0.9189385175704956
    return jnp.sum(log_prob, axis=-1, keepdims=True)

def squash_logprob(pi_pre, log_pi):
    """Tanh squash + log-prob correction."""
    pi = jnp.tanh(pi_pre)
    lp = log_pi - jnp.sum(jnp.log(jax.nn.relu(1.0 - pi**2) + 1e-6), axis=-1, keepdims=True)
    return pi, lp

def symlog(x):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def two_hot(x, vmin=-20, vmax=20, num_bins=101):
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

def soft_ce(pred, target):
    return -jnp.sum(target * jax.nn.log_softmax(pred, axis=-1), axis=-1)

def two_hot_inv(logits, vmin=-20, vmax=20, num_bins=101):
    probs = jax.nn.softmax(logits, axis=-1)
    bins = jnp.linspace(vmin, vmax, num_bins)
    return symexp(jnp.sum(probs * bins, axis=-1))


# ─── Models ───────────────────────────────────────────────────────────────────

class NormMLP(nn.Module):
    dims: tuple; out: int
    @nn.compact
    def __call__(self, x):
        for d in self.dims:
            x = nn.Dense(d)(x); x = nn.LayerNorm()(x); x = nn.silu(x)
        return nn.Dense(self.out)(x)

class Encoder(nn.Module):
    latent_dim: int; hidden: tuple = (128, 128); V: int = 8
    @nn.compact
    def __call__(self, obs):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)

class Dynamics(nn.Module):
    latent_dim: int; hidden: tuple = (128, 128); V: int = 8
    @nn.compact
    def __call__(self, z, a):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(jnp.concatenate([z, a], -1)), self.V)

class RewardHead(nn.Module):
    hidden: tuple = (128, 128)
    num_bins: int = 101
    @nn.compact
    def __call__(self, z, a):
        return NormMLP(self.hidden, self.num_bins)(jnp.concatenate([z, a], -1))

class QEnsemble(nn.Module):
    hidden: tuple = (128, 128)
    num_bins: int = 101
    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], -1)
        return jnp.stack([NormMLP(self.hidden, self.num_bins)(x),
                          NormMLP(self.hidden, self.num_bins)(x)], -2)

class Pi(nn.Module):
    action_dim: int; hidden: tuple = (128, 128)
    log_std_min: float = -10.0
    log_std_dif: float = 12.0
    @nn.compact
    def __call__(self, z):
        x = NormMLP(self.hidden, self.action_dim * 2)(z)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = log_std_fn(log_std, self.log_std_min, self.log_std_dif)
        return mean, log_std


# ─── Buffer ───────────────────────────────────────────────────────────────────

class MultiEnvBuffer:
    def __init__(self, cap, n_envs, obs_dim, act_dim, seq_len):
        self.cap = cap; self.N = n_envs; self.T = seq_len
        self.obs  = np.zeros((n_envs, cap, obs_dim), np.float32)
        self.acts = np.zeros((n_envs, cap, act_dim), np.float32)
        self.rews = np.zeros((n_envs, cap), np.float32)
        self.done = np.zeros((n_envs, cap), np.float32)
        self.ptr  = np.zeros(n_envs, np.int64)
        self.size = np.zeros(n_envs, np.int64)

    def add_batch(self, obs_b, acts_b, rews_b, done_b):
        p = self.ptr
        self.obs[np.arange(self.N), p]  = obs_b
        self.acts[np.arange(self.N), p] = acts_b
        self.rews[np.arange(self.N), p] = rews_b
        self.done[np.arange(self.N), p] = done_b
        self.ptr  = (p + 1) % self.cap
        self.size = np.minimum(self.size + 1, self.cap)

    def total_size(self): return int(self.size.sum())

    def sample(self, B, rng):
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        env_ids = rng.choice(valid, size=B, replace=True)
        sizes   = self.size[env_ids]
        starts  = (rng.random(B) * (sizes - self.T)).astype(np.int64)
        idx     = starts[:, None] + np.arange(self.T)[None, :]
        return (self.obs[env_ids[:, None], idx],
                self.acts[env_ids[:, None], idx],
                self.rews[env_ids[:, None], idx],
                self.done[env_ids[:, None], idx])

    def sample_k(self, K, B, rng):
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        KB = K * B
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
        return (obs_kb.reshape(K, B, self.T, obs_dim),
                acts_kb.reshape(K, B, self.T, act_dim),
                rews_kb.reshape(K, B, self.T),
                done_kb.reshape(K, B, self.T))


# ─── Update fn (Phase 2: stochastic pi + target-Q-only EMA) ──────────────────

def make_update_fn(enc, dyn, rew_net, q_net, pi_net, wm_tx, pi_tx,
                   gamma=0.99, rho=0.5, tau=0.01, entropy_coef=1e-4):
    """Phase 2 update function.

    Key changes vs v13:
    1. pi loss uses stochastic action (sampled) + entropy term
    2. tp only EMAs q params (not enc/dyn/rew/pi)
    3. Separate wm_tx (world model) and pi_tx (actor) optimizers
    """

    def loss_fn(params, tp_q, obs_b, act_b, rew_b, done_b, rng):
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B * T, -1)).reshape(B, T, -1)
        z0    = z_all[:, 0]
        acts_T = jnp.transpose(act_b[:, :-1], (1, 0, 2))

        def dyn_step(z, a): return dyn.apply(params["dyn"], z, a), z
        z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
        zs = jnp.concatenate([jnp.transpose(zs_prefix, (1, 0, 2)), z_final[:, None, :]], 1)

        z_tgt   = jax.lax.stop_gradient(z_all)
        weights = jnp.array([rho ** t for t in range(T - 1)])
        z_t_T   = jnp.transpose(zs[:, :-1],  (1, 0, 2))
        a_T     = acts_T
        r_T     = jnp.transpose(rew_b[:, :-1],  (1, 0))
        d_T     = jnp.transpose(done_b[:, :-1], (1, 0))
        z_t1_T  = jnp.transpose(z_tgt[:, 1:],   (1, 0, 2))
        zs_t1_T = jnp.transpose(zs[:, 1:],      (1, 0, 2))

        def step_loss(carry, inp):
            k, w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp
            # Consistency loss
            cl  = w * jnp.mean(jnp.sum((zs_t1 - z_tgt_t1) ** 2, -1))
            # Reward loss
            pr  = rew_net.apply(params["rew"], z_t, a_t)
            rl  = w * jnp.mean(soft_ce(pr, two_hot(r_t)))

            # TD target using TARGET Q only (Phase 2 change)
            z_n = jax.lax.stop_gradient(z_tgt_t1)
            k, sk1 = jax.random.split(k)
            # Sample stochastic action from pi for next-state value estimate
            pi_mean_n, pi_log_std_n = pi_net.apply(params["pi"], z_n)
            eps_n = jax.random.normal(sk1, pi_mean_n.shape)
            pi_pre_n = pi_mean_n + eps_n * jnp.exp(pi_log_std_n)
            pi_a_n, lp_n = squash_logprob(pi_pre_n, gaussian_logprob(eps_n, pi_log_std_n))
            # Use TARGET Q (tp_q) for TD target — Phase 2 change
            q_next_logits = q_net.apply(tp_q, z_n, jax.lax.stop_gradient(pi_a_n))
            q_next_vals = two_hot_inv(q_next_logits)
            v_n  = jnp.maximum(jnp.min(q_next_vals, -1), 0.0)
            td   = r_t + gamma * (1 - d_t) * jax.lax.stop_gradient(v_n)
            qp   = q_net.apply(params["q"], z_t, a_t)
            vl   = w * jnp.mean(jnp.mean(soft_ce(qp, two_hot(td)[:, None, :]), -1))  # mean over Q heads

            # Pi loss: stochastic action + entropy (Phase 2 change)
            k, sk2 = jax.random.split(k)
            z_t_sg = jax.lax.stop_gradient(z_t)
            pi_mean_t, pi_log_std_t = pi_net.apply(params["pi"], z_t_sg)
            eps_t = jax.random.normal(sk2, pi_mean_t.shape)
            pi_pre_t = pi_mean_t + eps_t * jnp.exp(pi_log_std_t)
            pi_a_t, lp_t = squash_logprob(pi_pre_t, gaussian_logprob(eps_t, pi_log_std_t))
            q_pi_logits = q_net.apply(jax.lax.stop_gradient(params["q"]),
                                      z_t_sg, pi_a_t)
            q_pi_vals = two_hot_inv(q_pi_logits)
            # entropy = -log_prob (stochastic, with tanh correction via squash_logprob)
            entropy = -lp_t.squeeze(-1)   # (B,)
            pl   = -w * jnp.mean(jnp.min(q_pi_vals, -1) + entropy_coef * entropy)

            return k, (cl, rl, vl, pl)

        keys = jax.random.split(rng, T - 1)
        _, (cls, rls, vls, pls) = jax.lax.scan(
            step_loss, rng, (keys, weights, z_t_T, a_T, r_T, d_T, z_t1_T, zs_t1_T))

        n = T - 1
        # Official loss coefficients (Phase 3): consistency dominant
        # pi term kept at coef=1.0 to match official pi learning rate scale
        # (pi grads separated to pi_opt; wm grads only see enc/dyn/rew/q via SG structure)
        return (20.0 * jnp.sum(cls) + 0.1 * jnp.sum(rls) + 0.1 * jnp.sum(vls) + 1.0 * jnp.sum(pls)) / n, \
               {"c": jnp.sum(cls)/n, "r": jnp.sum(rls)/n,
                "v": jnp.sum(vls)/n, "p": jnp.sum(pls)/n}

    @jax.jit
    def single_step(params, tp_q, wm_opt, pi_opt, ob, ab, rb, db, rng):
        """One gradient step for both world model and pi (compile/debug use)."""
        val_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = val_and_grad(params, tp_q, ob, ab, rb, db, rng)
        # World model update (enc, dyn, rew, q)
        wm_grads = {k: grads[k] for k in ["enc", "dyn", "rew", "q"]}
        wm_upd, new_wm_opt = wm_tx.update(wm_grads, wm_opt, {k: params[k] for k in wm_grads})
        # Pi update
        pi_grads = {"pi": grads["pi"]}
        pi_upd, new_pi_opt = pi_tx.update(pi_grads, pi_opt, {"pi": params["pi"]})
        # Apply updates
        wm_updates = {k: wm_upd[k] for k in wm_grads}
        new_params = optax.apply_updates(params, {**wm_updates, "pi": pi_upd["pi"]})
        # EMA target Q only (Phase 2: not full tp)
        new_tp_q = jax.tree_util.tree_map(lambda t, p: (1 - tau)*t + tau*p, tp_q, new_params["q"])
        return new_params, new_tp_q, new_wm_opt, new_pi_opt, loss, aux

    @jax.jit
    def multi_step(params, tp_q, wm_opt, pi_opt, all_obs, all_acts, all_rews, all_done, key):
        """K gradient steps via lax.scan (Phase 4 fused loop)."""
        def one_step(carry, batch):
            params, tp_q, wm_opt, pi_opt, key = carry
            ob, ab, rb, db = batch
            key, uk = jax.random.split(key)
            val_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = val_and_grad(params, tp_q, ob, ab, rb, db, uk)
            # World model update
            wm_grads = {k: grads[k] for k in ["enc", "dyn", "rew", "q"]}
            wm_upd, new_wm_opt = wm_tx.update(wm_grads, wm_opt, {k: params[k] for k in wm_grads})
            # Pi update
            pi_grads = {"pi": grads["pi"]}
            pi_upd, new_pi_opt = pi_tx.update(pi_grads, pi_opt, {"pi": params["pi"]})
            # Apply
            new_params = optax.apply_updates(params, {**{k: wm_upd[k] for k in wm_grads},
                                                       "pi": pi_upd["pi"]})
            # EMA target Q only
            new_tp_q = jax.tree_util.tree_map(
                lambda t, p: (1 - tau)*t + tau*p, tp_q, new_params["q"])
            return (new_params, new_tp_q, new_wm_opt, new_pi_opt, key), (loss, aux)

        batches = (all_obs, all_acts, all_rews, all_done)
        (params, tp_q, wm_opt, pi_opt, key), (losses, auxs) = jax.lax.scan(
            one_step, (params, tp_q, wm_opt, pi_opt, key), batches)
        last_aux = jax.tree_util.tree_map(lambda x: x[-1], auxs)
        return params, tp_q, wm_opt, pi_opt, key, losses[-1], last_aux

    return single_step, multi_step


# ─── MPPI fn (Phase 1: official-parity — unchanged from v13) ─────────────────

def make_mppi_fn(enc, dyn, rew_net, q_net, pi_net,
                 horizon=3, n_samples=512, num_elites=64, num_pi_trajs=24,
                 n_iter=6, min_std=0.05, max_std=2.0,
                 act_low=-1.0, act_high=1.0, act_dim=4,
                 gamma=0.99):
    n_noise = n_samples - num_pi_trajs
    _gammas  = jnp.array([gamma**t for t in range(horizon)])
    _gamma_H = float(gamma**horizon)

    @jax.jit
    def plan(params, tp_q, obs, mu, std, key, t0):
        mu  = jnp.where(t0, jnp.zeros_like(mu),  mu)
        std = jnp.where(t0, jnp.full_like(std, max_std), std)

        z0_single = enc.apply(params["enc"], obs[None])[0]

        def pi_rollout_stoch(key):
            def pi_step(z, k):
                mean_a, log_std_a = pi_net.apply(params["pi"], z[None])
                mean_a = mean_a[0]; log_std_a = log_std_a[0]
                eps = jax.random.normal(k, mean_a.shape)
                a = jnp.tanh(mean_a + eps * jnp.exp(log_std_a))
                z2 = dyn.apply(params["dyn"], z[None], a[None])[0]
                return z2, a
            keys_h = jax.random.split(key, horizon)
            _, traj = jax.lax.scan(pi_step, z0_single, keys_h)
            return traj

        key, pk = jax.random.split(key)
        pi_keys = jax.random.split(pk, num_pi_trajs)
        pi_trajs = jax.vmap(pi_rollout_stoch)(pi_keys)

        def one_iter(carry, _):
            mu_i, std_i, k = carry
            k, sk = jax.random.split(k)
            noise = jax.random.normal(sk, (n_noise, horizon, act_dim)) * std_i[None]
            noise_acts = jnp.clip(mu_i[None] + noise, act_low, act_high)
            acts = jnp.concatenate([pi_trajs, noise_acts], axis=0)
            z0_batch = jnp.tile(z0_single[None], (n_samples, 1))

            def rollout_one(z_i, a_seq):
                def env_step(z, a):
                    r_logits = rew_net.apply(params["rew"], z[None], a[None])
                    r = two_hot_inv(r_logits).squeeze()
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r
                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                pi_a_squashed = jnp.tanh(pi_a_mean)
                # Use tp_q (target Q) for terminal value in planning
                q_logits = q_net.apply(tp_q, zf[None], pi_a_squashed)
                vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                return jnp.sum(_gammas * rs) + _gamma_H * vt

            rets = jax.vmap(rollout_one)(z0_batch, acts)
            _, elite_idx = jax.lax.top_k(rets, num_elites)
            elite_acts = acts[elite_idx]
            new_mu  = jnp.mean(elite_acts, axis=0)
            new_std = jnp.std(elite_acts, axis=0) + 1e-6
            new_std = jnp.clip(new_std, min_std, max_std)
            return (new_mu, new_std, k), None

        (muf, stdf, _), _ = jax.lax.scan(one_iter, (mu, std, key), None, length=n_iter)
        action = jnp.clip(muf[0], act_low, act_high)
        new_mu  = jnp.concatenate([muf[1:], jnp.zeros((1, act_dim))], axis=0)
        new_std = jnp.concatenate([stdf[1:], jnp.full((1, act_dim), max_std)], axis=0)
        return action, new_mu, new_std

    return plan


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from mujoco_playground import registry, wrapper

    SEED            = 42
    N_ENVS          = 256
    TOTAL_ENV       = 4_000_000
    WARMUP_ENV      = 25_000
    BS              = 256
    H               = 3
    SEQ             = H + 1
    K_UPDATE        = 64
    LR              = 3e-4
    LATENT          = 128
    HIDDEN          = (128, 128)
    GAMMA           = 0.99
    TAU             = 0.01
    NS              = 512
    NUM_ELITES      = 64
    NUM_PI_TRAJS    = 24
    NI              = 6
    MIN_STD         = 0.05
    MAX_STD         = 2.0
    ENTROPY_COEF    = 1e-4    # Phase 2: entropy in actor loss
    EXPL_NOISE      = 0.3
    EXPL_UNTIL      = 25_000
    PI_EVAL_EVERY   = 250_000
    MPPI_EVAL_EVERY = 250_000
    EVAL_EPS        = 3

    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    env_raw = registry.load("HopperHop")
    env     = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)
    obs_dim = env.observation_size
    act_dim = env.action_size
    al, ah  = -1.0, 1.0
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  N_ENVS={N_ENVS}", flush=True)
    print(f"Phase3: consistency_coef=20  reward_coef=0.1  value_coef=0.1  (official)", flush=True)
    print(f"Phase2: entropy_coef={ENTROPY_COEF}  target-Q-only-EMA  separate-pi-opt", flush=True)
    print(f"Phase1: H={H}  NS={NS}  elites={NUM_ELITES}  pi_trajs={NUM_PI_TRAJS}", flush=True)

    @jax.jit
    def batch_reset(key):
        return env.reset(jax.random.split(key, N_ENVS))

    @jax.jit
    def batch_step(state, action):
        return env.step(state, action)

    # ── Models ──────────────────────────────────────────────────────────────
    enc = Encoder(LATENT, HIDDEN)
    dyn = Dynamics(LATENT, HIDDEN)
    rn  = RewardHead(HIDDEN)
    qn  = QEnsemble(HIDDEN)
    pn  = Pi(act_dim, HIDDEN)

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    do = jnp.zeros((1, obs_dim)); dz = jnp.zeros((1, LATENT)); da = jnp.zeros((1, act_dim))
    params = {"enc": enc.init(k1, do), "dyn": dyn.init(k2, dz, da),
              "rew": rn.init(k3, dz, da), "q": qn.init(k4, dz, da), "pi": pn.init(k5, dz)}

    # Phase 2: EMA only q params
    tp_q = jax.tree_util.tree_map(lambda x: x, params["q"])

    # Phase 2: Separate optimizers
    # World model (enc+dyn+rew+q): AdamW with weight decay
    wm_tx = optax.multi_transform(
        {'enc': optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(LR, weight_decay=1e-4)),
         'dyn': optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(LR, weight_decay=1e-4)),
         'rew': optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(LR, weight_decay=1e-4)),
         'q':   optax.chain(optax.clip_by_global_norm(1.0),  optax.adam(LR))},
        {'enc': 'enc', 'dyn': 'dyn', 'rew': 'rew', 'q': 'q'})
    wm_opt = wm_tx.init({k: params[k] for k in ["enc", "dyn", "rew", "q"]})

    # Pi: Adam with tighter gradient clipping
    pi_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    pi_opt = pi_tx.init({"pi": params["pi"]})

    single_upd, multi_upd = make_update_fn(enc, dyn, rn, qn, pn, wm_tx, pi_tx,
                                            GAMMA, tau=TAU, entropy_coef=ENTROPY_COEF)
    plan = make_mppi_fn(enc, dyn, rn, qn, pn, H, NS, NUM_ELITES, NUM_PI_TRAJS, NI,
                        MIN_STD, MAX_STD, al, ah, act_dim, GAMMA)

    @jax.jit
    def act_fn_single(params, obs):
        z = enc.apply(params["enc"], obs[None])
        mean, _ = pn.apply(params["pi"], z)
        return jnp.tanh(mean[0])

    @jax.jit
    def act_fn_batch(params, obs_batch):
        z = enc.apply(params["enc"], obs_batch)
        mean, _ = pn.apply(params["pi"], z)
        return jnp.tanh(mean)

    # ── Buffer ──────────────────────────────────────────────────────────────
    cap_per_env = max(12_000, TOTAL_ENV // N_ENVS + 2000)
    buf = MultiEnvBuffer(cap_per_env, N_ENVS, obs_dim, act_dim, SEQ)
    rng = np.random.default_rng(SEED)

    out_dir = "/workspace/helios-rl/exp/tdmpc_dmc"
    os.makedirs(out_dir, exist_ok=True)
    csv = f"{out_dir}/hopper-hop-v15.csv"
    with open(csv, "w") as f:
        f.write("step,reward,eval_type,seed\n")

    # ── Warmup ──────────────────────────────────────────────────────────────
    print("Warmup...", flush=True)
    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    obs_np = np.array(env_state.obs)
    env_steps = 0

    while env_steps < WARMUP_ENV:
        acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        env_state = batch_step(env_state, jnp.asarray(acts_np))
        buf.add_batch(obs_np, acts_np, np.array(env_state.reward),
                      np.array(env_state.done > 0.5, np.float32))
        obs_np = np.array(env_state.obs)
        env_steps += N_ENVS

    print(f"Warmup done. Buffer={buf.total_size()}  Compiling...", flush=True)

    # ── Compile ──────────────────────────────────────────────────────────────
    t_c = time.time()
    samp = buf.sample(BS, rng)
    ob, ab, rb, db = [jnp.asarray(x) for x in samp]
    key, uk = jax.random.split(key)
    params, tp_q, wm_opt, pi_opt, loss, _ = single_upd(
        params, tp_q, wm_opt, pi_opt, ob, ab, rb, db, uk)
    jax.block_until_ready(params["enc"])
    print(f"single_upd compiled in {time.time()-t_c:.1f}s  loss={float(loss):.4f}", flush=True)

    t_c = time.time()
    samp_k = buf.sample_k(K_UPDATE, BS, rng)
    ob_k, ab_k, rb_k, db_k = [jnp.asarray(x) for x in samp_k]
    key, uk2 = jax.random.split(key)
    params, tp_q, wm_opt, pi_opt, key, loss, aux = multi_upd(
        params, tp_q, wm_opt, pi_opt, ob_k, ab_k, rb_k, db_k, key)
    jax.block_until_ready(params["enc"])
    print(f"multi_upd (scan K={K_UPDATE}) compiled in {time.time()-t_c:.1f}s", flush=True)

    t_c = time.time()
    key, ek = jax.random.split(key)
    mu_e  = jnp.zeros((H, act_dim))
    std_e = jnp.full((H, act_dim), MAX_STD)
    act_e, _, _ = plan(params, tp_q, jnp.asarray(obs_np[0]), mu_e, std_e, ek, jnp.bool_(True))
    jax.block_until_ready(act_e)
    print(f"MPPI compiled in {time.time()-t_c:.1f}s", flush=True)

    # ── Eval helpers ────────────────────────────────────────────────────────
    @jax.jit
    def eval_reset(key):
        return env.reset(jax.random.split(key, 1))

    @jax.jit
    def eval_step(state, act):
        return env.step(state, act[None])

    def eval_pi_ep(key):
        key, rk = jax.random.split(key)
        state = eval_reset(rk)
        obs = jnp.asarray(state.obs[0])
        er = 0.0
        for _ in range(1000):
            act = act_fn_single(params, obs)
            state = eval_step(state, act)
            er += float(state.reward[0])
            if bool(state.done[0] > 0.5): break
            obs = jnp.asarray(state.obs[0])
        return er, key

    def eval_mppi_ep(key):
        key, rk = jax.random.split(key)
        state = eval_reset(rk)
        obs = jnp.asarray(state.obs[0])
        mu  = jnp.zeros((H, act_dim)); std = jnp.full((H, act_dim), MAX_STD)
        er = 0.0; t0 = jnp.bool_(True)
        key, ek = jax.random.split(key)
        for _ in range(1000):
            act, mu, std = plan(params, tp_q, obs, mu, std, ek, t0)
            t0 = jnp.bool_(False)
            state = eval_step(state, act)
            key, ek = jax.random.split(key)
            er += float(state.reward[0])
            if bool(state.done[0] > 0.5): break
            obs = jnp.asarray(state.obs[0])
        return er, key

    def eval_pi(n_eps):
        nonlocal key
        return np.mean([eval_pi_ep(key)[0] for _ in range(n_eps)])

    def eval_mppi(n_eps):
        nonlocal key
        rets = []
        for _ in range(n_eps):
            r, key = eval_mppi_ep(key)
            rets.append(r)
        return np.mean(rets)

    # ── Training loop ────────────────────────────────────────────────────────
    print("Training (Phase 3: official loss coefs + Phase 2 actor fixes)...", flush=True)
    t0_wall = time.time()
    log_every = N_ENVS * 4

    while env_steps < TOTAL_ENV:
        if env_steps < EXPL_UNTIL:
            acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        else:
            acts_jax = act_fn_batch(params, jnp.asarray(obs_np))
            noise    = np.random.normal(0, EXPL_NOISE, (N_ENVS, act_dim))
            acts_np  = np.clip(np.array(acts_jax) + noise, al, ah).astype(np.float32)

        env_state = batch_step(env_state, jnp.asarray(acts_np))
        new_obs   = np.array(env_state.obs)
        rews_np   = np.array(env_state.reward)
        done_np   = np.array(env_state.done > 0.5, np.float32)
        buf.add_batch(obs_np, acts_np, rews_np, done_np)
        obs_np = new_obs
        env_steps += N_ENVS

        samp_k = buf.sample_k(K_UPDATE, BS, rng)
        if samp_k is not None:
            ob_k, ab_k, rb_k, db_k = [jnp.asarray(x) for x in samp_k]
            params, tp_q, wm_opt, pi_opt, key, loss, aux = multi_upd(
                params, tp_q, wm_opt, pi_opt, ob_k, ab_k, rb_k, db_k, key)

        if env_steps % log_every < N_ENVS:
            elapsed = time.time() - t0_wall
            print(f"  es={env_steps:>11,}  sps={env_steps/elapsed:.0f}  loss={float(loss):.4f}", flush=True)

        if env_steps % PI_EVAL_EVERY < N_ENVS:
            pi_ret = eval_pi(EVAL_EPS)
            elapsed = time.time() - t0_wall
            print(f"step={env_steps:>11,}  pi={pi_ret:7.1f}  sps={env_steps/elapsed:.0f}  "
                  f"c={float(aux['c']):.3f} r={float(aux['r']):.3f} "
                  f"v={float(aux['v']):.3f} p={float(aux['p']):.3f}", flush=True)
            with open(csv, "a") as f:
                f.write(f"{env_steps},{pi_ret:.1f},pi,{SEED}\n")

        if env_steps % MPPI_EVAL_EVERY < N_ENVS:
            mr = eval_mppi(EVAL_EPS)
            elapsed = time.time() - t0_wall
            print(f"step={env_steps:>11,}  MPPI={mr:7.1f}  sps={env_steps/elapsed:.0f}", flush=True)
            with open(csv, "a") as f:
                f.write(f"{env_steps},{mr:.1f},mppi,{SEED}\n")

    elapsed = time.time() - t0_wall
    print(f"\nDone in {elapsed:.0f}s  ({elapsed/3600:.2f}h)  ->  {csv}", flush=True)


if __name__ == "__main__":
    main()
