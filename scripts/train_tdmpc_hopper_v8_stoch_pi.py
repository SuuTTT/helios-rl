"""Train TD-MPC2 on mujoco_playground HopperHop (JAX vectorised, fast iteration).

v4 design targets <30 min full run:
  - N_ENVS=1024  → 82k env-sps in pure env stepping (vs dm_control 92 sps)
  - HIDDEN=(128,128) → ~4x faster updates than (256,256)
  - K=64 updates per global step → UTD=64/1024=1/16
  - TOTAL_ENV=10M  → ~18 min wall time, ~625k gradient steps
  - Vectorised numpy buffer (no per-sample Python loops)
  - REW_SCALE=10.0 → reward head gets 100x larger gradient signal
  - SimNorm on encoder+dynamics → bounded latents, stable c<0.05
  - Eval on JAX env directly (no dm_control, no obs mismatch)
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

# --- Stoch Math & Scale ---
def log_std_fn(x, low=-10.0, dif=12.0):
    return low + 0.5 * dif * (jnp.tanh(x) + 1.0)

def gaussian_logprob(eps, log_std):
    residual = -0.5 * (eps ** 2) - log_std
    log_prob = residual - 0.9189385175704956
    return jnp.sum(log_prob, axis=-1, keepdims=True)

def squash(mu, pi, log_pi):
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    squashed_pi = jnp.log(jax.nn.relu(1 - pi**2) + 1e-6)
    log_pi = log_pi - jnp.sum(squashed_pi, axis=-1, keepdims=True)
    return mu, pi, log_pi

def q_percentile(q_vals):
    b, t, d = q_vals.shape
    q_flat = q_vals.reshape(-1)
    p5 = jnp.percentile(q_flat, 5.0)
    p95 = jnp.percentile(q_flat, 95.0)
    return jnp.maximum(p95 - p5, 1.0)

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
        
def sample_pi(params, z, key):
    mean, log_std = params
    eps = jax.random.normal(key, mean.shape)
    log_prob = gaussian_logprob(eps, log_std)
    size = eps.shape[-1]
    scaled_log_prob = log_prob * size
    action_pre = mean + eps * jnp.exp(log_std)
    mu, action, log_prob = squash(mean, action_pre, log_prob)
    entropy_scale = scaled_log_prob / (log_prob + 1e-8)
    scaled_entropy = -log_prob * entropy_scale
    return action, log_prob, scaled_entropy


# ─── Vectorised multi-env buffer ──────────────────────────────────────────────
# Stores N independent per-env ring buffers; sampling is fully vectorised numpy.

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
        p = self.ptr  # (N,)
        self.obs[np.arange(self.N), p] = obs_b
        self.acts[np.arange(self.N), p] = acts_b
        self.rews[np.arange(self.N), p] = rews_b
        self.done[np.arange(self.N), p] = done_b
        self.ptr = (p + 1) % self.cap
        self.size = np.minimum(self.size + 1, self.cap)

    def total_size(self): return int(self.size.sum())

    def sample(self, B, rng):
        valid = np.where(self.size >= self.T + 1)[0]
        if len(valid) == 0:
            return None
        # Fully-vectorised: no Python loops over B
        env_ids = rng.choice(valid, size=B, replace=True)             # (B,)
        sizes   = self.size[env_ids]                                   # (B,)
        starts  = (rng.random(B) * (sizes - self.T)).astype(np.int64) # (B,)
        idx     = starts[:, None] + np.arange(self.T)[None, :]        # (B, T)
        return (self.obs[env_ids[:, None], idx],   # (B, T, obs_dim)
                self.acts[env_ids[:, None], idx],  # (B, T, act_dim)
                self.rews[env_ids[:, None], idx],  # (B, T)
                self.done[env_ids[:, None], idx])  # (B, T)


# ─── Update fn ────────────────────────────────────────────────────────────────

def make_update_fn(enc, dyn, rew_net, q_net, pi_net, tx,
                   gamma=0.99, rho=0.5, tau=0.01, rew_scale=10.0):
    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b, rng, scale_val):
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
            cl  = w * jnp.mean(jnp.sum((zs_t1 - z_tgt_t1) ** 2, -1))
            pr  = rew_net.apply(params["rew"], z_t, a_t)
            rl  = w * jnp.mean(soft_ce(pr, two_hot(r_t)))
            
            z_n = jax.lax.stop_gradient(z_tgt_t1)
            k, sk1, sk2 = jax.random.split(k, 3)
            
            # Target Policy — use mean action (not sampled) for low-variance TD targets
            tp_mean_std = pi_net.apply(tp["pi"], z_n)
            pi_a_mean, _ = tp_mean_std          # deterministic mean for target
            pi_a_mean = jnp.tanh(pi_a_mean)    # squash to [-1,1]
            
            q_next_logits = q_net.apply(tp["q"], z_n, pi_a_mean)
            q_next_vals = two_hot_inv(q_next_logits)
            v_n  = jnp.maximum(jnp.min(q_next_vals, -1), 0.0)
            td   = r_t + gamma * (1 - d_t) * jax.lax.stop_gradient(v_n)
            qp   = q_net.apply(params["q"], z_t, a_t)
            vl   = w * jnp.mean(jnp.sum(soft_ce(qp, two_hot(td)[:, None, :]), -1))
            
            # Policy update — use MEAN action for Q (clean gradient, like v5/TD3),
            # sample separately only for entropy term (avoids noisy Q gradient from sampled action).
            pi_mean_std = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_t))
            pi2_mean = jnp.tanh(pi_mean_std[0])           # deterministic mean, squashed
            _, _, ent = sample_pi(pi_mean_std, jax.lax.stop_gradient(z_t), sk2)  # entropy only
            q_pi2_logits = q_net.apply(jax.lax.stop_gradient(params["q"]), jax.lax.stop_gradient(z_t), pi2_mean)
            q_pi2_vals = two_hot_inv(q_pi2_logits)
            
            # Running scale downscaling and entropy coef
            # entropy_coef=0.002: small but non-zero entropy bonus for log_std learning
            q_pi2_vals = q_pi2_vals / scale_val
            entropy_coef = 0.002
            pl   = -w * jnp.mean(jnp.min(q_pi2_vals, -1) + entropy_coef * ent.squeeze(-1))
            
            return k, (cl, rl, vl, pl, q_pi2_vals)

        keys = jax.random.split(rng, T - 1)
        _, (cls, rls, vls, pls, q_pi2_vals_T) = jax.lax.scan(
            step_loss, rng, (keys, weights, z_t_T, a_T, r_T, d_T, z_t1_T, zs_t1_T))
            
        n = T - 1
        return (2 * jnp.sum(cls) + 2 * jnp.sum(rls) + jnp.sum(vls) + 0.1 * jnp.sum(pls)) / n, \
               {"c": jnp.sum(cls)/n, "r": jnp.sum(rls)/n,
                "v": jnp.sum(vls)/n, "p": jnp.sum(pls)/n}

    @jax.jit
    def step(params, tp, opt, ob, ab, rb, db, rng, scale_val):
        val_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = val_and_grad(params, tp, ob, ab, rb, db, rng, scale_val)
        upd, nopt = tx.update(grads, opt, params)
        new_params = optax.apply_updates(params, upd)
        new_tp = jax.tree_util.tree_map(lambda t, p: (1 - tau)*t + tau*p, tp, new_params)
        return new_params, new_tp, nopt, loss, aux
    return step


# ─── MPPI fn ──────────────────────────────────────────────────────────────────

def make_mppi_fn(enc, dyn, rew_net, q_net, pi_net,
                 horizon=5, n_samples=256, n_iter=6, temp=0.5,
                 act_low=-1.0, act_high=1.0, act_dim=4,
                 gamma=0.99, rew_scale=10.0):
    _gammas  = jnp.array([gamma**t for t in range(horizon)])
    _gamma_H = float(gamma**horizon)

    @jax.jit
    def plan(params, obs, mu, key):
        z0_single = enc.apply(params["enc"], obs[None])[0]      # (latent,)
        z0 = jnp.tile(z0_single[None], (n_samples, 1))           # (N, latent)

        # Compute pi-guided trajectory from current state.
        def pi_step(z, _):
            mean_a, _  = pi_net.apply(params["pi"], z[None])
            a = mean_a[0]
            z2 = dyn.apply(params["dyn"], z[None], a[None])[0]
            return z2, a
        _, pi_traj = jax.lax.scan(pi_step, z0_single, None, length=horizon)  # (H, act_dim)

        mu_ws = mu.at[0].set(pi_traj[0])

        def one_iter(carry, _):
            mu_i, k = carry
            k, sk = jax.random.split(k)
            noise = jax.random.normal(sk, (n_samples, horizon, act_dim)) * 0.5
            acts  = jnp.clip(mu_i[None] + noise, act_low, act_high)
            acts  = acts.at[-1].set(pi_traj)

            def rollout_one(args):
                z_i, a_seq = args
                def env_step(z, a):
                    r_logits = rew_net.apply(params["rew"], z[None], a[None])
                    r = two_hot_inv(r_logits).squeeze()
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r
                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                q_logits = q_net.apply(params["q"], zf[None], pi_a_mean)
                vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                return jnp.sum(_gammas * rs) + _gamma_H * vt

            rets  = jax.vmap(rollout_one)((z0, acts))
            w     = jax.nn.softmax((rets - rets.max()) / (temp + 1e-8))
            new_mu = jnp.einsum("n,nha->ha", w, acts)
            return (new_mu, k), None

        (muf, _), _ = jax.lax.scan(one_iter, (mu_ws, key), None, length=n_iter)
        action  = jnp.clip(muf[0], act_low, act_high)
        new_mu  = jnp.concatenate([muf[1:], pi_traj[-1:]], 0)
        return action, new_mu
    return plan


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from mujoco_playground import registry, wrapper

    SEED            = 42
    N_ENVS          = 256         # parallel JAX envs
    TOTAL_ENV       = 4_000_000  # total env-steps 
    WARMUP_ENV      = 25_000      # env-steps before training
    BS              = 256
    SEQ             = 6
    K_UPDATE        = 256         # gradient steps per global step; UTD=256/256=1:1
    LR              = 3e-4
    LATENT          = 128
    HIDDEN          = (128, 128)  # 4x faster updates vs (256,256)
    GAMMA           = 0.99
    TAU             = 0.01        # target network EMA coefficient
    H               = 5
    NS              = 256
    NI              = 6
    TEMP            = 0.5
    REW_SCALE       = 10.0        # scale reward targets → strong reward-head gradient
    #                              # Q targets capped at 500 (=rew_scale×0.5/0.01)
    EXPL_NOISE      = 0.3
    EXPL_UNTIL      = 25_000

    PI_EVAL_EVERY   = 500_000     # env_steps between pi evals
    MPPI_EVAL_EVERY = 1_000_000   # env_steps between MPPI evals
    EVAL_EPS        = 5

    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    # ── JAX vectorised env ──────────────────────────────────────────────────
    env_raw = registry.load("HopperHop")
    env     = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)
    obs_dim = env.observation_size
    act_dim = env.action_size
    al, ah  = -1.0, 1.0
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  N_ENVS={N_ENVS}  K_UPDATE={K_UPDATE}", flush=True)

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
    tp = jax.tree_util.tree_map(lambda x: x, params)

    _labels = {'enc': 'world', 'dyn': 'world', 'rew': 'world', 'q': 'q', 'pi': 'pi'}
    tx = optax.multi_transform(
        {'world': optax.chain(optax.clip_by_global_norm(10.0), optax.adam(LR)),
         'q':     optax.chain(optax.clip_by_global_norm(1.0),  optax.adam(LR)),
         'pi':    optax.chain(optax.clip_by_global_norm(1.0),  optax.adam(LR))},
        _labels)
    opt = tx.init(params)

    upd  = make_update_fn(enc, dyn, rn, qn, pn, tx, GAMMA, tau=TAU, rew_scale=REW_SCALE)
    plan = make_mppi_fn(enc, dyn, rn, qn, pn, H, NS, NI, TEMP, al, ah, act_dim, GAMMA, REW_SCALE)

    @jax.jit
    def act_fn_single(params, obs):
        z = enc.apply(params["enc"], obs[None])
        mean, _ = pn.apply(params["pi"], z)
        return jnp.tanh(mean[0])  # squash to [-1,1] matching training

    @jax.jit
    def act_fn_batch(params, obs_batch):
        z = enc.apply(params["enc"], obs_batch)
        mean, _ = pn.apply(params["pi"], z)
        return jnp.tanh(mean)  # squash to [-1,1] matching training

    # ── Buffer ──────────────────────────────────────────────────────────────
    cap_per_env = max(12_000, TOTAL_ENV // N_ENVS + 2000)
    buf = MultiEnvBuffer(cap_per_env, N_ENVS, obs_dim, act_dim, SEQ)
    rng = np.random.default_rng(SEED)

    out_dir = "/workspace/helios-rl/exp/tdmpc_dmc"
    os.makedirs(out_dir, exist_ok=True)
    csv = f"{out_dir}/hopper-hop-twohot-v8.csv"
    with open(csv, "w") as f:
        f.write("step,reward,seed\n")

    # ── Warmup ──────────────────────────────────────────────────────────────
    print("Warmup...", flush=True)
    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    obs_np = np.array(env_state.obs)
    env_steps = 0

    while env_steps < WARMUP_ENV:
        acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        env_state = batch_step(env_state, jnp.asarray(acts_np))
        buf.add_batch(obs_np, acts_np, np.array(env_state.reward), np.array(env_state.done > 0.5, np.float32))
        obs_np = np.array(env_state.obs)
        env_steps += N_ENVS

    print(f"Warmup done. Buffer={buf.total_size()}  Compiling...", flush=True)

    # ── Compile ─────────────────────────────────────────────────────────────
    t_c = time.time()
    samp = buf.sample(BS, rng)
    ob, ab, rb, db = [jnp.asarray(x) for x in samp]
    key, uk = jax.random.split(key)
    params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)
    jax.block_until_ready(params["enc"])
    print(f"Update compiled in {time.time()-t_c:.1f}s  loss={float(loss):.4f}", flush=True)

    key, ek = jax.random.split(key)
    mu_e = jnp.zeros((H, act_dim))
    dummy_obs = jnp.asarray(obs_np[0])
    act_e, _ = plan(params, dummy_obs, mu_e, ek)
    jax.block_until_ready(act_e)
    print("MPPI compiled.", flush=True)

    # ── Eval helpers (on JAX env, single-episode stepping) ──────────────────
    @jax.jit
    def eval_reset(key):
        return env.reset(jax.random.split(key, 1))

    @jax.jit
    def eval_step(state, act):
        return env.step(state, act[None])

    def eval_pi_ep(key):
        """Run one episode with the policy, return total reward."""
        key, rk = jax.random.split(key)
        state = eval_reset(rk)
        obs = jnp.asarray(state.obs[0])
        er = 0.0
        for _ in range(1000):
            act = act_fn_single(params, obs)
            state = eval_step(state, act)
            er += float(state.reward[0])
            if bool(state.done[0] > 0.5):
                break
            obs = jnp.asarray(state.obs[0])
        return er, key

    def eval_pi(n_eps):
        nonlocal key
        rets = []
        for _ in range(n_eps):
            r, key = eval_pi_ep(key)
            rets.append(r)
        return np.mean(rets)

    def eval_mppi_ep(key):
        key, rk = jax.random.split(key)
        state = eval_reset(rk)
        obs = jnp.asarray(state.obs[0])
        mu = jnp.zeros((H, act_dim)); er = 0.0
        key, ek = jax.random.split(key)
        for _ in range(1000):
            act, mu = plan(params, obs, mu, ek)
            state = eval_step(state, act)
            key, ek = jax.random.split(key)
            er += float(state.reward[0])
            if bool(state.done[0] > 0.5):
                break
            obs = jnp.asarray(state.obs[0])
        return er, key

    def eval_mppi(n_eps):
        nonlocal key
        rets = []
        for _ in range(n_eps):
            r, key = eval_mppi_ep(key)
            rets.append(r)
        return np.mean(rets)

    # ── Training loop ────────────────────────────────────────────────────────
    print("Training...", flush=True)
    t0 = time.time()
    log_every = N_ENVS * 4

    while env_steps < TOTAL_ENV:
        # ---- collect N_ENVS steps ----
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

        # ---- K gradient updates to maintain reasonable UTD ----
        for _ in range(K_UPDATE):
            samp = buf.sample(BS, rng)
            if samp is not None:
                key, uk = jax.random.split(key)
                ob, ab, rb, db = [jnp.asarray(x) for x in samp]
                params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)

        # ---- logging ----
        if env_steps % log_every < N_ENVS:
            elapsed = time.time() - t0
            print(f"  es={env_steps:>11,}  sps={env_steps/elapsed:.0f}  loss={float(loss):.4f}", flush=True)

        if env_steps % PI_EVAL_EVERY < N_ENVS:
            pi_ret = eval_pi(EVAL_EPS)
            elapsed = time.time() - t0
            print(f"step={env_steps:>11,}  pi={pi_ret:7.1f}  sps={env_steps/elapsed:.0f}  "
                  f"c={float(aux['c']):.3f} r={float(aux['r']):.3f} "
                  f"v={float(aux['v']):.3f} p={float(aux['p']):.3f}", flush=True)

        if env_steps % MPPI_EVAL_EVERY < N_ENVS:
            mr = eval_mppi(EVAL_EPS)
            elapsed = time.time() - t0
            print(f"step={env_steps:>11,}  MPPI={mr:7.1f}  sps={env_steps/elapsed:.0f}", flush=True)
            with open(csv, "a") as f:
                f.write(f"{env_steps},{mr:.1f},{SEED}\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s  ({elapsed/3600:.2f}h)  ->  {csv}", flush=True)


if __name__ == "__main__":
    main()
