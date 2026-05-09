"""
Profiling script for JAX TD-MPC2 macro step timing.

Runs N_MACRO macro steps (= 1 env-step batch + K_UPDATE gradient updates each),
breaking down wall time into:
  - act_pi:    batched policy forward (act_fn_batch)
  - noise:     numpy exploration noise
  - env_step:  JAX batch_step + device→host obs transfer
  - buf_write: buffer.add_batch
  - buf_read:  buffer.sample per update call
  - h2d:       jnp.asarray (host-to-device transfer) per update call
  - update:    upd() GPU kernel time per update call
  - total:     full macro wall time

All GPU ops are flushed with jax.block_until_ready() so times are real.
MPPI is timed separately on a single call as a reference point.

Usage:
  MUJOCO_GL=egl python3 helios-rl/scripts/profile_tdmpc_timing.py 2>&1 | tee /tmp/tdmpc_timing.log
"""

import os, sys, time, collections
import numpy as np

sys.path.insert(0, "/workspace/wiki/learn_mujoco_playground/repo")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax, jax.numpy as jnp, optax
import flax.linen as nn

# ── Reuse model/buffer code from v10 by importing its components ──────────────
# Rather than importing the script (which calls main()), we reproduce only what
# we need: models, buffer, update_fn, mppi_fn. These are identical to v10.

# ─── Shared helpers (copy-minimised from v10) ─────────────────────────────────

def simnorm(x, V=8):
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1]//V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)

def log_std_fn(x, low=-10.0, dif=12.0):
    return low + 0.5*dif*(jnp.tanh(x)+1.0)

def gaussian_logprob(eps, log_std):
    return jnp.sum(-0.5*(eps**2)-log_std - 0.9189385175704956, axis=-1, keepdims=True)

def squash(mu, pi, log_pi):
    mu = jnp.tanh(mu); pi = jnp.tanh(pi)
    log_pi = log_pi - jnp.sum(jnp.log(jax.nn.relu(1-pi**2)+1e-6), axis=-1, keepdims=True)
    return mu, pi, log_pi

def symlog(x): return jnp.sign(x)*jnp.log(1+jnp.abs(x))
def symexp(x): return jnp.sign(x)*(jnp.exp(jnp.abs(x))-1)

def two_hot(x, vmin=-20, vmax=20, num_bins=101):
    x = jnp.clip(symlog(x), vmin, vmax)
    bsz = (vmax-vmin)/(num_bins-1)
    bi = (x-vmin)/bsz
    lo = jnp.floor(bi).astype(jnp.int32); hi = jnp.ceil(bi).astype(jnp.int32)
    pu = bi-lo; pl = 1.0-pu
    return jax.nn.one_hot(lo, num_bins)*pl[...,None] + jax.nn.one_hot(hi, num_bins)*pu[...,None]

def soft_ce(pred, target):
    return -jnp.sum(target*jax.nn.log_softmax(pred, axis=-1), axis=-1)

def two_hot_inv(logits, vmin=-20, vmax=20, num_bins=101):
    return symexp(jnp.sum(jax.nn.softmax(logits, axis=-1)*jnp.linspace(vmin, vmax, num_bins), axis=-1))

class NormMLP(nn.Module):
    dims: tuple; out: int
    @nn.compact
    def __call__(self, x):
        for d in self.dims:
            x = nn.Dense(d)(x); x = nn.LayerNorm()(x); x = nn.silu(x)
        return nn.Dense(self.out)(x)

class Encoder(nn.Module):
    latent_dim: int; hidden: tuple=(128,128); V: int=8
    @nn.compact
    def __call__(self, obs): return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)

class Dynamics(nn.Module):
    latent_dim: int; hidden: tuple=(128,128); V: int=8
    @nn.compact
    def __call__(self, z, a): return simnorm(NormMLP(self.hidden, self.latent_dim)(jnp.concatenate([z,a],-1)), self.V)

class RewardHead(nn.Module):
    hidden: tuple=(128,128); num_bins: int=101
    @nn.compact
    def __call__(self, z, a): return NormMLP(self.hidden, self.num_bins)(jnp.concatenate([z,a],-1))

class QEnsemble(nn.Module):
    hidden: tuple=(128,128); num_bins: int=101
    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z,a],-1)
        return jnp.stack([NormMLP(self.hidden, self.num_bins)(x),
                          NormMLP(self.hidden, self.num_bins)(x)], -2)

class Pi(nn.Module):
    action_dim: int; hidden: tuple=(128,128)
    @nn.compact
    def __call__(self, z):
        x = NormMLP(self.hidden, self.action_dim*2)(z)
        mean, log_std = jnp.split(x, 2, axis=-1)
        return mean, log_std_fn(log_std)

class MultiEnvBuffer:
    def __init__(self, cap, n_envs, obs_dim, act_dim, seq_len):
        self.cap=cap; self.N=n_envs; self.T=seq_len
        self.obs  = np.zeros((n_envs, cap, obs_dim), np.float32)
        self.acts = np.zeros((n_envs, cap, act_dim), np.float32)
        self.rews = np.zeros((n_envs, cap), np.float32)
        self.done = np.zeros((n_envs, cap), np.float32)
        self.ptr  = np.zeros(n_envs, np.int64)
        self.size = np.zeros(n_envs, np.int64)

    def add_batch(self, obs_b, acts_b, rews_b, done_b):
        p = self.ptr
        self.obs[np.arange(self.N), p] = obs_b
        self.acts[np.arange(self.N), p] = acts_b
        self.rews[np.arange(self.N), p] = rews_b
        self.done[np.arange(self.N), p] = done_b
        self.ptr  = (p+1) % self.cap
        self.size = np.minimum(self.size+1, self.cap)

    def total_size(self): return int(self.size.sum())

    def sample(self, B, rng):
        valid = np.where(self.size >= self.T+1)[0]
        if len(valid) == 0: return None
        env_ids = rng.choice(valid, size=B, replace=True)
        sizes   = self.size[env_ids]
        starts  = (rng.random(B)*(sizes-self.T)).astype(np.int64)
        idx     = starts[:,None]+np.arange(self.T)[None,:]
        return (self.obs[env_ids[:,None], idx],
                self.acts[env_ids[:,None], idx],
                self.rews[env_ids[:,None], idx],
                self.done[env_ids[:,None], idx])

def make_update_fn(enc, dyn, rew_net, q_net, pi_net, tx,
                   gamma=0.99, rho=0.5, tau=0.01, rew_scale=10.0):
    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b, rng, scale_val):
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B*T,-1)).reshape(B,T,-1)
        z0    = z_all[:,0]
        acts_T = jnp.transpose(act_b[:,:-1],(1,0,2))
        def dyn_step(z, a): return dyn.apply(params["dyn"], z, a), z
        z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
        zs = jnp.concatenate([jnp.transpose(zs_prefix,(1,0,2)), z_final[:,None,:]], 1)
        z_tgt   = jax.lax.stop_gradient(z_all)
        weights = jnp.array([rho**t for t in range(T-1)])
        z_t_T  = jnp.transpose(zs[:,:-1],  (1,0,2))
        a_T    = acts_T
        r_T    = jnp.transpose(rew_b[:,:-1],  (1,0))
        d_T    = jnp.transpose(done_b[:,:-1], (1,0))
        z_t1_T = jnp.transpose(z_tgt[:,1:],   (1,0,2))
        zs_t1_T= jnp.transpose(zs[:,1:],      (1,0,2))
        def step_loss(carry, inp):
            k, w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp
            cl = w*jnp.mean(jnp.sum((zs_t1-z_tgt_t1)**2,-1))
            pr = rew_net.apply(params["rew"], z_t, a_t)
            rl = w*jnp.mean(soft_ce(pr, two_hot(r_t)))
            z_n = jax.lax.stop_gradient(z_tgt_t1)
            tp_mean, _ = pi_net.apply(tp["pi"], z_n)
            pi_a_mean  = jnp.tanh(tp_mean)
            q_next = q_net.apply(tp["q"], z_n, pi_a_mean)
            v_n  = jnp.maximum(jnp.min(two_hot_inv(q_next),-1), 0.0)
            td   = r_t + gamma*(1-d_t)*jax.lax.stop_gradient(v_n)
            qp   = q_net.apply(params["q"], z_t, a_t)
            vl   = w*jnp.mean(jnp.sum(soft_ce(qp, two_hot(td)[:,None,:]),-1))
            pi_ms= pi_net.apply(params["pi"], jax.lax.stop_gradient(z_t))
            pi2  = jnp.tanh(pi_ms[0])
            q_pi = q_net.apply(jax.lax.stop_gradient(params["q"]), jax.lax.stop_gradient(z_t), pi2)
            pl   = -w*jnp.mean(jnp.min(two_hot_inv(q_pi),-1))
            return k, (cl, rl, vl, pl)
        keys = jax.random.split(rng, T-1)
        _, (cls, rls, vls, pls) = jax.lax.scan(
            step_loss, rng, (keys, weights, z_t_T, a_T, r_T, d_T, z_t1_T, zs_t1_T))
        n = T-1
        return (2*jnp.sum(cls)+2*jnp.sum(rls)+jnp.sum(vls)+0.1*jnp.sum(pls))/n, \
               {"c":jnp.sum(cls)/n,"r":jnp.sum(rls)/n,"v":jnp.sum(vls)/n,"p":jnp.sum(pls)/n}
    @jax.jit
    def step(params, tp, opt, ob, ab, rb, db, rng, scale_val):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, tp, ob, ab, rb, db, rng, scale_val)
        upd, nopt = tx.update(grads, opt, params)
        new_params = optax.apply_updates(params, upd)
        new_tp = jax.tree_util.tree_map(lambda t, p: (1-tau)*t+tau*p, tp, new_params)
        return new_params, new_tp, nopt, loss, aux
    return step

def make_mppi_fn(enc, dyn, rew_net, q_net, pi_net,
                 horizon=5, n_samples=256, n_iter=6, temp=0.5,
                 act_low=-1.0, act_high=1.0, act_dim=4, gamma=0.99, rew_scale=10.0):
    _gammas  = jnp.array([gamma**t for t in range(horizon)])
    _gamma_H = float(gamma**horizon)
    @jax.jit
    def plan(params, obs, mu, key):
        z0s = enc.apply(params["enc"], obs[None])[0]
        z0  = jnp.tile(z0s[None], (n_samples, 1))
        def pi_step(z, _):
            mn, _ = pi_net.apply(params["pi"], z[None])
            a = jnp.tanh(mn[0])
            return dyn.apply(params["dyn"], z[None], a[None])[0], a
        _, pi_traj = jax.lax.scan(pi_step, z0s, None, length=horizon)
        mu_ws = mu.at[0].set(pi_traj[0])
        def one_iter(carry, _):
            mu_i, k = carry
            k, sk = jax.random.split(k)
            noise = jax.random.normal(sk, (n_samples, horizon, act_dim))*0.5
            acts  = jnp.clip(mu_i[None]+noise, act_low, act_high)
            acts  = acts.at[-1].set(pi_traj)
            def rollout_one(args):
                z_i, a_seq = args
                def env_step(z, a):
                    r = two_hot_inv(rew_net.apply(params["rew"], z[None], a[None])).squeeze()
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r
                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a, _ = pi_net.apply(params["pi"], zf[None])
                vt = jnp.maximum(jnp.min(two_hot_inv(q_net.apply(params["q"], zf[None], jnp.tanh(pi_a)))), 0.0).squeeze()
                return jnp.sum(_gammas*rs)+_gamma_H*vt
            rets = jax.vmap(rollout_one)((z0, acts))
            w    = jax.nn.softmax((rets-rets.max())/(temp+1e-8))
            return (jnp.einsum("n,nha->ha", w, acts), k), None
        (muf, _), _ = jax.lax.scan(one_iter, (mu_ws, key), None, length=n_iter)
        return jnp.clip(muf[0], act_low, act_high), jnp.concatenate([muf[1:], pi_traj[-1:]], 0)
    return plan


# ─── Main profiling run ────────────────────────────────────────────────────────

def main():
    from mujoco_playground import registry, wrapper

    # ── Config — match v10 exactly so numbers are comparable ──────────────────
    SEED       = 42
    N_ENVS     = 256
    WARMUP_ENV = 25_000
    BS         = 256
    SEQ        = 6
    K_UPDATE   = 256
    LR         = 3e-4
    LATENT     = 128
    HIDDEN     = (128, 128)
    GAMMA      = 0.99
    TAU        = 0.01
    H, NS, NI  = 5, 256, 6
    TEMP       = 0.5
    REW_SCALE  = 10.0
    EXPL_NOISE = 0.3
    al, ah     = -1.0, 1.0

    # How many macro steps to profile (each = N_ENVS env steps + K_UPDATE updates)
    N_MACRO_PROFILE = 20  # enough to get stable averages; ~20-30s total

    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    # ── Env ──────────────────────────────────────────────────────────────────
    env_raw = registry.load("HopperHop")
    env     = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)
    obs_dim = env.observation_size
    act_dim = env.action_size
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  N_ENVS={N_ENVS}  K_UPDATE={K_UPDATE}", flush=True)

    @jax.jit
    def batch_reset(key): return env.reset(jax.random.split(key, N_ENVS))
    @jax.jit
    def batch_step(state, action): return env.step(state, action)

    # ── Models ───────────────────────────────────────────────────────────────
    enc=Encoder(LATENT,HIDDEN); dyn=Dynamics(LATENT,HIDDEN)
    rn=RewardHead(HIDDEN); qn=QEnsemble(HIDDEN); pn=Pi(act_dim,HIDDEN)

    key, k1,k2,k3,k4,k5 = jax.random.split(key,6)
    do=jnp.zeros((1,obs_dim)); dz=jnp.zeros((1,LATENT)); da=jnp.zeros((1,act_dim))
    params={"enc":enc.init(k1,do),"dyn":dyn.init(k2,dz,da),
            "rew":rn.init(k3,dz,da),"q":qn.init(k4,dz,da),"pi":pn.init(k5,dz)}
    tp = jax.tree_util.tree_map(lambda x: x, params)

    _labels={'enc':'world','dyn':'world','rew':'world','q':'q','pi':'pi'}
    tx = optax.multi_transform(
        {'world':optax.chain(optax.clip_by_global_norm(10.0),optax.adam(LR)),
         'q':    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR)),
         'pi':   optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))},
        _labels)
    opt = tx.init(params)

    upd  = make_update_fn(enc,dyn,rn,qn,pn,tx,GAMMA,tau=TAU,rew_scale=REW_SCALE)
    plan = make_mppi_fn(enc,dyn,rn,qn,pn,H,NS,NI,TEMP,al,ah,act_dim,GAMMA,REW_SCALE)

    @jax.jit
    def act_fn_batch(params, obs_batch):
        z = enc.apply(params["enc"], obs_batch)
        mean, _ = pn.apply(params["pi"], z)
        return jnp.tanh(mean)

    # ── Buffer ───────────────────────────────────────────────────────────────
    cap_per_env = max(12_000, 2_000_000 // N_ENVS + 2000)
    buf = MultiEnvBuffer(cap_per_env, N_ENVS, obs_dim, act_dim, SEQ)
    rng = np.random.default_rng(SEED)

    # ── Warmup ───────────────────────────────────────────────────────────────
    print("Warmup (random actions)...", flush=True)
    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    obs_np    = np.array(env_state.obs)
    env_steps = 0
    while env_steps < WARMUP_ENV:
        acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        env_state = batch_step(env_state, jnp.asarray(acts_np))
        buf.add_batch(obs_np, acts_np, np.array(env_state.reward), np.array(env_state.done>0.5, np.float32))
        obs_np    = np.array(env_state.obs)
        env_steps += N_ENVS
    print(f"Buffer filled: {buf.total_size()} transitions", flush=True)

    # ── Compile ──────────────────────────────────────────────────────────────
    print("Compiling update...", flush=True)
    samp = buf.sample(BS, rng)
    ob, ab, rb, db = [jnp.asarray(x) for x in samp]
    key, uk = jax.random.split(key)
    params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)
    jax.block_until_ready(params["enc"])

    print("Compiling MPPI...", flush=True)
    key, ek = jax.random.split(key)
    mu_e      = jnp.zeros((H, act_dim))
    dummy_obs = jnp.asarray(obs_np[0])
    act_e, _  = plan(params, dummy_obs, mu_e, ek)
    jax.block_until_ready(act_e)

    # ── Time one MPPI call as standalone reference ────────────────────────────
    N_MPPI_TIMING = 20
    mppi_times = []
    for _ in range(N_MPPI_TIMING):
        key, ek = jax.random.split(key)
        t0 = time.perf_counter()
        act_e, _ = plan(params, dummy_obs, mu_e, ek)
        jax.block_until_ready(act_e)
        mppi_times.append(time.perf_counter()-t0)
    mppi_ms = np.mean(mppi_times)*1e3
    print(f"\n[MPPI single call] mean={mppi_ms:.2f}ms  min={min(mppi_times)*1e3:.2f}ms  max={max(mppi_times)*1e3:.2f}ms", flush=True)

    # ── Time one full update call ─────────────────────────────────────────────
    N_UPDATE_TIMING = 20
    update_times = []
    for _ in range(N_UPDATE_TIMING):
        samp = buf.sample(BS, rng)
        ob, ab, rb, db = [jnp.asarray(x) for x in samp]
        key, uk = jax.random.split(key)
        t0 = time.perf_counter()
        params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)
        jax.block_until_ready(params["enc"])
        update_times.append(time.perf_counter()-t0)
    upd_ms = np.mean(update_times)*1e3
    print(f"[Update single call] mean={upd_ms:.2f}ms  min={min(update_times)*1e3:.2f}ms  max={max(update_times)*1e3:.2f}ms", flush=True)

    # ── Profile macro step components ─────────────────────────────────────────
    print(f"\nProfiling {N_MACRO_PROFILE} macro steps ({N_MACRO_PROFILE*N_ENVS} env steps total)...", flush=True)
    print(f"  Each macro = 1 env step batch ({N_ENVS} envs) + {K_UPDATE} update calls", flush=True)

    T = collections.defaultdict(list)  # component → list of times in ms

    for macro_i in range(N_MACRO_PROFILE):
        t_macro = time.perf_counter()

        # 1. Act (policy forward on GPU for all envs)
        t0 = time.perf_counter()
        acts_jax = act_fn_batch(params, jnp.asarray(obs_np))
        jax.block_until_ready(acts_jax)
        T["act_pi"].append(time.perf_counter()-t0)

        # 2. Exploration noise (CPU numpy)
        t0 = time.perf_counter()
        noise   = np.random.normal(0, EXPL_NOISE, (N_ENVS, act_dim))
        acts_np = np.clip(np.array(acts_jax)+noise, al, ah).astype(np.float32)
        T["noise"].append(time.perf_counter()-t0)

        # 3. Env step (JIT dispatch + GPU physics + device→host)
        t0 = time.perf_counter()
        env_state = batch_step(env_state, jnp.asarray(acts_np))
        new_obs   = np.array(env_state.obs)       # forces device→host
        rews_np   = np.array(env_state.reward)
        done_np   = np.array(env_state.done>0.5, np.float32)
        T["env_step"].append(time.perf_counter()-t0)

        # 4. Buffer write (numpy fancy index)
        t0 = time.perf_counter()
        buf.add_batch(obs_np, acts_np, rews_np, done_np)
        T["buf_write"].append(time.perf_counter()-t0)
        obs_np = new_obs

        # 5. K update calls — break into sub-components
        buf_read_t = 0.0
        h2d_t      = 0.0
        update_t   = 0.0
        for _ in range(K_UPDATE):
            t0 = time.perf_counter()
            samp = buf.sample(BS, rng)
            buf_read_t += time.perf_counter()-t0

            t0 = time.perf_counter()
            ob, ab, rb, db = [jnp.asarray(x) for x in samp]
            h2d_t += time.perf_counter()-t0

            key, uk = jax.random.split(key)
            t0 = time.perf_counter()
            params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)
            jax.block_until_ready(params["enc"])
            update_t += time.perf_counter()-t0

        T["buf_read"].append(buf_read_t)
        T["h2d"].append(h2d_t)
        T["update"].append(update_t)

        t_macro_total = time.perf_counter()-t_macro
        T["macro_total"].append(t_macro_total)

        sps = N_ENVS / t_macro_total
        print(f"  macro {macro_i+1:2d}/{N_MACRO_PROFILE}  total={t_macro_total*1e3:.0f}ms  sps={sps:.0f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PROFILING SUMMARY  (averages over last N_MACRO_PROFILE macro steps)")
    print("="*70)

    components = [
        ("act_pi",      "Policy forward (N_ENVS acts on GPU)"),
        ("noise",       "Exploration noise (numpy CPU)"),
        ("env_step",    "Env step JIT dispatch + d→h transfer"),
        ("buf_write",   "Buffer add_batch (numpy write)"),
        ("buf_read",    f"Buffer sample ×{K_UPDATE} (numpy fancy-idx, total)"),
        ("h2d",         f"Host→device transfer ×{K_UPDATE} (jnp.asarray, total)"),
        ("update",      f"GPU update ×{K_UPDATE} (total, block_until_ready)"),
        ("macro_total", "FULL MACRO STEP TOTAL"),
    ]

    macro_mean_ms = np.mean(T["macro_total"])*1e3

    print(f"\n{'Component':<35} {'Mean(ms)':>10} {'Min(ms)':>10} {'%macro':>8}")
    print("-"*67)
    for key_name, label in components:
        vals = np.array(T[key_name])*1e3
        mean_ms = np.mean(vals)
        min_ms  = np.min(vals)
        pct     = 100*mean_ms/macro_mean_ms if key_name!="macro_total" else 100.0
        marker  = " ◄ TOP" if pct > 20 else ""
        print(f"  {label:<33} {mean_ms:>10.1f} {min_ms:>10.1f} {pct:>7.1f}%{marker}")

    print(f"\n  Effective SPS:          {np.mean([N_ENVS/t for t in T['macro_total']]):.1f}")
    print(f"  One MPPI call (ref):    {mppi_ms:.2f}ms  (equiv to {mppi_ms/(np.mean(T['update'])*1e3/K_UPDATE):.1f}x update steps)")
    print(f"  One update call (ref):  {upd_ms:.2f}ms")
    print(f"  K updates total:        {np.mean(T['update'])*1e3:.1f}ms  ({K_UPDATE}×{upd_ms:.1f}ms)")
    print()
    print(f"  Bottleneck diagnosis:")

    update_pct  = 100*np.mean(T["update"])/np.mean(T["macro_total"])
    bufread_pct = 100*np.mean(T["buf_read"])/np.mean(T["macro_total"])
    h2d_pct     = 100*np.mean(T["h2d"])/np.mean(T["macro_total"])

    if update_pct > 70:
        print(f"    GPU update dominates ({update_pct:.0f}%). Reduce K_UPDATE or use faster loss.")
    if bufread_pct > 10:
        print(f"    CPU buffer sampling is significant ({bufread_pct:.0f}%). Consider device-side buffer.")
    if h2d_pct > 5:
        print(f"    Host→device transfer is significant ({h2d_pct:.0f}%). Move buffer to GPU.")
    print("="*70)


if __name__ == "__main__":
    main()
