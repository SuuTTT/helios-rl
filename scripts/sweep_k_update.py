"""
Sweep K_UPDATE values to measure speed vs. sample-efficiency tradeoff.

Each run:
  - HopperHop, N_ENVS=256, BS=256, SEQ=6, HIDDEN=(128,128)
  - Warmup 25k env-steps (random), then 500k env-steps training
  - Evaluates pi policy at 250k and 500k env-steps (5 episodes each)
  - Reports: SPS, wall time, pi reward at each eval point

K_UPDATE values swept: 16, 32, 64, 128, 256
  (UTD = K_UPDATE/N_ENVS = 1/16 ... 1:1)

Usage:
  MUJOCO_GL=egl python3 helios-rl/scripts/sweep_k_update.py 2>&1 | tee /tmp/k_update_sweep.log
"""

import os, sys, time, gc
import numpy as np

sys.path.insert(0, "/workspace/wiki/learn_mujoco_playground/repo")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")

import jax, jax.numpy as jnp, optax
import flax.linen as nn


# ─── Shared math ──────────────────────────────────────────────────────────────

def simnorm(x, V=8):
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1]//V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)

def log_std_fn(x, low=-10.0, dif=12.0):
    return low + 0.5*dif*(jnp.tanh(x)+1.0)

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


# ─── Models ───────────────────────────────────────────────────────────────────

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


# ─── Buffer ────────────────────────────────────────────────────────────────────

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


# ─── Update fn ────────────────────────────────────────────────────────────────

def make_update_fn(enc, dyn, rew_net, q_net, pi_net, tx,
                   gamma=0.99, rho=0.5, tau=0.01):
    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b, rng, _scale):
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B*T,-1)).reshape(B,T,-1)
        z0    = z_all[:,0]
        acts_T = jnp.transpose(act_b[:,:-1],(1,0,2))
        def dyn_step(z, a): return dyn.apply(params["dyn"], z, a), z
        z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
        zs = jnp.concatenate([jnp.transpose(zs_prefix,(1,0,2)), z_final[:,None,:]], 1)
        z_tgt   = jax.lax.stop_gradient(z_all)
        weights = jnp.array([rho**t for t in range(T-1)])
        z_t_T   = jnp.transpose(zs[:,:-1],  (1,0,2))
        a_T     = acts_T
        r_T     = jnp.transpose(rew_b[:,:-1],  (1,0))
        d_T     = jnp.transpose(done_b[:,:-1], (1,0))
        z_t1_T  = jnp.transpose(z_tgt[:,1:],   (1,0,2))
        zs_t1_T = jnp.transpose(zs[:,1:],      (1,0,2))
        def step_loss(carry, inp):
            k, w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp
            cl  = w*jnp.mean(jnp.sum((zs_t1-z_tgt_t1)**2,-1))
            pr  = rew_net.apply(params["rew"], z_t, a_t)
            rl  = w*jnp.mean(soft_ce(pr, two_hot(r_t)))
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


# ─── One training run ────────────────────────────────────────────────────────

def run_one(env, K_UPDATE, seed=42, total_env=500_000, warmup_env=25_000,
            eval_at=(250_000, 500_000), eval_eps=5):
    """Return dict with SPS, wall_time, and pi rewards at each eval point."""
    N_ENVS  = 256
    BS      = 256
    SEQ     = 6
    LR      = 3e-4
    LATENT  = 128
    HIDDEN  = (128, 128)
    GAMMA   = 0.99
    TAU     = 0.01
    EXPL_NOISE = 0.3
    EXPL_UNTIL = warmup_env
    al, ah  = -1.0, 1.0

    obs_dim = env.observation_size
    act_dim = env.action_size

    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    @jax.jit
    def batch_reset(k): return env.reset(jax.random.split(k, N_ENVS))
    @jax.jit
    def batch_step(state, action): return env.step(state, action)
    @jax.jit
    def eval_reset(k): return env.reset(jax.random.split(k, 1))
    @jax.jit
    def eval_step_fn(state, act): return env.step(state, act[None])

    # models
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

    upd = make_update_fn(enc,dyn,rn,qn,pn,tx,GAMMA,tau=TAU)

    @jax.jit
    def act_fn_batch(p, obs): 
        z = enc.apply(p["enc"], obs)
        return jnp.tanh(pn.apply(p["pi"], z)[0])

    @jax.jit
    def act_fn_single(p, obs):
        z = enc.apply(p["enc"], obs[None])
        return jnp.tanh(pn.apply(p["pi"], z)[0])

    # buffer
    cap = max(12_000, total_env // N_ENVS + 2000)
    buf = MultiEnvBuffer(cap, N_ENVS, obs_dim, act_dim, SEQ)

    # warmup
    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    obs_np = np.array(env_state.obs)
    env_steps = 0
    while env_steps < warmup_env:
        acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        env_state = batch_step(env_state, jnp.asarray(acts_np))
        buf.add_batch(obs_np, acts_np, np.array(env_state.reward), np.array(env_state.done>0.5, np.float32))
        obs_np = np.array(env_state.obs)
        env_steps += N_ENVS

    # compile
    samp = buf.sample(BS, rng)
    ob, ab, rb, db = [jnp.asarray(x) for x in samp]
    key, uk = jax.random.split(key)
    params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)
    jax.block_until_ready(params["enc"])

    def eval_pi():
        nonlocal key
        rets = []
        for _ in range(eval_eps):
            key, rk = jax.random.split(key)
            state = eval_reset(rk)
            obs = jnp.asarray(state.obs[0])
            er = 0.0
            for _ in range(1000):
                act = act_fn_single(params, obs)
                state = eval_step_fn(state, act)
                er += float(state.reward[0])
                if bool(state.done[0] > 0.5): break
                obs = jnp.asarray(state.obs[0])
            rets.append(er)
        return float(np.mean(rets))

    results = {"K_UPDATE": K_UPDATE, "eval": {}}
    t0 = time.time()
    eval_set = set(eval_at)

    # training loop
    while env_steps < total_env + warmup_env:
        if env_steps < EXPL_UNTIL + warmup_env:
            acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        else:
            acts_jax = act_fn_batch(params, jnp.asarray(obs_np))
            noise    = np.random.normal(0, EXPL_NOISE, (N_ENVS, act_dim))
            acts_np  = np.clip(np.array(acts_jax)+noise, al, ah).astype(np.float32)

        env_state = batch_step(env_state, jnp.asarray(acts_np))
        buf.add_batch(obs_np, acts_np, np.array(env_state.reward), np.array(env_state.done>0.5, np.float32))
        obs_np = np.array(env_state.obs)
        env_steps += N_ENVS

        # training steps counted from after warmup
        train_steps = env_steps - warmup_env

        for _ in range(K_UPDATE):
            samp = buf.sample(BS, rng)
            if samp is not None:
                key, uk = jax.random.split(key)
                ob, ab, rb, db = [jnp.asarray(x) for x in samp]
                params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db, uk, scale_val=50.0)

        for chk in list(eval_set):
            if train_steps >= chk:
                eval_set.discard(chk)
                elapsed = time.time()-t0
                pi_ret  = eval_pi()
                sps     = train_steps / elapsed
                results["eval"][chk] = {"pi": pi_ret, "sps": sps, "wall_s": elapsed}
                print(f"    [K={K_UPDATE:>3d}] train_steps={train_steps:>7,}  "
                      f"pi={pi_ret:6.1f}  sps={sps:.0f}  elapsed={elapsed:.0f}s  "
                      f"loss={float(loss):.4f}", flush=True)

    elapsed = time.time()-t0
    results["total_wall_s"] = elapsed
    results["final_sps"]    = total_env / elapsed
    return results


# ─── Main sweep ──────────────────────────────────────────────────────────────

def main():
    from mujoco_playground import registry, wrapper

    env_raw = registry.load("HopperHop")
    env     = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)
    print(f"obs_dim={env.observation_size}  act_dim={env.action_size}", flush=True)

    K_VALUES    = [16, 32, 64, 128, 256]
    TOTAL_ENV   = 500_000
    WARMUP_ENV  = 25_000
    EVAL_AT     = (250_000, 500_000)

    all_results = []
    for K in K_VALUES:
        print(f"\n{'='*60}", flush=True)
        print(f"K_UPDATE={K}  UTD={K/256:.3f}  (N_ENVS=256)", flush=True)
        print(f"{'='*60}", flush=True)
        res = run_one(env, K, total_env=TOTAL_ENV, warmup_env=WARMUP_ENV, eval_at=EVAL_AT)
        all_results.append(res)
        # Force GC between runs to free JAX memory
        gc.collect()
        print(f"  → final SPS={res['final_sps']:.0f}  total_wall={res['total_wall_s']:.0f}s", flush=True)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    header = f"{'K_UPDATE':>8}  {'UTD':>6}  {'SPS@500k':>9}  " + \
             "  ".join(f"{'pi@'+str(e//1000)+'k':>9}" for e in EVAL_AT)
    print(header)
    print("-"*70)
    for res in all_results:
        K   = res["K_UPDATE"]
        utd = K/256
        sps = res["final_sps"]
        pi_vals = "  ".join(
            f"{res['eval'].get(e, {}).get('pi', float('nan')):>9.1f}"
            for e in EVAL_AT)
        print(f"{K:>8}  {utd:>6.3f}  {sps:>9.0f}  {pi_vals}")
    print(f"{'='*70}")

    # Save CSV
    out_csv = "/tmp/k_update_sweep.csv"
    with open(out_csv, "w") as f:
        f.write("k_update,utd,final_sps,wall_s," +
                ",".join(f"pi_{e//1000}k" for e in EVAL_AT) + "\n")
        for res in all_results:
            K   = res["K_UPDATE"]
            sps = res["final_sps"]
            ws  = res["total_wall_s"]
            pi_vals = ",".join(
                f"{res['eval'].get(e, {}).get('pi', float('nan')):.1f}"
                for e in EVAL_AT)
            f.write(f"{K},{K/256:.3f},{sps:.0f},{ws:.0f},{pi_vals}\n")
    print(f"\nCSV saved to {out_csv}")


if __name__ == "__main__":
    main()
