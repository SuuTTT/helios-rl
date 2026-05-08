"""Train TD-MPC2 on mujoco_playground HopperHop (JAX vectorised envs).

Speed vs v2 (dm_control, single env):
  v2: ~92 sps (env bottleneck, Python dm_control)
  v3: N_ENVS=64 parallel JAX envs → ~5000-6000 sps expected

Key fixes vs v2:
  1. JAX-native env (mujoco_playground) → vectorised GPU env steps
  2. REW_SCALE=10.0 → reward head gradient amplified 100x
  3. Multi-env ring buffer tracks env_id to avoid cross-env sequences
  4. Reward head: predict scaled reward (REW_SCALE * r), MPPI uses same scale
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp, optax
import flax.linen as nn

sys.path.insert(0, "/workspace/wiki/learn_mujoco_playground/repo")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")


# ─── SimNorm ──────────────────────────────────────────────────────────────────

def simnorm(x, V=8):
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)


# ─── Model ────────────────────────────────────────────────────────────────────

class NormMLP(nn.Module):
    dims: tuple; out: int
    @nn.compact
    def __call__(self, x):
        for d in self.dims:
            x = nn.Dense(d)(x); x = nn.LayerNorm()(x); x = nn.silu(x)
        return nn.Dense(self.out)(x)

class Encoder(nn.Module):
    latent_dim: int; hidden: tuple = (256, 256); V: int = 8
    @nn.compact
    def __call__(self, obs):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)

class Dynamics(nn.Module):
    latent_dim: int; hidden: tuple = (256, 256); V: int = 8
    @nn.compact
    def __call__(self, z, a):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(jnp.concatenate([z, a], -1)), self.V)

class RewardHead(nn.Module):
    hidden: tuple = (256, 256)
    @nn.compact
    def __call__(self, z, a):
        return NormMLP(self.hidden, 1)(jnp.concatenate([z, a], -1)).squeeze(-1)

class QEnsemble(nn.Module):
    hidden: tuple = (256, 256)
    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], -1)
        return jnp.stack([NormMLP(self.hidden, 1)(x).squeeze(-1),
                          NormMLP(self.hidden, 1)(x).squeeze(-1)], -1)

class Pi(nn.Module):
    action_dim: int; hidden: tuple = (256, 256)
    @nn.compact
    def __call__(self, z): return jnp.tanh(NormMLP(self.hidden, self.action_dim)(z))


# ─── Multi-env ring buffer ─────────────────────────────────────────────────────
# Stores N_ENVS independent trajectories. Sequences are sampled within one env
# to avoid crossing episode / env boundaries.

class MultiEnvBuffer:
    def __init__(self, cap_per_env, n_envs, obs_dim, act_dim, seq_len):
        self.cap = cap_per_env
        self.N = n_envs
        self.T = seq_len
        self.obs  = np.zeros((n_envs, cap_per_env, obs_dim), np.float32)
        self.acts = np.zeros((n_envs, cap_per_env, act_dim), np.float32)
        self.rews = np.zeros((n_envs, cap_per_env), np.float32)
        self.done = np.zeros((n_envs, cap_per_env), np.float32)
        self.ptr  = np.zeros(n_envs, dtype=np.int64)
        self.size = np.zeros(n_envs, dtype=np.int64)

    def add_batch(self, obs_batch, acts_batch, rews_batch, done_batch):
        """Add one step from all N envs."""
        for i in range(self.N):
            p = self.ptr[i]
            self.obs[i, p] = obs_batch[i]
            self.acts[i, p] = acts_batch[i]
            self.rews[i, p] = rews_batch[i]
            self.done[i, p] = done_batch[i]
            self.ptr[i] = (p + 1) % self.cap
            self.size[i] = min(self.size[i] + 1, self.cap)

    def total_size(self):
        return int(self.size.sum())

    def sample(self, B, rng):
        """Sample B sequences of length T, each from a single env."""
        valid_envs = np.where(self.size >= self.T + 1)[0]
        if len(valid_envs) == 0:
            return None
        env_ids = rng.choice(valid_envs, size=B, replace=True)
        starts  = np.array([rng.integers(0, self.size[e] - self.T) for e in env_ids])
        idx = starts[:, None] + np.arange(self.T)[None, :]  # (B, T)
        obs_b  = np.array([self.obs[e, idx[i]]  for i, e in enumerate(env_ids)])
        acts_b = np.array([self.acts[e, idx[i]] for i, e in enumerate(env_ids)])
        rews_b = np.array([self.rews[e, idx[i]] for i, e in enumerate(env_ids)])
        done_b = np.array([self.done[e, idx[i]] for i, e in enumerate(env_ids)])
        return obs_b, acts_b, rews_b, done_b


# ─── Update fn ────────────────────────────────────────────────────────────────

def make_update_fn(enc, dyn, rew_net, q_net, pi_net, tx,
                   gamma=0.99, rho=0.5, tau=0.01, rew_scale=10.0):
    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b):
        B, T, _ = obs_b.shape
        z_all = enc.apply(params["enc"], obs_b.reshape(B * T, -1)).reshape(B, T, -1)
        z0 = z_all[:, 0]
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
            w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp
            cl = w * jnp.mean(jnp.sum((zs_t1 - z_tgt_t1) ** 2, -1))
            pr = rew_net.apply(params["rew"], z_t, a_t)
            # reward head predicts rew_scale * r  → large gradient even for small r
            rl = w * jnp.mean((pr - rew_scale * r_t) ** 2)
            z_n   = jax.lax.stop_gradient(z_tgt_t1)
            pi_a  = pi_net.apply(tp["pi"], z_n)
            v_n   = jnp.min(q_net.apply(tp["q"], z_n, pi_a), -1)
            v_n   = jnp.maximum(v_n, 0.0)
            td    = rew_scale * r_t + gamma * (1 - d_t) * jax.lax.stop_gradient(v_n)
            qp    = q_net.apply(params["q"], z_t, a_t)
            vl    = w * jnp.mean(jnp.sum((qp - td[:, None]) ** 2, -1))
            pi2   = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_t))
            pl    = -w * jnp.mean(jnp.min(
                q_net.apply(jax.lax.stop_gradient(params["q"]),
                            jax.lax.stop_gradient(z_t), pi2), -1))
            return carry, (cl, rl, vl, pl)

        _, (cls, rls, vls, pls) = jax.lax.scan(
            step_loss, None, (weights, z_t_T, a_T, r_T, d_T, z_t1_T, zs_t1_T))
        n = T - 1
        return (2 * jnp.sum(cls) + 2 * jnp.sum(rls) + jnp.sum(vls) + 0.1 * jnp.sum(pls)) / n, \
               {"c": jnp.sum(cls) / n, "r": jnp.sum(rls) / n,
                "v": jnp.sum(vls) / n, "p": jnp.sum(pls) / n}

    @jax.jit
    def step(params, tp, opt, ob, ab, rb, db):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, tp, ob, ab, rb, db)
        upd, nopt = tx.update(grads, opt, params)
        new_params = optax.apply_updates(params, upd)
        new_tp = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, tp, new_params)
        return new_params, new_tp, nopt, loss, aux
    return step


# ─── MPPI fn ──────────────────────────────────────────────────────────────────

def make_mppi_fn(enc, dyn, rew_net, q_net, pi_net,
                 horizon=5, n_samples=256, n_iter=6, temp=0.5,
                 act_low=-1.0, act_high=1.0, act_dim=4,
                 gamma=0.99, rew_scale=10.0):
    _gammas   = jnp.array([gamma ** t for t in range(horizon)])
    _gamma_H  = float(gamma ** horizon)

    @jax.jit
    def plan(params, obs, mu, key):
        z0 = jnp.tile(enc.apply(params["enc"], obs[None]), (n_samples, 1))

        def one_iter(carry, _):
            mu_i, k = carry
            k, sk = jax.random.split(k)
            noise = jax.random.normal(sk, (n_samples, horizon, act_dim)) * 0.5
            acts = jnp.clip(mu_i[None] + noise, act_low, act_high)

            def rollout_one(args):
                z_i, a_seq = args
                def env_step(z, a):
                    # reward head predicts rew_scale * r  → divide back to real scale for return
                    r = rew_net.apply(params["rew"], z[None], a[None]).squeeze() / rew_scale
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r
                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a = pi_net.apply(params["pi"], zf[None])
                # Q is in rew_scale space; divide back
                vt = jnp.maximum(jnp.min(q_net.apply(params["q"], zf[None], pi_a)), 0.0) / rew_scale
                return jnp.sum(_gammas * rs) + _gamma_H * vt

            rets = jax.vmap(rollout_one)((z0, acts))
            w = jax.nn.softmax((rets - rets.max()) / (temp + 1e-8))
            new_mu = jnp.einsum("n,nha->ha", w, acts)
            return (new_mu, k), None

        (muf, _), _ = jax.lax.scan(one_iter, (mu, key), None, length=n_iter)
        action = jnp.clip(muf[0], act_low, act_high)
        new_mu = jnp.concatenate([muf[1:], jnp.zeros((1, act_dim))], 0)
        return action, new_mu
    return plan


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from mujoco_playground import registry, wrapper

    SEED        = 42
    N_ENVS      = 16        # parallel JAX envs; do N_ENVS updates per global step → 1:1 UTD
    TOTAL_ENV   = 2_000_000 # total env-steps across all envs
    WARMUP_ENV  = 25_000    # env-steps before training (spread across N_ENVS envs)
    BS          = 256
    SEQ         = 6
    LR          = 3e-4
    LATENT      = 128
    HIDDEN      = (256, 256)
    GAMMA       = 0.99
    TAU         = 0.01
    H           = 5
    NS          = 256
    NI          = 6
    TEMP        = 0.5
    REW_SCALE   = 10.0      # amplify reward head gradient + Q scale
    EXPL_NOISE  = 0.3
    EXPL_UNTIL  = 25_000    # env-steps

    # Eval every this many env-steps (across all envs)
    PI_EVAL_EVERY   = 100_000
    MPPI_EVAL_EVERY = 200_000
    EVAL_EPS        = 5

    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    # ── Load JAX-vectorised env ──────────────────────────────────────────────
    env_raw = registry.load("HopperHop")
    env = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)

    obs_dim = env.observation_size
    act_dim = env.action_size
    al, ah  = -1.0, 1.0
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  N_ENVS={N_ENVS}", flush=True)

    @jax.jit
    def batch_reset(key):
        keys = jax.random.split(key, N_ENVS)
        return env.reset(keys)

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

    _param_labels = {'enc': 'world', 'dyn': 'world', 'rew': 'world', 'q': 'q', 'pi': 'world'}
    tx = optax.multi_transform(
        {'world': optax.chain(optax.clip_by_global_norm(10.0), optax.adam(LR)),
         'q':     optax.chain(optax.clip_by_global_norm(1.0),  optax.adam(LR))},
        _param_labels)
    opt = tx.init(params)

    upd  = make_update_fn(enc, dyn, rn, qn, pn, tx, GAMMA, tau=TAU, rew_scale=REW_SCALE)
    plan = make_mppi_fn(enc, dyn, rn, qn, pn, H, NS, NI, TEMP, al, ah, act_dim, GAMMA, REW_SCALE)

    @jax.jit
    def act_fn(params, obs):
        z = enc.apply(params["enc"], obs[None])
        return pn.apply(params["pi"], z)[0]

    # ── Buffer ──────────────────────────────────────────────────────────────
    cap_per_env = max(50_000, TOTAL_ENV // N_ENVS * 2)
    buf = MultiEnvBuffer(cap_per_env, N_ENVS, obs_dim, act_dim, SEQ)
    rng = np.random.default_rng(SEED)

    out_dir = "/workspace/helios-rl/exp/tdmpc_dmc"
    os.makedirs(out_dir, exist_ok=True)
    csv = f"{out_dir}/hopper-hop.csv"
    with open(csv, "w") as f:
        f.write("step,reward,seed\n")

    # ── Warmup (random actions across all envs) ──────────────────────────────
    print("Warmup...", flush=True)
    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    obs_np = np.array(env_state.obs)  # (N_ENVS, obs_dim)
    env_steps = 0

    while env_steps < WARMUP_ENV:
        key, ak = jax.random.split(key)
        acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        env_state = batch_step(env_state, jnp.asarray(acts_np))
        new_obs = np.array(env_state.obs)
        rews    = np.array(env_state.reward)
        dones   = np.array(env_state.done > 0.5, dtype=np.float32)
        buf.add_batch(obs_np, acts_np, rews, dones)
        obs_np = new_obs
        env_steps += N_ENVS

    print(f"Warmup done. Buffer total={buf.total_size()}  Compiling...", flush=True)

    # ── Compile ─────────────────────────────────────────────────────────────
    t_c = time.time()
    samp = buf.sample(BS, rng)
    ob, ab, rb, db = [jnp.asarray(x) for x in samp]
    params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db)
    jax.block_until_ready(params["enc"])
    print(f"Update compiled in {time.time()-t_c:.1f}s  loss={float(loss):.4f}", flush=True)

    te_state = batch_reset(jax.random.PRNGKey(SEED + 1))
    te_obs = jnp.asarray(te_state.obs[0])
    mu_e = jnp.zeros((H, act_dim))
    key, ek = jax.random.split(key)
    act_e, _ = plan(params, te_obs, mu_e, ek)
    jax.block_until_ready(act_e)
    print("MPPI compiled.", flush=True)

    # ── Eval helpers (using single dm_control env for accurate eval) ─────────
    # We use the JAX env for eval too — just step single env-state through sequentially
    from dm_control import suite as dmc_suite

    def extract_obs_dmc(ts):
        return np.concatenate([np.asarray(v, np.float32).flatten()
                               for v in ts.observation.values()])

    eval_env = dmc_suite.load("hopper", "hop", task_kwargs={"random": SEED + 1})

    def eval_pi(n_eps):
        rets = []
        for _ in range(n_eps):
            ts = eval_env.reset(); obs = extract_obs_dmc(ts); er = 0.0
            while not ts.last():
                a = np.array(act_fn(params, jnp.asarray(obs)))
                ts = eval_env.step(np.clip(a, al, ah))
                er += float(ts.reward); obs = extract_obs_dmc(ts)
            rets.append(er)
        return np.mean(rets)

    def eval_mppi(n_eps, _key):
        rets = []
        for _ in range(n_eps):
            _key, ek_ = jax.random.split(_key)
            ts = eval_env.reset(); oe = extract_obs_dmc(ts)
            mu = jnp.zeros((H, act_dim)); er = 0.0
            while not ts.last():
                act, mu = plan(params, jnp.asarray(oe), mu, ek_)
                _key, ek_ = jax.random.split(_key)
                ts = eval_env.step(np.array(act))
                er += float(ts.reward); oe = extract_obs_dmc(ts)
            rets.append(er)
        return np.mean(rets), _key

    # ── Training loop ────────────────────────────────────────────────────────
    print("Training...", flush=True)
    t0 = time.time()
    log_every = max(N_ENVS * 100, 5_000)  # print every ~100 global steps

    while env_steps < TOTAL_ENV:
        # ---- collect N_ENVS steps ----
        if env_steps < EXPL_UNTIL:
            acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
        else:
            # vectorised policy action
            acts_jax = jax.vmap(lambda o: act_fn(params, o))(jnp.asarray(obs_np))
            acts_np  = np.clip(
                np.array(acts_jax) + np.random.normal(0, EXPL_NOISE, (N_ENVS, act_dim)),
                al, ah).astype(np.float32)

        env_state = batch_step(env_state, jnp.asarray(acts_np))
        new_obs = np.array(env_state.obs)
        rews    = np.array(env_state.reward)
        dones   = np.array(env_state.done > 0.5, dtype=np.float32)

        buf.add_batch(obs_np, acts_np, rews, dones)
        obs_np = new_obs
        env_steps += N_ENVS

        # ---- N_ENVS updates to maintain 1:1 update-to-data (UTD) ratio ----
        for _ in range(N_ENVS):
            samp = buf.sample(BS, rng)
            if samp is not None:
                ob, ab, rb, db = [jnp.asarray(x) for x in samp]
                params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db)

        # ---- logging ----
        if env_steps % log_every < N_ENVS:
            elapsed = time.time() - t0
            sps = env_steps / elapsed
            print(f"  es={env_steps:>9,}  sps={sps:.0f}  loss={float(loss):.4f}", flush=True)

        if env_steps % PI_EVAL_EVERY < N_ENVS:
            pi_ret = eval_pi(EVAL_EPS)
            elapsed = time.time() - t0
            print(f"step={env_steps:>9,}  pi={pi_ret:7.1f}  sps={env_steps/elapsed:.0f}  "
                  f"c={float(aux['c']):.3f} r={float(aux['r']):.3f} "
                  f"v={float(aux['v']):.3f} p={float(aux['p']):.3f}", flush=True)

        if env_steps % MPPI_EVAL_EVERY < N_ENVS:
            mr, key = eval_mppi(EVAL_EPS, key)
            elapsed = time.time() - t0
            print(f"step={env_steps:>9,}  MPPI={mr:7.1f}  sps={env_steps/elapsed:.0f}", flush=True)
            with open(csv, "a") as f:
                f.write(f"{env_steps},{mr:.1f},{SEED}\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s  ({elapsed/3600:.2f}h)  ->  {csv}", flush=True)


if __name__ == "__main__":
    main()
