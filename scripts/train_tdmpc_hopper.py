"""Train TD-MPC2 on dm_control hopper-hop."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp, optax
import flax.linen as nn


def simnorm(x, V=8):
    """Simplicial normalization (TD-MPC2): partition latent into V groups, softmax each."""
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*s)

# ─── Model components (same as train_tdmpc_dmc.py) ────────────────────────────

class FastBuffer:
    def __init__(self, cap, obs_dim, act_dim, seq_len):
        self.cap = cap; self.T = seq_len
        self.obs  = np.zeros((cap, obs_dim), np.float32)
        self.acts = np.zeros((cap, act_dim), np.float32)
        self.rews = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.ptr = 0; self.size = 0
    def add(self, o, a, r, d):
        self.obs[self.ptr]=o; self.acts[self.ptr]=a
        self.rews[self.ptr]=r; self.done[self.ptr]=float(d)
        self.ptr=(self.ptr+1)%self.cap; self.size=min(self.size+1,self.cap)
    def sample(self, B, rng):
        s=rng.integers(0,self.size-self.T,size=B)
        idx=s[:,None]+np.arange(self.T)[None,:]
        return self.obs[idx],self.acts[idx],self.rews[idx],self.done[idx]

class NormMLP(nn.Module):
    dims: tuple; out: int
    @nn.compact
    def __call__(self, x):
        for d in self.dims:
            x=nn.Dense(d)(x); x=nn.LayerNorm()(x); x=nn.silu(x)
        return nn.Dense(self.out)(x)

class Encoder(nn.Module):
    latent_dim:int; hidden:tuple=(256,256); V:int=8
    @nn.compact
    def __call__(self, obs):
        x = NormMLP(self.hidden,self.latent_dim)(obs)
        return simnorm(x, self.V)

class Dynamics(nn.Module):
    latent_dim:int; hidden:tuple=(256,256); V:int=8
    @nn.compact
    def __call__(self, z, a):
        x = NormMLP(self.hidden,self.latent_dim)(jnp.concatenate([z,a],-1))
        return simnorm(x, self.V)

class RewardHead(nn.Module):
    hidden:tuple=(256,256)
    @nn.compact
    def __call__(self, z, a):
        return NormMLP(self.hidden,1)(jnp.concatenate([z,a],-1)).squeeze(-1)

class QEnsemble(nn.Module):
    hidden:tuple=(256,256)
    @nn.compact
    def __call__(self, z, a):
        x=jnp.concatenate([z,a],-1)
        return jnp.stack([NormMLP(self.hidden,1)(x).squeeze(-1),
                          NormMLP(self.hidden,1)(x).squeeze(-1)],-1)

class Pi(nn.Module):
    action_dim:int; hidden:tuple=(256,256)
    @nn.compact
    def __call__(self, z): return jnp.tanh(NormMLP(self.hidden,self.action_dim)(z))


# ─── Update + MPPI (same logic as cartpole) ───────────────────────────────────

def make_update_fn(enc, dyn, rew_net, q_net, pi_net, tx, gamma=0.99, rho=0.5, tau=0.01):
    def loss_fn(params, tp, obs_b, act_b, rew_b, done_b):
        B,T,_=obs_b.shape
        z_all=enc.apply(params["enc"],obs_b.reshape(B*T,-1)).reshape(B,T,-1)
        z0=z_all[:,0]
        acts_T=jnp.transpose(act_b[:,:-1],(1,0,2))
        def dyn_step(z, a): return dyn.apply(params["dyn"],z,a),z
        z_final,zs_prefix=jax.lax.scan(dyn_step,z0,acts_T)
        zs=jnp.concatenate([jnp.transpose(zs_prefix,(1,0,2)),z_final[:,None,:]],1)
        z_tgt=jax.lax.stop_gradient(z_all)
        weights=jnp.array([rho**t for t in range(T-1)])
        z_t_T   =jnp.transpose(zs[:,:-1],(1,0,2))
        a_T     =acts_T
        r_T     =jnp.transpose(rew_b[:,:-1],(1,0))
        d_T     =jnp.transpose(done_b[:,:-1],(1,0))
        z_t1_T  =jnp.transpose(z_tgt[:,1:],(1,0,2))
        zs_t1_T =jnp.transpose(zs[:,1:],(1,0,2))
        def step_loss(carry, inp):
            w,z_t,a_t,r_t,d_t,z_tgt_t1,zs_t1=inp
            cl=w*jnp.mean(jnp.sum((zs_t1-z_tgt_t1)**2,-1))
            pr=rew_net.apply(params["rew"],z_t,a_t)
            rl=w*jnp.mean((pr-r_t)**2)
            z_n=jax.lax.stop_gradient(z_tgt_t1)
            pi_a=pi_net.apply(tp["pi"],z_n)
            v_n=jnp.min(q_net.apply(tp["q"],z_n,pi_a),-1)
            v_n=jnp.maximum(v_n,0.0)
            td=r_t+gamma*(1-d_t)*jax.lax.stop_gradient(v_n)
            qp=q_net.apply(params["q"],z_t,a_t)
            vl=w*jnp.mean(jnp.sum((qp-td[:,None])**2,-1))
            pi2=pi_net.apply(params["pi"],jax.lax.stop_gradient(z_t))
            pl=-w*jnp.mean(jnp.min(
                q_net.apply(jax.lax.stop_gradient(params["q"]),jax.lax.stop_gradient(z_t),pi2),-1))
            return carry,(cl,rl,vl,pl)
        _,(cls,rls,vls,pls)=jax.lax.scan(
            step_loss,None,(weights,z_t_T,a_T,r_T,d_T,z_t1_T,zs_t1_T))
        n=T-1
        return (2*jnp.sum(cls)+2*jnp.sum(rls)+jnp.sum(vls)+0.1*jnp.sum(pls))/n, \
               {"c":jnp.sum(cls)/n,"r":jnp.sum(rls)/n,"v":jnp.sum(vls)/n,"p":jnp.sum(pls)/n}

    @jax.jit
    def step(params, tp, opt, ob, ab, rb, db):
        (loss,aux),grads=jax.value_and_grad(loss_fn,has_aux=True)(params,tp,ob,ab,rb,db)
        upd,nopt=tx.update(grads,opt,params)
        new_params=optax.apply_updates(params,upd)
        new_tp=jax.tree_util.tree_map(lambda t,p:(1-tau)*t+tau*p,tp,new_params)
        return new_params,new_tp,nopt,loss,aux
    return step


def make_mppi_fn(enc, dyn, rew_net, q_net, pi_net,
                 horizon=5, n_samples=512, n_iter=6, temp=0.5,
                 act_low=-1.0, act_high=1.0, act_dim=4, gamma=0.99):
    _gammas = jnp.array([gamma**t for t in range(horizon)])
    _gamma_H = float(gamma**horizon)
    @jax.jit
    def plan(params, obs, mu, key):
        z0=jnp.tile(enc.apply(params["enc"],obs[None]),(n_samples,1))
        def one_iter(carry,_):
            mu_i,k=carry; k,sk=jax.random.split(k)
            noise=jax.random.normal(sk,(n_samples,horizon,act_dim))*0.5
            acts=jnp.clip(mu_i[None]+noise,act_low,act_high)
            def rollout_one(args):
                z_i,a_seq=args
                def env_step(z,a):
                    r=rew_net.apply(params["rew"],z[None],a[None]).squeeze()
                    z2=dyn.apply(params["dyn"],z[None],a[None]).squeeze(0)
                    return z2,r
                zf,rs=jax.lax.scan(env_step,z_i,a_seq)
                pi_a=pi_net.apply(params["pi"],zf[None])
                vt=jnp.maximum(jnp.min(q_net.apply(params["q"],zf[None],pi_a)),0.0)
                return jnp.sum(_gammas*rs)+_gamma_H*vt
            rets=jax.vmap(rollout_one)((z0,acts))
            w=jax.nn.softmax((rets-rets.max())/(temp+1e-8))
            new_mu=jnp.einsum("n,nha->ha",w,acts)
            return (new_mu,k),None
        (muf,_),_=jax.lax.scan(one_iter,(mu,key),None,length=n_iter)
        action=jnp.clip(muf[0],act_low,act_high)
        new_mu=jnp.concatenate([muf[1:],jnp.zeros((1,act_dim))],0)
        return action,new_mu
    return plan


# ─── mujoco_playground env helpers ────────────────────────────────────────────

def extract_obs(ts):
    return np.concatenate([np.asarray(v, np.float32).flatten()
                           for v in ts.observation.values()])


def eval_pi(eval_env, act_fn_jit, params, n_eps):
    rets = []
    for _ in range(n_eps):
        ts = eval_env.reset(); obs = extract_obs(ts); er = 0.0
        while not ts.last():
            a = np.array(act_fn_jit(params, jnp.asarray(obs)))
            if not np.all(np.isfinite(a)):
                a = np.zeros_like(a)
            ts = eval_env.step(np.clip(a, -1.0, 1.0))
            er += float(ts.reward); obs = extract_obs(ts)
        rets.append(er)
    return np.mean(rets)


def eval_mppi(eval_env, plan, params, n_eps, H, act_dim, key):
    rets = []
    for _ in range(n_eps):
        key, ek = jax.random.split(key)
        te = eval_env.reset(); oe = extract_obs(te)
        mu = jnp.zeros((H, act_dim)); er = 0.0
        while not te.last():
            act, mu = plan(params, jnp.asarray(oe), mu, ek)
            key, ek = jax.random.split(key)
            a = np.array(act)
            if not np.all(np.isfinite(a)):
                a = np.zeros_like(a)
            te = eval_env.step(a)
            er += float(te.reward); oe = extract_obs(te)
        rets.append(er)
    return np.mean(rets), key


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from dm_control import suite

    SEED        = 42
    TOTAL       = 2_000_000
    WARMUP      = 25_000
    BS          = 256
    SEQ         = 6
    PI_EVAL_EVERY   = 50_000
    MPPI_EVAL_EVERY = 100_000
    EVAL_EPS    = 5
    LR          = 3e-4
    LATENT      = 128
    HIDDEN      = (256, 256)
    GAMMA       = 0.99
    TAU         = 0.01
    H           = 5
    NS          = 256
    NI          = 6
    TEMP        = 0.5
    EXPL_NOISE  = 0.3
    EXPL_UNTIL  = 25_000

    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    env      = suite.load("hopper", "hop", task_kwargs={"random": SEED})
    eval_env = suite.load("hopper", "hop", task_kwargs={"random": SEED + 1})
    obs_dim  = sum(np.prod(v.shape) for v in env.observation_spec().values())
    act_dim  = env.action_spec().shape[0]
    al = float(env.action_spec().minimum[0])
    ah = float(env.action_spec().maximum[0])
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  act=[{al},{ah}]")

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

    _param_labels = {'enc':'world','dyn':'world','rew':'world','q':'q','pi':'world'}
    tx = optax.multi_transform(
        {'world': optax.chain(optax.clip_by_global_norm(10.0), optax.adam(LR)),
         'q':     optax.chain(optax.clip_by_global_norm(1.0),  optax.adam(LR))},
        _param_labels)
    opt = tx.init(params)

    upd  = make_update_fn(enc, dyn, rn, qn, pn, tx, GAMMA, tau=TAU)
    plan = make_mppi_fn(enc, dyn, rn, qn, pn, H, NS, NI, TEMP, al, ah, act_dim, GAMMA)

    @jax.jit
    def act_fn(params, obs):
        z = enc.apply(params["enc"], obs[None])
        return pn.apply(params["pi"], z)[0]

    buf = FastBuffer(2_000_000, obs_dim, act_dim, SEQ)
    rng = np.random.default_rng(SEED)

    out_dir = "/workspace/helios-rl/exp/tdmpc_dmc"
    os.makedirs(out_dir, exist_ok=True)
    csv = f"{out_dir}/hopper-hop.csv"
    with open(csv, "w") as f:
        f.write("step,reward,seed\n")

    # ── Warmup ──────────────────────────────────────────────────────────────
    print("Warmup...", flush=True)
    ts = env.reset(); obs = extract_obs(ts)
    for _ in range(WARMUP):
        a = np.random.uniform(al, ah, (act_dim,)).astype(np.float32)
        ts2 = env.step(a)
        buf.add(obs, a, float(ts2.reward), ts2.last())
        obs = extract_obs(env.reset()) if ts2.last() else extract_obs(ts2)

    # ── Compile ─────────────────────────────────────────────────────────────
    print(f"Buffer size: {buf.size}  Compiling...", flush=True)
    t_c = time.time()
    ob, ab, rb, db = buf.sample(BS, rng)
    params, tp, opt, loss, aux = upd(params, tp, opt,
        jnp.asarray(ob), jnp.asarray(ab), jnp.asarray(rb), jnp.asarray(db))
    jax.block_until_ready(params["enc"])
    print(f"Update compiled in {time.time()-t_c:.1f}s  loss={float(loss):.4f}", flush=True)

    te = eval_env.reset(); oe = extract_obs(te)
    mu_e = jnp.zeros((H, act_dim)); key, ek = jax.random.split(key)
    act_e, _ = plan(params, jnp.asarray(oe), mu_e, ek)
    jax.block_until_ready(act_e)
    print("MPPI compiled.", flush=True)

    # ── Training loop ────────────────────────────────────────────────────────
    ts = env.reset(); obs = extract_obs(ts); gs = WARMUP; t0 = time.time()
    print("Training...", flush=True)

    while gs < TOTAL:
        if gs < EXPL_UNTIL:
            a = np.random.uniform(al, ah, (act_dim,)).astype(np.float32)
        else:
            ap = np.array(act_fn(params, jnp.asarray(obs)))
            if not np.all(np.isfinite(ap)):
                ap = np.zeros(act_dim, np.float32)
            a = np.clip(ap + np.random.normal(0, EXPL_NOISE, (act_dim,)), al, ah).astype(np.float32)

        try:
            ts2 = env.step(a)
        except Exception:
            ts = env.reset(); obs = extract_obs(ts); continue

        buf.add(obs, a, float(ts2.reward), ts2.last())
        obs = extract_obs(env.reset()) if ts2.last() else extract_obs(ts2)
        gs += 1

        ob, ab, rb, db = buf.sample(BS, rng)
        params, tp, opt, loss, aux = upd(params, tp, opt,
            jnp.asarray(ob), jnp.asarray(ab), jnp.asarray(rb), jnp.asarray(db))

        if gs % 5_000 == 0 and gs % PI_EVAL_EVERY != 0:
            elapsed = time.time() - t0
            print(f"  gs={gs:>9,}  sps={gs/elapsed:.0f}  loss={float(loss):.4f}", flush=True)

        if gs % PI_EVAL_EVERY == 0:
            pi_ret = eval_pi(eval_env, act_fn, params, EVAL_EPS)
            elapsed = time.time() - t0
            print(f"step={gs:>9,}  pi={pi_ret:7.1f}  sps={gs/elapsed:.0f}  "
                  f"c={float(aux['c']):.3f} r={float(aux['r']):.3f} v={float(aux['v']):.3f} p={float(aux['p']):.3f}",
                  flush=True)

        if gs % MPPI_EVAL_EVERY == 0:
            mr, key = eval_mppi(eval_env, plan, params, EVAL_EPS, H, act_dim, key)
            elapsed = time.time() - t0
            print(f"step={gs:>9,}  MPPI={mr:7.1f}  sps={gs/elapsed:.0f}", flush=True)
            with open(csv, "a") as f:
                f.write(f"{gs},{mr:.1f},{SEED}\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s  ->  {csv}", flush=True)


if __name__ == "__main__":
    main()

