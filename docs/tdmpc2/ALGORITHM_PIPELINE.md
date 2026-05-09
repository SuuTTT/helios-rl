# TD-MPC2 JAX — Complete Algorithm Pipeline

**Implementation**: `helios-rl/scripts/train_tdmpc_hopper_v4.py`  
**Status**: Working v4f — MPPI=182 at 10M steps (reference: 449 at 4M, UTD=1:1)

---

## 0. Architecture at a Glance

Five networks, one shared Adam optimizer, one EMA target copy:

```
obs_t ──enc──► z_t ──dyn(a_t)──► z_{t+1} ──dyn(a_{t+1})──► ... z_{t+H}
               │                   │
               ├──rew(a_t)──► r̂_t  ├──rew(a_{t+1})──► r̂_{t+1}
               ├──Q(a_t)──►  Q̂_t  ├──Q(a_{t+1})──► Q̂_{t+1}
               └──π ──► â_t        └──π ──► â_{t+1}   (training noise, or MPPI)
```

| Network | Class | Input → Output | Activation |
|---------|-------|----------------|-----------|
| Encoder `enc` | `Encoder` | obs (15) → z (128) | SiLU + LayerNorm + **SimNorm** |
| Dynamics `dyn` | `Dynamics` | (z,a) (132) → z' (128) | SiLU + LayerNorm + **SimNorm** |
| Reward `rew` | `RewardHead` | (z,a) (132) → r̂ (scalar) | SiLU + LayerNorm, no final act |
| Critic `q` | `QEnsemble` | (z,a) (132) → [Q₁,Q₂] | SiLU + LayerNorm, no final act |
| Policy `pi` | `Pi` | z (128) → a (4) | SiLU + LayerNorm + **tanh** |

**Target network `tp`**: EMA copy of all five — `tp ← (1−τ)·tp + τ·params`, τ=0.01.  
Used **only** for the Bellman bootstrap in the Q loss; never in MPPI.

---

## 1. SimNorm — Bounded Latent Space

```python
# train_tdmpc_hopper_v4.py:26-31
def simnorm(x, V=8):
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)  # (batch, V, latent//V)
    x = jax.nn.softmax(x, axis=-1)          # softmax per group
    return x.reshape(*s)                    # (batch, latent)
```

Applied to **both** `Encoder` and `Dynamics` output layers.

**Why it is non-negotiable**: Without bounded latents, the consistency loss
`||dyn(z,a) − enc(o')||²` grows as scale², causing gradient explosions
within the first 100k steps (empirically: `c=77` vs healthy `c≈0.05–0.15`).
SimNorm with V=8 partitions 128 dims into 8 groups of 16, applies softmax within
each group → values ∈ [0,1], max MSE per sample ≤ 128.

---

## 2. Replay Buffer

```python
# MultiEnvBuffer, train_tdmpc_hopper_v4.py:73-111
self.obs  = np.zeros((N_ENVS=1024, cap=12000, obs_dim=15), np.float32)
self.acts = np.zeros((N_ENVS,      cap,       act_dim=4),  np.float32)
self.rews = np.zeros((N_ENVS,      cap),                   np.float32)
self.done = np.zeros((N_ENVS,      cap),                   np.float32)
```

- `N_ENVS=1024` independent ring buffers (capacity ~12k each, total ~737 MB).
- **Write** (`add_batch`): all N environments written in one vectorised numpy op.
- **Sample** (`sample`): B=256 sequences of T=6, fully vectorised (no Python loop):
  ```python
  env_ids = rng.choice(valid, size=B)               # pick B envs
  starts  = (rng.random(B) * (size - T)).astype(int)
  idx     = starts[:, None] + np.arange(T)[None, :] # (B, T) index grid
  return obs[env_ids[:, None], idx], ...             # (B, T, 15)
  ```

---

## 3. One Global Step — Collection

```python
# train_tdmpc_hopper_v4.py:375-400 (main training loop)
if env_steps < EXPL_UNTIL:                           # first 25k steps
    acts_np = np.random.uniform(al, ah, (N_ENVS, 4)) # pure random
else:
    acts_jax = act_fn_batch(params, obs_jax)         # π(enc(obs)) for all 1024 envs
    noise    = np.random.normal(0, σ=0.3, size=(N_ENVS, 4))
    acts_np  = np.clip(acts_jax + noise, -1, 1)      # policy + Gaussian exploration
env_state = batch_step(env_state, acts_jax)
buf.add_batch(obs_np, acts_np, rew_np, done_np)
env_steps += N_ENVS   # 1024 env transitions added
```

Each global step adds **N_ENVS=1024** transitions to the buffer.

---

## 4. K_UPDATE Gradient Steps per Global Step

```python
# train_tdmpc_hopper_v4.py:402-410
for _ in range(K_UPDATE=64):
    samp = buf.sample(BS=256, rng)
    ob, ab, rb, db = [jnp.asarray(x) for x in samp]  # CPU → GPU
    params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db)
```

**UTD ratio** = 64 / 1024 = **1/16**.  
Reference TD-MPC2 uses UTD=1:1 (single env, 1 update/step → 4× more gradient steps at same env count).

---

## 5. Loss Function — `make_update_fn` (the heart of the algorithm)

### 5.1 Encode the full batch

```python
# train_tdmpc_hopper_v4.py:118-122
B, T, _ = obs_b.shape  # (256, 6, 15)
z_all = enc.apply(params["enc"], obs_b.reshape(B*T, -1)).reshape(B, T, -1)  # (256, 6, 128)
z0    = z_all[:, 0]                                                          # (256, 128)
```

All T=6 observations are encoded in a single batched forward pass.  
`z_all[b, t]` = **ground-truth** latent for sample b, timestep t.

### 5.2 Dynamics rollout (lax.scan)

```python
# train_tdmpc_hopper_v4.py:123-126
acts_T = jnp.transpose(act_b[:, :-1], (1, 0, 2))  # (T-1=5, B, 4)

def dyn_step(z, a): return dyn.apply(params["dyn"], z, a), z
z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
zs = cat([transpose(zs_prefix), z_final[:, None, :]], axis=1)  # (B, T, 128)
```

`zs[b, t]` = latent predicted by **dynamics** from z0 with real actions through t.  
`z_all[b, t]` = latent from **encoder** applied to real obs[t].  
These diverge because dynamics is imperfect — the consistency loss closes the gap.

### 5.3 Stop-gradient boundary (asymmetric consistency)

```python
z_tgt = jax.lax.stop_gradient(z_all)   # train_tdmpc_hopper_v4.py:128
```

Consistency loss gradients update **dynamics** to match encoder, but do **not** pull
the encoder toward dynamics. Without `stop_gradient`, the encoder degenerates to a
constant (trivially making consistency loss = 0).

### 5.4 Per-timestep loss with geometric decay (lax.scan)

```python
# train_tdmpc_hopper_v4.py:130-165
weights = [ρ^0, ρ^1, ..., ρ^{T-2}]  # ρ=0.5 → [1.0, 0.5, 0.25, 0.125, 0.0625]

def step_loss(carry, inp):
    w, z_t, a_t, r_t, d_t, z_tgt_{t+1}, zs_{t+1} = inp
    ...

_, (cls, rls, vls, pls) = jax.lax.scan(step_loss, None, stacked_inputs)
```

Weight ρ^t decays each timestep's contribution. Earlier timesteps (closer to real obs)
have more reliable dynamics predictions and contribute more to the loss.

#### 5.4.1 Consistency Loss (world model)

```python
cl = w * mean( sum( (zs_{t+1} - z_tgt_{t+1})^2, axis=-1 ) )
```

MSE between dynamics-predicted next-latent and encoder's next-latent.  
Trains the dynamics model to be consistent with what the encoder sees.

#### 5.4.2 Reward Loss (world model)

```python
pr = rew_net.apply(params["rew"], z_t, a_t)           # predicted reward
rl = w * mean( (pr - rew_scale * r_t)^2 )             # MSE vs SCALED target
```

`rew_scale=10.0` multiplies the reward target, giving 100× larger gradient signal.  
Without scaling: per-step reward ≈ 0.01 → MSE ≈ 10⁻⁴ → gradient ≈ 0 → `r=0.000`.  
With scaling: target ≈ 0.1–0.5 → MSE ≈ 0.01–0.25 → healthy learning signal.

**z_t here is the dynamics-predicted latent** (`zs[:, t]`), not encoder's latent.
This matches MPPI inference, where only dynamics latents are available.

#### 5.4.3 Q Loss — Bellman TD Target

```python
z_n  = stop_gradient(z_tgt_{t+1})           # true next-latent, detached
pi_a = pi_net.apply(tp["pi"], z_n)           # target policy action
v_n  = max(min(q_net.apply(tp["q"], z_n, pi_a), axis=-1), 0.0)
td   = rew_scale * r_t + γ * (1 - d_t) * stop_gradient(v_n)   # TD target

qp   = q_net.apply(params["q"], z_t, a_t)   # predicted Q, shape (B, 2)
vl   = w * mean( sum( (qp - td[:, None])^2, axis=-1 ) )
```

- TD target uses `tp` (target network) for stable bootstrap.
- `z_tgt_{t+1}` (encoder's latent) gives better next-state estimate than dynamics latent.
- `v_n ≥ 0` clamp: Q should be non-negative (rewards are non-negative for locomotion).
- Both Q heads trained to the same scalar target.

**Why tp["pi"] and tp["q"] (not params)?**  
If training Q's own bootstrap target, the target shifts every gradient step → instability.
Target network provides a slowly-moving stable reference (τ=0.01 EMA).

#### 5.4.4 Policy Loss (actor update)

```python
pi2 = pi_net.apply(params["pi"], stop_gradient(z_t))
pl  = -w * mean( min(
    q_net.apply(stop_gradient(params["q"]), stop_gradient(z_t), pi2),
    axis=-1) )
```

Maximise the pessimistic Q (min of twin critics) under the current policy.  
`stop_gradient` on `z_t` and `params["q"]`: policy gradient only updates `params["pi"]`.  
`min(Q₁, Q₂)` avoids exploiting Q overestimation.

### 5.5 Combined loss and optimizer step

```python
# train_tdmpc_hopper_v4.py:162-178
total_loss = (2*sum(cls) + 2*sum(rls) + sum(vls) + 0.1*sum(pls)) / (T-1)
```

Loss weights: consistency×2, reward×2, Q×1, policy×0.1.  
Policy gets 0.1× because its gradient magnitude is larger and would otherwise
dominate the encoder representation.

```python
@jax.jit
def step(params, tp, opt, ob, ab, rb, db):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ...)
    upd, nopt = tx.update(grads, opt, params)    # Adam + gradient clip
    new_params = optax.apply_updates(params, upd)
    new_tp = tree_map(λ t,p: (1-τ)t + τp, tp, new_params)   # EMA target update
    return new_params, new_tp, nopt, loss, aux
```

**Optimizer** (`optax.multi_transform`):
- `enc, dyn, rew, pi` → `clip_global_norm(10.0)` + `adam(3e-4)`
- `q` → `clip_global_norm(1.0)` + `adam(3e-4)` (tighter clip; Q gradients are ×10–100 larger)

---

## 6. MPPI Planning — `make_mppi_fn`

Called at evaluation time; also used to generate actions in `plan_eval`.

### 6.1 Encode current observation

```python
z0_single = enc.apply(params["enc"], obs[None])[0]   # (latent,)
z0 = tile(z0_single, (N_SAMPLES=256, 1))             # (256, latent)
```

### 6.2 π warm-start (prevents zero-action collapse)

```python
def pi_step(z, _):
    a  = pi_net.apply(params["pi"], z[None])[0]
    z2 = dyn.apply(params["dyn"], z[None], a[None])[0]
    return z2, a
_, pi_traj = jax.lax.scan(pi_step, z0_single, None, length=H=5)  # (5, 4)

mu_ws = mu.at[0].set(pi_traj[0])   # warm-start mean with first policy action
```

Without this: all 256 sample returns ≈ equal early in training → softmax ≈ uniform →
weighted mean ≈ 0 → MPPI outputs zero action every step.

### 6.3 Iterative CEM refinement (N_ITER=6 × lax.scan)

```python
def one_iter(carry, _):
    mu_i, key = carry
    noise = jax.random.normal(key, (256, H=5, 4)) * σ=0.5
    acts  = clip(mu_i[None] + noise, -1, 1)       # (256, 5, 4)
    acts  = acts.at[-1].set(pi_traj)               # last sample = noiseless π baseline
    ...
```

The last sample is always the noiseless policy trajectory — guarantees
MPPI return ≥ policy return in the learned model.

### 6.4 Vectorised H-step rollout (vmap over 256 samples)

```python
def rollout_one(z_i, a_seq):                       # single trajectory
    def env_step(z, a):
        r  = rew_net.apply(params["rew"], z[None], a[None]).squeeze() / rew_scale
        z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
        return z2, r
    zf, rs = jax.lax.scan(env_step, z_i, a_seq)    # H steps through world model
    pi_a = pi_net.apply(params["pi"], zf[None])
    vt   = max(min(q_net.apply(params["q"], zf[None], pi_a)), 0.0) / rew_scale
    return sum(γ^t * r_t, t=0..H-1) + γ^H * vt    # discounted return + terminal V

rets = jax.vmap(rollout_one)((z0, acts))            # (256,) — all rollouts in parallel
```

All network calls inside `rollout_one` use `params` (current weights, **not** `tp`).
Using `tp["q"]` here would mismatch with `params["enc"]` — latents from the current
encoder are not interpretable by the lagged Q network.

### 6.5 Softmax weighting and mean update

```python
w       = softmax((rets - rets.max()) / (temp=0.5 + 1e-8))   # (256,)
new_mu  = einsum("n,nha->ha", w, acts)                        # (H, 4)
```

`rets.max()` subtraction: numerically stable softmax.  
Low temperature (temp=0.5) concentrates weight on the best trajectories.

### 6.6 Receding horizon warm-start for next step

```python
action  = clip(muf[0], -1, 1)                         # take first planned action
new_mu  = concat([muf[1:], pi_traj[-1:]], axis=0)     # shift plan left, fill with π
```

After acting, the remaining plan `muf[1:]` seeds the next MPPI call.
This temporal consistency gives MPPI a significant advantage over replanning from scratch.

---

## 7. Complete Training Loop (one epoch = one global step)

```
┌─────────────────────────────────────────────────────────────┐
│ Warmup (25k env steps, pure random)                         │
│   ▶ fill buffer with diverse initial data                   │
│   ▶ no gradient updates                                     │
├─────────────────────────────────────────────────────────────┤
│ Repeat until env_steps = 10M:                               │
│                                                             │
│  [COLLECT] env_steps += 1024                                │
│   for i in range(N_ENVS=1024):                              │
│     a_i = π(enc(obs_i)) + N(0, 0.3)    ← policy + noise   │
│   env_state = batch_step(env_state, acts)  ← JAX vmap      │
│   buf.add_batch(obs, acts, rews, dones)                     │
│                                                             │
│  [UPDATE × 64]                                              │
│   for k in range(K_UPDATE=64):                              │
│     (obs,acts,rews,dones) = buf.sample(B=256, T=6)         │
│     params, tp, opt = step(params, tp, opt, ...)            │
│       ├─ enc: encode all B×T obs in one batched call        │
│       ├─ dyn: unroll T-1 steps via lax.scan                 │
│       ├─ loss: consistency + reward + Q + policy            │
│       ├─ grad: jax.value_and_grad                           │
│       ├─ opt: Adam(3e-4) + gradient clip                    │
│       └─ tp: EMA update (τ=0.01)                            │
│                                                             │
│  [EVAL every 500k steps] — deterministic policy             │
│   pi_return = run_episode(act_fn = π(enc(o)))               │
│                                                             │
│  [EVAL every 1M steps] — MPPI planning                      │
│   mppi_return = run_episode(act_fn = plan(params, o, μ, k)) │
│   write step,mppi_return,seed to CSV                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Key Design Decisions and Their Justifications

| Decision | Code Location | Justification |
|----------|--------------|---------------|
| **SimNorm(V=8)** on enc+dyn | `Encoder.__call__`, `Dynamics.__call__` | Without it: `c=77`, divergence. With it: `c≈0.1`. |
| **rew_scale=10** on reward target | `step_loss` line `rl=` | Per-step reward ≈0.01; raw MSE≈1e-4≈0. Scale gives `r≈0.03+`. |
| **Asymmetric stop_gradient** on z_tgt | `z_tgt = stop_gradient(z_all)` | Prevents encoder collapse (minimize consistency = map all obs to same z). |
| **π warm-start** in MPPI | `mu_ws = mu.at[0].set(pi_traj[0])` | Early training: all returns ≈ equal → softmax uniform → MPPI=0. Fix makes MPPI ≥ π. |
| **params["q"] in MPPI** (not tp["q"]) | `rollout_one` q call | Mixing current enc with lagged Q breaks latent alignment; empirically: MPPI 64→6. |
| **Pessimistic Q** (min of twins) | `jnp.min(qp, axis=-1)` | Reduces Q overestimation → prevents inflated MPPI terminal values. |
| **Q grad clip 1.0 vs 10.0** | `optax.multi_transform` | Q values ∈ [0,500]; without tighter clip, Q gradients dominate the encoder. |
| **Multi-head gradients to encoder** | All losses flow to enc | Enc needs Q-gradient to learn value-predictive latents. stop_grad on enc → `pi≈3`. |
| **lax.scan for dyn unroll + loss** | `dyn_step`, `step_loss` scans | Single compiled XLA op; no Python loop overhead; enables JIT compilation. |
| **Geometric weights ρ^t** | `weights = [0.5^t ...]` | Dynamics error compounds; later timesteps less reliable; down-weight accordingly. |

---

## 9. Known Gaps vs Official TD-MPC2 (Priority Order)

| # | Gap | Our Code | Official Code | Impact |
|---|-----|----------|--------------|--------|
| 1 | **UTD ratio** | 1/16 (N=1024, K=64) | ~1:1 (N=1, K=1) | 🔴 PRIMARY — 16× fewer grad steps at 4M |
| 2 | **Reward+Q loss function** | MSE + manual `rew_scale` | Two-hot categorical on symlog grid | 🔴 Removes need for rew_scale hack |
| 3 | **Policy type** | Deterministic tanh | Stochastic Gaussian + entropy bonus | 🔴 Less exploration, earlier collapse |
| 4 | **Q normalisation in π loss** | None | `RunningScale` (5th-95th %ile) | 🟠 π gradient grows 100× as Q scales |
| 5 | **Q ensemble size** | 2 (fixed) | 5 (random 2-subset per call) | 🟠 More overestimation variance |
| 6 | **MPPI sampling** | Full softmax, fixed σ=0.5 | Elite top-64, dynamic σ update | 🟠 Bad trajectories dilute weights |
| 7 | **MPPI π trajectories** | 1 deterministic | 24 stochastic | 🟡 Only 1 baseline candidate |
| 8 | **Weight init** | Glorot uniform | trunc_normal(std=0.02) + zero output heads | 🟡 Noisy early bootstrap |
| 9 | **Discount γ** | Fixed 0.99 | Episode-length-scaled (≈0.995 at T=1000) | 🟢 Negligible |
| 10 | **Activation** | SiLU | Mish | 🟢 Negligible |

**Performance at 4M env steps**: Ours 138 MPPI vs reference 449. Gap is almost entirely
explained by 16× UTD deficit (250k vs 4M gradient updates).

**Recommended fix (Phase 1)**:
1. N_ENVS=256, K_UPDATE=256 → UTD=1:1 at same wall time.
2. Implement two-hot loss (removes rew_scale sensitivity).
3. Stochastic policy + entropy coef.

---

## 10. Metrics Glossary (logged every 500k steps as `c= r= v= p=`)

| Metric | Meaning | Healthy Range | Problem Indicator |
|--------|---------|--------------|-------------------|
| `c` | Consistency MSE: `||dyn(z,a) − enc(o')||²` | 0.05–0.15 | `>5`: latent explosion; `≈0`: collapse |
| `r` | Reward loss: `(rew_net(z,a) − rew_scale·r)²` | 0.001–0.5 | `=0.000`: reward head dead |
| `v` | Q/value loss: `(Q(z,a) − TD_target)²` | 1–30 | Monotone rise: Q divergence |
| `p` | Policy loss: `−mean(min_Q(z, π(z)))` | grows negative | Stagnation while `v` spikes → Q corruption |
| `pi` | Undiscounted return, deterministic policy | 50+ at 1M | `≈0` at 300k: UTD too low |
| `MPPI` | Undiscounted return, MPPI planning | `≥ pi` ideally | `MPPI < pi`: world model misleading planner |
