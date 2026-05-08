# TD-MPC2 JAX — Code Walkthrough

**Script**: `helios-rl/scripts/train_tdmpc_hopper_v4.py`  
**Task**: mujoco_playground HopperHop (JAX/Warp GPU backend)  
**Final results**: `/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop.csv`

---

## Pipeline Overview (read this first)

```
obs_t  ──enc──►  z_t  ──dyn(a_t)──►  z_{t+1}  ──dyn(a_{t+1})──►  ...  z_{t+H}
                  │                     │
                  └──rew(a_t)──► r̂_t   └──rew(a_{t+1})──► r̂_{t+1}
                  │                     │
                  └──Q(a_t)──► Q̂_t     └──Q(a_{t+1})──► Q̂_{t+1}
                  │
                  └──π──► â_t   (greedy policy, used both for training and as MPPI warm-start)
```

Five networks are trained jointly on sequences of length T=6 from the replay buffer:

1. **Encoder** `enc(obs) → z`: maps raw obs (dim 15) to latent (dim 128)
2. **Dynamics** `dyn(z, a) → z'`: predicts next latent given current latent + action
3. **Reward** `rew(z, a) → r̂`: predicts per-step reward in latent space
4. **Q-ensemble** `Q(z, a) → [Q₁, Q₂]`: two Q-networks (twin critics)
5. **Policy** `π(z) → a`: deterministic greedy actor (tanh-bounded to [-1, 1])

**Target network** `tp`: a slow EMA copy of all five networks (`tp ← (1-τ)*tp + τ*params`).
Used only to bootstrap TD targets in the Q loss; not used in MPPI.

**Training loss** (summed over T-1 time steps, decayed by ρ^t):
```
L = 2·L_consistency + 2·L_reward + 1·L_Q + 0.1·L_policy
```

**Action selection at training time**: Gaussian noise (σ=0.3) added to π output,
clipped to [-1,1]. Pure random during warmup.

**Action selection at eval time (MPPI)**: Model-Predictive Path Integral.
256 action sequences are sampled around the current mean `mu`, each is rolled out
H=5 steps through the world model (dyn+rew), returns are computed, softmax
weights the sequences by their return, and `mu` is updated as the weighted mean.
The first action of the updated `mu` is taken.

---

## Section 1: Imports and Environment Setup (lines 1–21)

```python
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp, optax
import flax.linen as nn
```

- `jax`: XLA-compiled numerical computing. Enables `jit`, `vmap`, `lax.scan`.
- `jax.numpy`: Drop-in numpy replacement that runs on GPU via XLA.
- `optax`: Gradient transformation library (Adam, gradient clipping, etc.).
- `flax.linen`: Neural network module system for JAX. Parameters are explicit dicts,
  not hidden inside module state. `nn.compact` allows defining layers in `__call__`.

```python
sys.path.insert(0, "/workspace/wiki/learn_mujoco_playground/repo")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
```

- `MUJOCO_GL=egl`: Use headless OpenGL (EGL) for rendering on a GPU server without display.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: Don't pre-allocate all GPU memory at startup.
  Prevents OOM when running alongside other processes.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`: Limit JAX to 60% of GPU VRAM.

---

## Section 2: SimNorm (lines 26–31)

```python
def simnorm(x, V=8):
    s = x.shape
    x = x.reshape(*s[:-1], V, s[-1] // V)  # (batch, latent) → (batch, V, latent//V)
    x = jax.nn.softmax(x, axis=-1)          # softmax within each group
    return x.reshape(*s)                    # back to (batch, latent)
```

**What it does**: Partitions the last dimension into V groups of equal size, applies
softmax within each group, then flattens back.

For latent_dim=128 and V=8:
- Groups: 8 groups × 16 dimensions each
- Each group sums to 1.0 (softmax)
- All values in [0,1]

**Why this is necessary**: Without normalization, the encoder's output scale grows
over training (encoder weights drift). The consistency loss `||dyn(z,a) - enc(o')||²`
then grows as `scale²` because the latent magnitude is unbounded. This causes
gradient explosions early in training. SimNorm forces a bounded output permanently.

**Why softmax over alternatives**: Unlike L2 normalization (`z/||z||`), softmax
preserves the relative magnitude structure within each group. It's differentiable
everywhere. The partition into V groups prevents all information collapsing to a
single softmax (which would be a probability distribution over 128 classes).

---

## Section 3: Neural Network Modules (lines 34–70)

### NormMLP — shared building block

```python
class NormMLP(nn.Module):
    dims: tuple; out: int
    @nn.compact
    def __call__(self, x):
        for d in self.dims:
            x = nn.Dense(d)(x)    # linear projection
            x = nn.LayerNorm()(x) # normalize across features (not batch)
            x = nn.silu(x)        # smooth activation: x * sigmoid(x)
        return nn.Dense(self.out)(x)  # output projection, no activation
```

`LayerNorm` is used rather than `BatchNorm` because:
- Works correctly with batch size 1 during eval
- Does not require tracking running statistics
- Stable under JAX JIT (no mutable state)

`silu` (Swish) is smoother than ReLU, avoids the dying neuron problem, and
empirically performs better in model-based RL where the MLP must be
differentiable for planning.

### Encoder

```python
class Encoder(nn.Module):
    latent_dim: int; hidden: tuple = (128, 128); V: int = 8
    def __call__(self, obs):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(obs), self.V)
```

Maps `obs ∈ ℝ^15` → `z ∈ [0,1]^128` (bounded by SimNorm).
Two hidden layers of 128 units. Final output goes through SimNorm before being
returned — the latent is *always* normalized, even during forward passes at eval time.

### Dynamics

```python
class Dynamics(nn.Module):
    def __call__(self, z, a):
        return simnorm(NormMLP(self.hidden, self.latent_dim)(jnp.concatenate([z, a], -1)), self.V)
```

Maps `(z, a) ∈ [0,1]^128 × [-1,1]^4` → `z' ∈ [0,1]^128`.
Concatenates latent and action, passes through NormMLP, applies SimNorm.
The SimNorm here ensures the predicted next-latent lives in the same space
as the encoder output — both are bounded to [0,1]^128, so the consistency
loss `||z_pred - z_enc||²` is always in [0, latent_dim].

### RewardHead

```python
class RewardHead(nn.Module):
    def __call__(self, z, a):
        return NormMLP(self.hidden, 1)(jnp.concatenate([z, a], -1)).squeeze(-1)
```

Maps `(z, a)` → scalar `r̂`. No activation on output — unconstrained real value.
Trained to predict `rew_scale × r_true` (not `r_true` directly — see Section 6).

### QEnsemble (Twin Critics)

```python
class QEnsemble(nn.Module):
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], -1)
        return jnp.stack([NormMLP(self.hidden, 1)(x).squeeze(-1),
                          NormMLP(self.hidden, 1)(x).squeeze(-1)], -1)
```

Two independent Q-networks sharing the same input `(z,a)`.
Output shape: `(batch, 2)`.
Uses `min(Q₁, Q₂)` at eval time (pessimistic Q) to avoid Q overestimation.
Both networks receive the same input but have independently initialized weights
due to Flax's parameter naming (`Dense_0` vs `Dense_1` within different submodule
instances).

### Policy (Pi)

```python
class Pi(nn.Module):
    action_dim: int; hidden: tuple = (128, 128)
    def __call__(self, z): return jnp.tanh(NormMLP(self.hidden, self.action_dim)(z))
```

Maps `z ∈ [0,1]^128` → `a ∈ (-1,1)^4`.
`tanh` bounds the output to the action space. Deterministic (no stochasticity).
Used both for greedy rollouts in training and as a warm-start in MPPI planning.

---

## Section 4: Multi-Environment Replay Buffer (lines 73–111)

### Structure

```python
self.obs  = np.zeros((n_envs, cap, obs_dim), np.float32)  # (N, C, 15)
self.acts = np.zeros((n_envs, cap, act_dim), np.float32)  # (N, C, 4)
self.rews = np.zeros((n_envs, cap),          np.float32)  # (N, C)
self.done = np.zeros((n_envs, cap),          np.float32)  # (N, C)
self.ptr  = np.zeros(n_envs, np.int64)                    # write pointer per env
self.size = np.zeros(n_envs, np.int64)                    # fill level per env
```

N=1024 independent ring buffers, one per environment.
Each environment writes to its own circular buffer of capacity `cap = TOTAL_ENV//N + 2000 ≈ 12000`.
Total buffer size: 1024 × 12000 × 15 × 4 bytes ≈ 737 MB.

### Writing — `add_batch`

```python
def add_batch(self, obs_b, acts_b, rews_b, done_b):
    p = self.ptr                            # current write positions (N,)
    self.obs[np.arange(self.N), p] = obs_b  # write all N envs at once
    ...
    self.ptr = (p + 1) % self.cap           # advance all pointers
    self.size = np.minimum(self.size + 1, self.cap)
```

All N environments are written simultaneously via advanced numpy indexing.
No Python loop. `ptr` wraps around when it reaches `cap` (ring buffer).

### Sampling — `sample`

```python
def sample(self, B, rng):
    valid = np.where(self.size >= self.T + 1)[0]  # envs with enough data
    env_ids = rng.choice(valid, size=B, replace=True)             # (B,)
    sizes   = self.size[env_ids]                                   # (B,)
    starts  = (rng.random(B) * (sizes - self.T)).astype(np.int64) # (B,)
    idx     = starts[:, None] + np.arange(self.T)[None, :]        # (B, T)
    return (self.obs[env_ids[:, None], idx], ...)                  # (B, T, obs_dim)
```

Key: `self.obs[env_ids[:, None], idx]` is a single numpy fancy-index operation.
`env_ids[:, None]` is shape `(B,1)`, `idx` is shape `(B,T)` — broadcasting selects
`obs[env_ids[i], idx[i,t]]` for all `(i,t)` simultaneously. No Python loops.

Why `size >= T+1`: We need T consecutive obs to form a (obs, action, reward, done)
sequence of length T=6, which includes T-1=5 transitions. Plus one extra for the
final "next obs" in the last transition.

---

## Section 5: Update Function — `make_update_fn` (lines 114–168)

This is the core training step. It's closed over the model objects and optimizer,
then compiled once by `@jax.jit`.

### 5.1 Encode the full sequence

```python
def loss_fn(params, tp, obs_b, act_b, rew_b, done_b):
    B, T, _ = obs_b.shape                                         # (256, 6, 15)
    z_all = enc.apply(params["enc"], obs_b.reshape(B*T, -1))
    z_all = z_all.reshape(B, T, -1)                               # (256, 6, 128)
    z0    = z_all[:, 0]                                           # (256, 128) — initial latent
```

`enc.apply(params["enc"], x)` is the Flax way of calling a module with explicit
parameters. It does NOT modify any state — parameters are passed in, output is returned.
All T=6 observations are encoded in one batched call (B×T=1536 parallel forward passes).

`z_all` contains the *ground truth* latents computed from real observations.
These are used as consistency targets (but with `stop_gradient` — see 5.3).

### 5.2 Unroll the dynamics model

```python
acts_T = jnp.transpose(act_b[:, :-1], (1, 0, 2))  # (T-1, B, 4)

def dyn_step(z, a): return dyn.apply(params["dyn"], z, a), z
z_final, zs_prefix = jax.lax.scan(dyn_step, z0, acts_T)
zs = jnp.concatenate([jnp.transpose(zs_prefix, (1,0,2)), z_final[:, None, :]], 1)
# zs shape: (B, T, 128) — latents predicted purely by dynamics
```

`jax.lax.scan` is JAX's equivalent of a for-loop but compiled into a single XLA op.
`dyn_step` takes the current latent `z` and action `a`, returns `(next_z, z)`.
The second return value accumulates into `zs_prefix` (the input latents at each step).

After scan: `zs_prefix` has shape `(T-1, B, 128)` (the latents *entering* each step),
and `z_final` is the latent after the last step. Concatenating gives a full sequence
`zs[t]` = the dynamics-predicted latent at step t (starting from the *real* z0).

The distinction: `z_all[t]` = encoder applied to real obs[t]; `zs[t]` = dynamics
unrolled from z0 with real actions. These differ because dynamics is imperfect.
The consistency loss minimizes this difference.

### 5.3 Prepare targets (stop_gradient boundary)

```python
z_tgt = jax.lax.stop_gradient(z_all)
```

Critical: the consistency loss targets (`z_tgt`) are *detached* from the computational
graph. Gradients from the consistency loss only update the dynamics network (and the
encoder via `z0 = z_all[:, 0]` which IS in the graph). The loss penalizes the
dynamics for being inconsistent with the encoder, but does NOT penalize the encoder
for being inconsistent with the dynamics. This asymmetry prevents latent collapse
(where the encoder degenerates to a constant to make consistency loss = 0).

### 5.4 Per-timestep loss with geometric discount

```python
weights = jnp.array([rho ** t for t in range(T-1)])  # [1, 0.5, 0.25, ...]

def step_loss(carry, inp):
    w, z_t, a_t, r_t, d_t, z_tgt_t1, zs_t1 = inp
    ...
_, (cls, rls, vls, pls) = jax.lax.scan(step_loss, None, (...))
```

Another `lax.scan` — this one iterates over the T-1=5 time steps within each sequence.
Each step contributes a weighted loss (weight = ρ^t, ρ=0.5), so earlier steps
in the sequence matter more. This is standard in TD-MPC2 to handle the compounding
error of multi-step dynamics rollouts.

### 5.5 Consistency loss

```python
cl = w * jnp.mean(jnp.sum((zs_t1 - z_tgt_t1) ** 2, -1))
```

MSE between the dynamics-predicted next-latent (`zs_t1 = zs[:, t+1]`) and the
encoder's latent for the true next-obs (`z_tgt_t1 = z_tgt[:, t+1]`).
`jnp.sum(..., -1)` sums over latent dimensions → gives per-sample MSE.
`jnp.mean(...)` averages over the batch.
With SimNorm, values are in [0,1]^128, so max possible MSE per sample = 128.
Healthy range: `c ≈ 0.05–0.15` (after normalization).

### 5.6 Reward loss

```python
pr = rew_net.apply(params["rew"], z_t, a_t)     # predicted reward
rl = w * jnp.mean((pr - rew_scale * r_t) ** 2)  # MSE vs SCALED target
```

The reward head predicts `rew_scale × r_true`. At rew_scale=10 and r_true≈0.03,
the target is 0.3 and the MSE is ~0.09 — large enough for a real gradient signal.
Without scaling (target≈0.03), MSE≈0.001, gradient≈2×(0.03−0)×0.03≈0.0018 → negligible.

Note: `z_t = zs[:, t]` — the *predicted* latent from dynamics, not the encoder's latent.
So the reward head sees the dynamics-predicted latent, not the ground-truth encoded latent.
This is intentional: at MPPI time, we only have the dynamics-predicted latent.

### 5.7 Q loss (TD Bellman target)

```python
z_n   = jax.lax.stop_gradient(z_tgt_t1)    # true next latent, detached
pi_a  = pi_net.apply(tp["pi"], z_n)          # target policy action at next state
v_n   = jnp.maximum(jnp.min(q_net.apply(tp["q"], z_n, pi_a), -1), 0.0)
# min over twin critics (pessimistic); clamp at 0 (Q must be non-negative)

td = rew_scale * r_t + gamma * (1 - d_t) * stop_gradient(v_n)
# Bellman target: scaled_reward + γ × V(next) × (1 - done)
# Everything except the current Q is stop_grad — standard TD target

qp = q_net.apply(params["q"], z_t, a_t)     # predicted Q, shape (B, 2)
vl = w * jnp.mean(jnp.sum((qp - td[:, None]) ** 2, -1))
# MSE between both Q heads and the same scalar TD target
```

Why `tp["pi"]` and `tp["q"]` for the Bellman target: Using the training parameters
for bootstrap creates the "moving target" problem — the Q estimate and its bootstrap
are both changing simultaneously, leading to instability. The target network provides
a stable bootstrap by moving slowly (EMA with τ=0.01).

Why `z_tgt_t1` (encoder's latent) and not `zs_t1` (dynamics' latent) for the target:
The Q target should be computed from the best available estimate of the next state.
The encoder has direct access to the true next observation, so it gives a more
accurate latent. The dynamics latent is used for the Q prediction (to match MPPI).

### 5.8 Policy loss

```python
pi2 = pi_net.apply(params["pi"], jax.lax.stop_gradient(z_t))
pl  = -w * jnp.mean(jnp.min(
    q_net.apply(jax.lax.stop_gradient(params["q"]), jax.lax.stop_gradient(z_t), pi2), -1))
```

Maximize Q by minimizing its negative. `stop_gradient(z_t)` — policy gradient
does NOT flow back into the encoder through z_t. `stop_gradient(params["q"])` —
policy gradient does NOT update the Q network (Q is treated as a fixed critic).
Only `params["pi"]` gets gradients from `pl`.

The `jnp.min(..., -1)` over twin critics makes the policy maximize the *pessimistic*
Q estimate — a conservative policy gradient that avoids exploiting Q overestimation.

### 5.9 Combined loss and optimizer step

```python
return (2*sum(cls) + 2*sum(rls) + sum(vls) + 0.1*sum(pls)) / n, {metrics}
```

Weights: 2×consistency, 2×reward, 1×Q, 0.1×policy.
Consistency and reward get 2× because they define the world model quality;
policy gets 0.1× because its gradient is larger in magnitude and would otherwise
dominate the encoder representation.

```python
@jax.jit
def step(params, tp, opt, ob, ab, rb, db):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ...)
    upd, nopt = tx.update(grads, opt, params)
    new_params = optax.apply_updates(params, upd)
    new_tp = jax.tree_util.tree_map(lambda t, p: (1-tau)*t + tau*p, tp, new_params)
    return new_params, new_tp, nopt, loss, aux
```

`jax.value_and_grad`: computes both forward pass (loss) and backward pass (gradients)
in one call. `has_aux=True` means the function returns `(primary_output, aux_dict)`.

`jax.tree_util.tree_map`: applies the EMA function to every leaf in the parameter
pytree simultaneously. JAX parameter dicts are "pytrees" — nested dict/list structures
where leaves are arrays. `tree_map` recurses and applies the lambda to every array pair.

---

## Section 6: MPPI Planning — `make_mppi_fn` (lines 171–221)

MPPI (Model-Predictive Path Integral) is a sampling-based planning algorithm.
At each decision step, it:
1. Samples N action sequences around the current mean
2. Rolls each sequence forward through the world model
3. Weights sequences by `softmax(returns / temp)`
4. Updates the mean as the weighted sum of sequences
5. Repeats for NI iterations
6. Returns the first action of the final mean

### 6.1 Pi warm-start

```python
def pi_step(z, _):
    a  = pi_net.apply(params["pi"], z[None])[0]
    z2 = dyn.apply(params["dyn"], z[None], a[None])[0]
    return z2, a
_, pi_traj = jax.lax.scan(pi_step, z0_single, None, length=horizon)  # (H, 4)

mu_ws = mu.at[0].set(pi_traj[0])
```

`lax.scan` with `xs=None` and `length=H` runs the scan H times without any external
input — the state just carries forward. `pi_traj` is the H-step greedy policy rollout.

`mu.at[0].set(pi_traj[0])`: JAX arrays are immutable. `.at[idx].set(val)` returns a
new array with the value at `idx` replaced by `val`. This sets the first slot of the
MPPI mean to the policy's action, ensuring the center of the sample distribution
is not at zero.

### 6.2 Sampling and baseline injection

```python
noise = jax.random.normal(sk, (n_samples, horizon, act_dim)) * 0.5
acts  = jnp.clip(mu_i[None] + noise, act_low, act_high)  # (N, H, 4)
acts  = acts.at[-1].set(pi_traj)                           # last sample = noiseless pi
```

`mu_i[None]` broadcasts the `(H, 4)` mean to `(1, H, 4)` for addition with `(N, H, 4)` noise.
The last sample is replaced with the noiseless pi trajectory — this guarantees MPPI
returns at least what the policy returns in the model, preventing full collapse.

### 6.3 Vectorised rollout

```python
def rollout_one(args):
    z_i, a_seq = args
    def env_step(z, a):
        r  = rew_net.apply(params["rew"], z[None], a[None]).squeeze() / rew_scale
        z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
        return z2, r
    zf, rs = jax.lax.scan(env_step, z_i, a_seq)
    pi_a = pi_net.apply(params["pi"], zf[None])
    vt   = jnp.maximum(jnp.min(q_net.apply(params["q"], zf[None], pi_a)), 0.0) / rew_scale
    return jnp.sum(_gammas * rs) + _gamma_H * vt

rets = jax.vmap(rollout_one)((z0, acts))
```

`rollout_one` computes the return for a single (initial_latent, action_sequence) pair.
The H-step dynamics + reward rollout is `lax.scan`, so it's a single compiled op.
Terminal value: `V(z_H) = max(min(Q₁(z_H, π(z_H)), Q₂(z_H, π(z_H))), 0)`.
Both predicted rewards and terminal V are divided by `rew_scale` to recover true scale.

`jax.vmap(rollout_one)((z0, acts))` maps `rollout_one` over the N=256 samples in parallel.
`z0` is `(N, latent)`, `acts` is `(N, H, 4)`. `vmap` vectorizes over the leading axis.
This runs all 256 rollouts simultaneously as a single batched GPU computation.

### 6.4 Softmax weighting and mean update

```python
w      = jax.nn.softmax((rets - rets.max()) / (temp + 1e-8))  # (N,)
new_mu = jnp.einsum("n,nha->ha", w, acts)                      # (H, 4)
```

`rets.max()` subtraction: numerically stabilizes the softmax (same result mathematically,
avoids exp overflow from large returns). `temp=0.5` controls sharpness: lower temp →
more weight on the best trajectory, less diversity.

`einsum("n,nha->ha")`: weighted sum over N samples; equivalent to `(w[:, None, None] * acts).sum(0)`.

### 6.5 Mu shift for next step

```python
new_mu = jnp.concatenate([muf[1:], pi_traj[-1:]], 0)
```

After taking action `muf[0]`, shift the remaining plan by one: use `muf[1:]` (the
remaining H-1 planned actions) as the warm-start for the next MPPI call. Append
`pi_traj[-1]` (policy's last-step action) as the filler for the new last slot.
This "receding horizon" warm-start significantly improves MPPI consistency across steps.

---

## Section 7: Main Function — Setup (lines 224–299)

### 7.1 Hyperparameters

| Name | Value | Meaning |
|------|-------|---------|
| `N_ENVS` | 1024 | parallel environments; ~82k env-sps on Warp GPU |
| `TOTAL_ENV` | 10M | total env steps (not gradient steps) |
| `WARMUP_ENV` | 25k | random exploration before any training |
| `BS` | 256 | batch size per gradient step |
| `SEQ` | 6 | sequence length T (5 transitions per sample) |
| `K_UPDATE` | 64 | gradient steps per global step; UTD = 64/1024 |
| `LR` | 3e-4 | Adam learning rate (all heads) |
| `LATENT` | 128 | latent dimension |
| `HIDDEN` | (128,128) | MLP hidden layer sizes (2 layers × 128 units) |
| `GAMMA` | 0.99 | discount factor |
| `TAU` | 0.01 | target network EMA rate (1% blend per step) |
| `H` | 5 | MPPI horizon (steps to look ahead) |
| `NS` | 256 | MPPI number of samples |
| `NI` | 6 | MPPI refinement iterations |
| `TEMP` | 0.5 | MPPI softmax temperature |
| `REW_SCALE` | 10.0 | reward scaling factor |
| `EXPL_NOISE` | 0.3 | Gaussian noise σ for training-time exploration |
| `EXPL_UNTIL` | 25k | env steps of pure random exploration |

### 7.2 Environment setup

```python
env_raw = registry.load("HopperHop")
env     = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)
```

`registry.load` returns the raw MuJoCo MJX environment.
`wrap_for_brax_training` wraps it with Brax-compatible API:
- `env.reset(keys)` → vectorized reset; keys determines initial state randomness
- `env.step(state, actions)` → vectorized step; returns (new_state with .obs/.reward/.done)
- `episode_length=1000` → done=1 after 1000 steps even without falling
- `action_repeat=1` → one physics step per action

```python
@jax.jit
def batch_reset(key):
    return env.reset(jax.random.split(key, N_ENVS))

@jax.jit
def batch_step(state, action):
    return env.step(state, action)
```

These are JIT-compiled wrappers. JAX JIT traces the function once, producing an XLA
computation graph that runs directly on GPU. All N=1024 envs step in parallel.

### 7.3 Parameter initialization

```python
params = {"enc": enc.init(k1, do), "dyn": dyn.init(k2, dz, da),
          "rew": rn.init(k3, dz, da), "q": qn.init(k4, dz, da), "pi": pn.init(k5, dz)}
tp = jax.tree_util.tree_map(lambda x: x, params)
```

`module.init(key, dummy_input)`: Flax module initialization. Traces the `__call__`
method with dummy inputs to determine layer shapes, then initializes all weights.
Returns a nested parameter dict (pytree).

`tp` starts as an exact copy of `params`. Both are separate pytrees in JAX — there is
no shared memory. Over training, `tp` will lag behind `params` via EMA.

### 7.4 Optimizer with separate gradient clipping

```python
_labels = {'enc': 'world', 'dyn': 'world', 'rew': 'world', 'q': 'q', 'pi': 'world'}
tx = optax.multi_transform(
    {'world': optax.chain(optax.clip_by_global_norm(10.0), optax.adam(LR)),
     'q':     optax.chain(optax.clip_by_global_norm(1.0),  optax.adam(LR))},
    _labels)
```

`multi_transform` applies different gradient transformations to different parameter
subtrees. The 'q' subtree (Q-network weights) gets tighter gradient clipping (1.0 vs
10.0). This is because Q-value gradients can be large (Q values are in [0, 500] scaled
space) while world model gradients are smaller. Without the tighter clip on Q, Q
gradients dominate and corrupt the shared latent representation.

---

## Section 8: Warmup and Compilation (lines 315–355)

### Warmup

```python
while env_steps < WARMUP_ENV:
    acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
    env_state = batch_step(env_state, jnp.asarray(acts_np))
    buf.add_batch(obs_np, acts_np, ...)
    env_steps += N_ENVS
```

25k random steps to fill the buffer before any training. At N=1024 per step,
this is 25 global steps (≈ 0.3s wall time). Ensures the buffer has enough diversity
before the first gradient update.

`jnp.asarray(acts_np)`: copies the numpy array to GPU memory. JAX and numpy have
different memory spaces; conversions happen explicitly.

### Compilation trigger

```python
params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db)
jax.block_until_ready(params["enc"])
```

The first call to `upd` triggers JIT compilation (XLA traces and compiles the
computation graph). This takes ~60s for HIDDEN=(128,128). All subsequent calls
reuse the compiled graph — compilation is a one-time cost.

`jax.block_until_ready`: JAX operations are asynchronous by default (they return
before the GPU finishes computing). This forces a synchronization point to accurately
measure compilation time.

---

## Section 9: Training Loop (lines 375–430)

```
for each global step:
    1. Collect N_ENVS=1024 environment steps (policy + noise or random)
    2. Store (obs, act, rew, done) in buffer
    3. Do K_UPDATE=64 gradient steps from buffer samples
    4. Log metrics every 200k steps
    5. Evaluate pi every 500k steps
    6. Evaluate MPPI every 1M steps, write to CSV
```

### Exploration strategy

```python
if env_steps < EXPL_UNTIL:
    acts_np = np.random.uniform(al, ah, (N_ENVS, act_dim))   # pure random
else:
    acts_jax = act_fn_batch(params, jnp.asarray(obs_np))     # policy actions
    noise    = np.random.normal(0, EXPL_NOISE, (N_ENVS, act_dim))
    acts_np  = np.clip(np.array(acts_jax) + noise, al, ah)   # policy + noise
```

Two phases: pure random for the first 25k steps (same as warmup, so training starts
on somewhat diverse data), then policy + Gaussian noise (σ=0.3) for the rest.
The 0.3 noise is large — the action space is [-1,1] so 0.3 is 15% of the full range.
This keeps the data collection on-policy while maintaining exploration diversity.

### The inner K_UPDATE loop

```python
for _ in range(K_UPDATE):
    samp = buf.sample(BS, rng)
    if samp is not None:
        ob, ab, rb, db = [jnp.asarray(x) for x in samp]
        params, tp, opt, loss, aux = upd(params, tp, opt, ob, ab, rb, db)
```

64 gradient steps after each global step (1024 new env transitions).
UTD = 64/1024 = 1/16. Each `jnp.asarray` copies the sampled numpy batch to GPU.
The compiled `upd` call then runs on GPU (~1-2ms at HIDDEN=(128,128)).

---

## Section 10: Evaluation (lines 358–373)

### Policy eval

```python
for _ in range(1000):
    act = act_fn_single(params, obs)  # z = enc(obs); a = π(z)
    state = eval_step(state, act)
    er += float(state.reward[0])
    if bool(state.done[0] > 0.5): break
    obs = jnp.asarray(state.obs[0])
```

Single environment, deterministic rollout. Uses `act_fn_single` which adds no noise.
Note: `float(state.reward[0])` and `bool(state.done[0] > 0.5)` each trigger a
GPU→CPU sync. In an episode of 1000 steps, this is 2000 syncs — slow for training
but acceptable for eval (runs once per 500k steps).

### MPPI eval

```python
for _ in range(1000):
    act, mu = plan(params, obs, mu, ek)   # full MPPI planning call
    state = eval_step(state, act)
    key, ek = jax.random.split(key)       # new random key for next MPPI call
    er += float(state.reward[0])
    if bool(state.done[0] > 0.5): break
```

Same structure but replaces `act_fn_single` with `plan` (MPPI).
`mu` is carried across steps — the receding horizon warm-start.
`ek` is a fresh key for each MPPI call's noise sampling.

---

## Tricks Summary

These are the non-obvious implementation choices that significantly affect results.

### T1: SimNorm — bounded latent space
Group softmax on encoder + dynamics outputs. Prevents consistency loss explosion.
Without this, `c` diverges and training fails. Parameters: V=8 (8 groups of 16).

### T2: rew_scale=10 — amplified reward gradient
Multiply reward targets by 10 before the reward loss. Divide by 10 in MPPI rollouts.
Without this, `r=0.000` throughout training (MSE too small for any effective learning).
Calibrate as: `rew_scale ≈ 1 / avg_per_step_reward` for the task.

### T3: Pi warm-start in MPPI
Inject the greedy policy trajectory as (a) the initial mu[0] value and (b) the last
MPPI sample. Prevents MPPI collapse to zero-action when reward predictions are similar
across samples early in training.

### T4: Receding horizon mu shift
After taking action `mu[0]`, shift mu → `[mu[1:], pi_traj[-1]]` for the next step.
Reuses the planned trajectory rather than re-planning from scratch every step.
Provides temporal consistency in action selection.

### T5: Consistent parameter snapshot in MPPI
All networks in the MPPI rollout (enc, dyn, rew, q, pi) must use the same `params`
dict. Mixing `params["enc"]` with `tp["q"]` breaks the latent alignment trained into
the networks — they expect latents from the same encoder they were co-trained with.

### T6: Pessimistic twin critics
`min(Q₁, Q₂)` in TD target bootstrap and in policy gradient. Reduces Q overestimation
which would inflate MPPI terminal values and cause uniform softmax weights (→ MPPI collapse).

### T7: Geometric sequence weights ρ^t
Losses at earlier timesteps in the sequence (closer to real obs) are weighted higher.
ρ=0.5 means t=0 has weight 1.0, t=1 has 0.5, t=4 has 0.0625. Compensates for the
compounding dynamics error at later timesteps reducing training signal quality.

### T8: Asymmetric gradient from consistency loss
The consistency target `z_tgt = stop_gradient(enc(obs))` ensures consistency loss
only trains the dynamics model to match the encoder, not the other way around.
Without this, the encoder collapses to a constant to minimize consistency loss to 0.

### T9: Separate Q gradient clipping (1.0 vs 10.0)
Q-value gradients operate in a scaled space (Q ∈ [0, ~500]) while world model
gradients are in [0, ~5]. Tighter clipping on Q gradients prevents them from
dominating the shared computation graph through the encoder.

### T10: N=1024 + HIDDEN=(128,128) for fast iteration
The Warp GPU backend costs ~12ms per batch regardless of N. N=1024 gives 82k env-sps.
HIDDEN=(128,128) gives ~4× faster updates vs (256,256). The combination achieves
~1200-1900 effective env-sps including 64 gradient updates per step. Total: 2.3h for 10M steps.
