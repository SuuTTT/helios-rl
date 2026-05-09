# SAC Design Document — Two Implementations

**Scope:** This document covers the architecture and design decisions for both SAC variants used in the DMC Suite benchmark: the official Brax SAC wrapper (reference baseline) and our custom GPU-native SAC.

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      SAC Implementations                         │
│                                                                  │
│  ┌─────────────────────┐         ┌────────────────────────────┐  │
│  │  Official Brax SAC  │         │  Custom GPU SAC            │  │
│  │  (Reference)        │         │  (run_sac_custom.py)       │  │
│  └─────────────────────┘         └────────────────────────────┘  │
│  GPU · full XLA · pmap            GPU · custom loop · lax.scan   │
│  ~3900–4300 sps                   ~3200–3600 sps                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Variant A — Official Brax SAC (Reference Baseline)

### 2.1 File
`helios-rl/scripts/run_sac_official.py`

### 2.2 Architecture

```
Observation (obs_dim,)
      │
  ┌───▼────────────────────────────────┐
  │  Shared preprocessing (normalize) │
  └───┬────────────────────────────────┘
      │
  ┌───▼────────────────┐    ┌───▼────────────────┐
  │  Actor π           │    │  Twin Critics Q1,Q2 │
  │  Dense(256) relu   │    │  Dense(256) relu    │
  │  Dense(256) relu   │    │  Dense(256) relu    │
  │  Dense(2*a_dim)    │    │  + LayerNorm        │
  └────────────────────┘    └────────────────────-┘
  ↓ (mean, log_std)             ↓ Q(s,a) scalar each
  TanhNormal distribution
```

**Key characteristics:**
- `brax.training.replay_buffers.UniformSamplingQueue` — GPU ring buffer, zero CPU transfers
- Full epoch compiled as `jax.lax.scan` — no Python overhead per step
- `jax.pmap` over available devices
- Network: 256×2 hidden, relu, LayerNorm on Q only (not actor)
- Distribution: TanhNormal (squashed Gaussian)

### 2.3 Training Loop (Brax internals)

```
outer: lax.scan over num_epochs
  inner: lax.scan over num_training_steps_per_epoch
    1. Collect 1 step (all num_envs in parallel)
    2. Insert into GPU UniformSamplingQueue
    3. If buffer full enough:
       for k in range(grad_updates_per_step):
         sample batch (512 transitions)
         update Q1, Q2
         update actor π
         update temperature α (auto-tuning)
         soft-update Q̄ ← (1-τ)Q̄ + τQ
```

### 2.4 JAX 0.10.0 Compatibility Patch

Brax calls `jax.device_put_replicated` which was removed in JAX 0.10.0. Patch applied in `run_sac_official.py`:

```python
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(val, devices):
        n = len(devices)
        return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), val)
    jax.device_put_replicated = _device_put_replicated
```

Also: remove `num_resets_per_eval` from training params (not accepted by current `sac.train`):
```python
sac_training_params.pop("num_resets_per_eval", None)
```

### 2.5 Performance (HopperStand, RTX 3090)

| Metric | Value |
|--------|-------|
| sps (steady-state) | ~3900–4300 |
| JIT warmup | ~80s (first call) |
| 10M steps wall time | ~2742s (~46 min) |
| HopperStand s1 reward | **841** |
| HopperStand s2 reward | 90 (high variance) |
| HopperStand s3 reward | **922** (best seen) |

---

## 3. Variant B — Custom GPU SAC

### 3.1 File
`helios-rl/scripts/run_sac_custom.py`

### 3.2 Architecture

```
Observation (obs_dim,)
      │
  ┌───▼────────────────────────────────┐
  │  Running mean/var normalization    │  (online, CPU-side update)
  └───┬────────────────────────────────┘
      │
  ┌───▼────────────────┐    ┌───▼────────────────┐
  │  Actor π           │    │  Twin Critics Q1,Q2 │
  │  Dense(512) relu   │    │  Dense(512) relu    │
  │  Dense(512) relu   │    │  Dense(512) relu    │
  │  Dense(a_dim)×2    │    │  + LayerNorm        │
  └────────────────────┘    └────────────────────-┘
  ↓ (mu, log_std)               ↓ Q(s,a) scalar each
  TanhNormal: a = tanh(μ + σε)
```

**Key characteristics:**
- `brax.training.replay_buffers.UniformSamplingQueue` — same GPU buffer as official
- `lax.scan` over `collect_steps` env steps per JIT call
- `lax.scan` over `k_updates = collect_steps × grad_updates_per_step` gradient steps
- Network: **512×2 hidden** (vs official 256×2) — more capacity
- Three separate `@jax.jit` update functions (critic, actor, alpha) — fast individual compilation

### 3.3 Custom Training Loop

```python
# Phase 1: Warmup (random actions, fill buffer to min_replay_size)
while buf_state.insert_position < min_replay_size:
    act = random.uniform(...)
    transitions = env_step(es, act)
    buf_state = buf.insert(buf_state, transitions)

# Phase 2: Main loop
while total_steps < total_timesteps:
    # Collect collect_steps steps (lax.scan, stays on GPU)
    es, rng, (obs, action, reward, next_obs, done) = collect_fn(
        es, actor_p, obs_mean, obs_var, rng
    )
    # Insert batch into GPU replay buffer (no CPU transfer)
    buf_state = buf.insert(buf_state, transitions)
    # Update running obs stats (one GPU→CPU per iter)
    obs_mean, obs_var = update_running_stats(obs)
    # k_updates gradient steps via lax.scan (stays on GPU)
    actor_p, critic_p, target_p, log_alpha, buf_state = scan_update(
        ..., buf_state, rng
    )
```

**Critical improvement over old `run_sac_mjx_old.py`:**
- Old: CPU numpy buffer → 1 GPU→CPU copy per step (obs_np = np.array(obs)) + 1 CPU→GPU upload (sample_bulk) → ~1037 sps
- New: GPU `UniformSamplingQueue` → zero CPU roundtrips for transitions → ~3200–3600 sps

### 3.4 Separate JIT Update Functions

Three separate `@jax.jit` functions (fast compilation each, no monolithic graph):

```python
@jax.jit
def update_critic(critic_p, critic_opt_s, actor_p, target_p, log_alpha,
                  obs_n, next_obs_n, actions, rewards, dones, key):
    # Bellman backup with entropy-regularized target
    ...

@jax.jit
def update_actor(actor_p, actor_opt_s, critic_p, log_alpha, obs_n, key):
    # Maximize entropy-augmented Q
    ...

@jax.jit
def update_alpha(log_alpha, alpha_opt_s, actor_p, obs_n, key):
    # Auto-tune temperature to target entropy
    ...
```

Wrapped in `make_scan_update` → `lax.scan` over `k_updates` steps per JIT call.

### 3.5 Performance (HopperStand s1, RTX 3090)

| Metric | Value |
|--------|-------|
| sps (steady-state) | ~3200–3600 |
| JIT warmup (512×2) | ~64s |
| JIT warmup (256×2) | ~55s |
| 10M steps wall time | ~2866s (~48 min) |
| HopperStand s1 reward (512×2, g/step=8) | **653** best / **645** final |
| HopperStand s1 reward (256×2, g/step=16) | **449** peak / **262** final (collapsed) |

### 3.6 Key Design Choices vs Official

| Choice | Official | Custom | Notes |
|--------|----------|--------|-------|
| Network size | 256×2 | 512×2 | More capacity; faster early learning |
| LR schedule | constant | constant | Both use 1e-3 |
| Replay buffer | GPU UniformSamplingQueue | GPU UniformSamplingQueue | Same |
| Obs normalization | RunningStatisticsState (XLA) | CPU numpy update | Official fully XLA |
| grad_updates_per_step | 8 | 8 | 16 causes collapse |
| pmap | Yes | No | Single GPU; pmap overhead for 1 device is zero |
| scan structure | full epoch in scan | collect_steps in scan | Ours is coarser-grained |

---

## 4. TanhNormal Distribution (both implementations)

Actions are bounded to $[-1, 1]$ via squashed Gaussian:

$$a = \tanh(\mu + \sigma \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, I)$$

**Log-probability** with Jacobian correction (numerically stable form):

$$\log \pi(a|s) = \log \mathcal{N}(u|\mu, \sigma) - \sum_i \log(1 - \tanh^2(u_i) + \epsilon)$$

where $u = \tanh^{-1}(a)$ is the pre-squash sample. The $\epsilon=10^{-7}$ prevents $\log(0)$ at the boundary.

**Why this matters for SAC:** The entropy term in the SAC objective requires accurate log-probabilities. Using `clip(action, -1, 1)` instead of TanhNormal makes log-prob undefined at the boundaries and breaks the IS ratio.

---

## 5. Automatic Temperature Tuning

The temperature $\alpha$ is not fixed — it's automatically optimized to match a **target entropy** $\bar{H}$:

$$\mathcal{L}_\alpha = \mathbb{E}[\alpha \cdot (-\log \pi(a|s) - \bar{H})]$$

**Target entropy:** $\bar{H} = -0.5 \times |\mathcal{A}|$ (half the action dimension, negative).

For HopperStand (4-dim action): $\bar{H} = -2.0$.

When entropy is too high (too random), $\alpha$ decreases → policy focuses.  
When entropy is too low (too deterministic), $\alpha$ increases → more exploration.

**Observed values in experiments:**
- Early training: α ≈ 0.002–0.009 (low, exploring structure)
- Mid-training: α ≈ 0.008–0.017 (rising, adapting to task)
- The 256×2 + g/step=16 run showed rising α (0.009) correlating with onset of policy collapse

---

## 6. Replay Buffer: GPU UniformSamplingQueue

```python
from brax.training import replay_buffers

buf = replay_buffers.UniformSamplingQueue(
    max_replay_size=4_194_304,   # 4M transitions
    dummy_data_sample={
        "obs": jnp.zeros(obs_size),
        "action": jnp.zeros(action_size),
        "reward": jnp.zeros(()),
        "next_obs": jnp.zeros(obs_size),
        "done": jnp.zeros(()),
    },
    sample_batch_size=512,
)

buf_state = buf.init(key)                          # allocate on GPU
buf_state = buf.insert(buf_state, transitions)     # circular write
buf_state, batch = buf.sample(buf_state)           # uniform random sample
```

**Key API details:**
- `insert` accepts batches of any leading size; fills circularly (oldest overwritten when full)
- `sample` uses `insert_position` and `sample_position` to track fill level
- `buf_state.insert_position` = number of total transitions ever inserted (mod max_replay_size)
- Entire buffer lives on GPU VRAM — all operations are XLA-compiled
