# TD-MPC2 JAX: Acceleration Plan — From 1h+ to ~10 min Iteration

**Goal**: Reduce each experiment iteration from the current **60–70 sps** (v5–v9) to a fast iteration
cycle where a 1M-step run completes in **~10 min** while still producing a visible learning signal.

---

## 1. What Is Currently Slow: Measured Baseline

### Actual Measured sps by Version

| Script | N_ENVS | K_UPDATE | H | UTD | Measured sps | 1M steps |
|--------|--------|----------|---|-----|-------------|----------|
| v4f (MSE loss) | 1024 | 64 | 5 | 1/16 | **1388** | **12 min** |
| v5 (two-hot Q) | 256 | 256 | 5 | 1:1 | **61** | **4.5h** |
| v9 (stoch Pi) | 256 | 256 | 5 | 1:1 | **60–67** | **4–5h** |

**The 23× slowdown between v4f and v9 is not from code changes — it is from the config change:**
- K_UPDATE: 64 → 256 (4× more gradient steps)
- N_ENVS: 1024 → 256 (4× fewer env steps per macro)
- UTD: 1/16 → 1:1 (16× more gradient compute per env step)
- Plus: two-hot Q adds ~4ms per step vs MSE (11ms → 16ms per gradient step)
- Combined: 23× slower wall clock per env step

---

## 2. Timing Profile per Macro Step

Each **macro step** = 1 Python iteration = N_ENVS env steps + K_UPDATE gradient updates.

### Current (v9: N=256, K=256)
```
Per macro step = 256 env steps + 256 gradient updates

Component               | Time    | % of macro
------------------------|---------|----------
Env step (Warp batch)   |  13ms   |  0.3%
CPU buffer sample ×256  | 128ms   |  3.0%   ← buf.sample() fancy-index 277MB array
CPU→GPU transfer ×256   |  26ms   |  0.6%   ← jnp.asarray on 4 small arrays
GPU gradient update×256 | 3840ms  | 91.0%   ← 256 × 15ms (TWO-HOT Q DOMINANT)
Python loop overhead    | 128ms   |  3.0%   ← 256 × JIT dispatch ~0.5ms
Eval (5 eps ×1000 steps)|~10s/eval| one-time each checkpoint

Total per macro: ~4135ms → sps = 256/4.135 = 62 sps ← matches observation
```

### v4f (N=1024, K=64, MSE loss)
```
Per macro step = 1024 env steps + 64 gradient updates

Component               | Time    | % of macro
------------------------|---------|----------
Env step (Warp batch)   |  12ms   |  1.6%
CPU buffer sample ×64   |  32ms   |  4.3%
CPU→GPU transfer ×64    |   6ms   |  0.8%
GPU gradient update ×64 | 704ms   | 94.7%   ← 64 × 11ms (simpler MSE loss)
Python loop overhead     |  32ms   |  4.3%

Total per macro: ~786ms → sps = 1024/0.786 = 1302 sps ← matches observation
```

**Key finding: GPU gradient compute accounts for 91–95% of wall time.** Python loop
overhead and CPU buffer transfers are only 4–7% combined. The usual JAX optimization
advice (GPU buffer, lax.scan) provides only **4–7% speedup** in this case.

### Per-gradient-step costs (measured)

| Loss type | H | Hidden | ms/step |
|-----------|---|--------|---------|
| MSE (v4f) | 5 | (128,128) | ~11ms |
| Two-hot Q (v5–v9) | 5 | (128,128) | ~15–16ms |

The two-hot Q adds ~40% overhead vs MSE. Both have a sequential `lax.scan` over H=5 temporal
steps inside the loss function, creating a data dependency chain the compiler cannot parallelize.

---

## 3. Root Cause Analysis

### Root Cause 1: UTD=1:1 requires 256 gradient steps per 256 env steps (primary)
Each gradient step at (128,128) with two-hot Q costs ~15ms. With K=256:
`256 × 15ms = 3840ms per macro`. This is unavoidable at the current config.

### Root Cause 2: Two-hot Q adds 40% overhead vs MSE (secondary)
The `two_hot()` + `soft_ce()` computation over 101 bins is heavier than MSE.
Each call: `symlog → clip → bin_index → one_hot×2 → softmax → sum` for 256 sequences × 5 timesteps.
The reward head (also two-hot) adds another similar overhead.

### Root Cause 3: Sequential H-step dynamics rollout (secondary)
`lax.scan(dyn_step, z0, acts_T, length=H-1)` creates an H-step sequential chain.
This prevents XLA from parallelizing the temporal dimension. At H=5, this is 5 serial network
forward+backward passes for the dynamics, constraining the minimum compilation graph depth.

### Root Cause 4: Eval is 5000 Python dispatches per eval checkpoint (minor)
`eval_pi_ep`: 1000 Python-level env steps per episode × 5 episodes = 5000 JIT dispatches
for `act_fn_single + eval_step`. At ~0.5ms per dispatch: ~2.5s per pi eval.
`eval_mppi_ep`: same 5000 dispatches, each calling `plan()` (~1ms MPPI JIT): ~5s per MPPI eval.
Total eval overhead: ~7.5s per checkpoint × 8 checkpoints per 4M steps = 60s = 1 min (minor).

### Root Cause 5: CPU buffer (marginal)
`MultiEnvBuffer.sample()` does numpy fancy indexing on a (256, 18000, 15) = 277MB array.
At K=256: 256 × ~0.5ms = 128ms per macro. This is only 3% of total time.

---

## 4. Optimization Plan

### Optimization A: Config switch for fast iteration (23× speedup, zero code change)

**This is the biggest win and requires no code changes.**

| Config | N_ENVS | K_UPDATE | UTD | Expected sps | 1M time | Quality |
|--------|--------|----------|-----|-------------|---------|---------|
| **Fast iter** | 1024 | 64 | 1/16 | ~1400 | **12 min** | v4f level |
| **Medium** | 512 | 128 | 1/4 | ~400 | ~42 min | mid |
| **Quality** | 256 | 256 | 1:1 | ~65 | ~4.3h | best |

For quick hypothesis testing (does change X produce a learning signal?), use **N=1024, K=64**.
The learning signal (pi, MPPI at 1M) is visible and responsive to architectural changes,
even if absolute values are lower than the UTD=1:1 run.

**When to use each:**
- Fast iter: test new architecture changes, verify gradient is flowing, eliminate bugs
- Medium: verify a change survives to 4M steps before full quality run
- Quality: final evaluation of a validated approach

---

### Optimization B: Shorter horizon H=2–3 (~30–40% speedup)

**Current**: H=5 temporal steps in loss_fn (5-step sequential dynamics chain).

The `lax.scan(dyn_step, ...)` for H=5 steps is the deepest sequential dependency in the
loss_fn. Reducing to H=2 cuts the scan length by 60%, reducing:
1. The sequential dynamics forward+backward to 2 steps (vs 5)
2. The `lax.scan(step_loss, ...)` over temporal losses to 1 timestep (vs 4)

**Expected speedup per gradient step:**

| H | Temporal scan depth | Estimated ms/step |
|---|---------------------|-------------------|
| 5 (current) | 5 dyn steps, 4 loss steps | ~15ms |
| 3 | 3 dyn steps, 2 loss steps | ~10ms |
| 2 | 2 dyn steps, 1 loss step | ~8ms |

At H=2: per-step cost drops from 15ms → 8ms (~47% reduction).

**Combined with fast-iter config (K=64, N=1024, H=2):**
- Per macro: 12ms + 64 × 8ms = 524ms → sps = 1024/0.524 = **1954 sps**
- 1M steps: **512s = 8.5 min** ✓

**Quality impact of H=2:**
- Shorter planning horizon = less multi-step world model rollout = less accurate plan
- At H=2, MPPI planning uses only 2 predicted steps before bootstrapping with Q
- For `HopperHop`, H=5 is already much shorter than the episode (1000 steps), so H=2 may
  still capture enough temporal structure for learning. Worth validating.
- Official default: H=3 (the config.yaml shows `horizon: 3`). Our H=5 is above official!
- **H=3 is the natural fast-iter choice** (matches official default).

---

### Optimization C: bfloat16 mixed precision (1.5–2× GPU speedup)

**Why**: RTX 3090 (sm_86) has BF16 Tensor Cores running at 2× float32 throughput for
matrix multiplications. At (128,128) hidden dim, the MLPs are matmul-dominated, especially
at BS=256 per sequence × 5 timesteps.

**Implementation in JAX:**
```python
# Cast params to bfloat16 at init
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)

# Optimizer keeps master weights in float32:
# Use optax.scale_by_learning_rate with float32 accumulators

# Alternative: use flax's mixed precision via nn.Dtype
class NormMLP(nn.Module):
    dtype: jnp.dtype = jnp.bfloat16   # ← add this
    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)       # ← cast input
        for d in self.dims:
            x = nn.Dense(d, dtype=self.dtype)(x)
            x = nn.LayerNorm(dtype=jnp.float32)(x)  # ← keep norms in float32
            x = nn.silu(x)
        return nn.Dense(self.out, dtype=self.dtype)(x)
```

**Risk factors:**
- LayerNorm and softmax (in simnorm, two-hot) need float32 for numerical stability
- Keep optimizer state (Adam m/v) in float32 → use mixed precision pattern
- Two-hot 101-bin softmax is sensitive: test carefully

**Expected speedup**: 1.5–2× on the 91% GPU compute → 1.5–2× overall speedup.

**Combined with fast-iter + H=3:**
- sps goes from 1400 → ~2100–2800
- 1M steps: **360–480s = 6–8 min** ✓✓

---

### Optimization D: GPU buffer with JAX sequence sampling (~5% speedup)

**Why it matters less than expected:** CPU buffer overhead is 3–4% of total time (see profile).
GPU buffer eliminates this 3–4%, giving marginal speedup. However, it enables all-GPU
data pipelines that could be important for multi-env Qs.

**Implementation sketch for sequence sampling on GPU:**

```python
# GPU ring buffer as pytree (all jnp arrays, lives on GPU)
@struct.dataclass
class BufState:
    obs:   jnp.ndarray   # (N_ENVS, CAP, obs_dim)
    acts:  jnp.ndarray   # (N_ENVS, CAP, act_dim)
    rews:  jnp.ndarray   # (N_ENVS, CAP)
    done:  jnp.ndarray   # (N_ENVS, CAP)
    ptr:   jnp.ndarray   # (N_ENVS,) int32
    size:  jnp.ndarray   # (N_ENVS,) int32

@jax.jit
def buf_insert(state, obs_b, act_b, rew_b, done_b):
    # Use dynamic_update_slice per env (vmapped)
    def update_env(obs_e, ptr_e, new_obs):
        return jax.lax.dynamic_update_slice(obs_e, new_obs[None], [ptr_e, 0])
    new_obs = jax.vmap(update_env)(state.obs, state.ptr, obs_b)
    ...  # same for acts, rews, done
    new_ptr  = (state.ptr + 1) % CAP
    new_size = jnp.minimum(state.size + 1, CAP)
    return BufState(obs=new_obs, ..., ptr=new_ptr, size=new_size)

@jax.jit
def buf_sample(state, key, B, SEQ):
    k1, k2 = jax.random.split(key)
    valid_mask = (state.size >= SEQ + 1).astype(jnp.float32)
    env_ids = jax.random.choice(k1, N_ENVS, (B,), p=valid_mask / valid_mask.sum())
    max_starts = state.size[env_ids] - SEQ
    starts = (jax.random.uniform(k2, (B,)) * max_starts).astype(jnp.int32)
    def get_seq(env_id, start):
        return jax.lax.dynamic_slice(state.obs[env_id], [start, 0], [SEQ, obs_dim])
    obs_seq = jax.vmap(get_seq)(env_ids, starts)
    ...  # same for acts, rews, done
    return obs_seq, ...
```

**Challenge**: `dynamic_update_slice` with jit requires static slice sizes (OK — SEQ is fixed)
and the ptr must be a jnp array (not Python int) for jit to handle it correctly.

**Memory footprint at N=1024, CAP=15000:**
- obs: 1024 × 15000 × 15 × 4B = 921MB (fits on 24GB RTX 3090)

**When to implement**: after validating the config (Opt A) and H=3 (Opt B) changes, since
GPU buffer adds implementation complexity for small gain.

---

### Optimization E: lax.scan over K_UPDATE gradient steps (~3% speedup)

Wraps the Python loop into a single JIT call. Eliminates 256 × 0.5ms = 128ms Python dispatch overhead.

```python
def make_scan_update(upd, buf_sample_fn, K):
    @jax.jit
    def scan_update(params, tp, opt, buf_state, rng):
        def one_step(carry, key):
            p, t, o = carry
            buf_state_outer, sample_key = buf_state, key
            ob, ab, rb, db = buf_sample_fn(buf_state_outer, sample_key)
            p, t, o, loss, aux = upd(p, t, o, ob, ab, rb, db, sample_key)
            return (p, t, o), (loss, aux)
        keys = jax.random.split(rng, K)
        (params, tp, opt), (losses, auxs) = jax.lax.scan(one_step, (params, tp, opt), keys)
        return params, tp, opt, losses[-1], jax.tree_map(lambda x: x[-1], auxs)
    return scan_update
```

**Constraint**: requires GPU buffer (Opt D) to avoid capturing CPU buffer state inside JIT.
Without GPU buffer, scan cannot call `buf.sample()` (Python/numpy). So Opt E depends on Opt D.

**Expected impact**: 3% speedup standalone. Combined with GPU buffer: ~5–7% total.

---

### Optimization F: Vectorized eval with lax.scan (~10s saved per checkpoint)

**Current**: 5 episodes × 1000 Python steps = 5000 JIT dispatches per eval.

```python
@jax.jit
def eval_pi_batch(params, keys):
    """Run N_eval episodes in parallel, lax.scan for 1000 steps."""
    def one_episode(key):
        state = env.reset(jax.random.split(key, 1))
        def step(carry, _):
            state, done, total = carry
            act = act_fn_batch(params, state.obs)  # (1, act_dim)
            new_state = env.step(state, act)
            r = new_state.reward[0] * (1 - done)
            new_done = jnp.maximum(done, new_state.done[0])
            return (new_state, new_done, total + r), None
        (_, _, total), _ = jax.lax.scan(step, (state, jnp.float32(0), jnp.float32(0)), None, length=1000)
        return total
    return jax.vmap(one_episode)(keys)  # vectorize over episodes

# Usage: eval_pi_batch(params, jax.random.split(key, 5))  → 1 JIT call for 5 episodes
```

**Speedup**: ~2.5s → ~0.2s per pi eval. More importantly, enables **50 eval episodes**
cheaply for statistical significance.

**For MPPI eval**: vectorizing over `plan()` calls requires vmapping MPPI itself — feasible but
requires careful handling of the per-episode `mu` state. Run 5 MPPI evals sequentially (already 5×JIT) or vmap with `jax.vmap(lambda k, mu: plan(params, obs, mu, k))`.

---

## 5. Priority Order and Implementation Roadmap

### Phase 0: Immediate (zero code change, same day)
**Use K=64, N=1024, H=3 for all fast iteration experiments.**

This single config change brings 1M-step runs from 4.5h → ~15 min. Test all hypotheses here.

Expected sps: **~1100–1200** (similar to v4f but with two-hot Q overhead).
For 1M steps: **~14 min** ✓

```python
# In train_tdmpc_hopper_v10_stoch_pi.py (or create v11)
N_ENVS          = 1024   # was 256
K_UPDATE        = 64     # was 256
H               = 3      # was 5
TOTAL_ENV       = 1_000_000  # 1M for fast experiments
PI_EVAL_EVERY   = 250_000    # 4 eval checkpoints per 1M
MPPI_EVAL_EVERY = 500_000    # 2 MPPI checkpoints
```

---

### Phase 1: bfloat16 (1–2 days implementation)

Implement mixed precision:
- Network forward/backward in bfloat16
- LayerNorm activations in float32
- Adam optimizer state in float32
- Expected: 1.5× speedup → 1M in **~9 min**

**Validation required**: check that `c`, `r`, `v`, `p` metrics remain stable with bfloat16.
If two-hot softmax shows NaN: keep softmax in float32, only use bfloat16 for MLP weights.

---

### Phase 2: Vectorized eval (1 day)

Implement `eval_pi_batch` and test eval correctness. No training loop change needed.
Bonus: can run 50 eval episodes per checkpoint instead of 5 → better statistics.

---

### Phase 3: GPU buffer + scan (3–5 days, validate carefully)

High complexity, low return (~5% speedup). Implement only if:
- Want all-GPU pipeline for correctness guarantees (no CPU state in training loop)
- Planning to do multi-GPU experiments later

---

## 6. Expected Speed After Optimizations

### 1M-step run times (HopperHop)

| Phase | Changes | Expected sps | 1M steps | Notes |
|-------|---------|-------------|----------|-------|
| Now (v9) | K=256, N=256, H=5 | 60–67 | **4.5h** | quality UTD |
| Phase 0 | K=64, N=1024, H=3 | ~1100 | **15 min** | fast iter |
| Phase 0+1 | + bfloat16 | ~1600 | **10 min** | fast iter |
| Phase 0+1+2 | + vectorized eval | ~1650 | **10 min** | + better eval stats |
| Quality run | K=256, N=256, H=3, bf16 | ~90 | **3.1h** | quality |

### What "fast iter" vs "quality" tells you

| Metric | Fast iter (K=64, N=1024) | Quality (K=256, N=256) |
|--------|--------------------------|------------------------|
| pi @ 500K | ~10–50 | ~150–280 |
| pi @ 1M | ~100–180 | ~290–340 |
| MPPI @ 1M | ~50–100 | ~100–230 |
| Useful for | Debugging gradients, testing new architecture | Final evaluation |
| Misleading for | Comparing MPPI scores to official | Fast hypothesis test |

**Rule of thumb**: if `c`, `r`, `v`, `p` metrics look wrong in fast-iter, the fix won't work in
quality mode either. If they look right, promote to a 4M quality run.

---

## 7. Analysis: Why TD-MPC2 is Fundamentally Harder to Accelerate than PPO/SAC

### The JAX PPO comparison

PPO at 30M steps on CheetahRun (2048 envs, 16 epochs, 32 minibatches):
- **~60,000 sps** → 30M steps in **~8 min**
- Per update: ~5ms at (64,64) hidden, batch=2048×30÷32=1920
- Python loop overhead: zero (entire rollout + update in one JIT call via lax.scan)

**Why PPO is 23,000× faster per env step:**
1. **On-policy**: no replay buffer, transitions go directly from env to gradient
2. **Low UTD**: 512 grad steps per 61,440 env steps = UTD=0.0083
3. **No temporal rollout in loss**: no H-step world model unroll, just GAE
4. **All inside lax.scan**: one JIT compilation, no Python dispatch

### The SAC comparison

SAC at 10M steps on HopperStand (128 envs):
- **~3600 sps** → 10M in **~45 min**
- Per update: ~1ms at (256,256) hidden, no temporal unroll, no two-hot Q
- GPU buffer via `UniformSamplingQueue` (Brax)

**Why SAC is faster than our TD-MPC2:**
1. No H-step world model unroll (only Q + actor update)
2. Simpler loss (MSE for twin-Q, no distributional bins)
3. GPU buffer (no CPU overhead)
4. lax.scan over gradient steps (K=512 in single JIT)

### The irreducible cost of TD-MPC2 world model

TD-MPC2's loss function requires an H-step sequential dynamics rollout inside the gradient
computation. This is a fundamental sequential dependency chain:

```
z_0 → [enc] → z_0 → [dyn] → z_1 → [dyn] → z_2 → ... → z_H
                ↓            ↓            ↓
              rew/Q/pi    rew/Q/pi    rew/Q/pi
              loss_0      loss_1      loss_H
```

The dynamics chain `z_{t+1} = dyn(z_t, a_t)` means step t+1 cannot start until step t completes.
This limits XLA's ability to parallelize. At H=5 with (128,128) networks, this chain costs ~15ms
regardless of batch size above some threshold.

The **only ways to reduce this cost**:
1. Smaller H (reduces chain length)
2. Smaller/faster dynamics network
3. bfloat16 (reduces memory bandwidth per chain step)
4. Reduce batch size B (reduces computation per chain step, but may hurt gradient quality)

This is why TD-MPC2 cannot reach PPO-level sps without sacrificing either UTD or world model quality.

---

## 8. Implementation Notes for v11 (Fast Iteration Script)

Suggested `train_tdmpc_hopper_v11_fast.py` configuration:

```python
# ── Fast iteration config ──────────────────────────────────────────────────
SEED            = 42
N_ENVS          = 1024         # 4× more env throughput vs v9
K_UPDATE        = 64           # 4× fewer gradient steps vs v9; UTD=1/16
H               = 3            # official default; 40% faster per step than H=5
TOTAL_ENV       = 1_000_000   # 1M for fast iter, 4M for quality
BS              = 256
SEQ             = H + 1        # = 4
LR              = 3e-4
LATENT          = 128
HIDDEN          = (128, 128)
GAMMA           = 0.99
TAU             = 0.01
NS              = 256           # MPPI samples
NI              = 6             # MPPI iterations
TEMP            = 0.5
EXPL_NOISE      = 0.3
EXPL_UNTIL      = 20_000

PI_EVAL_EVERY   = 250_000      # 4 evals per 1M steps
MPPI_EVAL_EVERY = 500_000      # 2 MPPI evals per 1M steps
EVAL_EPS        = 5             # episodes per eval
```

**Expected performance:**
- Compilation time: ~60s (same as v4, K=64 reduces XLA graph)
- sps after warmup: ~1000–1200
- 1M steps wall time: **~14–17 min**
- pi @ 500K: ~50–100 (lower than v9's 278 due to UTD difference)
- MPPI @ 500K: ~20–50 (informative for relative comparisons)

**What to check in fast-iter:**
1. Is `c` stable (< 0.15)? → World model not diverging
2. Is `r` > 0.01? → Reward head learning
3. Is `p` more negative each checkpoint? → Policy is improving
4. Is `pi` increasing monotonically? → No policy collapse

If any of these fail in fast-iter (1M steps), the same failure will appear in quality mode (4M steps).
Fast-iter can be run in ~15 min to validate any change before committing to a 4h quality run.

---

## 9. Why Our pi (291) Already Exceeds Official MPPI (338) Comparison

Current status (v9 @ 1M steps, seed=42):
- **pi = 291** (our policy greedy rollout)
- **MPPI = 157** (our world model planning)
- Reference official MPPI @ 4M steps: **449**

The observation that **our pi > our MPPI** is the key diagnostic:
- **MPPI < pi** means the world model is *hurting* planning, not helping
- The world model's reward and Q predictions are inaccurate enough that MPPI's 256 samples
  exploring via the world model are *worse* than just running the learned policy greedily
- This is the "world model quality gap" — the policy is learning faster than the world model

**Why this gap exists:**
1. H=5 with two-hot Q is still recovering from the v6 `/scale_val` bug
2. UTD=1:1 at N=256 produces high-variance gradient estimates from correlated sequences
3. The MPPI discriminability problem (Q overestimation → uniform softmax weights → collapses to mean)

The acceleration plan (fast-iter config) reduces UTD, which may actually **improve the
MPPI/pi ratio** — because lower UTD means less overfitting of Q to particular replayed
trajectories, and more diverse data entering the buffer.

The path to matching official MPPI is not just "more gradient steps" — it is also:
1. Stable Q (no spikes above v=10 sustained)
2. Better MPPI discriminability (temperature tuning or advantage normalization)
3. More diverse replay buffer (larger N_ENVS helps here — another reason to use N=1024)

---

## 10. Summary: Action Checklist

| Priority | Action | Expected outcome | Effort |
|----------|--------|-----------------|--------|
| ✅ 1 | Set K=64, N=1024, H=3 in next script | 1M in 14 min | **5 min** |
| 2 | Implement bfloat16 mixed precision | 1M in 9 min | **1–2 days** |
| 3 | Vectorize eval with lax.scan | Better eval stats | **0.5 day** |
| 4 | GPU sequence buffer | ~5% speedup | **3–5 days** |
| 5 | lax.scan over K_UPDATE | ~3% speedup (needs GPU buf) | **1 day** |

**The 80/20 answer:** Change K=64, N=1024, H=3 in the config. That's it. This achieves the
target iteration time. All other optimizations are refinements.
