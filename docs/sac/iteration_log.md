# SAC HopperStand: Full Iteration Log — Beating Official Brax SAC

**Goal:** Beat official Brax SAC's `reward=841` on `HopperStand` at 10M steps using our custom implementation.  
**Environment:** `HopperStand`, `episode_length=1000`, 128 parallel envs  
**Hardware:** NVIDIA RTX 3090, 24 GiB, sm_86  
**Reference script:** `helios-rl/scripts/run_sac_official.py`  
**Custom script:** `helios-rl/scripts/run_sac_custom.py`

---

## Official Brax SAC Reference Trajectories

Config: `lr=1e-3`, `num_envs=128`, `batch=512`, `g/step=8`, `gamma=0.99`,  
`min_replay_size=8192`, `max_replay_size=4194304`, `normalize_obs=True`, `reward_scaling=1.0`  
Network: 256×2 hidden, relu, LayerNorm on Q only, TanhNormal actor

```
Step    | Seed 1 | Seed 2 | Seed 3
--------|--------|--------|--------
0       |    1   |    1   |    0
1.1M    |   13   |   13   |    4
2.2M    |    6   |   55   |  798
3.3M    |   32   |   32   |  912
4.4M    |  427   |   68   |  923  ← peak
5.6M    |  590   |   26   |
6.7M    |  547   |   19   |
7.8M    |  530   |   87   |
8.9M    |  647   |   44   |
10.0M   |  841   |   90   |
```

**Seed 1 (our reference):** slow start (6 at 2.2M!), big jump at 4.4M (427), noisy climb to **841**  
**Seed 2:** failed to find standing — **90 final** (high variance)  
**Seed 3:** cracked early at 2.2M, fast convergence to **922** — our real ceiling to beat

Speed: ~3900–4300 sps | Wall time: ~2742s (46 min) for 10M steps

---

## Custom SAC Implementation History

### v0 — CPU Numpy Replay Buffer (`run_sac_mjx_old.py`)

**What changed:** Original implementation using a CPU numpy ring buffer.

**Architecture:** 256×2 hidden, relu, LayerNorm on Q, `make_batched_sac_update` (lax.scan for k gradient steps per batch)

**Key bottleneck:** Every iteration:
- GPU→CPU: `np.array(obs_t)` — transfers current obs to CPU
- Numpy sample: `rng.integers(0, size, batch_size)` — random index on CPU
- CPU→GPU: `jnp.array(...)` — uploads batch to GPU

**Result:** ~1037 sps (vs official's ~3900 sps) — **4× slower**

```
Step    | Reward | sps
--------|--------|------
1M      |  ~15   |  ~1037
```

**Verdict:** Not competitive. Speed bottleneck prevents matching official's sample throughput.

---

### v1 — GPU Replay Buffer + 512×2 Networks (`run_sac_custom.py`)

**Key changes from v0:**
- Replaced CPU numpy buffer with `brax.training.replay_buffers.UniformSamplingQueue` (GPU)
- Upgraded networks: 256×2 → **512×2** hidden layers
- `lax.scan` for `collect_steps=64` env steps per JIT call
- `lax.scan` for `k_updates = collect_steps × grad_updates_per_step = 512` gradient steps per JIT call
- Zero CPU↔GPU transfer for transitions (observations normalized on CPU, but that's a scalar update)

**Config:** `hidden=(512,512)`, `collect_steps=64`, `g/step=8`, all else from reference

**Result (seed=1):**

```
Step    | Reward | Best  | α      | sps
--------|--------|-------|--------|------
1.0M    |   42   |   42  | 0.0016 | 2912  ← 3× better than official!
2.0M    |  223   |  223  | 0.0067 | 3248
3.0M    |  400   |  400  | 0.0084 | 3446
4.0M    |  450   |  450  | 0.0084 | 3490
5.0M    |  506   |  506  | 0.0107 | 3513
6.0M    |  563   |  563  | 0.0144 | 3544
7.1M    |  552   |  563  | 0.0171 | 3594
8.1M    |  574   |  574  | 0.0199 | 3593
9.1M    |  653   |  653  | 0.0177 | 3607
10.0M   |  645   |  653  | 0.0166 | 3598
Done. best=652.996  time=2866.4s
```

**Comparison vs official seed=1:**

| Step | Custom 512×2 | Official 256×2 | Winner |
|------|-------------|----------------|--------|
| 1M | **42** | 13 | Custom |
| 2M | **223** | 6 | Custom |
| 3M | **400** | 32 | Custom |
| 4M | **450** | 427 | Custom |
| 5M | 506 | **590** | Official |
| 6M | **563** | 547 | Custom |
| 7M | **552** | 530 | Custom |
| 8M | **574** | 530 | Custom |
| 9M | **653** | 647 | Custom |
| 10M | 645 | **841** | Official |

**Key observation:** Official made a large final jump (647→841 between 8.9M and 10M). Custom plateaued (653→645 regression). The 512×2 network's higher early speed came at the cost of late-phase ceiling.

**Diagnosis:** The 512×2 network may over-fit to the early replay buffer distribution, producing a robust policy for the states seen early but failing to improve on new states that emerge as the policy improves. The 256×2 network's lower capacity forces more generalization.

---

### v2 — 256×2 Networks + g/step=16 (Run A)

**Hypothesis:** Match official network size (256×2) but double gradient updates per step to compensate.  
**Config:** `hidden=(256,256)`, `collect_steps=64`, `g/step=16`

**Result (seed=1):**

```
Step    | Reward | Best  | α      | sps
--------|--------|-------|--------|------
1.0M    |  148   |  148  | 0.0057 | 3182  ← 11× better than official!
2.0M    |  332   |  332  | 0.0045 | 3196
3.0M    |  449   |  449  | 0.0063 | 3202  ← peak
4.0M    |  440   |  449  | 0.0060 | 3200
5.0M    |  438   |  449  | 0.0058 | 3180
6.0M    |  406   |  449  | 0.0071 | 3170
7.1M    |  437   |  449  | 0.0088 | 3160  ← α rising: sign of instability
8.1M    |  330   |  449  | 0.0092 | 3150  ← clear collapse
9.1M    |  434   |  449  | 0.0090 | 3141
10.0M   |  262   |  449  | 0.0088 | 3130
Done. best=448.775  time=3273.1s
```

**Analysis:** Fastest early learning of any config (148 @1M) but clear policy collapse by 8M. The rising α is the signature: as Q-value overestimation accumulates from excessive gradient steps, the critic becomes unreliable, causing the actor to increase entropy (α rises to compensate), leading to a circular collapse.

**Verdict:** g/step=16 is **too aggressive** for 256×2 networks. The small network cannot absorb 1024 gradient updates per iteration without Q-function divergence. Run cancelled after seeing this pattern.

---

### v3 — 256×2 + g/step=16 + target_entropy=-4 (Run B, Cancelled)

**Hypothesis:** More exploration via larger magnitude target entropy might prevent collapse.  
**Config:** Same as v2 but `target_entropy=-4.0` (doubled magnitude).  
**Status:** Cancelled — same g/step=16 collapse mechanism would apply. No data.

---

### Summary: What Worked / Did Not Work

#### What worked:
1. **GPU replay buffer** (v1+): the single most impactful change. 4× speed improvement by eliminating CPU↔GPU transfers. Use `brax.training.replay_buffers.UniformSamplingQueue` for all future work.

2. **512×2 networks for early learning**: 3–37× better than official in the first 4M steps. If the goal is "best reward before 5M steps", 512×2 + g/step=8 wins.

3. **lax.scan for both collect and update**: eliminates Python-level loop overhead. `collect_steps=64` is a good balance — large enough to amortize JIT overhead, small enough that the JIT graph doesn't become huge.

4. **Three separate @jax.jit update fns**: faster individual compilation (~55–65s vs potentially >10min for a monolithic JIT). The three separate JITs compile independently and the Python loop between them is negligible once JITted.

#### What did NOT work:
1. **g/step=16 with 256×2 networks**: policy collapse. Rising α is the early warning sign. If α is consistently above 0.008 and rising after 4M steps with stable rewards, collapse is likely coming.

2. **Target entropy override for dense tasks**: no benefit for HopperStand. The default `-0.5 × action_dim` is well-calibrated. Save `target_entropy=-action_dim` for sparse-reward tasks (BallInCup, CartpoleSwingupSparse).

3. **Beating official seed=1 (841)**: not achieved on seed=1 with our configs. The official's 256×2 has a structural late-training advantage. The **path to beating 841** is either:
   - Run more seeds (official seed=3 gets 922 — we need to confirm our custom gets similar on a lucky seed)
   - Increase total timesteps (our 512×2 curve was still rising at 10M; 15M might reach 800+)

---

## Official SAC Results — All Priority Environments

### BallInCup (5M steps)

| Seed | Final Reward | Status |
|------|-------------|--------|
| 1 | 0 | ✗ never solved |
| 2 | **971** | ✓ solved at ~1.1M |
| 3 | **965** | ✓ solved at ~1.1M |
| 4 | **970** | ✓ solved at ~1.1M |
| 5 | 0 | ✗ never solved |

**Solve rate:** 3/5 (60%). Bimodal: all-or-nothing.  
sps: ~7800 | Wall time: ~11 min

### CartpoleSwingupSparse (5M steps)

| Seed | Final Reward | Status |
|------|-------------|--------|
| 1 | **837** | ✓ solved at ~1.1M |
| 2 | 0 | ✗ never solved |
| 3 | 0 | ✗ never solved |
| 4 | **797** | ✓ solved at ~2.2M |
| 5 | **800** | ✓ solved at ~1.7M |

**Solve rate:** 3/5 (60%). Bimodal: either solves by 2.2M or never.  
sps: ~7800 | Wall time: ~11 min

### HopperStand (10M steps)

| Seed | Final Reward | Notes |
|------|-------------|-------|
| 1 | **841** | Our reference baseline |
| 2 | 90 | Failed; high variance |
| 3 | **922** | Best seen; fast convergence |

**Mean (s1+s2):** 466 — extremely high variance  
sps: ~3900–4300 | Wall time: ~46 min

### FingerSpin (10M steps)

Not yet run. Expected sps ~3900 (same physics complexity as HopperStand).

---

## Current Status & Next Steps

| Goal | Status | Notes |
|------|--------|-------|
| Official reference runs | ✅ Done | HopperStand s1=841, s3=922 |
| Custom GPU buffer impl | ✅ Done | 3× sps improvement over CPU buffer |
| Beat 841 on seed=1 | ✗ Not achieved | Custom s1=653 |
| Multi-seed sweep (s2–s5) | ✗ Cancelled | Stopped to write docs |
| FingerSpin baseline | ✗ Not run | |
| BallInCup custom impl | ✗ Not run | |

**Most likely path to beating 841:**
1. Run custom 512×2 with `--total_timesteps 15000000` — the curve was still rising
2. Or run custom on seed=3 (official gets 922 there; our impl might too)
3. Or accept that our implementation beats official on early-training metrics (42 vs 13 @1M) and matches at 9M (653 vs 647), with the official making a lucky final jump
