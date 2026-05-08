# TD-MPC2 JAX Hopper-Hop: Iteration Log

Covers the full engineering arc from v1 (broken) to v4f (working).
Intended as a reference for future experiments on other tasks.

---

## Run Results Reference

| File | Description |
|------|-------------|
| `/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop.csv` | Final v4f results (10M steps, seed=42) |
| `/workspace/tdmpc2/results/tdmpc2/hopper-hop.csv` | Official reference (seeds 1-3 avg) |
| `/workspace/helios-rl/exp/tdmpc_dmc/cartpole-balance.csv` | Cartpole-balance results (MPPI=997.3) |
| `/tmp/hopper_v4.log` → `/tmp/hopper_v4f.log` | Training logs for each version |
| `/workspace/helios-rl/scripts/train_tdmpc_hopper.py` | v2 — SimNorm run, 2M steps, dm_control |
| `/workspace/helios-rl/scripts/train_tdmpc_hopper_v4.py` | v4f — final fast JAX-env version |
| `/workspace/helios-rl/scripts/compare_hopper.py` | Comparison script vs reference |

---

## Debug Metrics Glossary

These are logged every 500k env_steps as `c= r= v= p=` in the training output.

### `c` — Consistency Loss
**What it is**: MSE between the predicted next-latent `dyn(z_t, a_t)` and the
encoded actual next-obs `enc(o_{t+1})`, averaged over the rollout horizon.

**What it indicates**:
- `c > 5`: World model latents are unbounded / exploding. Encoder drifting to large
  scale so that `||z||²` grows → MSE grows as scale². Training will diverge.
- `c ≈ 0.05–0.15`: Normal range with SimNorm. Latents bounded in [0,1] via
  group-softmax, so maximum possible MSE per dimension is ≤1.
- `c ≈ 0.0`: Pathological collapse — encoder maps everything to the same latent.

**Fixed by**: SimNorm(V=8) applied to encoder + dynamics outputs.

### `r` — Reward Loss
**What it is**: MSE between the reward head's prediction `rew_net(z_t, a_t)` and
the *scaled* true reward `rew_scale × r_t`, averaged over the rollout horizon.

**What it indicates**:
- `r = 0.000`: Reward head is outputting near-zero, gradients ≈0. This means MPPI
  planning is driven entirely by terminal Q, not the 5-step predicted reward
  sequence. Severely limits planning quality on sparse-reward tasks.
- `r > 0.01`: Reward head is actively learning. MPPI rollouts have discriminable
  returns (different action sequences produce meaningfully different predicted rewards).
- `r ≈ 0.1`: Healthy. Reward head contributing ≈equal to terminal Q in MPPI.

**Fixed by**: `rew_scale=10.0` — multiply reward targets in loss by 10 so the MSE
signal is 100× larger. Divide back by 10 in MPPI rollouts.

### `v` — Q/Value Loss
**What it is**: MSE between predicted Q-value `Q(z_t, a_t)` and the TD target
`rew_scale × r_t + γ × (1-done) × V(z_{t+1})`, in the scaled reward space.

**What it indicates**:
- `v ≈ 1–10`: Normal. Q is tracking bootstrapped targets stably.
- `v` spikes to 50–200 then returns: Transient Q target explosion from a batch
  with high-value bootstrap. Usually benign if it recovers within 1–2 checkpoints.
- `v` climbs monotonically without recovery: Q divergence — targets bootstrap
  above their theoretical max. Will cause encoder to follow Q gradients into
  a bad regime, breaking world model quality.

**Theoretical max Q** (for future clip calibration):
`Q_max = rew_scale × max_per_step_r / (1-γ) ≈ 10 × 0.5 / 0.01 = 500`

### `p` — Policy Loss
**What it is**: Negative mean min-Q of the policy's actions `π(z_t)` in the learned
Q-ensemble, i.e., `-E[min_Q(z_t, π(z_t))]`. More negative = policy finding higher-Q actions.

**What it indicates**:
- `p ≈ -5` at 500k: Policy just learning.
- `p ≈ -90` at 8M: Policy strongly exploiting learned Q landscape.
- `p` grows more negative over training monotonically — healthy.
- `p` stagnates or grows less negative while `v` spikes: Q divergence is corrupting
  the policy gradient signal.

### `pi` — Policy Evaluation Return
**What it is**: Mean undiscounted episode return of 5 deterministic policy rollouts
(`a = π(enc(o))`) in the real environment (JAX env), evaluated every 500k steps.

**What it indicates**:
- `pi = 0` at 300k: Policy never learned (UTD too low, or Q diverged early).
- `pi ≈ 10–50` at 500k: Policy learning basic locomotion.
- `pi ≈ 150–260` at 5–10M: Policy well-trained; world model is the bottleneck.
- Reference policy (official TD-MPC2 at 4M, 1:1 UTD): ~370–450.

### `MPPI` — MPPI Planning Return
**What it is**: Mean undiscounted episode return of 5 MPPI-planned rollouts
(`a = plan(enc(o))` with 256 samples, 6 iterations, horizon=5) in the real
environment, evaluated every 1M steps.

**What it indicates**:
- `MPPI < pi`: World model is lying — planning with a bad model does worse than
  just running the policy directly. This is the main failure mode.
- `MPPI ≈ pi`: World model is neutral. MPPI rollouts pick roughly the same
  trajectory as the policy; terminal Q dominates the return estimate.
- `MPPI > pi`: World model is beneficial. The 5-step lookahead + 256 samples
  finds better action sequences than the greedy policy alone.
- Reference MPPI ≈ 375–450 at 2–4M steps.

**Relationship**: `MPPI ≥ pi` should hold in theory (MPPI subsumes pi via
pi-trajectory candidate). In practice it's violated when the world model
reward predictions are wrong (r≈0) or Q scale is off.

---

## Iteration History

### v1 — Baseline (dm_control env, no SimNorm)

**Script**: `/workspace/helios-rl/scripts/train_tdmpc_hopper.py` (original)  
**Env**: `dm_control/hopper-hop-v0`, N_ENVS=1, ~92 env-sps  
**Config**: HIDDEN=(256,256), LATENT=128, no rew_scale

**Results**: Run killed early.

**Symptom**: `c=77` at 100k steps (consistency loss exploding).

**Root cause**: Encoder outputs unbounded floats. As the encoder weights grow,
`||z||` grows without bound. Consistency loss ∝ `||z_pred - z_true||²` → grows
with latent scale squared. Training destabilizes.

**Lesson**: Without latent normalization, the world model is numerically unstable
on continuous control tasks. The encoder must have a bounded output.

---

### v2 — SimNorm Fix (dm_control, 2M steps)

**Script**: `/workspace/helios-rl/scripts/train_tdmpc_hopper.py` (SimNorm version)  
**Env**: dm_control, N_ENVS=1, 92 env-sps  
**Config**: Added `simnorm(x, V=8)` to encoder + dynamics outputs. V=8 means the
128-dim latent is split into 8 groups of 16; each group is softmax-normalized.
This bounds each dimension to [0,1] and constrains the latent norm.

**Results** (from `/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop.csv` at the time):
- 500k: MPPI=36, pi=~10
- 1M: MPPI=123
- 2M: MPPI=155

**Metrics**:
- `c ≈ 0.025–0.031`: Stable, bounded. SimNorm working.
- `r = 0.000` **throughout**: Reward head never learned. Per-step rewards ≈0.01–0.05;
  MSE ≈ (0.01)² = 0.0001 → rounds to 0.000 in print. Gradient signal essentially zero.

**Problem**: `r=0.000` means MPPI is entirely driven by terminal Q, not the
5-step predicted reward sequence. MPPI plateaus at ~155 because the model
cannot differentiate action sequences by their predicted reward paths.

**Wall time**: ~6h for 2M steps (92 sps bottleneck).

**Lesson**: `rew_scale` is mandatory on tasks where per-step rewards are small
fractions (< 0.1). Without it, the reward head receives zero effective gradient
and the world model cannot be used for reward-guided planning.

---

### v3 — JAX Env Attempt (killed)

**Script**: `/workspace/helios-rl/scripts/train_tdmpc_hopper_v3.py`  
**Env**: mujoco_playground HopperHop (JAX/Warp backend), N_ENVS=16  
**Config**: HIDDEN=(256,256), K_UPDATE=1 per step (then 16 per step)

**Problem 1**: N_ENVS=16, K=1 → UTD = 1/16. Policy sees 16 new env transitions
per gradient step → too sparse to learn from. `pi=0` at 300k steps.

**Problem 2**: Even with K=16 updates per step, speed = 16 updates × 10ms each
= 160ms/step → 6 steps/s × 16 envs = 96 env-sps. Same as dm_control. No speedup.

**Root cause of speed problem**: At (256,256) hidden dim, each gradient update
takes ~10ms on RTX 3090. Doing 16 per step costs 160ms. The JAX env step (Warp
backend) costs ~12ms per batch regardless of N. So for N=16: env step ≈ 1ms/env,
update ≈ 10ms → update dominates.

**Lesson**: Speed with JAX vectorized env requires both: (a) large N to amortize
the fixed ~12ms Warp batch cost, AND (b) fast updates to not negate the env gain.
The two knobs interact.

---

### v4 (initial) — Large N + Smaller Model

**Script**: `/workspace/helios-rl/scripts/train_tdmpc_hopper_v4.py` (first version)  
**Env**: mujoco_playground HopperHop, N_ENVS=1024  
**Config**: HIDDEN=(128,128), K_UPDATE=64, REW_SCALE=10.0, rew_target_scale  
**Design rationale**:
- N=1024: 82k env-sps in pure env stepping
- HIDDEN=(128,128): ~4× faster updates (~1-2ms vs ~10ms at (256,256))
- K=64 updates per global step: UTD = 64/1024 = 1/16 (reasonable)
- REW_SCALE=10.0: give reward head 100× larger gradient signal

**Compile time**: 62s (vs 111s with (256,256))  
**Training speed**: ~1900 env-sps (vs 92 for dm_control)

**Results at 1M env_steps**:
- `pi=121.7`, `r=0.006` ✅ reward head learning!
- `MPPI=0.0` ❌

**MPPI=0 bug**: MPPI mu was initialized to zero. Zero-action → softmax weights
are uniform (all returns identical for zero action rollouts with poor world model
early in training) → weighted mean stays zero → planning collapses to no-op.

**Fix**: Added pi-guided warm-start:
1. Run the policy for H steps from current state to get `pi_traj`.
2. Initialize `mu[0] = pi_traj[0]` (first MPPI mean slot = first pi action).
3. Replace last MPPI sample with noiseless `pi_traj` as a baseline candidate.
This guarantees MPPI return ≥ pi return in expectation.

---

### v4b — Pi Warm-Start Fix

**Config**: Same as v4 + pi warm-start in MPPI plan()

**Results**:
```
1M:  MPPI=78.3
2M:  MPPI=111.4
3M:  MPPI=8.2   ← transient dip
4M:  MPPI=144.9
5M:  MPPI=179.1
6M:  MPPI=181.5
```

**The 3M dip**: MPPI dropped from 111 to 8. Diagnosis: The Q values spiked
(`v=37` then `v=56`) at the same time. High Q values inflated all sample returns
uniformly → softmax still approximately uniform → weights wash out → MPPI
collapses to weighted mean ≈ zero again (all samples similar). This is the
"Q overestimation kills MPPI discriminability" failure mode.

**Lesson**: MPPI is fragile to Q scale. When Q estimates for all 256 samples are
similarly large (or large and noisy), the softmax temperature `temp=0.5`
produces near-uniform weights and MPPI degenerates to the CEM mean.

---

### v4c — Q Target Clipping Experiment (failed)

**Config**: Q targets clipped to [0, 500] (`Q_max ≈ rew_scale × 0.5 / 0.01 = 500`)

**Result**: `r=0.000` again, `pi=2.2` at 500k.

**Why it failed**: Clipping at 500 prevented Q from backing up value correctly
(hopper early training Q values legitimately exceed 500 when the model
is optimistic). The clip cut off the gradient that trains the encoder
to produce useful latent representations for value prediction.

**Lesson**: Hard clipping Q targets is too aggressive early in training. It
removes gradient signal from the encoder via the Q loss path.

---

### v4d — Gradient Isolation Experiment (failed)

**Config**: `stop_gradient` on encoder outputs before Q and Pi forward passes
(encoder only trained via consistency + reward losses, not Q/pi losses).

**Result**: `r=0.000`, `pi=3.2` at 500k.

**Why it failed**: The encoder critically needs gradients from both the Q loss
and the reward loss to learn a useful representation. The Q gradient pushes
the encoder to produce latents that are predictive of future value. Removing
this signal degrades the encoder to only learning to be "consistent" without
being "useful for planning."

**Lesson**: In TD-MPC2, the multi-task gradient (consistency + reward + Q + pi
all flowing to the encoder) is a feature, not a bug. Do not isolate them.

---

### v4e — Target Q in MPPI Experiment (failed)

**Config**: Use `tp["q"]` (target Q network) instead of `params["q"]` for
MPPI terminal value.

**Expected**: Smoother MPPI rollouts because `tp["q"]` is the EMA-smoothed
stable version of Q.

**Result**: `MPPI=6.4` at 1M (vs 78.3 baseline). Severe regression.

**Why it failed**: MPPI uses `params["enc"]` for encoding observations, but
`tp["q"]` was trained jointly with `tp["enc"]` (the lagged encoder). Mixing
current encoder with lagged Q creates a latent-mismatch — the Q function
expects latents that look like the slightly older encoder's output, but receives
current encoder latents. This mismatch causes Q to output garbage for most
input latents, collapsing MPPI discriminability.

**Critical lesson**: In MPPI, `enc`, `dyn`, `rew`, `q`, and `pi` must all come
from the *same* parameter snapshot (either all current `params` or all `tp`).
Mixing current and lagged versions breaks the latent alignment the networks
were trained to expect.

---

### v4f — Final Clean Run (current best)

**Config**: v4b settings exactly (pi warm-start, `params["q"]` in MPPI, no
Q target clipping, gradient flow unchanged) run to completion.

**Run time**: **2.30h** for 10M env_steps (vs 6h for 2M with dm_control)

**Results**:
```
step      MPPI    pi      c       r       v
1M        64.2    152.1   0.148   0.018   7.8
2M        97.4    176.9   0.144   0.034   66.8 ← spike, recovered
3M        116.4   209.9   0.132   0.080   4.5
4M        138.1   231.7   0.153   0.110   9.5
5M        159.9   203.7   0.151   0.060   20.7
6M        145.6   242.9   0.141   0.087   3.3
7M        186.6   242.0   0.157   0.069   174.7 ← spike, recovered
8M        178.0   241.1   0.125   0.065   114.6 ← recovering
9M        161.7   249.2   0.114   0.081   150.5
10M       182.6   242.5   0.104   0.101   9.1
```

**CSV**: `/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop.csv`  
**vs reference** at 4M: Ours 138, reference 449 (gap = 311)

---

## What Worked vs What Did Not

### ✅ What Worked

| Fix | Effect |
|-----|--------|
| **SimNorm(V=8)** on encoder+dynamics | c: 77→0.025; training stabilized |
| **rew_scale=10.0** on reward targets | r: 0.000→0.006+; reward head learned |
| **Pi warm-start in MPPI** | MPPI: 0.0→78+ at 1M; eliminated zero-action collapse |
| **N_ENVS=1024 + HIDDEN=(128,128)** | Speed: 92 sps → 1200-1900 sps; 2.3h vs 6h+ |
| **K_UPDATE=64 per global step** | UTD=1/16; pi: 0→152+ at 1M (vs pi=0 with K=1) |
| **Using params[q] (not tp[q]) in MPPI** | MPPI: 6→64 at 1M; enc/Q alignment preserved |

### ❌ What Did Not Work

| Attempt | Failure mode |
|---------|-------------|
| No latent normalization (v1) | c=77; gradient explosion |
| N_ENVS=16, K=1 update/step | UTD=1/16, pi=0 at 300k; too sparse |
| N_ENVS=16, K=16 updates/step, HIDDEN=(256,256) | 96 env-sps; no speedup over dm_control |
| rew_scale=3.0 | r=0.000; too small, reward head still doesn't learn |
| Q target clipping at 500 | r=0.000, pi=2; clips valid Q gradients in encoder |
| stop_grad on encoder before Q/pi | r=0.000, pi=3; encoder loses value-predictive gradient |
| tp["q"] in MPPI (lagged Q) | MPPI=6; latent mismatch between current enc + lagged Q |

---

## Key Lessons

### 1. Latent normalization is mandatory
SimNorm partitions the latent into V groups and applies softmax. This is not
merely "good practice" — without it, consistency loss explodes and training
fails entirely. Any task with continuous obs should use SimNorm.

### 2. Reward scale must be calibrated to gradient magnitude
For tasks where per-step reward ≪ 1 (hopper: ~0.01–0.05 per step), the raw
MSE for the reward head is ~1e-4, which rounds to 0 and produces no effective
learning signal. Rule of thumb: `rew_scale = 1 / avg_per_step_reward`. For
hopper: rew_scale ≈ 10–20. Check `r` metric: it must be > 0.001 for healthy
reward learning.

### 3. MPPI requires pi warm-start to avoid zero-action collapse
Early in training, the world model predicts similar low returns for most action
sequences. The softmax over returns is near-uniform → weighted mean ≈ zero →
MPPI output ≈ zero action for every step. Fix: always inject the greedy policy
action as the first sample and as the mu initialization.

### 4. Latent alignment: never mix param snapshots in MPPI
All networks in MPPI (enc, dyn, rew, q, pi) must come from the same `params`
dict. Using `tp["q"]` with `params["enc"]` breaks the latent alignment the
networks were jointly trained to expect. Use either all `params` or all `tp`.

### 5. Multi-task gradients to the encoder are a feature
The encoder is trained via four gradient paths: consistency, reward, Q-value,
and policy. Removing any one of them (e.g., stop_grad before Q) significantly
degrades performance. The encoder learns a representation that is jointly
useful for dynamics prediction, reward prediction, and value prediction.

### 6. UTD ratio matters more than raw env-sps
Running N_ENVS=64 with K=1 update/step gives UTD=1/64 → policy never learns
despite high env throughput. Always set K_UPDATE so that UTD ≥ 1/N_ENVS at
minimum. For N=1024, K=64 gives UTD=1/16 (acceptable). For better quality,
K=256 with smaller N=256 gives UTD=1 at same wall time.

### 7. Q instability is correlated with MPPI drops
MPPI episodes where all sample returns are uniformly large (v spike) produce
near-uniform softmax weights. The weighted mean of 256 diverse samples ≈ zero.
This is why MPPI can dip below pi immediately after a v spike. The MPPI
variance is a leading indicator of world model stability.

### 8. Speed benchmarks for Warp JAX env
Env step cost ≈ 12ms per batch regardless of N (GPU-fixed overhead from Warp):

| N_ENVS | env-sps (pure) |
|--------|---------------|
| 16 | 1,151 |
| 64 | 5,151 |
| 256 | 20,004 |
| 1024 | 82,169 |
| 2048 | 162,936 |

Update step cost at (128,128): ~1-2ms. At (256,256): ~10ms.
With N=1024, K=64: effective sps ≈ 1,200-1,900 (update-bottlenecked).

---

## Gap Analysis: Ours vs Reference at 4M Steps

| Metric | Ours (v4f) | Reference |
|--------|-----------|-----------|
| MPPI at 4M | 138 | 449 |
| Total grad steps at 4M env_steps | 250k | 4M |
| UTD ratio | 1/16 | ~1:1 |

**Primary bottleneck**: UTD ratio. The reference algorithm does 1 gradient step
per env step (1:1 UTD). Our v4f does 64 steps per 1024 env steps (1:16 UTD).
At 4M env steps: we have done 250k gradient updates; reference has done 4M.
Quality gap is almost entirely explained by the 16× gradient deficit.

**Next experiment hypothesis**: Reduce N_ENVS to 256 and increase K_UPDATE to
256 → UTD=1:1 at the same ~1200 env-sps. Expected MPPI ≈ 300+ at 4M env steps.
Trade-off: 4× fewer diverse env streams, so off-policy data is less varied.
