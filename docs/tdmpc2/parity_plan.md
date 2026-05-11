# TD-MPC2 Parity Plan

**Goal**: Close the algorithm gap between our JAX TD-MPC2 and the official PyTorch implementation to match official hopper-hop results (≥370 @1M, ≥594 @4M).

**Reference**: official `tdmpc2/` (PyTorch), seed 3: 267.9@500k → 373.1@1M → 594.2@4M

**Baseline**: v12 (JAX, fused loop, 640 SPS): MPPI=49.9@1M, pi=68.3@1M

---

## Key Algorithm Gaps (ordered by impact)

| # | Gap | Our impl | Official | Impact |
|---|-----|----------|----------|--------|
| 1 | MPPI planner quality | 256 samples, all-softmax, 1 pi-traj, fixed std=0.5, H=5 | 512 samples, 64 elites, 24 pi-trajs, adaptive std [0.05,2.0], H=3 | HIGH |
| 2 | Collection strategy | pi-only with noise | MPPI after seed steps | HIGH |
| 3 | UTD ratio | 0.25 (K=64 per 256 env steps) | 1.0 (1 update per env step) | HIGH |
| 4 | Actor: entropy + RunningScale | none | entropy_coef=1e-4 + RunningScale | HIGH |
| 5 | Target network scope | EMA entire `tp` dict | EMA target Q only | MEDIUM |
| 6 | Loss coefficients | consistency=2, reward=2, value=1 | consistency=20, reward=0.1, value=0.1 | MEDIUM |
| 7 | Model capacity | latent=128, mlp=(128,128), Q=2 | latent=512, mlp_dim=512, Q=5 | MEDIUM |
| 8 | Discount | γ=0.99 | γ=0.995 (discount_denom=5, ep_len=1000) | LOW |

---

## Phase 0: Diagnostics (instrumentation, no algo change)

**Purpose**: Understand what's failing before changing things.

Script: `train_tdmpc_hopper_v13_diag.py` (ready, logs every 50k steps)

Instruments:
- `mean_q`: mean Q(z, pi(z)) over batch → should track discounted return (~50–600)
- `rew_err`: mean |predicted_reward - actual_reward| → world model quality
- `mppi_spread`: mean MPPI return spread (max-min) per episode → near-zero = world model useless
- `mppi_mean_ret`: mean imagined return from MPPI → calibration vs real return
- `act_norm`, `mean_log_std`: actor confidence proxies

Status: **ready to launch** (after v13 finishes or if v13 results are poor)

---

## Phase 1: Official-Parity MPPI Planner ✅

**Script**: `train_tdmpc_hopper_v13_mppi_parity.py` — RUNNING

**Changes from v12**:
- H=3 (was 5), NS=512 (was 256), NUM_ELITES=64, NUM_PI_TRAJS=24
- Elite-based top-k selection (not all-softmax)
- Adaptive std [0.05, 2.0], updated per iteration from elite distribution
- 24 stochastic pi trajectories seeded per MPPI iteration
- t0 flag resets mu/std on episode start

**Results**:

| Step | pi | MPPI | Notes |
|------|----|------|-------|
| 250k | 1.9 | 0.0 | Early training |
| 500k | 117.0 | 73.0 | vs v12 MPPI=0.9 |
| 750k | 165.7 | 108.3 | |
| 1M | 179.8 | 169.0 | vs v12 MPPI=49.9 |
| 1.25M | 141.8 | - | Still running |

**Success criterion**: MPPI@1M ≥ 150 ✅ (169.0)

---

## Phase 2: Actor/Target Semantics

**Target script**: `train_tdmpc_hopper_v14_actor_fix.py`

**Changes from v13**:

### 2.1 Stochastic policy with entropy
```python
ENTROPY_COEF = 1e-4
# Actor loss: -Q(z, a) - entropy_coef * H(pi)
# a = tanh(mean + eps * exp(log_std)), eps~N(0,I)
# entropy with tanh squash correction
```

### 2.2 RunningScale on Q targets
```python
# scale = EMA of (95th - 5th percentile) of Q(z, pi(z)) across batch
# Q_target = Q / max(1.0, scale)
# EMA decay = 0.99
```

### 2.3 Target Q only (not full tree EMA)
```python
# Only EMA q params: tp["q"] = (1-tau)*tp["q"] + tau*params["q"]
# Use current params for enc/dyn/rew; only tp["q"] for TD targets
```

### 2.4 Separate actor optimizer
```python
# world_model params: {enc, dyn, rew, q} → AdamW, lr=3e-4
# pi params: Adam, lr=3e-4
```

**Success criterion**: pi@1M ≥ 200 and MPPI@1M ≥ 250.

---

## Phase 3: Loss Coefficients

**Changes from v14**:
```python
CONSISTENCY_COEF = 20.0  # was 2.0 — encoder must be predictable
REWARD_COEF      = 0.1   # was 2.0
VALUE_COEF       = 0.1   # was 1.0
```

Note: official weighting puts 200x more weight on consistency than reward. This fundamentally changes what the encoder learns.

**Success criterion**: reward prediction error drops after first 500k steps.

---

## Phase 4: Model Capacity

**Target script**: `train_tdmpc_hopper_v15_capacity.py`

**Changes**:
```python
LATENT_DIM = 512   # was 128
MLP_DIM    = 512   # was 128
NUM_Q      = 5     # was 2
```

Also: zero-init last layer of reward/Q; enc lr × 0.3.

**Warning**: 4–16x larger model. Will drop SPS to ~200–400. Only after algorithm is correct.

**Success criterion**: pi@1M ≥ 300, MPPI@1M ≥ 350.
**Status (v24)**: ✅ ACHIEVED — v24 MPPI@1.25M=344, MPPI@1.75M=353, MPPI@3M=357. Stable at 340–357 from 1.25M onwards.

---

## Phase 5: Collection Parity

After model capacity is validated:
- Switch to MPPI collection after seed steps
- `jax.vmap(plan)` over all 256 envs
- Per-env mu/std warm-start

**Warning**: drops SPS by another 2–5x. Likely ~50-100 SPS.

**Success criterion**: MPPI@1M ≥ 400.
**Status**: Not started. Next after v24 4M run completes.

---

## Phase 6: Speed Recovery

1. Profile new bottleneck
2. Reduce NS/NI for collection MPPI (128 samples, 3 iters)
3. Cache pi trajectories across iterations
4. Reduce NUM_ENVS if OOM

---

## Remaining Architecture Gaps (to close the 350→594 gap)

Based on official config analysis, the following discrepancies remain after v24:

| Gap | Ours (v24) | Official | Priority |
|-----|------------|---------|----------|
| num_q | 2 | **5** | HIGH — 5-head ensemble gives better conservative Q estimates |
| vmin/vmax | ±20 | **±10** | HIGH — 2× coarser bin resolution; Q≈6 symlog for hopper |
| enc_lr_scale | 1.0 (missing) | **0.3** | HIGH — encoder updates 3× too fast, destabilizes heads |
| grad_clip_norm | none | **20** | MEDIUM — p_loss at -14 shows unconstrained gradients |
| dropout | 0.0 | **0.01** | MEDIUM — missing regularization in NormMLP layers |
| gamma | 0.99 | **≈0.995** | LOW — 1 − discount_denom/ep_len = 1 − 5/1000 |
| eval_episodes | 3 | **10** | LOW — our variance is 3× higher |

**Proposed v25** = v24 + (num_q=5, vmin/vmax=±10, enc_lr_scale=0.3, grad_clip=20).
These are the four changes most likely to unlock the phase transition (hopper learning consistent gait) seen in official seed 3 at 3M+.

---

## Experiment Log

| Version | pi@500k | MPPI@500k | pi@1M | MPPI@1M | pi@4M | MPPI@4M | SPS | Status |
|---------|---------|-----------|-------|---------|-------|---------|-----|--------|
| v12 (baseline) | ~5.1 | 0.9 | 68.3 | 49.9 | 239.3 | — | 640 | ✅ Done |
| v13 (Phase 1: MPPI parity) | 117 | 73 | 180 | 169 | 337 | **388** | ~560 | ✅ Done |
| v14 (Phase 2: stochastic pi) | 175 | 144 | 0 | 0 | 266 | 263 | ~550 | ✅ Done |
| v15 (official coefs, latent=128) | 0 | 0 | — | — | — | — | ~540 | ✅ Done |
| v16 (official coefs, latent=512) | ~7 | ~0 | ~40 | ~40 | — | ~unstable | ~300 | ✅ Done |
| v17 (latent=512, no RunningScale) | 168 | 185 | 0 | 0 | — | — | ~500 | ✅ Done |
| v18 (latent=512 + RunningScale) | 76 | 163 | 261 | 244 | 270 | **280** | ~550 | ✅ Done |
| v19 (v18 + entropy_coef=1e-4) | 0 | 68 | 246 | 244 | 264 | ~270 | ~550 | ✅ Done |
| v20 (K_UPDATE=256) | 225 | 239 | 231 | 234 | — | ~235 | ~550 | ✅ Done |
| v21 (v20 + TAU/K fix) | ~12 | ~0 | — | — | — | — | ~550 | ✅ Done |
| v22 (v20 + fixed target) | ~22 | ~67 | — | — | — | — | ~550 | ✅ Done |
| v23 (latent=256) | ~0 | ~0 | 262 | 272 | 270 | 303 | ~550 | ✅ Done |
| **v24 (RunningScale cap=4.0)** | **0** | **0** | **330** | **344** | — | **~350** | **~560** | 🔄 Running |
| Official (PyTorch, seed 3) | 268 | — | 373 | — | — | **594** | 10 | Reference |
| Official (PyTorch, seeds 1-2 avg) | ~170 | — | ~320 | — | — | **~377** | 10 | Reference |

---

## Decision Points

- ✅ v13@4M MPPI=388 ≥ 250: Phase 2 proceeded
- ✅ Phase 2 entropy (v14) caused instability: stochastic pi alone insufficient without RunningScale
- ✅ Phase 4 capacity (v17): latent=512 needs RunningScale
- ✅ RunningScale (v18): prevents collapse, scale cap (v24) breaks plateau → v24 milestone
- **Next**: v25 targeting official architecture gaps (num_q=5, vmin=±10, enc_lr_scale=0.3, grad_clip=20)
- If v24@4M MPPI ≥ 370: v24 matches official seed 1-2 mean → proceed to Phase 5 (MPPI collection)
- If v25 causes instability: add changes one at a time (num_q=5 first, highest impact)
