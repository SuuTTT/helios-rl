# TD-MPC2 Parity Plan

**Goal**: Close the algorithm gap between our JAX TD-MPC2 and the official PyTorch implementation to match official hopper-hop results (≥370 @1M, ≥594 @4M).

**Reference**: official `tdmpc2/` (PyTorch), seed 3: 267.9@500k → 373.1@1M → 594.2@4M

**Baseline**: v12 (JAX, fused loop, 640 SPS): MPPI=49.9@1M, pi=68.3–144.1@750k–1M

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

Instruments to add:
- Log per-step mean Q value → check if Q is on the right scale (should track discounted return ~50–600)
- Log MPPI candidate return spread (max - min across 256 samples) → if near-zero, world model is useless
- Log reward prediction error (predicted vs actual 1-step reward on held-out batch)
- Log MPPI vs pi action distance (if MPPI agrees with pi, planner adds nothing)

Success criterion: at least one diagnostic shows a clear failure mode before Phase 1.

---

## Phase 1: Official-Parity MPPI Planner

**Target script**: `train_tdmpc_hopper_v13_mppi_parity.py`

**Changes from v12**:

### 1.1 Planning parameters
```python
H = 3           # was 5
NS = 512        # was 256 (total samples incl. pi trajs)
NUM_ELITES = 64 # was N/A (all-softmax)
NUM_PI_TRAJS = 24  # was 1
MIN_STD = 0.05  # was N/A
MAX_STD = 2.0   # was 0.5 fixed
NI = 6          # unchanged (iterations)
```

### 1.2 Elite-based MPPI (replace all-softmax with top-k elite selection)
```
1. Initialize mu = zeros(H, act_dim), std = full(H, act_dim, MAX_STD)
2. On t0 (episode start): reset mu to zeros
3. Each MPPI iteration:
   a. Sample 24 stochastic pi trajectories: eps~N(0,I), a = tanh(pi_mean + eps * exp(pi_log_std))
   b. Sample 488 Gaussian noise trajectories: a ~ mu + eps * std, clipped to [-1,1]
   c. Stack to (512, H, act_dim)
   d. Evaluate all 512: roll out H steps in world model, sum Q(z_H) + sum discounted rewards
   e. Select top-64 elite indices: elite_idx = argsort(rets)[-64:]
   f. elite_actions = actions[elite_idx]  # shape (64, H, act_dim)
   g. Update: mu = mean(elite_actions, axis=0), std = std(elite_actions, axis=0) + 1e-6
   h. Clamp: std = clip(std, MIN_STD, MAX_STD)
4. Final action: mu[0] (not sampled)
5. Shift: mu = roll(mu, -1, axis=0); mu[-1] = 0  # warm-start next step
```

### 1.3 Keep everything else from v12
- Training loop: fused lax.scan, K_UPDATE=64
- Collection: pi-only with noise (not changing yet)
- Actor: deterministic (not changing yet)
- Target: EMA entire tp (not changing yet)

### 1.4 Evaluation
- Eval both pi and MPPI every 250k steps, 3 episodes each
- Log `eval/mppi_reward` and `eval/pi_reward` separately

**Success criterion**: MPPI@1M ≥ 150, and MPPI consistently ≥ pi after 500k.

---

## Phase 2: Actor/Target Semantics

**Target script**: `train_tdmpc_hopper_v14_actor_fix.py`

**Changes from v13**:

### 2.1 Stochastic policy with entropy
```python
# In pi network output: (mean, log_std)
# log_std clamped to [-10, 2]
# Sample: a = tanh(mean + eps * exp(log_std))
# Entropy: -sum(log_prob) with tanh correction
# Actor loss: -Q(z, a) - entropy_coef * entropy
ENTROPY_COEF = 1e-4
```

### 2.2 RunningScale on Q targets
```python
# Before computing TD targets, scale Q values:
# scale = running 5/95th percentile spread of Q(z, a_pi)
# Q_scaled = Q / max(1.0, scale)
# Use EMA-updated percentiles (decay 0.99)
```

### 2.3 Target Q only (not full tree EMA)
```python
# Only EMA the Q-network parameters, not encoder/dynamics/reward
# tp = {q_params: ema_update(tp.q_params, params.q_params, tau)}
# Use tp.q_params for TD targets; use params.encoder for encoding targets
```

### 2.4 Separate actor optimizer
```python
# world_model_opt: AdamW for encoder + dynamics + reward + q
# actor_opt: Adam for pi, separate lr (default same)
```

**Success criterion**: pi@1M ≥ 200 and MPPI@1M ≥ 250.

---

## Phase 3: Loss Coefficients

**Changes from v14**:
```python
CONSISTENCY_COEF = 20.0  # was 2.0
REWARD_COEF      = 0.1   # was 2.0
VALUE_COEF       = 0.1   # was 1.0
```

Note: consistency loss dominates — this forces the encoder to learn a predictable latent space. The shift from reward/value to consistency is one of the biggest structural differences.

**Success criterion**: world model reward prediction error drops; consistency loss converges faster.

---

## Phase 4: Model Capacity

**Target script**: `train_tdmpc_hopper_v15_capacity.py`

**Changes from v13/v14 (after coefficients are right)**:
```python
LATENT_DIM = 512      # was 128
MLP_DIM    = 512      # was 128
NUM_Q      = 5        # was 2
```

Also:
- Zero-init last layer of reward/Q networks (as in official `weight_init(zero_std=...)`
- Encoder learning rate scaled by 0.3 relative to other params (if needed for stability)

**Warning**: this is a 4–16x larger model. Will reduce SPS significantly (possibly 200–400 SPS vs 640). Only do this after algorithm is correct.

**Success criterion**: pi@1M ≥ 300, MPPI@1M ≥ 350.

---

## Phase 5: Collection Parity

**Changes from v15**:
- After `seed_steps` (default 5000 env steps), switch from pi-collection to MPPI-collection
- Use same MPPI planner as eval but during data collection
- Maintain per-env MPPI warm-start (mu per env)
- Increase seed_steps to avoid garbage data dominating early replay

**Cost**: MPPI-collection at 256 envs simultaneously is expensive — need batched plan across envs.
```python
# vmap plan() over environments
# plan_batch = jax.vmap(plan, in_axes=(None, 0, 0, 0))
# actions = plan_batch(params, obs_batch, mu_batch, key_batch)
```

**Warning**: this likely drops SPS by another 2–5x compared to pi-collection. Major tradeoff.

**Success criterion**: MPPI@1M ≥ 400. If pi@1M also rises substantially, collection parity matters.

---

## Phase 6: Speed Recovery

After algorithm parity is confirmed, recover speed:

1. **Profile new bottleneck** — likely MPPI-collection (if Phase 5 done) or large model update
2. **Reduce NS for collection MPPI** — use 128 samples for collection, 512 for eval
3. **Cache pi trajectories** — reuse stochastic pi samples across iterations
4. **Reduce NI for collection MPPI** — use 3 iterations for collection, 6 for eval
5. **Reduce NUM_ENVS if OOM** — large model at 256 envs may OOM
6. **Profile update scan** — with larger model, K=64 scan cost increases; may need K=32

---

## Experiment Log

| Version | Key changes | pi@1M | MPPI@1M | SPS | Notes |
|---------|-------------|-------|---------|-----|-------|
| v12 | Phase 4 fused (baseline) | 68.3 | 49.9 | 640 | Running |
| v13 | Phase 1: MPPI parity | ? | ? | ~580 | To implement |
| v14 | Phase 2: actor/target | ? | ? | ~550 | After v13 |
| v15 | Phase 3: loss coefs | ? | ? | ~540 | After v14 |
| v16 | Phase 4: model capacity | ? | ? | ~300 | After v15 |
| v17 | Phase 5: MPPI collection | ? | ? | ~100 | After v16 |

---

## Reference Checkpoints

Official TD-MPC2, hopper-hop (from `results/tdmpc2/hopper-hop.csv`):
- seed 3: 267.9@500k, 373.1@1M, 594.2@4M
- seed 2: 306.3@500k, 271.2@1M (high variance seed)
- **Target**: ≥350@1M, ≥550@4M

---

## Decision Points

- If v13 MPPI@1M < 100: likely world model is not predictive enough — add Phase 0 diagnostics, check reward prediction error
- If v13 MPPI > v12 but still < pi: actor is learning well, planner underexplores — try more iterations or more diverse seeds
- If Phase 2 actor entropy causes instability: lower entropy_coef to 1e-5
- If Phase 4 capacity causes OOM: reduce NUM_ENVS to 128 or use gradient checkpointing
- If Phase 5 MPPI-collection tanks SPS too much: use hybrid (50% pi, 50% MPPI) collection
