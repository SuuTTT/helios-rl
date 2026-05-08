# Implementation Gap: Custom JAX PPO vs Brax PPO

**Date:** 2026-05-07 (updated 2026-05-08)  
**Goal:** Close the gap between our 465 peak and Brax PPO's 922 peak on CheetahRun.

---

## 1. Results Summary

| Variant | Peak Return | Total Steps | Wall-clock |
|---------|------------|-------------|-----------|
| CleanRL PyTorch PPO | 550 | 3 M | 2.5 hours |
| Custom JAX PPO (original) | 465 | 30 M | 12 min |
| **Brax PPO (reference)** | **922** | **88 M** | **~27 min (290s JIT + 967s train)** |
| **Custom JAX PPO v8 (fixed)** | **619** | **90 M** | **~10 min (47s JIT + 579s train)** |

---

## 2. Hyperparameter Diff

| Parameter | Our Best | Brax PPO | Δ Impact |
|-----------|---------|---------|---------|
| `num_steps` / `unroll_length` | 10 | **30** | +3× horizon |
| `gamma` / `discounting` | 0.99 | **0.995** | longer effective horizon |
| `learning_rate` | 2e-4 | **1e-3** | 5× higher |
| `update_epochs` / `num_updates_per_batch` | 8 | **16** | 2× gradient updates per rollout |
| `ent_coef` / `entropy_cost` | 0.0 | **0.01** | exploration bonus |
| `num_envs` | 512 | **2048** | 4× more diversity |
| `normalize_observations` | ❌ **MISSING** | ✓ (Welford) | **HIGH** |
| `reward_scaling` | ❌ **MISSING** | **10.0** | **HIGH** |
| `clip_coef` / `clipping_epsilon` | 0.2 | **0.3** | wider trust region |
| `vf_coef` | 0.5 | *(separate value net)* | — |

---

## 3. Architecture Diff

| Component | Our Custom JAX | Brax PPO |
|-----------|---------------|---------|
| Shared trunk | ✓ 2×256 swish | ✓ (default 2×256 swish via `MLP`) |
| `log_std` | Learnable global parameter | Same |
| Value network | Shares trunk | Shares trunk |
| **Obs normalisation** | ✓ (Welford, added v5+) | ✓ Running Welford mean/std |
| **Reward scaling** | ✓ (10.0, added v5+) | `reward × reward_scaling` before GAE |

---

## 4. Root Cause Analysis

### 4.1 Missing Obs Normalisation — Estimated Impact: HIGH

CheetahRun's observation space contains velocities, joint angles, and positions across multiple scales. Without normalisation:
- The value network receives unnormalised inputs → poor regression accuracy → biased advantages
- Policy gradient variance is high → slow, unstable learning

Brax normalises observations online using Welford's algorithm (numerically stable running mean/variance):
```
mean ← running mean over all obs seen so far
std  ← running std over all obs seen so far
norm_obs = (obs - mean) / std
```

### 4.2 Missing Reward Scaling — Estimated Impact: HIGH

Brax multiplies all rewards by `reward_scaling=10.0` before computing GAE advantages and returns. This:
- Scales the value function target into a larger numeric range → easier for the MLP to fit
- Increases the effective policy gradient signal magnitude

### 4.3 Gamma 0.99 vs 0.995

Effective horizon = `1 / (1 - γ)`:
- γ=0.99 → 100 step horizon
- γ=0.995 → 200 step horizon

For a 1000-step episode this matters a lot: γ=0.995 allows return signals to propagate 2× further through the rollout.

### 4.4 LR 2e-4 vs 1e-3

With obs normalisation and reward scaling, a 5× higher LR is stable because:
- Normalised inputs give consistent gradient magnitudes regardless of obs scale
- Scaled rewards give a more consistent value target scale

### 4.5 Entropy Cost 0.0 vs 0.01

`ent_coef=0.0` means the policy can collapse to deterministic actions early. For CheetahRun (which rewards sustained running speed), exploration helps escape local optima in the first few million steps.

---

## 5. Iteration History

### Runs v1–v2 (failed)

**Root causes:**
- v1 (`ppo_jax_brax_match`): `reward_scaling=10` + `lr=1e-3` → value gradient overwhelms policy gradient → collapse to 0.798
- v2 (`ppo_jax_v2`): obs_norm updated *inside* scan (per-step), stored `norm_obs` at step-t stats but GAE used step-T stats → inconsistent advantages → collapse to ~2

### Runs v3–v6 (new bugs introduced, still collapsing ~6–17)

Three separate bugs were introduced/discovered during attempts to fix normalisation:

| Bug | Run | Description |
|-----|-----|-------------|
| **IS ratio mismatch** | v3–v8 | `action = clip(raw_action)` stored in storage, but `logprob` computed at unclipped `raw_action`. PPO re-evaluates `logN(clip(z); μ', σ')` — different point under Gaussian → IS ratio corrupted. |
| **Stale stored values** | v3–v5 | `storage.values` from rollout-time critic; after epoch 1 of 16 the critic improves but `returns = adv + stored_values` stay stale → returns target shifts → v_loss instability |
| **obs_norm eps** | v3–v5 | Used `sqrt(var) + 1e-8` (eps outside sqrt). For near-zero variance: denominator ≈ 1e-8 → normalized values explode to millions. Fixed to `sqrt(var + eps)` with clip to `[1e-6, 1e6]`. |

### Run v8 — SUCCESS (2026-05-08)

**Fixes applied:**
1. **Store raw unclipped action** in storage; send `clip(action, -1, 1)` to env separately. PPO recomputes `logN(raw_action; μ', σ')` → correct IS ratio.
2. **Per-epoch fresh GAE**: all T×N values recomputed from current critic at start of each epoch → returns target stays consistent with value loss.
3. **obs_norm_apply**: `std = clip(sqrt(var + 1e-6), 1e-6, 1e6)` — matches Brax, prevents denominator explosion.
4. **obs_norm order**: update stats from raw batch *after* PPO update (for next rollout). Normalise current rollout with pre-update stats (consistent with stored logprobs).

**Result:** 619 at 90 M steps, 579s training (vs Brax 922, 967s). **Gap to Brax: -303 (33% below).**

---

## 9. Iteration v9–v10 (2026-05-08)

### v9 — lr=1e-3 (Brax exact)

**Peak: 612** — similar to v8 (619) but noisier (large drops: 391→14→165). Higher LR amplifies gradient noise at high performance. Not clearly better than lr=3e-4.

### v10 — Added truncation bootstrap fix

**Peak: 592** — added `truncation` from `state.info['truncation']` to GAE:
```python
nextnonterminal = clip(1.0 - done + truncation, 0, 1)  # 1 for timeouts, 0 for true terminals
```
Without this, every ~33rd rollout (when all 2048 envs time out together) gets a massive negative bias: `δ_t = r_t - V(s_t)` instead of `r_t + γV(s_{t+1}) - V(s_t)`. The missing `γV(s_{t+1}) ≈ 617` term suppresses policy gradient every ~33 rollouts.

Despite the fix being theoretically correct, peak didn't improve, likely masked by lr=1e-3 noise.

---

## 10. Active Experiment: v11 (2026-05-08)

**Config:** All v10 fixes + lr=3e-4 + **180 M steps** (2× longer) + early_stop_patience=50  
**Rationale:** v8 was still improving at step 81M (619 at step 81M, end of 90M run). More steps should push past 619.

---

## 7. Log Format

```
{global_step}: reward={mean_return:.3f}  [SPS={sps}]
```

---

## 8. Files

| File | Role |
|------|------|
| `/workspace/run_ppo_continuous_mjx.py` | Our script (v8+) |
| `/workspace/run_brax_ppo.py` | Brax PPO wrapper (with JAX 0.10 shim) |
| `/workspace/runs/brax_ppo_cheetahrun.log` | Brax PPO reference results |
| `/workspace/runs/ppo_jax_v8.log` | v8 run: 619 peak |

---

## 1. Results Summary

| Variant | Peak Return | Total Steps | Wall-clock |
|---------|------------|-------------|-----------|
| CleanRL PyTorch PPO | 550 | 3 M | 2.5 hours |
| Custom JAX PPO (best) | 465 | 30 M | 12 min |
| **Brax PPO** | **922** | **88 M env-steps** | **~27 min (290s JIT + 967s train)** |

> Note: Brax reports env-steps = `num_timesteps × num_envs / num_envs = num_timesteps`. Its 88 M at the last eval comes from rounding: each eval interval = `num_envs × unroll_length × N_batches_per_eval` steps.

---

## 2. Hyperparameter Diff

| Parameter | Our Best | Brax PPO | Δ Impact |
|-----------|---------|---------|---------|
| `num_steps` / `unroll_length` | 10 | **30** | +3× horizon |
| `gamma` / `discounting` | 0.99 | **0.995** | longer effective horizon |
| `learning_rate` | 2e-4 | **1e-3** | 5× higher |
| `update_epochs` / `num_updates_per_batch` | 8 | **16** | 2× gradient updates per rollout |
| `ent_coef` / `entropy_cost` | 0.0 | **0.01** | exploration bonus |
| `num_envs` | 512 | **2048** | 4× more diversity |
| `normalize_observations` | ❌ **MISSING** | ✓ (Welford) | **HIGH** |
| `reward_scaling` | ❌ **MISSING** | **10.0** | **HIGH** |
| `clip_coef` / `clipping_epsilon` | 0.2 | **0.3** | wider trust region |
| `vf_coef` | 0.5 | *(separate value net)* | — |

---

## 3. Architecture Diff

| Component | Our Custom JAX | Brax PPO |
|-----------|---------------|---------|
| Shared trunk | ✓ 2×256 swish | ✓ (default 2×256 swish via `MLP`) |
| `log_std` | Learnable global parameter | Same |
| Value network | Shares trunk | Shares trunk |
| **Obs normalisation** | ❌ None | ✓ Running Welford mean/std updated every rollout |
| **Reward scaling** | ❌ None (raw reward) | `reward × reward_scaling` before GAE |

---

## 4. Root Cause Analysis

### 4.1 Missing Obs Normalisation — Estimated Impact: HIGH

CheetahRun's observation space contains velocities, joint angles, and positions across multiple scales. Without normalisation:
- The value network receives unnormalised inputs → poor regression accuracy → biased advantages
- Policy gradient variance is high → slow, unstable learning

Brax normalises observations online using Welford's algorithm (numerically stable running mean/variance):
```
mean ← running mean over all obs seen so far
std  ← running std over all obs seen so far
norm_obs = (obs - mean) / (std + ε)
```
State is updated every rollout (not per-step) to keep it JAX-compatible.

### 4.2 Missing Reward Scaling — Estimated Impact: HIGH

Brax multiplies all rewards by `reward_scaling=10.0` before computing GAE advantages and returns. This:
- Scales the value function target into a larger numeric range → easier for the MLP to fit
- Increases the effective policy gradient signal magnitude
- Combined with the clipping ε and entropy cost, this acts similarly to a reward-weighted LR boost

Without this, our value function is fitting returns in [0, 465] while Brax's is fitting [0, 9220].

### 4.3 Gamma 0.99 vs 0.995

Effective horizon = `1 / (1 - γ)`:
- γ=0.99 → 100 step horizon
- γ=0.995 → 200 step horizon

For a 1000-step episode this matters a lot: γ=0.995 allows return signals to propagate 2× further through the rollout.

### 4.4 LR 2e-4 vs 1e-3

With obs normalisation and reward scaling, a 5× higher LR is stable because:
- Normalised inputs give consistent gradient magnitudes regardless of obs scale
- Scaled rewards give a more consistent value target scale

Without normalisation, high LR would destabilise training (and did in our earlier sweeps).

### 4.5 Entropy Cost 0.0 vs 0.01

`ent_coef=0.0` means the policy can collapse to deterministic actions early. For CheetahRun (which rewards sustained running speed), exploration helps escape local optima in the first few million steps.

---

## 5. Iteration Plan

Priority order based on estimated impact:

| Priority | Change | Expected Δ Return |
|----------|--------|------------------|
| 1 | Add obs normalisation (Welford, JAX-compatible) | +100–200 |
| 2 | Add reward scaling (default 10.0) | +50–100 |
| 3 | Match gamma=0.995 | +20–50 |
| 4 | Increase LR to 1e-3, update_epochs=16, num_steps=30 | +30–100 |
| 5 | Add entropy_cost=0.01 | +10–30 |
| 6 | Increase num_envs to 2048 | +10–20 |

**Target:** Reproduce Brax's 922 with our script architecture.

---

## 6. Log Format

Brax PPO log format (adopted):
```
{global_step}: reward={mean_return:.3f}
```
Our previous format: `step=XX, episodic_return=XX.XXXX, SPS=XX`  
Updated format (matches Brax, adds SPS for diagnostics):
```
{global_step}: reward={mean_return:.3f}  [SPS={sps}]
```

---

## 8. Iteration 1 — Failure Analysis (2026-05-07)

**Run:** `ppo_jax_brax_match`, 90 M steps, log at `/workspace/runs/ppo_jax_brax_match.log`  
**Result:** Peak 17 at step 6M → collapsed to 0.798 → never recovered.

### Root Cause A: LR too high relative to reward_scaling

`reward_scaling=10` scales all rewards by 10× before GAE. This means:
- Value function targets are in [0, ~10,000] instead of [0, ~1,000]
- Initial value prediction ≈ 0, so initial value error = 10,000
- Value loss gradient ∝ `(pred - target)` ≈ -10,000 per step
- With `vf_coef=0.5` and `lr=1e-3`: effective value LR acts like `lr=1e-2` on the unscaled problem

Despite `max_grad_norm=0.5` clipping, the value gradients dominate the total gradient, shrinking the policy gradient near zero. The value function fails to bootstrap → advantages are noisy → policy collapses.

**Fix:** Lower LR to `1e-4` when using `reward_scaling=10`. Rule of thumb: `lr ≈ 1e-3 / reward_scaling^0.5 ≈ 3e-4`.

### Root Cause B: Obs normalisation updated outside scan (stale stats)

The obs_norm_state was updated once per iteration from only the LAST `next_obs` (2048 samples). The rollout scan used fixed, stale stats from the previous iteration. Early in training (iterations 1-50), normalisation stats are built from only ~100K samples instead of 3M → poorly normalised inputs → unstable value regression → policy collapse.

**Fix:** Move obs_norm update INSIDE `step_once` scan. Each step updates stats from the current `num_envs` observations. This means obs_norm is fresh at every step, matching Brax's approach.

### Iteration Plan (Revised)

| Priority | Change | Status |
|----------|--------|--------|
| 1 | Fix obs_norm update inside scan | next |
| 2 | Lower LR: `lr=1e-4` with `reward_scaling=10` | next |
| 3 | Run with Brax-matched config | next |
| 4 | Tune LR upward if stable | after |

---

## 9. Files

| File | Role |
|------|------|
| `/workspace/run_ppo_continuous_mjx.py` | Our script (being updated) |
| `/workspace/run_brax_ppo.py` | Brax PPO wrapper (with JAX 0.10 shim) |
| `/workspace/runs/brax_ppo_cheetahrun.log` | Brax PPO reference results |
| `/workspace/runs/ppo_jax_ue8_fixed.log` | Our best pre-fix results |
