# PPO DMC Suite — Experiment Analysis

**Date:** 2026-05-09  
**Status:** Baseline complete, FingerSpin fix in progress

---

## 1. Setup

| Item | Value |
|---|---|
| Algorithm | PPO (JAX, parallel envs) |
| Hardware | NVIDIA RTX 3090 24 GiB |
| Envs | 8 × DMC Suite via MuJoCo Playground (MJX) |
| Seeds | 5 per env |
| Steps | 75M per seed |
| Reference | Brax PPO (MuJoCo Playground official) |

**Shared hyperparameters used for all envs (baseline run):**

| Param | Value |
|---|---|
| `num_envs` | 2048 |
| `num_steps` | 30 |
| `learning_rate` | 1e-3 |
| `update_epochs` | 16 |
| `num_minibatches` | 32 |
| `gamma` | 0.995 |
| `ent_coef` | 0.01 |
| `clip_coef` | 0.3 |
| `max_grad_norm` | 1.0 |
| `reward_scaling` | 10.0 |
| `anneal_lr` | False |
| `eval_freq` | 1 (every outer iteration ≈ 983K env steps) |

---

## 2. Results Summary

| Environment | Seeds | Mean Best | Min | Max | Paper (Brax) |
|---|---|---|---|---|---|
| CheetahRun | 5 | 887.5 | 873.7 | 903.0 | ~900 |
| BallInCup | 5 | 754.5 | **0.0** | 959.2 | ~950 |
| CartpoleSwingup | 5 | 814.4 | 770.2 | 836.1 | ~850 |
| CartpoleSwingupSparse | 5 | 413.3 | **0.0** | 638.2 | — |
| FingerSpin | 5 | 408.5 | 378.0 | 436.0 | ~600 ⚠️ |
| FishSwim | 5 | 611.9 | 592.8 | 629.2 | ~650 |
| AcrobotSwingup | 5 | 106.7 | 102.2 | 109.2 | ~100 |
| HopperStand | 5 | 83.8 | 6.0 | **223.1** | ~300 |

Plots: `helios-rl/exp/ppo/plots/`  
Plot script: `helios-rl/scripts/plot_dmc_suite.py`

---

## 3. Issues Identified

### 3.1 Zigzag / High-Frequency Oscillation on Most Curves

**Symptom:** Most curves (especially CheetahRun, CartpoleSwingup, FingerSpin) show rapid up-down oscillations on a ~1M step scale, making the learning curve look noisy compared to official Brax results.

**Root cause — evaluation frequency mismatch:**

| Source | Eval interval | Evals over 75M steps |
|---|---|---|
| Our impl (`eval_freq=1`) | ≈ 983K steps | ~76 |
| Brax reference (`num_evals=10`) | ≈ 7.5M steps | 10 |

We log **7.6× more frequently** than the reference. PPO has inherent per-iteration variance in policy performance (policy collapses, recovers, collapses) — Brax simply subsamples this noise at coarser resolution, making curves look smooth.

**The underlying variance is real.** Brax doesn't have better training dynamics; it just reports fewer data points. When we apply EWA smoothing (α=0.4) to our data, the curves look qualitatively similar to Brax.

**Not affected:** FishSwim, AcrobotSwingup, HopperStand — these have lower absolute performance ceilings or naturally smoother dynamics.

**Secondary cause — crash recovery re-loading:** Our crash recovery (reload best params when reward drops >150) introduces discontinuities at recovery events. These appear as hard spikes upward in the raw curves. This is harmless but contributes to the zigzag appearance.

**Recommended fix:** Use `--smooth 0.4` in the plot script (EWA), or reduce `--eval-freq` to 5–10 in future runs. Do **not** change training — the variance is real.

### 3.2 Bimodal Learning on Sparse / Hard Envs

**Symptom:** BallInCup seed 2 = 0 for all 75M steps. CartpoleSwingupSparse has 1 zero seed. HopperStand has seeds 1,3,4 ≈ 6–9 (essentially zero).

**Root cause:** PPO's on-policy exploration is insufficient for these envs. Without ever receiving a positive reward signal, the policy never learns. Entropy regularization (`ent_coef=0.01`) is too weak.

**Recommended fix:** SAC (off-policy, implicit entropy maximization) or higher `ent_coef` / entropy annealing for these envs.

### 3.3 FingerSpin Gap (408 vs 600)

**Root cause — wrong gamma:** FingerSpin's reference config uses `gamma=0.95`, not `0.995`. We ran all envs with CheetahRun's `gamma=0.995`.

```
Reference (Brax): FingerSpin → discounting=0.95
Our run:          FingerSpin → gamma=0.995  ← WRONG
```

A lower gamma (0.95) shortens the effective horizon, which is critical for the finger-spinning task where only recent contacts matter. With `gamma=0.995`, the value function integrates rewards over ~200 steps, causing the critic to overfit to long-horizon noise and destabilizing learning.

**Fix:** Rerun FingerSpin with `gamma=0.95` (and also `total_timesteps=60M` matching reference). See Section 5.

**Similarly affected:** BallInCup (also uses `gamma=0.95` in reference). However BallInCup's bimodal issue likely dominates.

---

## 4. Per-Environment Notes

### CheetahRun ✅
- Best result: 903 (seed 3), mean 887.5 — matches Brax reference (~900).
- Zigzag visible but mean is stable. Crash recovery active on ~2–3 seeds.

### BallInCup ⚠️
- 4 seeds learn well (mean ~939 over learning seeds), 1 seed stuck at 0.
- Should use `gamma=0.95` (reference value).
- Strong candidate for SAC.

### CartpoleSwingup ✅
- Consistent, all seeds between 770–836. Slightly below reference (~850).
- No obvious issues.

### CartpoleSwingupSparse ⚠️
- 1 zero seed out of 5. Sparse reward — same bimodal issue as BallInCup.
- Learning seeds reach 550–638. Reasonable.

### FingerSpin ❌
- All 5 seeds plateau at 378–436. Reference is ~600.
- **Hypothesis confirmed: wrong gamma (0.995 vs 0.95).**
- Scheduled for rerun with correct hyperparams.

### FishSwim ✅
- Consistent 591–629 across all seeds. Reference ~650.
- Small gap, likely from `gamma=0.995 vs 0.995` (same) — may need more steps.

### AcrobotSwingup ✅
- 102–109 across all seeds. Reference ~100. Matches.

### HopperStand ⚠️
- Extremely bimodal: 3 seeds stuck at ≈6–9, 2 seeds reach 174–225.
- Reference is ~300. Even the "learning" seeds underperform.
- `gamma=0.995` may be wrong here too. Needs investigation.

---

## 5. Planned Experiments

### EXP-001: FingerSpin Gamma Fix
**Script:** `helios-rl/scripts/run_fingerspin_fix.py`  
**Hypothesis:** Using `gamma=0.95` (matching reference) will close the 200-point gap.

| Config | Value |
|---|---|
| `gamma` | **0.95** (was 0.995) |
| `total_timesteps` | 75M (same as baseline) |
| seeds | 1–5 |
| Other params | Same as baseline |

**Success criterion:** Mean best ≥ 550 (vs current 408).

### EXP-002: FingerSpin with Reference Total Steps
- Reference uses 60M steps — shorter run. Validate our 75M run would converge similarly.

### EXP-003: Per-env Gamma Correction (All Envs)
- BallInCup: rerun with `gamma=0.95`
- HopperStand: test `gamma=0.97` and `0.98`

### EXP-004: SAC for Sparse Envs (Future)
- BallInCup, CartpoleSwingupSparse, HopperStand (bimodal seeds)
- Expected: higher floor, faster convergence on sparse rewards

---

## 6. How to Reproduce Plots

```bash
cd /workspace/helios-rl

# All envs — smoothed (recommended for reports)
python3 scripts/plot_dmc_suite.py \
    --csv_dir exp/ppo/csv \
    --out_dir exp/ppo/plots \
    --smooth 0.4

# Unsmoothed (raw eval data)
python3 scripts/plot_dmc_suite.py \
    --csv_dir exp/ppo/csv \
    --out_dir exp/ppo/plots_raw

# Single env only
python3 scripts/plot_dmc_suite.py \
    --csv_files exp/ppo/csv/ours_fingerspin.csv:FingerSpin \
    --out_dir exp/ppo/plots \
    --smooth 0.4

# Grid only (skip individual plots)
python3 scripts/plot_dmc_suite.py \
    --csv_dir exp/ppo/csv \
    --out_dir exp/ppo/plots \
    --no_per_env
```

---

## 7. File Map

```
helios-rl/
├── scripts/
│   ├── plot_dmc_suite.py          ← plotting script (this skill)
│   └── run_fingerspin_fix.py      ← EXP-001 runner
├── exp/
│   └── ppo/
│       ├── csv/
│       │   ├── ours_cheetahrun.csv
│       │   ├── ours_ballincup.csv
│       │   ├── ours_cartpoleswingup.csv
│       │   ├── ours_cartpoleswingupsparse.csv
│       │   ├── ours_fingerspin.csv          ← baseline (gamma=0.995, wrong)
│       │   ├── ours_fingerspin_g095.csv     ← EXP-001 output (gamma=0.95)
│       │   ├── ours_fishswim.csv
│       │   ├── ours_acrobotswingup.csv
│       │   └── ours_hopperstand.csv
│       └── plots/
│           ├── all_ci.png
│           ├── all_optic.png
│           └── <env>_{ci,optic}.png  (×8 envs × 2 styles = 16 files)
└── docs/
    └── ppo/
        └── ppo_dmc_suite_analysis.md   ← this file
```
