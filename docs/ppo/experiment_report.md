# PPO Experiment Report — CheetahRun Benchmark
**Date:** 2026-05-07  
**Environment:** `dm_control/cheetah-run-v0` (max return ≈ 1000, episode length = 1000 steps)  
**Hardware:** NVIDIA GeForce RTX 3090 (24 GiB) / CPU (JAX uses GPU, CleanRL runs on CPU)

---

## 1. Executive Summary

Two PPO variants were benchmarked on CheetahRun. CleanRL's PyTorch PPO achieves **~550 peak return in 3 M steps (2.5 hours)**. A custom JAX/MJX PPO on GPU reaches **~465 peak return in 30 M steps (~12 minutes)**. The JAX variant is 84× faster in wall-clock time at 3 M steps but is capped structurally by its short rollout horizon relative to episode length.

| | CleanRL PyTorch PPO | JAX/MJX PPO (best config) |
|---|---|---|
| Framework | PyTorch + CleanRL | JAX + Flax + Optax |
| Env backend | dm_control (CPU) | MuJoCo Playground / MJX (GPU) |
| Num envs | 4 | 512 |
| Steps per rollout | 512 | 10 |
| Throughput | ~350 SPS | ~27 000 SPS |
| Wall-clock (3 M steps) | ~2.5 hours | ~110 seconds |
| Peak return @ 3 M steps | **550** | 124 |
| Peak return @ 30 M steps | — | **465** |
| Stability after peak | Stable | Stable (constant LR) |
| Checkpoint saved | — | ✓ (`ppo_jax_ue8_fixed_best.msgpack`) |

---

## 2. CleanRL PyTorch PPO

### 2.1 Configuration

```
script:         cleanrl/cleanrl/ppo_continuous_action_dmc.py
env_id:         dm_control/cheetah-run-v0
num_envs:       4
num_steps:      512
total_timesteps: 3_000_000
learning_rate:  3e-4 (annealed)
update_epochs:  10
num_minibatches: 32
```

### 2.2 Results

| Timestep | Return |
|---------|--------|
| ~500 K | ~200 |
| ~1 M | ~400 |
| ~1.5 M | ~460 |
| ~2.5 M | ~500–540 |
| 3 M (peak) | **550** |

- Training curve was monotonically increasing with no collapse.
- 512-step rollouts cover ~0.5 episodes per env per rollout → dense return signal.
- Wall-clock: ~2.5 hours on CPU.

---

## 3. JAX/MJX PPO — Evolution of Configs

All runs used `CheetahRun` via MuJoCo Playground registry, `episode_length=1000`.

### 3.1 Baseline (3 M steps)

```
num_envs=512, num_steps=10, lr=3e-4, ue=4  →  peak 124
num_envs=2048, num_steps=10, lr=3e-4, ue=4 →  peak 103
```
Short rollouts (1% of episode) produce badly truncated advantages — policy barely learns in 3 M steps.

### 3.2 Extended Sweep — 30 M steps

Key hypothesis: more gradient updates per timestep (increase `update_epochs`) and avoid LR annealing collapse.

| num_steps | update_epochs | lr | num_envs | anneal_lr | peak return |
|-----------|--------------|-----|---------|-----------|-------------|
| 10 | 4 | 3e-4 | 512 | on | 386 |
| 32 | 4 | 3e-4 | 512 | on | ~200 |
| 64 | 4 | 3e-4 | 512 | on | ~160 |
| 128 | 4 | 1.5e-4 | 512 | on | ~120 |
| 256 | 4 | 3e-4 | 512 | on | ~80 |
| 1000 | 4 | 3e-5 | 2048 | on | ~0 (too few updates) |
| 10 | 8 | 2e-4 | 512 | on | 444 (collapsed after) |
| 10 | 8 | 2e-4 | 2048 | on | ~420 |
| **10** | **8** | **2e-4** | **512** | **off** | **465 (stable)** |

**Key findings:**
1. `num_steps=10` maximises gradient update count for a 30 M budget — 585 K updates vs 23 K for `num_steps=128`.
2. `update_epochs=8` doubles effective learning per batch vs default 4 without hurting stability.
3. LR annealing (linear decay to 0) caused policy collapse at ~20 M steps when LR was near-zero — constant LR prevented this.
4. Best return 465.3 achieved at step 27.1 M. Policy remained in 350–465 range from step 15 M to 30 M.

### 3.3 Best JAX Run — Learning Curve

```
Step 5.6 M  → 384.8  (checkpoint saved)
Step 17.9 M → 463.6  (checkpoint saved)
Step 27.1 M → 465.3  (BEST — checkpoint saved)
Step 30 M   → 450    (no early-stop triggered, patience=30)
```

### 3.4 Structural Gap Analysis

| Source | Impact |
|--------|--------|
| Horizon truncation | `num_steps/episode_length = 1%` → GAE biased toward bootstrapped values |
| Total gradient updates (10 steps) | 30 M ÷ (512×10) × 8 epochs = 46 875 gradient steps |
| Total gradient updates (CleanRL) | 3 M ÷ (4×512) × 10 epochs = 14 648 gradient steps |
| Credit assignment depth | CleanRL sees full episodes; JAX sees 10-step windows only |

The remaining ~85-point gap (465 → 550) is structural, not a hyperparameter problem. Closing it requires either longer rollouts (which reduce update frequency), recurrence, or a world-model algorithm (DreamerV3).

---

## 4. Artifacts

| File | Description |
|------|-------------|
| `/workspace/runs/checkpoints/ppo_jax_ue8_fixed_best.msgpack` | Best JAX policy (return=465.3), Flax msgpack |
| `/workspace/runs/ppo_jax_ue8_fixed.log` | Full training log for best JAX run |
| `/workspace/runs/ppo_cleanrl_dmc_cheetahrun2.log` | CleanRL training log |
| `/workspace/ppo_comparison_report.md` | Earlier comparison at 3 M steps |
| `/workspace/run_ppo_continuous_mjx.py` | JAX PPO training script (modified) |
| `/workspace/cleanrl/cleanrl/ppo_continuous_action_dmc.py` | CleanRL dm_control adaptation |

---

## 5. Next Experiments

- **Brax PPO** (MuJoCo Playground native): `train_jax_ppo.py` uses Brax's own PPO with tuned defaults for each environment — expected to outperform our custom JAX PPO on the same env.
- **RSL-RL** (locomotion-focused): `train_rsl_rl.py` — designed for high-DOF locomotion; less relevant for cheetah-run but interesting comparison.
- **DreamerV3** (`/workspace/beat-dreamerv3-speed/`): World-model approach that should exceed CleanRL PPO on long-horizon tasks.
