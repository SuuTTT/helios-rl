# PPO Launch Guide

**Environment:** CheetahRun / `dm_control/cheetah-run-v0`  
**Workspace root:** `/workspace`

---

## Prerequisites

```bash
# Verify GPU
nvidia-smi

# Verify JAX sees GPU
python3 -c "import jax; print(jax.devices())"

# Verify MuJoCo Playground
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 -c "from mujoco_playground import registry; print(registry.ALL_ENVS[:5])"
```

---

## 1. Variant A — CleanRL PyTorch PPO

**Script:** `cleanrl/cleanrl/ppo_continuous_action_dmc.py`

### 1.1 Standard Run (3 M steps, ~2.5 hours)

```bash
MUJOCO_GL=egl python3 /workspace/cleanrl/cleanrl/ppo_continuous_action_dmc.py \
  --env-id "dm_control/cheetah-run-v0" \
  --total-timesteps 3000000 \
  --num-envs 4 \
  --num-steps 512 \
  --exp-name ppo_cleanrl_dmc \
  2>&1 | tee /workspace/runs/ppo_cleanrl_dmc.log
```

### 1.2 Key Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--env-id` | — | Must use `"dm_control/cheetah-run-v0"` (shimmy syntax) |
| `--num-envs` | 4 | Increase for more diversity; CPU-bound |
| `--num-steps` | 512 | 512 steps ≈ half an episode per rollout |
| `--learning-rate` | 3e-4 | Annealed to 0 by default |
| `--total-timesteps` | 1M | Set 3M for a serious run |

### 1.3 View TensorBoard

```bash
tensorboard --logdir /workspace/runs/ --port 6006
```

---

## 2. Variant B — Custom JAX/MJX PPO

**Script:** `/workspace/run_ppo_continuous_mjx.py`

### 2.1 Best Config Run (30 M steps, ~12 min)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/run_ppo_continuous_mjx.py \
  --env-id CheetahRun \
  --total-timesteps 30000000 \
  --num-envs 512 \
  --num-steps 10 \
  --learning-rate 2e-4 \
  --update-epochs 8 \
  --no-anneal-lr \
  --early-stop-patience 30 \
  --checkpoint-dir /workspace/runs/checkpoints \
  --exp-name ppo_jax_ue8 \
  2>&1 | tee /workspace/runs/ppo_jax_ue8.log
```

### 2.2 Quick Smoke Test (3 M steps, ~2 min)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/run_ppo_continuous_mjx.py \
  --env-id CheetahRun \
  --total-timesteps 3000000 \
  --num-envs 512 \
  --num-steps 10 \
  --exp-name ppo_jax_smoke \
  2>&1 | tee /workspace/runs/ppo_jax_smoke.log
```

### 2.3 Key Flags

| Flag | Best Value | Notes |
|------|-----------|-------|
| `--env-id` | `CheetahRun` | MuJoCo Playground registry name |
| `--num-envs` | 512 | ~8 GB GPU RAM; use 2048 if memory allows |
| `--num-steps` | 10 | Short rollout maximises update frequency |
| `--update-epochs` | 8 | Doubles gradient steps vs default 4 |
| `--learning-rate` | 2e-4 | Slightly lower than default 3e-4 for stability |
| `--no-anneal-lr` | ✓ required | Constant LR prevents late-training collapse |
| `--early-stop-patience` | 30 | Stop if no improvement for 30 eval intervals |
| `--checkpoint-dir` | any path | Saves `{exp_name}_best.msgpack` on improvement |

### 2.4 Load a Checkpoint

```python
import flax, jax, jax.numpy as jnp

# Build template params from your model first
with open("/workspace/runs/checkpoints/ppo_jax_ue8_fixed_best.msgpack", "rb") as f:
    params = flax.serialization.from_bytes(template_params, f.read())
```

### 2.5 All Environment IDs

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 -c "from mujoco_playground import registry; print(registry.ALL_ENVS)"
```

Notable: `CartpoleBalance`, `CheetahRun`, `HopperHop`, `WalkerWalk`, `HumanoidStand`, `PandaPickCube`

---

## 3. Variant C — MuJoCo Playground Brax PPO (Next Step)

**Script:** `/workspace/wiki/learn_mujoco_playground/repo/learning/train_jax_ppo.py`

### 3.1 CheetahRun Run

```bash
cd /workspace/wiki/learn_mujoco_playground/repo

PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 learning/train_jax_ppo.py \
  --env_name=CheetahRun \
  --num_timesteps=30000000 \
  --episode_length=1000 \
  2>&1 | tee /workspace/runs/brax_ppo_cheetahrun.log
```

### 3.2 CartpoleBalance Quick Test

```bash
cd /workspace/wiki/learn_mujoco_playground/repo

PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 learning/train_jax_ppo.py \
  --env_name=CartpoleBalance \
  --num_timesteps=5000000 \
  2>&1 | tee /workspace/runs/brax_ppo_cartpole.log
```

### 3.3 Key Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--env_name` | `LeapCubeReorient` | Use any name from `registry.ALL_ENVS` |
| `--num_timesteps` | 50M | Brax PPO default; reduce for quick tests |
| `--learning_rate` | 5e-4 | Brax default |
| `--discounting` | 0.97 | Slightly lower than 0.99 — tunes returns faster |
| `--entropy_cost` | 0.005 | Encourages exploration |
| `--batch_size` | 256 | Brax minibatch size |
| `--episode_length` | 1000 | Match env's episode length |
| `--log_training_metrics` | true | TensorBoard logging |

### 3.4 RSL-RL (Locomotion)

For high-DOF locomotion tasks (G1, Go1, Spot, H1):

```bash
cd /workspace/wiki/learn_mujoco_playground/repo

PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 learning/train_rsl_rl.py \
  --env_name=HopperHop \
  2>&1 | tee /workspace/runs/rslrl_hopper.log
```

---

## 4. Comparison Experiment (All Three)

Run all three sequentially and compare logs:

```bash
#!/usr/bin/env bash
set -e

REPO=/workspace/wiki/learn_mujoco_playground/repo

# --- Variant A: CleanRL (background) ---
MUJOCO_GL=egl python3 /workspace/cleanrl/cleanrl/ppo_continuous_action_dmc.py \
  --env-id "dm_control/cheetah-run-v0" --total-timesteps 3000000 \
  --num-envs 4 --num-steps 512 --exp-name ppo_cleanrl_dmc \
  2>&1 | tee /workspace/runs/compare_cleanrl.log

# --- Variant B: Custom JAX ---
PYTHONPATH=$REPO python3 /workspace/run_ppo_continuous_mjx.py \
  --env-id CheetahRun --total-timesteps 30000000 --num-envs 512 \
  --num-steps 10 --learning-rate 2e-4 --update-epochs 8 --no-anneal-lr \
  --early-stop-patience 30 --checkpoint-dir /workspace/runs/checkpoints \
  --exp-name ppo_jax_compare \
  2>&1 | tee /workspace/runs/compare_jax.log

# --- Variant C: Brax PPO ---
cd $REPO
PYTHONPATH=$REPO python3 learning/train_jax_ppo.py \
  --env_name=CheetahRun --num_timesteps=30000000 --episode_length=1000 \
  2>&1 | tee /workspace/runs/compare_brax.log

echo "All done. View results:"
echo "  tensorboard --logdir /workspace/runs/"
```

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MUJOCO_GL` error / rendering crash | EGL not set | Prefix command with `MUJOCO_GL=egl` |
| `ModuleNotFoundError: mujoco_playground` | PYTHONPATH missing | Set `PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo` |
| OOM on GPU | `num_envs` too large | Reduce to 256 or set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` |
| Policy collapses after peak | LR annealing | Add `--no-anneal-lr` |
| `--anneal-lr False` has no effect | Tyro boolean syntax | Use `--no-anneal-lr` (not `--anneal-lr False`) |
| Brax `XLA_FLAGS` warning | Triton flag missing | Script sets it automatically; safe to ignore |
| Very slow training (JAX) | XLA recompilation | Keep `num_envs` constant across runs in same process |
