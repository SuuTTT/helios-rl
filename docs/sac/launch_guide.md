# SAC Launch Guide

**Environment:** Priority DMC Suite envs: BallInCup, CartpoleSwingupSparse, HopperStand, FingerSpin  
**Workspace root:** `/workspace`  
**PYTHONPATH:** `/workspace/wiki/learn_mujoco_playground/repo`

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

# Verify brax replay buffers accessible
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 -c "from brax.training import replay_buffers; print('OK')"
```

---

## 1. Variant A — Official Brax SAC (Reference Baseline)

**Script:** `helios-rl/scripts/run_sac_official.py`

### 1.1 Single Env, Single Seed

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/helios-rl/scripts/run_sac_official.py \
  --env_id HopperStand \
  --seed 1 \
  2>&1 | tee /workspace/runs/sac_HopperStand_s1.log
```

### 1.2 Priority Env Suite (all 4 envs, seeds 1–5)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/helios-rl/scripts/run_sac_official.py \
  --priority \
  --seeds 1 2 3 4 5 \
  2>&1 | tee /workspace/runs/sac_official_priority.log
```

### 1.3 Key Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--env_id` | `BallInCup` | MuJoCo Playground registry name |
| `--seed` | 1 | RNG seed |
| `--total_timesteps` | from config | Override (e.g. `5000000`) |
| `--priority` | off | Run all 4 priority envs sequentially |
| `--seeds` | `[1]` | Seeds to run (used with `--priority`) |

### 1.4 Expected Performance

| Env | Steps | sps | Wall time | Notes |
|-----|-------|-----|-----------|-------|
| HopperStand | 10M | ~3900–4300 | ~46 min | High seed variance |
| CartpoleSwingupSparse | 5M | ~7800 | ~11 min | Bimodal: solved or 0 |
| BallInCup | 5M | ~7800 | ~11 min | Bimodal: solved or 0 |
| FingerSpin | 10M | ~3900 | ~46 min | Not yet run |

---

## 2. Variant B — Custom GPU SAC

**Script:** `helios-rl/scripts/run_sac_custom.py`

### 2.1 Standard Run (10M steps, HopperStand)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/helios-rl/scripts/run_sac_custom.py \
  --env_id HopperStand \
  --seed 1 \
  --hidden 512 512 \
  --collect_steps 64 \
  2>&1 | tee /workspace/runs/sac_custom_HopperStand_s1.log
```

### 2.2 Quick Smoke Test (1M steps, ~5 min)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/helios-rl/scripts/run_sac_custom.py \
  --env_id HopperStand \
  --seed 1 \
  --total_timesteps 1000000 \
  --hidden 256 256 \
  --collect_steps 64 \
  2>&1 | tee /workspace/runs/sac_custom_smoke.log
```

### 2.3 Multi-Seed Run (beat-official attempt)

```bash
bash /workspace/helios-rl/scripts/run_sac_multiseed_beat.sh \
  2>&1 | tee /workspace/runs/sac_multiseed_beat.log
```

This runs seeds 2–5 with `512×2, g/step=8` sequentially (each ~48 min).

### 2.4 Key Flags

| Flag | Best Value | Notes |
|------|-----------|-------|
| `--env_id` | `HopperStand` | MuJoCo Playground registry name |
| `--seed` | 1 | RNG seed |
| `--total_timesteps` | 0 (from config) | Override with integer |
| `--hidden` | `512 512` | Network hidden layer sizes |
| `--collect_steps` | 64 | Env steps per lax.scan collect call |
| `--grad_updates_per_step` | 0 (from config = 8) | **Do not exceed 8** — 16 causes collapse |
| `--batch_size` | 0 (from config = 512) | Replay sample batch size |
| `--lr` | 0 (from config = 1e-3) | Adam learning rate |
| `--target-entropy` | None (auto = -0.5×a_dim) | Override for more exploration |
| `--csv_log` | auto | Path to CSV output |

### 2.5 Output Files

| File | Description |
|------|-------------|
| `/workspace/runs/sac_custom_{Env}_s{seed}.log` | Full training log |
| `/workspace/helios-rl/exp/sac/csv/sac_custom_{env}.csv` | Step/reward CSV |
| `/workspace/helios-rl/exp/sac/csv/sac_{env}.csv` | Official reference CSV |

### 2.6 Reading the Log

```
Custom SAC  env=HopperStand  seed=1
  total=10,000,000  gamma=0.99  lr=0.001
  envs=128  batch=512  g/step=8
  ...
  SAC JIT: 63.8s                        ← one-time compilation
  Warmup done: 8,192 steps              ← random-action buffer fill
  step=  1,007,616  reward=  42.430     ← eval point
    best=  42.430  α=0.0016  sps=2912   ← best, temperature, speed
    elapsed=346s
Done. best=652.996  time=2866.4s
```

---

## 3. Multiseed Reference Runner (Official)

**Script:** `helios-rl/scripts/run_sac_multiseed.py`

Runs official SAC on priority envs sequentially:

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/helios-rl/scripts/run_sac_multiseed.py \
  --env_id HopperStand \
  --seeds 1 2 3 \
  --total_timesteps 10000000 \
  2>&1 | tee /workspace/runs/sac_multiseed_official.log
```

---

## 4. Monitoring

```bash
# Watch latest eval points in real time
tail -f /workspace/runs/sac_custom_HopperStand_s1.log | grep "step="

# Check all CSV results
cat /workspace/helios-rl/exp/sac/csv/sac_hopperstand.csv
cat /workspace/helios-rl/exp/sac/csv/sac_custom_hopperstand.csv

# GPU utilization
nvidia-smi -l 1

# Process check
ps aux | grep run_sac | grep -v grep
```

---

## 5. Stopping Runs

```bash
# Kill all SAC processes
pkill -f "run_sac_custom.py|run_sac_official.py|run_sac_multiseed"
```

---

## 6. Available Environments

All MuJoCo Playground DMC Suite envs work. Priority 4:

| Env | Config steps | Solved threshold |
|-----|-------------|-----------------|
| `BallInCup` | 5M | ~950 (catching) |
| `CartpoleSwingupSparse` | 5M | ~800 (sparse balance) |
| `HopperStand` | 10M | ~800–900 |
| `FingerSpin` | 10M | ~500–700 |

Other available: `AcrobotSwingup`, `CartpoleBalance`, `CartpoleSwingup`, `CheetahRun`, `FingerTurnEasy`, `FingerTurnHard`, `FishSwim`, `HopperHop`, `HumanoidStand`, `HumanoidWalk`, `PendulumSwingup`, `ReacherEasy`, `ReacherHard`, `WalkerRun`, `WalkerStand`, `WalkerWalk`.
