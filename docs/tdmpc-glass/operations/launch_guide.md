# TD-MPC-Glass Launch Guide

This guide launches the current TD-MPC-Glass implementation from
`scripts/run_benchmark.py`.

## Environment

Run from the repository root:

```bash
cd /workspace/helios-rl
```

Use this `PYTHONPATH` so `helios` and `mujoco_playground` are importable:

```bash
export PYTHONPATH=/workspace/helios-rl/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
```

`run_benchmark.py` inserts the local mujoco playground repo path internally:

```text
/workspace/wiki/learn_mujoco_playground/repo
```

## Smoke Run

Use this to validate compile, training, MPPI eval, and Glass matrix dumps:

```bash
PYTHONPATH=/workspace/helios-rl/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 250000 \
  --seed 42 \
  --no_plot
```

Expected runtime on the current RTX 3090 setup:

- JIT compile: about 140-150s.
- Full 250k run: about 25-30 minutes.

## 500k Comparison Run

This is the current matched-checkpoint comparison against v24:

```bash
PYTHONPATH=/workspace/helios-rl/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 500000 \
  --seed 42 \
  --no_plot
```

Expected runtime on the current RTX 3090 setup:

- About 50-55 minutes.

## 1M Full-Speed Run

When the GPU is free, use a larger XLA memory fraction:

```bash
PYTHONPATH=/workspace/helios-rl/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 1000000 \
  --seed 42 \
  --no_plot
```

Observed runtime:

- JIT compile: `141.5s`
- Full run: `42.1 min`

Preserved output:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass-1m-fullspeed.csv
```

## Output Paths

Primary benchmark CSV:

```text
/workspace/helios-rl/exp/benchmark/tdmpc-glass_HopperHop.csv
```

Hopper-style comparison CSV with `pi` and `mppi` rows:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass.csv
```

Preserved best 250k fixed-graph run:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass-250k-fixed.csv
```

Glass transition matrix diagnostics:

```text
/workspace/helios-rl/exp/benchmark/glass_diag/HopperHop/seed_42/
```

Each eval checkpoint writes:

```text
step_<env_steps>.npz
```

The `.npz` contains:

- `P`: prototype transition matrix.
- `A`: symmetrized adjacency.
- `S`: prototype-to-cluster assignment probabilities.

## Checkpointing

Model-only checkpoints are written for TD-MPC-Glass HopperHop runs:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/
```

Files:

```text
best_mppi.pkl
latest_eval.pkl
final.pkl
```

To save exact-resume checkpoints including replay buffer, vectorized env state,
current observations, JAX key, and NumPy RNG states, add:

```bash
--save_full_state
```

This creates:

```text
best_mppi_full.pkl
latest_full.pkl
final_full.pkl
```

Full-state checkpoints are large because they include replay arrays.

Resume from any checkpoint:

```bash
PYTHONPATH=/workspace/helios-rl/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 4000000 \
  --seed 42 \
  --no_plot \
  --resume_checkpoint /workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/best_mppi.pkl
```

Exact resume should use a `*_full.pkl` checkpoint. Model-only checkpoints resume
weights and optimizer but restart replay/env state.

## Baseline For Comparison

v24 milestone CSV:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-v24.csv
```

Matched checkpoint rows from v24:

```text
250112,2.2,pi,42
250112,0.0,mppi,42
500224,0.1,pi,42
500224,0.0,mppi,42
```

## Notes

- HopperHop eval cadence is fixed at `250_000` env steps to match v24.
- TD-MPC-Glass currently runs through `scripts/run_benchmark.py`, not the older
  one-off `train_tdmpc_hopper_v24_scale_cap.py` script.
- Matrix dumps are intentionally saved as compressed artifacts instead of
  printed in full to stdout.
- For long 4M runs, prefer `--save_full_state` at least once after the model is
  stable if disk space allows.
