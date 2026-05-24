# TD-MPC-Glass Environment Setup

Use the repo-root requirement file that matches the VastAI GPU:

- `requirements-rtx3090.txt`: RTX 3090 / Ampere. Uses JAX CUDA 12 wheels.
- `requirements-rtx50series.txt`: RTX 50-series / Blackwell, including RTX 5060 Ti. Uses JAX CUDA 13 wheels.

## Install

```bash
cd /workspace/helios-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-rtx3090.txt
python -m pip install -e .
```

For RTX 5060 Ti / 50-series:

```bash
cd /workspace/helios-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-rtx50series.txt
python -m pip install -e .
```

The benchmark imports `mujoco_playground` from:

```text
/workspace/wiki/learn_mujoco_playground/repo
```

Clone or copy that repo before launching TD-MPC-Glass runs.

## Verify

```bash
python - <<'PY'
import jax, mujoco, warp
print("jax", jax.__version__)
print("devices", jax.devices())
print("mujoco", mujoco.__version__)
print("warp", warp.__version__)
PY
```

Expected: `jax.devices()` should include a CUDA GPU, not only CPU.

## Runtime Env

RTX 3090 / 24 GB:

```bash
export PYTHONPATH=/workspace/helios-rl/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
```

RTX 5060 Ti / 50-series:

```bash
export PYTHONPATH=/workspace/helios-rl/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
```

Increase the 50-series memory fraction only after a smoke run passes. For 8 GB 5060 Ti instances, prefer smaller sweeps or one seed at a time.

## Smoke Run

```bash
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks FishSwim \
  --total_steps 250000 \
  --seed 1 \
  --no_plot
```

Output should appear under:

```text
/workspace/helios-rl/exp/tdmpc_glass/<Task>/seed_<seed>.csv
/workspace/helios-rl/exp/tdmpc_glass/<Task>/seed_<seed>/checkpoints/
```

## Notes

JAX's official install guide recommends CUDA installed from pip wheels and recommends CUDA 13 wheels for current NVIDIA installs. It also states CUDA 13 needs driver `>= 580`; CUDA 12 needs driver `>= 525`. Use the 3090 file on older images, and the 50-series file on modern Blackwell-compatible images.

