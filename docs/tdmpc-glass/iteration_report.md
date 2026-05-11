# TD-MPC-Glass Iteration Report

Date: 2026-05-11

Task:

```text
HopperHop
```

Baseline:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-v24.csv
```

Experimental implementation:

```text
/workspace/helios-rl/src/helios/algorithms/tdmpc_glass.py
```

## Baseline Reference

v24 matched checkpoint rows:

```text
250112,2.2,pi,42
250112,0.0,mppi,42
500224,0.1,pi,42
500224,0.0,mppi,42
```

v24 later performance:

```text
3000064,331.3,pi,42
3000064,356.6,mppi,42
4000000,338.5,pi,42
4000000,354.0,mppi,42
```

The current iteration target was to beat v24 at matched early checkpoints before
continuing longer runs.

## Iteration 0: Proposal

Created:

```text
/workspace/helios-rl/docs/tdmpc2/tdmpc_glass_proposal.md
```

Initial design choices:

- Separate module, not invasive edits to `tdmpc2.py`.
- Prototype transition matrix instead of batch-state `B^2` graph.
- Low-frequency Glass auxiliary loss.
- Eval-time matrix dumps, not full stdout matrix printing.

## Iteration 1: First Implementation

Added:

- `src/helios/algorithms/tdmpc_glass.py`
- `--algos tdmpc-glass` support in `scripts/run_benchmark.py`
- MPPI eval output for TD-MPC2/TD-MPC-Glass HopperHop runs
- eval matrix dumps under `exp/benchmark/glass_diag/...`

First 250k run:

```text
250112,33.0,pi,42
250112,26.5,mppi,42
```

Result:

- Beat v24 at 250k.
- Diagnostics were not healthy:

```text
glass se=-0.4997 ent=2.079 active=8 max_mass=0.125 cut=0.000
```

Matrix inspection:

```text
P (32, 32) min=0.0 max=1.0 mean=0.0009765625
P row sums min/max: 0.0 / 1.0
```

Problem:

- Dead prototype rows created zero-volume graph nodes.
- Structural entropy became negative.
- `cut=0.000` was not useful.

## Iteration 2: Fixed Prototype Graph

Patch:

- Reduced `num_prototypes` from 32 to 16.
- Initialized prototypes through SimNorm-shaped random latents.
- Added row smoothing before row normalization.
- Added prototype usage balance.

Small CPU check:

```text
P shape: (16, 16)
P row sum min: 0.9999976
structural entropy: 3.2131
prototype balance: 0.1435
```

250k fixed-graph run:

```text
250112,138.1,pi,42
250112,98.0,mppi,42
```

Diagnostics:

```text
glass se=3.6250 ent=2.079 active=8 max_mass=0.125 cut=0.859
```

Matrix inspection:

```text
P (16, 16) min=0.06174 max=0.06345 mean=0.06250
A (16, 16) min=0.06182 max=0.06345 mean=0.06250
S (16, 8)  min=0.12171 max=0.12774 mean=0.12500
P row sums min/max: 0.99999994 / 1.00000012
S argmax counts: [2, 2, 3, 1, 1, 3, 2, 2]
```

Preserved artifact:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass-250k-fixed.csv
```

## Iteration 3: 500k Matched Run

Command:

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

Runtime:

```text
JIT compiled in 142.4s
TD-MPC-Glass HopperHop done in 2888s
All runs completed in 51.0 min
```

Results:

```text
250112,10.2,pi,42
250112,17.3,mppi,42
500224,186.5,pi,42
500224,182.0,mppi,42
```

Comparison against v24:

```text
step 250112
  pi:   glass=10.2   v24=2.2   delta=+8.0
  mppi: glass=17.3   v24=0.0   delta=+17.3

step 500224
  pi:   glass=186.5  v24=0.1   delta=+186.4
  mppi: glass=182.0  v24=0.0   delta=+182.0
```

Diagnostics at 500k:

```text
glass se=3.6250 ent=2.079 active=8 max_mass=0.125 cut=0.859
```

Primary output:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass.csv
```

## Conclusion

TD-MPC-Glass currently beats v24 at the matched 250k and 500k checkpoints on
HopperHop.

## Iteration 4: 1M Full-Speed Run

Command:

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

Runtime:

```text
JIT compiled in 141.5s
TD-MPC-Glass HopperHop done in 2357s
All runs completed in 42.1 min
```

Preserved output:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass-1m-fullspeed.csv
```

Results:

```text
250112,167.7,pi,42
250112,129.6,mppi,42
500224,331.1,pi,42
500224,336.5,mppi,42
750080,401.3,pi,42
750080,426.3,mppi,42
1000192,415.4,pi,42
1000192,411.8,mppi,42
```

Comparison against v24:

```text
step 250112
  pi:   glass=167.7  v24=2.2    delta=+165.5
  mppi: glass=129.6  v24=0.0    delta=+129.6

step 500224
  pi:   glass=331.1  v24=0.1    delta=+331.0
  mppi: glass=336.5  v24=0.0    delta=+336.5

step 750080
  pi:   glass=401.3  v24=309.3  delta=+92.0
  mppi: glass=426.3  v24=302.0  delta=+124.3

step 1000192
  pi:   glass=415.4  v24=6.7    delta=+408.7
  mppi: glass=411.8  v24=42.3   delta=+369.5
```

Diagnostics remained stable at eval:

```text
glass se=3.6250 ent=2.079 active=8 max_mass=0.125 cut=0.859
```

This run beat v24's later best MPPI before 1M:

```text
v24 best observed MPPI: 356.6 at 3,000,064 steps
TD-MPC-Glass MPPI:      426.3 at   750,080 steps
```

## Current Conclusion

TD-MPC-Glass beats v24 at every matched checkpoint through 4M in the current
best run. The best observed TD-MPC-Glass checkpoint is:

```text
3,500,032: pi=553.5, MPPI=565.4
```

v24's best observed checkpoint was:

```text
3,000,064: pi=331.3, MPPI=356.6
```

The remaining engineering gap is exact long-run resume. Model-only resume works,
but exact resume should use the new `--save_full_state` path.

## Iteration 5: 4M Run With Model-Only Resume

Initial 4M command:

```bash
PYTHONPATH=/workspace/helios-rl/src \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 4000000 \
  --seed 42 \
  --no_plot
```

The run crashed after the 3M eval with a MuJoCo/Warp CUDA capture error:

```text
Warp CUDA error 901: operation failed due to a previous error during capture
```

Saved pre-crash result:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass-3m-interrupted.csv
```

Best pre-crash checkpoint:

```text
3,000,064: pi=453.8, MPPI=505.2
```

Resumed from:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/best_mppi.pkl
```

Because the checkpoint was model-only, replay/env state restarted fresh. This
caused a temporary performance drop:

```text
3,250,176: pi=309.8, MPPI=352.0
```

The model recovered as replay refilled:

```text
3,500,032: pi=553.5, MPPI=565.4
3,750,144: pi=543.2, MPPI=561.4
4,000,000: pi=537.4, MPPI=548.2
```

Preserved 4M output:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-tdmpc-glass-4m-resumed.csv
```

Best checkpoint after the 4M run:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/best_mppi.pkl
```

Metadata:

```text
env_steps: 3,500,032
pi:        553.5
MPPI:      565.4
```

Final checkpoint:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/final.pkl
```

Metadata:

```text
env_steps: 4,000,000
best_mppi: 565.4
```

## Iteration 6: Exact Resume Support

Implemented after diagnosing the model-only resume drop:

```text
--resume_checkpoint
--save_full_state
```

Full-state checkpoints include:

- model params
- target params
- optimizer state
- RunningScale
- Glass step
- JAX PRNG key
- replay buffer
- vectorized environment state
- current observation batch
- NumPy generator state
- global NumPy random state

This is now the recommended mode for long 4M+ runs if disk space allows.
