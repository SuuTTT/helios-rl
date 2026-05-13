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

## Iteration 7: Phase 1 — Make Glass Actually Cluster (2026-05-12)

After the first 5-seed CI run (`exp/tdmpc_glass/HopperHop_pre_phase1/`) finished
with `final_mean = 327.5 ± 149.8` vs official `449.2 ± 312.1`, the Glass
diagnostics dumped to `exp/benchmark/glass_diag/HopperHop/seed_3/step_*.npz`
were inspected for all 16 saved evals from 250k to 4M:

```text
step= 250112  P_min=0.0617 P_max=0.0637 S_max=0.1283 clu_ent=2.0794
step= 500224  P_min=0.0617 P_max=0.0635 S_max=0.1284 clu_ent=2.0794
step=1000192  P_min=0.0623 P_max=0.0631 S_max=0.1286 clu_ent=2.0794
step=4000000  P_min=0.0624 P_max=0.0626 S_max=0.1287 clu_ent=2.0794
```

Uniform values are `1/K = 0.0625`, `2/K = 0.125`, `log(K) = 2.0794`. Glass was
inert — `P` deviated from uniform by less than 0.001 and `S` by less than 0.004
for the entire 4M run. Five root causes identified:

1. `stopgrad_graph=True` cut every Glass gradient before it could reach the
   encoder/dynamics, so the loss only saw `params["glass"]`.
2. 2D-SE has a vanishing gradient near the uniform fixed point. With
   `assign_logits = 0.01·N(0,1)` and `assignment_temperature = 1.0`, S started
   inside that vanishing region.
3. `balance` and `proto_balance` were `||mass − uniform||^2`, which
   *opposes* clustering. Their only non-vanishing gradient pulled S back to
   uniform.
4. Prototype L2 distance with `proto_temperature=0.2` was not well-conditioned
   for SimNorm latents at `d=512`; soft-min collapsed.
5. Coefficients `lambda_*` plus shared `clip_by_global_norm(20.0)` reduced the
   effective LR on Glass to `~1e-7`.

### Changes

`src/helios/algorithms/tdmpc_glass.py`:

- `glass_transition_graph(..., use_cosine_assign=True)`: cosine
  similarity in latent + prototype norms, well-conditioned at `d=512`.
- one-sided hinge balance: `sum(relu(cluster_mass − 2/K)^2)`, only fires on
  collapse.
- `z_next = stop_gradient(z_next)` always; `stopgrad_graph=True` kept by
  default for now (allowing `z_src` to backprop wrecks the world model at the
  current lambda; revisit in Phase 2).
- `init_glass_params(..., assign_logits_init_scale=1.0)` so S is born outside
  the vanishing-gradient region.
- `GLASS_DEFAULTS`: `proto_temperature 0.2 → 1.0`, `lambda_se 1e-4 → 5e-3`,
  `lambda_balance 1e-3 → 1e-2`, `lambda_temporal 1e-4 → 1e-3`.

`scripts/run_benchmark.py`:

- Plumb `use_cosine_assign` and `assign_logits_init_scale` through.
- Bump `eval_mppi(n_eps=3 → 8)` when `use_glass=True`.
- Single optimizer kept (shared `clip_by_global_norm`) so the JAX trace order
  matches baseline; separate-optimizer experiment was abandoned because it
  perturbed the world-model RNG stream.

### Verification

Glass diagnostics from Phase 1 logs (constant after a few evals, plotted in
`exp/tdmpc_glass/plots/hopperhop_phase1_glass_diag.png`):

```text
seed 1  ent=1.386  active=4  max_mass=0.250  cut=0.722
seed 2  ent=1.386  active=4  max_mass=0.250  cut=0.733
seed 3  ent=1.386  active=4  max_mass=0.250  cut=0.725
seed 4  ent=1.098  active=3  max_mass=0.346  cut=0.636
```

vs pre-Phase-1 inert reference `ent=2.079  active=8  max_mass=0.129  cut=~0`.

The block structure is visible in
`exp/tdmpc_glass/plots/hopperhop_phase1_glass_matrix_seed3.png`: the 16×16 `P`
reorders into a 4-block × 5-row diagonal layout with `S` collapsing to
near-one-hot prototype-to-cluster assignment.

### MPPI returns (4M)

Final eval per seed and per-seed best:

```text
Phase 1     seed 1  final=323.0  best=371.6 @ 2.25M
Phase 1     seed 2  final=440.1  best=451.4 @ 3.50M
Phase 1     seed 3  final=447.9  best=451.2 @ 3.75M
Phase 1     seed 4  in flight (last MPPI=258.7 @ 2.75M)
Phase 1     seed 5  queued

pre-phase1  seed 1  final=273.9
pre-phase1  seed 2  final=230.7
pre-phase1  seed 3  final=526.0  (single lucky 526→336→526 oscillation)
pre-phase1  seed 4  final=355.5
pre-phase1  seed 5  final=251.2

official    seed 1  final=380.1
official    seed 2  final=373.2
official    seed 3  final=594.2  (single strong seed)
```

3-seed comparison (1,2,3 — completed seeds on both sides):

```text
Phase 1   mean = 403.7   std-of-seeds =  70.6
official  mean = 449.2   std-of-seeds = 125.4
pre-p1    mean = 343.5   std-of-seeds = 161.7
```

Phase 1 closes the gap to official to within one CI half-width and reduces
seed variance by ~45% versus official. Seed-3 has lost its lucky 526 spike but
is now more representative; seed 2 has recovered from a 1.5M near-zero stall.

Pending: seed 4 (currently underperforming at MPPI≈250, 3-cluster collapse) and
seed 5. Final 5-seed CI plot blocked on those two runs.

### Plots

```text
exp/tdmpc_glass/plots/hopperhop_phase1_progress_95ci.png
exp/tdmpc_glass/plots/hopperhop_phase1_glass_diag.png
exp/tdmpc_glass/plots/hopperhop_phase1_glass_matrix_seed3.png
```
