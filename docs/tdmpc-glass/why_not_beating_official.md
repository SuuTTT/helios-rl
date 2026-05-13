# Why TD-MPC-Glass Is Not Yet Beating Official TD-MPC2

Date: 2026-05-12

## Current Result

Five HopperHop TD-MPC-Glass seeds were compared against the official TD-MPC2
HopperHop CSV at:

```text
/workspace/tdmpc2/results/tdmpc2/hopper-hop.csv
```

Plot and summary:

```text
/workspace/helios-rl/exp/tdmpc_glass/plots/hopperhop_tdmpc_glass_vs_official_95ci.png
/workspace/helios-rl/exp/tdmpc_glass/plots/hopperhop_tdmpc_glass_vs_official_95ci_summary.csv
```

Final-return comparison:

| Series | Seeds | Final mean | 95% CI half-width | Best-per-seed mean | 95% CI half-width |
|---|---:|---:|---:|---:|---:|
| TD-MPC-Glass pi | 5 | 266.9 | 227.9 | 339.8 | 125.9 |
| TD-MPC-Glass MPPI | 5 | 327.5 | 149.8 | 354.0 | 128.5 |
| Official TD-MPC2 | 3 | 449.2 | 312.1 | 456.5 | 315.9 |

The current TD-MPC-Glass sweep does not clearly beat official TD-MPC2 under
95% CI. Seed 3 is strong, but seeds 1, 2, and 5 are weak.

## Seed-Level Evidence

| Seed | Pi final | Pi best | MPPI final | MPPI best |
|---:|---:|---:|---:|---:|
| 1 | 250.3 | 271.9 | 273.9 | 283.8 |
| 2 | 0.1 | 321.1 | 230.7 | 340.8 |
| 3 | 507.2 | 507.2 | 526.0 | 526.0 |
| 4 | 336.4 | 348.6 | 355.5 | 355.5 |
| 5 | 240.3 | 250.0 | 251.2 | 263.7 |

Key observations:

- Seed 3 is milestone-grade.
- Seed 4 roughly matches local v24 but does not beat official TD-MPC2.
- Seeds 1 and 5 plateau around 250-280 MPPI.
- Seed 2 has a severe final pi drop even though its best pi was 321.1.

## Likely Causes

1. Glass is currently almost inert.

   Matrix diagnostics show the assignment matrix `S` remains almost uniform:

   ```text
   S_max ~= 0.128
   cluster_mass ~= 0.125
   cluster_entropy ~= log(8)
   ```

   The transition graph `P` is also nearly uniform:

   ```text
   P_row_entropy ~= log(16)
   P_max ~= 0.0625
   ```

   This means the current cluster module is not discovering meaningful HopperHop
   transition regions. The performance gain in seed 3 is therefore not yet
   strong evidence of a reliable Glass mechanism.

2. Weak-seed plateau dominates the confidence interval.

   The mean is pulled down by three low seeds. To beat official TD-MPC2, the
   target should be not only a high best seed, but a weak-seed floor above about
   380-400 final MPPI.

3. MPPI eval is noisy.

   MPPI occasionally collapses while pi remains strong, for example seed 3 at
   3.5M:

   ```text
   pi=474.6, MPPI=336.8
   ```

   It later recovered to 526.0. This suggests evaluation variance or planner
   brittleness, not model collapse.

4. Exploration was not fully reproducible.

   The training loop used global `np.random` for exploration actions/noise while
   using a seeded generator for replay sampling. This made runs less reproducible
   than the seed labels implied. Fixed in `scripts/run_benchmark.py` by routing
   exploration through `rng_np`.

## Implemented Before Next Run

1. Seeded exploration.

   `scripts/run_benchmark.py` now uses `rng_np.uniform` and `rng_np.normal` for
   collection-time random actions/noise.

2. Glass override CLI.

   Added flags for fast ablations:

   ```text
   --glass_warmup_env_steps
   --glass_every_k_updates
   --glass_proto_temperature
   --glass_assignment_temperature
   --glass_lambda_se
   --glass_lambda_balance
   --glass_lambda_temporal
   --glass_stopgrad_graph
   --glass_num_prototypes
   --glass_num_clusters
   ```

## How To Beat Official

The next iteration should optimize for the weak-seed floor.

### Ablation A: Make clusters non-uniform

Goal: `S_max` and `P` should move away from uniform without collapse.

Run 1M first:

```bash
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 1000000 \
  --seed 1 \
  --no_plot \
  --glass_proto_temperature 0.05 \
  --glass_assignment_temperature 0.3 \
  --glass_lambda_balance 1e-4
```

Continue only if:

```text
MPPI >= 300 by 750k or 1M
S_max > 0.16
active_clusters >= 4
```

### Ablation B: Let temporal Glass shape latents

Current graph latents are stop-gradient. Try allowing the Glass temporal term to
shape encoder/dynamics lightly:

```bash
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 1000000 \
  --seed 1 \
  --no_plot \
  --glass_stopgrad_graph false \
  --glass_lambda_temporal 3e-5 \
  --glass_lambda_se 1e-5 \
  --glass_lambda_balance 1e-4
```

Stop if loss spikes or `active_clusters <= 1`.

### Ablation C: Delay Glass

If early Glass regularization hurts weak seeds, delay activation until model
rollouts are less noisy:

```bash
python3 scripts/run_benchmark.py \
  --algos tdmpc-glass \
  --tasks HopperHop \
  --total_steps 1000000 \
  --seed 1 \
  --no_plot \
  --glass_warmup_env_steps 250000
```

### Milestone Gate

Do not spend a full 5-seed 4M run until a 3-seed 1M probe satisfies:

```text
mean MPPI at 1M >= 350
min-seed MPPI at 1M >= 250
no seed has pi final near zero
Glass diagnostics are not uniform
```

Then run 5 seeds to 4M and compare final and best-per-seed CI.

