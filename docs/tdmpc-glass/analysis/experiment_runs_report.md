# TD-MPC-Glass Existing Experiment Runs

Date: 2026-05-11

This report summarizes TD-MPC-Glass HopperHop CSV artifacts under:

```text
/workspace/helios-rl/exp/tdmpc_dmc
```

Baseline reference:

```text
/workspace/helios-rl/exp/tdmpc_dmc/hopper-hop-v24.csv
```

v24 best observed MPPI:

```text
356.6 at 3,000,064 env steps
```

## Run Summary

| File | Rows | Best MPPI | Final PI | Final MPPI | Notes |
|---|---:|---:|---:|---:|---|
| `hopper-hop-tdmpc-glass-250k-fixed.csv` | 2 | 98.0 @ 250,112 | 138.1 @ 250,112 | 98.0 @ 250,112 | First fixed-graph 250k run. |
| `hopper-hop-tdmpc-glass-500k-fixed.csv` | 4 | 182.0 @ 500,224 | 186.5 @ 500,224 | 182.0 @ 500,224 | 500k fixed-graph run. |
| `hopper-hop-tdmpc-glass-1m-fullspeed.csv` | 8 | 426.3 @ 750,080 | 415.4 @ 1,000,192 | 411.8 @ 1,000,192 | Full-speed 1M run. |
| `hopper-hop-tdmpc-glass-1m-before-4m.csv` | 8 | 426.3 @ 750,080 | 415.4 @ 1,000,192 | 411.8 @ 1,000,192 | Preserved before launching 4M. |
| `hopper-hop-tdmpc-glass-3m-interrupted.csv` | 24 | 505.2 @ 3,000,064 | 453.8 @ 3,000,064 | 505.2 @ 3,000,064 | Original 4M run interrupted by Warp CUDA error after 3M eval. |
| `hopper-hop-tdmpc-glass-4m-resumed.csv` | 32 | 565.4 @ 3,500,032 | 537.4 @ 4,000,000 | 548.2 @ 4,000,000 | Resumed from 3M best model checkpoint; fresh replay initially, then recovered. |
| `hopper-hop-tdmpc-glass.csv` | 32 | 565.4 @ 3,500,032 | 537.4 @ 4,000,000 | 548.2 @ 4,000,000 | Current active output path, same content as 4M resumed. |

## 4M Current Best

Best checkpoint:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/best_mppi.pkl
```

Checkpoint metadata:

```text
env_steps:   3,500,032
pi_reward:   553.5
mppi_reward: 565.4
```

Latest eval checkpoint:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/latest_eval.pkl
```

Metadata:

```text
env_steps:   4,000,000
pi_reward:   537.4
mppi_reward: 548.2
```

Final model-only checkpoint:

```text
/workspace/helios-rl/exp/tdmpc_dmc/checkpoints/tdmpc-glass/HopperHop/seed_42/final.pkl
```

Metadata:

```text
env_steps: 4,000,000
best_mppi: 565.4
```

## Crash And Resume

The original 4M run crashed after the 3M eval with a MuJoCo/Warp CUDA capture
error:

```text
Warp CUDA error 901: operation failed due to a previous error during capture
FFI callback error ... mjwarp.step(...)
```

The best model checkpoint had already been saved:

```text
3,000,064: pi=453.8, MPPI=505.2
```

The resumed run loaded model, target params, optimizer state, RunningScale,
Glass step, and PRNG key. That resume did not include replay/env state because
the earlier checkpoint format did not save them yet. The resumed run temporarily
dropped at 3.25M:

```text
3,250,176: pi=309.8, MPPI=352.0
```

It recovered after the replay buffer refilled:

```text
3,500,032: pi=553.5, MPPI=565.4
3,750,144: pi=543.2, MPPI=561.4
4,000,000: pi=537.4, MPPI=548.2
```

## Checkpoint Format Update

After this run, `scripts/run_benchmark.py` was updated with `--save_full_state`.
When enabled, checkpoints additionally store:

- replay buffer arrays and metadata
- vectorized environment state
- current observation batch
- JAX PRNG key
- NumPy `Generator` state
- global `np.random` state

This is the correct format for exact off-policy resume. It is opt-in because
the replay buffer can make each checkpoint large.

