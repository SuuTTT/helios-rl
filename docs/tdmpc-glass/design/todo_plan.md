# TD-MPC-Glass TODO Plan

## Immediate

1. Add checkpoint/resume. Completed for model-only checkpoints and implemented
   opt-in full-state checkpoints.

   Long comparisons to v24's 3M-4M plateau need checkpointing for:

   - `params`
   - `target params`
   - optimizer state
   - replay buffer
   - PRNG key
   - RunningScale
   - Glass step counter
   - env step count

   Implemented:

   ```text
   --resume_checkpoint
   --save_full_state
   ```

   Model-only checkpoint files:

   ```text
   best_mppi.pkl
   latest_eval.pkl
   final.pkl
   ```

   Full-state checkpoint files:

   ```text
   best_mppi_full.pkl
   latest_full.pkl
   final_full.pkl
   ```

2. Run to 1M. Completed on 2026-05-11.

   Target comparison rows from v24:

   ```text
   750080,309.3,pi,42
   750080,302.0,mppi,42
   1000192,6.7,pi,42
   1000192,42.3,mppi,42
   ```

   The 1M v24 checkpoint is weak, but 750k is strong. TD-MPC-Glass beat both.

3. Preserve every result file with a unique suffix.

   Current `run_benchmark.py` overwrites:

   ```text
   exp/tdmpc_dmc/hopper-hop-tdmpc-glass.csv
   ```

   Add timestamp or run tag support.

4. Log Glass aux fields to CSV.

   Current console diagnostics are useful but not enough for sweeps. Add columns
   or a sidecar CSV for:

   ```text
   glass_se
   glass_balance
   glass_proto_balance
   glass_temp
   glass_entropy
   glass_active_clusters
   glass_max_cluster_mass
   glass_transition_cut_mass
   ```

## Next Experiments

1. Run another 4M with `--save_full_state`.

   The completed 4M resumed run reached:

   ```text
   best MPPI: 565.4 @ 3,500,032
   final MPPI: 548.2 @ 4,000,000
   ```

   A new 4M run with full-state checkpoints will allow exact recovery from
   runtime crashes.

2. Add automatic run tags.

   Current active output path is overwritten:

   ```text
   hopper-hop-tdmpc-glass.csv
   ```

   Add a CLI tag to write:

   ```text
   hopper-hop-tdmpc-glass-<tag>.csv
   ```

3. Warmup sweep:

   ```text
   25k, 50k, 100k, 200k
   ```

   Current default is `100k`.

4. Coefficient sweep:

   ```text
   lambda_se:       1e-5, 1e-4, 3e-4
   lambda_balance:  1e-4, 1e-3, 3e-3
   lambda_temporal: 0, 1e-4, 3e-4
   ```

5. Prototype temperature sweep:

   ```text
   proto_temperature: 0.05, 0.1, 0.2, 0.5
   ```

6. Prototype count sweep:

   ```text
   num_prototypes: 8, 16, 32
   ```

7. Glass frequency sweep:

   ```text
   every_k_updates: 1, 4, 8
   ```

## Stability Gates

Do not continue long runs if any of these occur:

- consistency loss `c` grows materially above the v24 healthy range
- `P` row sums become zero or non-finite
- `glass_se` is non-finite
- `glass_max_cluster_mass > 0.8`
- `active_clusters <= 1`
- MPPI is consistently below policy return after 500k

## Comparison Targets

Completed milestone:

```text
750080:  tdmpc-glass pi=401.3, MPPI=426.3  vs v24 pi=309.3, MPPI=302.0
1000192: tdmpc-glass pi=415.4, MPPI=411.8  vs v24 pi=6.7,   MPPI=42.3
```

Completed longer milestone:

```text
3000064: tdmpc-glass pi=453.8, MPPI=505.2  vs v24 pi=331.3, MPPI=356.6
4000000: tdmpc-glass pi=537.4, MPPI=548.2  vs v24 pi=338.5, MPPI=354.0
```
