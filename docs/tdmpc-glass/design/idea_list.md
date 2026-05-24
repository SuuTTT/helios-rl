# TD-MPC-Glass Idea List

## Representation Ideas

1. Let temporal Glass loss update latents.

   Current default stops graph latents. A second stage could allow
   `lambda_temporal` to update encoder/dynamics with one-sided stop-gradient.

2. Use encoded next latents for diagnostics.

   Current graph uses predicted rollout transitions. Add a parallel diagnostic:

   ```text
   z_next = stopgrad(enc(o_{t+1}))
   ```

   Compare predicted-graph and observed-graph structural entropy.

3. EMA prototype tracking.

   Instead of learning prototypes only through gradient descent, update them
   toward batch latent centroids using EMA. This may produce more meaningful
   transition matrices than free prototypes.

4. Sharpen assignment temperature over training.

   Anneal:

   ```text
   assignment_temperature: 1.0 -> 0.3
   ```

   Keep a collapse guard through balance loss.

## Planner Ideas

1. Cluster-aware MPPI diagnostics.

   For each planned trajectory, log abstract cluster sequence and dwell time.
   This is diagnostic only and should not affect action selection initially.

2. Cluster transition bonus or penalty.

   Add a small planning-time term that penalizes high-cut transitions or rewards
   controlled bottleneck crossing. This should only be tested after the cluster
   graph becomes visibly meaningful.

3. Option-level rollout priors.

   Use clusters to bias MPPI samples toward action sequences that maintain useful
   abstract regions for a few steps.

## Objective Ideas

1. Directed transition objective.

   Glass-SE currently uses symmetrized `A`. Try directed flow objectives on `P`
   once the undirected baseline is stable.

2. Modularity comparison.

   Add a modularity auxiliary loss on the same prototype graph as an ablation.

3. Map-equation comparison.

   Use the directed transition matrix for a flow-style map-equation objective.

4. Reward/value-aware features.

   Move from free assignment logits to a small assignment network or Glass-GNN
   using:

   ```text
   prototype latent
   predicted reward statistics
   value estimates
   transition in/out degree
   ```

## Engineering Ideas

1. Add a dedicated `train_tdmpc_glass_hopper.py`.

   The generic benchmark runner is useful, but a dedicated script can expose
   Glass-specific sweeps and checkpoint/resume cleanly.

2. Add run tags.

   Avoid overwriting:

   ```text
   hopper-hop-tdmpc-glass.csv
   ```

   Use:

   ```text
   hopper-hop-tdmpc-glass_<tag>.csv
   ```

3. Add matrix visualization script.

   Read `.npz` files and write heatmaps for:

   ```text
   P
   A
   S
   ```

4. Add automatic v24 comparison report.

   Given a run CSV and `hopper-hop-v24.csv`, emit a Markdown table with nearest
   checkpoint deltas.

## Experiment Ideas

1. Multi-seed early checkpoint.

   Run seeds:

   ```text
   1, 2, 3, 42
   ```

   at 500k to check whether the early improvement is robust or seed-specific.

2. Task transfer.

   Test:

   ```text
   HopperStand
   CheetahRun
   CartpoleBalance
   ```

   Use policy return first, then add MPPI comparison where relevant.

3. Longer HopperHop run.

   Run 1M, then 3M if 750k/1M remains competitive.

