# TD-MPC-Glass Proposal

This note proposes a fast-iteration variant of `src/helios/algorithms/tdmpc2.py`
that adds Glass-JAX clustering losses to TD-MPC2 latent training. The working
name is `tdmpc-glass`.

The goal is not to replace TD-MPC2 planning. The first version should add a
small structural regularizer that encourages the latent world model to discover
reusable transition regions, while preserving TD-MPC2's current speed and
stability profile.

## Source Context

Glass reference:

- `/workspace/glass-jax/docs/rl/tdmpc2_transition_matrix_glass_se.md`

TD-MPC2 implementation:

- `/workspace/helios-rl/src/helios/algorithms/tdmpc2.py`

Current TD-MPC2 has one main integration point:

- `make_update_fn(...).loss_fn`: encodes replay sequences, rolls out dynamics
  latents, computes consistency, reward, value, and policy losses, then returns
  `total, aux`.

The Glass loss should be inserted in this loss function after:

```text
z_all = enc(obs sequence)
zs    = dynamics rollout from z0 and replay actions
```

and before:

```text
total = TD-MPC2 weighted loss
```

## Core Idea

Construct a small transition graph from TD-MPC2 latent rollouts:

```text
z_t       = latent state
a_t       = replay action
z_pred+   = dyn(z_t, a_t)
z_enc+    = stopgrad(enc(o_{t+1}))
```

Then cluster these states by transition structure using Glass-SE:

```text
L_total = L_tdmpc2
        + lambda_se   * H2(A, S_logits)
        + lambda_bal  * L_balance(S)
        + lambda_temp * L_temporal_consistency(S_t, S_t+1)
```

The intended effect is to bias latents toward abstract regions that have
coherent incoming and outgoing transition structure, instead of only matching
one-step latent MSE and reward/value targets.

## First Iteration Design

Use the prototype transition matrix path first. Avoid batch-state `B^2`
transition matrices in the initial implementation.

Recommended config:

```yaml
glass:
  enabled: false
  warmup_env_steps: 100000
  every_k_updates: 4
  num_prototypes: 32
  num_clusters: 8
  proto_temperature: 0.2
  assignment_temperature: 1.0
  lambda_se: 1.0e-4
  lambda_balance: 1.0e-3
  lambda_temporal: 1.0e-4
  stopgrad_graph: true
  diag_dump_matrices: true
```

Keep `enabled: false` by default until the branch has a smoke-tested training
path. For experiment scripts, enable it explicitly as `tdmpc-glass`.

### Why Prototypes First

The current default batch is `B=256`, sequence length `T=H+1`, and fused
`K_UPDATE=64`. A dense pairwise graph over all rollout states can make each
update noticeably heavier and may force recompilation for shape changes.

Prototype construction keeps the graph size fixed:

```text
N = B * (T - 1) transition samples
K = num_prototypes

c_t      = softmax(-||z_t - p_k||^2 / tau_p)
c_t_next = softmax(-||z_next - p_k||^2 / tau_p)
P        = row_normalize(sum_n outer(c_t[n], c_t_next[n]))
A        = 0.5 * (P + P.T)
```

This makes Glass-SE operate on a `K x K` graph, for example `32 x 32`, rather
than a `N x N` graph.

## Cluster Parameters

Add one small module or parameter group:

```text
params["glass"]["prototypes"]    -> (num_prototypes, latent_dim)
params["glass"]["assign_logits"] -> (num_prototypes, num_clusters)
```

For the first version, prefer free assignment logits over a Glass-GNN. This is
cheaper, easier to debug, and enough to test whether structural entropy helps
the world model.

Possible initialization:

- prototypes: uniform or normal around the SimNorm latent scale
- assignment logits: small normal noise

Because TD-MPC2 latents are SimNorm-bounded, prototype distance scales should be
stable enough for a simple fixed `proto_temperature` in the first pass.

## Loss Details

### Structural Entropy

Call Glass-JAX once the prototype adjacency is built:

```python
from glass.objectives.structural_entropy import two_dimensional_structural_entropy

se_loss = two_dimensional_structural_entropy(A, assign_logits)
```

Use `A = (P + P.T) / 2` in the first version because the referenced Glass-SE
path is undirected. Directed flow/map-equation variants can come later.

### Balance Loss

Prevent collapse to one cluster:

```text
S = softmax(assign_logits / assignment_temperature)
cluster_mass = mean_k S[k, :]
target = uniform(num_clusters)
L_balance = sum((cluster_mass - target)^2)
```

This term is cheap and should be on from the start whenever `lambda_se > 0`.

### Temporal Consistency Loss

Encourage adjacent latent states to stay in compatible abstract regions:

```text
s_t      = c_t @ S
s_t_next = c_t_next @ S
L_temp   = mean(||s_t - stopgrad(s_t_next)||^2)
```

Keep this coefficient smaller than balance. We do not want to suppress real
bottleneck transitions or force every transition to remain in the same cluster.

## Gradient Boundaries

Default first iteration:

- Build `P` from `stop_gradient(z_t)` and `stop_gradient(z_next)`.
- Let Glass-SE update only `params["glass"]`.
- Do not allow Glass-SE to reshape the encoder or dynamics until diagnostics
  show stable cluster behavior.

Second iteration:

- Allow `lambda_temp` to affect encoder/dynamics through `z_t`.
- Keep one side of each transition stopped, matching the Glass note's safeguard.
- Only then consider a small `lambda_se` path into latents if it does not
  increase consistency loss or degrade MPPI.

This staged boundary is conservative. It tests whether the clustering signal is
useful before letting it change the world model representation.

## Fast Feedback Loop

The implementation should keep iteration time low:

1. Gate Glass with `glass.enabled` and `env_steps >= glass.warmup_env_steps`.
2. Compute Glass only every `glass.every_k_updates` inside the fused update scan.
3. Use prototype graphs with fixed `num_prototypes <= 32` for the first pass.
4. Log scalar diagnostics only; do not log full matrices during hot training.
5. Dump small prototype matrices only at eval checkpoints, not every update.
6. Avoid planner changes in v1. MPPI should continue to use the same reward,
   dynamics, critic, and policy networks.
7. Add a short smoke script or benchmark mode that runs for `50k-100k` env steps
   before launching long Hopper runs.

Expected overhead target for v1:

- Less than 5-10% wall-clock overhead versus `tdmpc2` when
  `glass.every_k_updates >= 4`.

If overhead exceeds that, increase `every_k_updates`, lower prototypes to `16`,
or compute Glass only in `single_step` experiments before reintroducing fused
`multi_step`.

## Transition Matrix Logging

Printing transition matrices at eval time is useful for diagnosis and
visualization, but it should be bounded. A `32 x 32` prototype matrix is cheap
to transfer occasionally; printing it to stdout every eval is still noisy and
can make logs hard to compare across seeds. A batch-state matrix over all latent
states should not be printed in the training loop.

Recommended policy:

- Print scalar summaries at every eval:
  `glass_se`, `glass_entropy`, `glass_active_clusters`,
  `glass_max_cluster_mass`, and `glass_transition_cut_mass`.
- Save compact matrices as compressed artifacts:
  `P`, `A`, and `S` in `.npz` files under an experiment diagnostics directory.
- Use prototype matrices only for routine eval diagnostics.
- Smooth prototype transition rows and track prototype usage balance so dead
  prototypes do not create zero-volume graph nodes or misleading entropy values.
- Print the full matrix to console only in manual debugging runs with
  `num_prototypes <= 16`.

This gives visualization data without blocking the hot feedback loop on large
device-to-host transfers or overwhelming the terminal.

Initial implementation target:

- `scripts/run_benchmark.py --algos tdmpc-glass`
- Eval-time diagnostics saved under:
  `exp/benchmark/glass_diag/<task>/seed_<seed>/step_<env_steps>.npz`

## Diagnostics To Add

Add these fields to `aux` when Glass is enabled:

```text
glass_se
glass_balance
glass_temp
glass_total
glass_entropy
glass_active_clusters
glass_max_cluster_mass
glass_transition_cut_mass
```

Definitions:

- `glass_active_clusters`: number of clusters with mean assignment mass above a
  small threshold, for example `0.05 / num_clusters`.
- `glass_max_cluster_mass`: collapse detector.
- `glass_transition_cut_mass`: transition probability whose source and target
  most-likely clusters differ.

Use these alongside existing TD-MPC2 metrics:

```text
c, r, v, p, scale, pi return, MPPI return
```

Regression rule for early experiments: if `c` rises materially or MPPI falls
below the TD-MPC2 baseline for the same seed and wall-clock budget, reduce
`lambda_se` or keep Glass gradients isolated to cluster parameters.

## Development Plan

### Phase 0: Documentation and Interface

- Add this proposal.
- Decide whether `tdmpc-glass` is a new module or a config-gated variant.
- Preferred first implementation: new module
  `src/helios/algorithms/tdmpc_glass.py` that imports/reuses TD-MPC2 components
  where practical. This protects the v24 baseline from experimental churn.

### Phase 1: Non-Invasive Cluster Logging

- Add prototypes and assignment logits.
- Build `P` and `A` from stopped latents.
- Compute Glass losses and diagnostics.
- Set Glass coefficients to zero or update only the Glass parameters.
- Confirm compile succeeds and diagnostics are non-degenerate.

### Phase 2: Auxiliary Training

- Turn on `lambda_se`, `lambda_balance`, and `lambda_temporal`.
- Keep Glass gradients out of encoder/dynamics initially.
- Run short feedback jobs:
  - CartpoleBalance: `50k-100k`
  - HopperStand or HopperHop: `100k-300k`
- Compare loss curves and wall-clock overhead to TD-MPC2.

### Phase 3: Representation Coupling

- Allow temporal consistency to update encoder/dynamics with one-sided
  stop-gradient.
- Sweep:
  - `lambda_se`: `1e-5`, `1e-4`, `1e-3`
  - `num_prototypes`: `16`, `32`, `64`
  - `num_clusters`: `4`, `8`, `16`
  - `every_k_updates`: `1`, `4`, `8`

### Phase 4: Planner-Aware Use

Only after the auxiliary loss improves or preserves baseline returns:

- Expose cluster IDs to logging and trajectory analysis.
- Test option-level dwell time and transition bottleneck statistics.
- Consider cluster-aware MPPI priors, but keep that out of v1.

## Open Questions

- Does Glass-JAX expose a JAX-pure objective with no Python-side shape-dependent
  work inside `jit`?
- Should prototypes be learned solely through Glass losses, or should they also
  track latent centroids with an EMA-style update outside gradient descent?
- Does structural entropy prefer more clusters than are useful for short-horizon
  MPPI tasks?
- Should `tdmpc-glass` use predicted next latents `dyn(z, a)` or encoded next
  latents `enc(o_next)` for `c_t_next` in the first experiment?

Recommendation for the first run: use `dyn(z, a)` for source-to-target
transition structure and log a parallel diagnostic using `enc(o_next)`. This
aligns the graph with the model MPPI actually plans through while still checking
against observed latent transitions.
