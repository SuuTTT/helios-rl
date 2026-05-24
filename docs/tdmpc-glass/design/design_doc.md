# TD-MPC-Glass Design

TD-MPC-Glass is an experimental TD-MPC2 variant that adds Glass-style structural
clustering over latent transition graphs.

Implementation:

```text
/workspace/helios-rl/src/helios/algorithms/tdmpc_glass.py
```

Launch integration:

```text
/workspace/helios-rl/scripts/run_benchmark.py --algos tdmpc-glass
```

## Goal

TD-MPC2 already learns a latent world model with:

- encoder `h(o_t) -> z_t`
- dynamics `d(z_t, a_t) -> z_{t+1}`
- reward model
- Q ensemble
- policy
- MPPI planner

TD-MPC-Glass adds a structural auxiliary loss so the latent model also exposes
coherent transition regions. The intended benefit is faster emergence of useful
abstract dynamics and better early planning behavior.

## Current Implementation

The baseline TD-MPC2 file is left untouched:

```text
src/helios/algorithms/tdmpc2.py
```

The experimental variant is isolated in:

```text
src/helios/algorithms/tdmpc_glass.py
```

The module copies the TD-MPC2 v24 architecture and adds:

- `GLASS_DEFAULTS`
- `init_glass_params`
- `glass_transition_graph`
- `glass_loss_and_aux`
- local JAX implementation of 2D structural entropy
- `make_glass_diag_fn`
- Glass-aware `make_update_fn`

## Prototype Transition Graph

For each replay sequence batch, TD-MPC-Glass builds a graph over learned
prototypes, not over all batch states.

```text
z_src  = predicted rollout latents z_t
z_next = predicted rollout latents z_{t+1}
p_k    = learned prototypes
```

Soft prototype assignments:

```text
c_t      = softmax(-||z_t - p_k||^2 / tau_p)
c_t_next = softmax(-||z_next - p_k||^2 / tau_p)
```

Transition matrix:

```text
P_counts = sum_n outer(c_t[n], c_t_next[n])
P        = row_normalize(P_counts + smoothing)
A        = 0.5 * (P + P.T)
```

Current defaults:

```text
num_prototypes = 16
num_clusters = 8
proto_temperature = 0.2
assignment_temperature = 1.0
```

Prototype count was reduced from 32 to 16 after iteration showed dead prototype
rows and unnecessary diagnostic size.

## Glass Loss

The total loss is:

```text
L = L_tdmpc2
  + lambda_se       * H2(A, S_logits)
  + lambda_balance  * (cluster_balance + prototype_usage_balance)
  + lambda_temporal * temporal_consistency
```

Current defaults:

```text
lambda_se = 1e-4
lambda_balance = 1e-3
lambda_temporal = 1e-4
```

`S_logits` maps prototypes to clusters:

```text
params["glass"]["assign_logits"] -> (num_prototypes, num_clusters)
```

Prototypes are stored as:

```text
params["glass"]["prototypes"] -> (num_prototypes, latent_dim)
```

Prototype initialization is SimNorm-shaped so it lives on a scale compatible
with TD-MPC2 latents.

## Safeguards

The first implemented version uses:

- Glass warmup: `warmup_env_steps = 100_000`
- Glass frequency: `every_k_updates = 4`
- stopped graph latents by default: `stopgrad_graph = True`
- transition-row smoothing
- prototype usage balance
- scalar diagnostics during eval
- compressed matrix artifacts at eval

The planner is not cluster-aware yet. MPPI still uses the same TD-MPC2 reward,
dynamics, critic, and policy networks.

## Diagnostics

Eval prints:

```text
glass se=<...> ent=<...> active=<...> max_mass=<...> cut=<...>
```

Saved matrices:

```text
P, A, S
```

Useful checks:

- `P` row sums should be near `1.0`.
- `glass_se` should be finite and usually positive.
- `active` should not collapse to `1`.
- `max_mass` should not approach `1.0`.
- `cut` near `0.0` may indicate degenerate cluster routing.

## Known Limitations

- Current structural graph is nearly uniform in the 500k run despite good RL
  return improvement. Glass may be acting mostly as a mild regularizer so far.
- No checkpoint/resume path is wired for TD-MPC-Glass.
- `run_benchmark.py` reruns from scratch for each horizon.
- The design only beats v24 at 250k and 500k matched checkpoints so far; it has
  not yet been run to v24's 3M-4M plateau.

