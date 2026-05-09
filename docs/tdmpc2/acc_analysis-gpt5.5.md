# TD-MPC2 Acceleration Analysis

Date: 2026-05-09

## Short Answer

Yes, we can make it faster, but there are two different bottlenecks:

1. Environment stepping is already mostly broken open by MuJoCo Playground.
2. Planning and world-model updates are now the real bottlenecks.

The current JAX TD-MPC2 hopper scripts already use MuJoCo Playground in the same broad way as `scripts/run_sac_official.py`: they call `registry.load("HopperHop")` and `wrapper.wrap_for_brax_training(...)`, then step a JIT-compiled batched environment. That is why pure env stepping is no longer the limiting factor.

What we do not yet have is the full Brax SAC style training loop: SAC keeps rollout, replay queue, sampling, and update orchestration inside XLA. Our TD-MPC scripts still use a Python loop, a NumPy replay buffer, Python-side exploration noise, and repeated JIT dispatches for K updates per macro step. Moving those pieces into XLA will help, but it will not erase the cost of MPPI or the H-step world-model loss.

## Current Speed Picture

Official TD-MPC2 PyTorch baseline on hopper-hop, measured by `benchmark_official_tdmpc2_sps.py`:

| Path | Measured speed | Meaning |
|------|----------------|---------|
| Random action seed collection | ~1290 SPS | Env is not inherently slow. |
| Official train step with MPPI + update | 10.5 SPS | MPPI dominates. |
| Projected official 1M steps | ~26.4 hours | Single-env planning is the wall. |

Our JAX v9 path:

| Path | Measured speed | Meaning |
|------|----------------|---------|
| MuJoCo Playground batched env | Tens of thousands of raw transitions/sec possible | Env bottleneck mostly solved. |
| v9 training with N_ENVS=256, K_UPDATE=256 | ~60-70 logged env SPS | Update-heavy 1:1 UTD dominates. |
| Effective env transitions/sec | ~15k-17k transitions/sec | Still far faster than official single-env MPPI collection. |

The key difference: our collection path is policy-based, not MPPI-based. We only run MPPI for evaluation checkpoints. That is already the main speed trick.

## Are We Using MuJoCo Playground Like SAC?

Partially yes.

`scripts/run_sac_official.py` uses:

```python
env = registry.load(env_id)
sac.train(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    ...
)
```

The TD-MPC hopper scripts use the same Playground registry and Brax training wrapper pattern:

```python
env_raw = registry.load("HopperHop")
env = wrapper.wrap_for_brax_training(env_raw, episode_length=1000, action_repeat=1)
```

So the environment backend is already the fast one.

The remaining difference is loop ownership:

| Component | Official Brax SAC | Current JAX TD-MPC |
|-----------|-------------------|--------------------|
| Env stepping | XLA | XLA batched step, called from Python |
| Rollout loop | XLA scan | Python while loop |
| Replay buffer | Device-side Brax queue | NumPy host buffer |
| Sampling | Device-side | NumPy fancy indexing, then host-to-device copy |
| Update loop | XLA scan | Python for loop over K JIT calls |
| Planning | None | JIT MPPI, currently eval-only |

Adopting the full SAC-style loop can reduce Python dispatch, host sampling, and transfer overhead. It will not by itself remove TD-MPC2's model unroll or MPPI planning cost.

## Why Planning Is the Hard Bottleneck

MPPI cost scales roughly as:

```text
cost ~= n_iter * n_samples * horizon * world_model_forward
```

Current JAX MPPI eval config:

```text
horizon = 5
n_samples = 256
n_iter = 6
```

That is 7680 imagined transitions per real action, plus terminal Q evaluation and policy warm-starts. Official TD-MPC2 uses a similar idea and gets 10.5 train SPS on a single env.

If we used MPPI for every action across 256 parallel envs, the compute would explode. The current design avoids that by collecting with the learned policy and using MPPI only as an evaluation/probing tool.

The planning bottleneck can be broken only by changing one of these:

1. Run fewer plans.
2. Make each plan cheaper.
3. Vectorize/batch plans harder.
4. Distill planning into the policy so planning is rarely needed.
5. Replace MPPI with a cheaper optimizer or a better policy prior.

## Acceleration Plan

### Phase 1: Keep Policy Collection, Use MPPI Sparingly

This is the safest near-term route.

Use the policy for training collection, as v9/v10 already do. Treat MPPI as:

- Evaluation metric.
- Debugger for world-model usefulness.
- Occasional policy improvement target.

Recommended experiment:

| Setting | Value |
|---------|-------|
| Collection action | policy mean + exploration noise |
| MPPI eval frequency | every 1M env steps |
| MPPI eval episodes | 3-5 |
| Training target | improve pi first, then MPPI |

This keeps the speed advantage while we fix algorithm quality.

### Phase 2: Reduce MPPI Cost Without Changing Semantics Too Much

Test a small grid on evaluation-only MPPI:

| Variant | Horizon | Samples | Iters | Relative cost | Risk |
|---------|---------|---------|-------|---------------|------|
| Current | 5 | 256 | 6 | 1.00x | Baseline |
| Official-ish fast | 3 | 256 | 6 | 0.60x | Usually reasonable; official horizon is 3. |
| Fewer samples | 5 | 128 | 6 | 0.50x | More noisy elite weights. |
| Fewer iters | 5 | 256 | 3 | 0.50x | Less CEM refinement. |
| Fast eval | 3 | 128 | 3 | 0.15x | Good for frequent monitoring, not final score. |

The best first test is `H=3, NS=256, NI=6` because it matches the official horizon and should cut planning and world-model update depth.

### Phase 3: Batch MPPI Evaluation Across Episodes

Current eval loops step one episode at a time in Python. The plan function itself is JIT-compiled, but evaluation still does repeated host dispatch.

Build `eval_mppi_batch(params, key, n_eps)`:

- Reset `n_eps` envs at once.
- Maintain per-env `mu` with shape `(n_eps, H, action_dim)`.
- Use `jax.vmap(plan)` across envs.
- Use `jax.lax.scan` across episode time.

Expected effect:

- MPPI eval wall time should drop substantially.
- Training SPS will not change much because MPPI is eval-only today.
- This makes frequent MPPI checks affordable.

This is the right way to make evaluation less annoying without changing learning.

### Phase 4: Fuse the TD-MPC Update Loop

Current training does:

```python
for _ in range(K_UPDATE):
    sample on CPU
    transfer to GPU
    upd(...)
```

A faster version should move toward:

```python
@jax.jit
def train_macro_step(state):
    state = collect_one_batched_env_step(state)
    state = lax.scan(update_once, state, None, length=K_UPDATE)
    return state
```

This requires replacing `MultiEnvBuffer` with a device-side ring buffer or Brax-style queue.

Expected effect:

- Removes Python dispatch per update.
- Removes NumPy sampling and host-to-device transfer.
- Likely 5-20% faster in the current v9 quality config.
- Bigger gain for fast-iteration configs where dispatch is a larger fraction.

Important caveat: this does not remove the H-step dynamics unroll inside each update. It improves orchestration, not the core model cost.

### Phase 5: Distill MPPI Into the Policy

This is the most promising way to break the planning bottleneck while keeping planning quality.

Run MPPI on a small subset of states, then train the policy to imitate the MPPI action:

```text
policy_loss = -Q(z, pi(z)) + lambda_bc * ||pi(z) - stop_gradient(mppi_action(z))||^2
```

Possible schedule:

| Step range | MPPI usage |
|------------|------------|
| Warmup to 500k | No MPPI for collection. |
| 500k onward | Run MPPI on 1-5% of replay states. |
| Every 100k | Add MPPI imitation minibatch. |
| Eval | Report both pi and MPPI. |

This amortizes planning: one expensive plan teaches many cheap policy actions.

Success criterion:

- `pi` rises toward `MPPI`.
- Gap `MPPI - pi` shrinks.
- Collection remains fast.

### Phase 6: Plan Only on Uncertain or High-Value States

Do not plan uniformly. Use planning where it is likely to matter.

Candidate triggers:

- Q ensemble disagreement above threshold.
- Low policy entropy collapse.
- Novel latent state by kNN or running latent variance.
- Periodic every K real env steps.

Example:

```text
95% actions: pi(obs) + noise
5% actions: MPPI(obs), store both action and MPPI target
```

This gives some on-policy MPPI improvement without paying official TD-MPC2's every-step planning cost.

## What Not To Expect

MuJoCo Playground will not further break the main bottleneck because env stepping is no longer the main bottleneck. It can still help if we move the whole loop into the Brax-style XLA training architecture, but the large wins now come from:

1. Reducing UTD for fast iteration.
2. Reducing horizon from 5 to 3.
3. Making MPPI eval batched.
4. Distilling MPPI into the policy.
5. Planning on a small subset of states instead of every state.

## Recommended Next Implementation

Build `train_tdmpc_hopper_v11_accel.py` with these changes:

| Change | Value |
|--------|-------|
| Base | v10 |
| N_ENVS | 1024 for fast iteration, 256 for quality |
| K_UPDATE | 64 for fast iteration, 256 for quality |
| Horizon | 3 |
| MPPI samples/iters | 256/6 for final eval, 128/3 for frequent eval |
| Eval | batched `eval_pi_batch`, then batched `eval_mppi_batch` |
| Collection | pi-only with noise |
| Optional | MPPI distillation on replay subset |

Run order:

1. `v11_fast_h3`: N=1024, K=64, H=3, pi-only collection.
2. `v11_quality_h3`: N=256, K=256, H=3, same algorithm.
3. `v12_mppi_distill`: add sparse MPPI imitation targets.
4. `v13_device_buffer`: move replay/update loop toward full XLA if the algorithm is worth scaling.

## Bottom Line

We already got the big environment win by using MuJoCo Playground. The next speed breakthrough is not another env backend change; it is planning amortization.

The fastest viable strategy is:

```text
collect with pi, evaluate with MPPI, occasionally distill MPPI into pi
```

That keeps the JAX/Playground throughput advantage while using planning as a teacher instead of paying the official TD-MPC2 cost on every environment step.