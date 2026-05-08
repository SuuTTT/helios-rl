# PPO Design Document — Two Implementations

**Scope:** This document covers the architecture and design decisions for both PPO variants used in the CheetahRun benchmark. It also introduces the third target: MuJoCo Playground native Brax PPO.

---

## 1. System Overview

```
┌───────────────────────────────────────────────────────────────┐
│                      PPO Implementations                      │
│                                                               │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────┐ │
│  │  CleanRL PyTorch  │    │  Custom JAX/MJX   │    │ Brax PPO│ │
│  │  (Variant A)      │    │  (Variant B)      │    │ (Target)│ │
│  └──────────────────┘    └──────────────────┘    └─────────┘ │
│  CPU · dm_control         GPU · MJX playground    GPU · Brax  │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Variant A — CleanRL PyTorch PPO

### 2.1 File
`cleanrl/cleanrl/ppo_continuous_action_dmc.py`

### 2.2 Architecture

```
Observation (obs_dim,)
      │
  ┌───▼───────────────┐
  │  Linear(64)       │  ← orthogonal init, tanh activation
  │  Linear(64)       │
  └───┬───────────────┘
      │
  ┌───▼───────┐    ┌───▼───────────┐
  │  Actor    │    │   Critic      │
  │  Linear   │    │   Linear(1)   │
  │  (action) │    │               │
  └───────────┘    └───────────────┘
  ↓ mean + log_std (learnable scalar)
  Normal distribution → clipped action
```

**Network:** MLP with two 64-unit hidden layers, `tanh` activations, orthogonal initialisation.  
**Actor:** Outputs action mean; `log_std` is a separate learned parameter (not input-dependent).  
**Critic:** Outputs scalar value.

### 2.3 PPO Loop

```
for each iteration (num_iterations = total_timesteps / (num_envs × num_steps)):
    collect num_steps transitions from num_envs envs
    compute GAE advantages (gamma=0.99, gae_lambda=0.95)
    for update_epoch in range(10):
        shuffle and split into num_minibatches=32 minibatches
        compute surrogate loss + value loss + entropy bonus
        clip gradients at max_norm=0.5
        Adam step (lr annealed from 3e-4 → 0)
```

**Episode logging fix:** dm_control does not use `final_info`; instead it populates `infos["episode"]` directly. The CleanRL script was patched to detect this and correctly extract `r[i]` using the `next_done` mask.

### 2.4 Key Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| `num_steps` | 512 | Covers ~0.5 episodes/env/rollout for 1000-step episodes |
| `num_envs` | 4 | CPU-bound; 4 is enough for diversity without memory pressure |
| LR schedule | Linear anneal 3e-4→0 | Standard PPO; annealing over full budget |
| `update_epochs` | 10 | Extra data reuse helps on CPU-bottlenecked training |
| Backend | dm_control via shimmy | Mature, well-tested physics; CPU multiprocess |

---

## 3. Variant B — Custom JAX/MJX PPO

### 3.1 File
`/workspace/run_ppo_continuous_mjx.py`

### 3.2 Architecture

```
Observation (obs_dim,)
      │
  ┌───▼───────────────┐
  │  Dense(256) swish │  ← orthogonal(√2) init
  │  Dense(256) swish │
  └───┬───────────────┘
      │
  ┌───▼───────────┐  ┌───▼──────────────┐
  │  Actor head   │  │  Critic head     │
  │  Dense(a_dim) │  │  Dense(1)        │
  └───────────────┘  └──────────────────┘
  ↓ mean + log_std (learnable param, orthogonal(0.01))
  Normal distribution
```

**Network:** Two 256-unit hidden layers, `swish` activations (vs `tanh` in CleanRL). Wider and deeper than CleanRL's 64-unit layers.

**State container:**
```python
@flax.struct.dataclass
class AgentParams:
    network_params:   FrozenDict  # shared trunk
    actor_params:     FrozenDict  # actor head
    critic_params:    FrozenDict  # critic head
    actor_logstd:     jnp.ndarray # learnable log-std
```

### 3.3 JAX Training Loop Design

The entire rollout and update step is compiled with `@jax.jit`. Vectorisation over `num_envs` uses `jax.vmap` implicitly through the MJX environment's batched `step` function.

```python
# Compiled training step (runs entirely on GPU)
@jax.jit
def train_step(runner_state, _):
    # Rollout: num_steps × num_envs transitions
    # GAE computation (vectorised)
    # PPO loss with clipped surrogate + value clip + entropy
    # Optax gradient update
    return runner_state, metrics

# lax.scan over num_iterations avoids Python-level loops
runner_state, metrics = jax.lax.scan(train_step, runner_state, None, num_iterations)
```

**Rollout collection** collects `(obs, action, logprob, reward, done, value)` for `num_steps` steps across all envs simultaneously.

### 3.4 Optimiser Design (Modified)

```python
# LR schedule: toggled by --anneal-lr / --no-anneal-lr
if args.anneal_lr:
    lr_schedule = optax.linear_schedule(
        init_value=args.learning_rate,
        end_value=0.0,
        transition_steps=args.num_iterations,
    )
else:
    lr_schedule = optax.constant_schedule(args.learning_rate)

tx = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adam(lr_schedule, eps=1e-5),
)
```

**Why constant LR wins here:** With `num_steps=10` and 30 M total steps, the policy is still improving at step 25 M. Annealing to zero by then freezes gradients and prevents further learning, causing stagnation or instability as the value function can no longer track the improving policy.

### 3.5 New Features Added This Session

#### Early Stopping
```python
# Args
early_stop_patience: int = 0  # 0 = disabled; N = stop after N non-improving evals

# Training loop
if ret > best_return and ret > 0:
    best_return = ret
    no_improve_count = 0
elif ret > 0:
    no_improve_count += 1
if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
    break
```

#### Best Checkpoint Saving
```python
# Args
checkpoint_dir: str = ""  # empty = disabled

# On improvement
if args.checkpoint_dir:
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = f"{args.checkpoint_dir}/{args.exp_name}_best.msgpack"
    with open(ckpt_path, "wb") as f:
        f.write(flax.serialization.to_bytes(best_params))
```

Checkpoint format: `flax.serialization.to_bytes` (msgpack binary). Load with:
```python
with open(ckpt_path, "rb") as f:
    params = flax.serialization.from_bytes(template_params, f.read())
```

### 3.6 Key Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| `num_steps` | 10 | Maximises gradient update count for fixed timestep budget |
| `num_envs` | 512 | Fills GPU VRAM (RTX 3090), amortises JIT compilation |
| `update_epochs` | 8 | Doubles learning vs default 4 without reducing sample diversity |
| `anneal_lr` | **off** | Prevents value-function stagnation at high step counts |
| Backend | MuJoCo Playground / MJX | Native GPU physics, no Python overhead in rollout |
| Network width | 256 | Larger than CleanRL's 64; captures more state features |
| Activation | swish | Smooth gradient flow; empirically preferred for continuous control |

---

## 4. Variant C (Planned) — Brax PPO from MuJoCo Playground

### 4.1 File
`wiki/learn_mujoco_playground/repo/learning/train_jax_ppo.py`

### 4.2 Key Differences vs Variant B

| | Variant B (Custom JAX) | Variant C (Brax PPO) |
|--|------------------------|----------------------|
| PPO implementation | Hand-written with Flax+Optax | Brax's battle-tested PPO |
| Network | MLP 2×256 swish | Configurable (per-env presets) |
| Rollout | Manual scan loop | Brax internal (optimised) |
| Hyperparameters | Manual CLI flags | Per-env config dicts in `mujoco_playground.config` |
| Evaluation | Inline return logging | Separate eval callback |
| Env config | `registry.load(env_id)` | Same registry + domain randomisation support |
| Logging | TensorBoard | TensorBoard + optional WandB |

### 4.3 Expected Config for CheetahRun

From `mujoco_playground/config/dm_control_suite_params.py` the Brax PPO defaults for CheetahRun are optimised by DeepMind and expected to outperform hand-tuned Variant B.

---

## 5. Design Comparison Matrix

| Property | CleanRL (A) | Custom JAX (B) | Brax PPO (C) |
|----------|------------|----------------|--------------|
| Gradient updates / M steps | ~5 K | ~47 K | TBD |
| Episode horizon coverage | ~50% | ~1% | TBD |
| Reproducibility (seed) | ✓ | ✓ | ✓ |
| Multi-env vectorisation | Python multiprocess | JAX vmap | Brax vmap |
| Best observed return | 550 | 465 | — |
| Development overhead | Low (CleanRL template) | Medium (custom) | Low (library) |
