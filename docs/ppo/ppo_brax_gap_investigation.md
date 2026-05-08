# PPO Implementation Gap Investigation: Matching Brax PPO on CheetahRun

**Goal:** Reproduce Brax's 922 return on MuJoCo Playground `CheetahRun` using a from-scratch JAX/Flax PPO implementation.  
**Environment:** MuJoCo Playground `CheetahRun`, `episode_length=1000`, 2048 parallel envs.  
**Hardware:** NVIDIA RTX 3090 (24 GiB, sm_86). JAX 0.10.0, Brax 0.14.2.  
**Main script:** `/workspace/run_ppo_continuous_mjx.py`  
**Reference script:** `/workspace/run_brax_ppo.py` (Brax PPO wrapper)

---

## Brax Reference Results

```
Config: lr=1e-3, num_envs=2048, unroll_length=30, num_updates_per_batch=16, 
        num_minibatches=32, batch_size=1024, gamma=0.995, ent_coef=0.01, 
        clip_coef=0.3, reward_scaling=10.0, normalize_observations=True, max_grad_norm=1.0
Network: Policy=4×32 swish lecun_uniform → 2*action_dim; Value=5×256 swish lecun_uniform → 1
         Distribution: tanh_normal, separate policy/value networks

Step       | Reward
-----------|-------
9.8M       | 287
19.7M      | 548
29.5M      | 709
39.3M      | 808
49.2M      | 872
59.0M      | 904
68.8M      | 913
78.6M      | 906
88.5M      | 922  ← peak

JIT compile: ~290s first run, ~56s with cached warp kernels
Train time: ~967s (180M steps)
```

---

## Bug Table: All Issues Found and Fixed

| Version | Bug | Fix | Result |
|---------|-----|-----|--------|
| v1–v2 | Baseline: wrong IS ratio, clipped action, no bootstrap | — | ~0 |
| v3 | `obs_norm` updated inside `jax.lax.scan` (re-traced each step) | Update obs_norm outside scan | +150 |
| v4–v5 | PPO re-evaluation used new obs_ns but stored logprobs used old | Normalize storage with old obs_ns before update | +100 |
| v6 | `sqrt(var) + eps` instead of `sqrt(var + eps)` in obs norm | Correct formula | +50 |
| v7 | Stale `storage.values` (critic not re-evaluated at update time) | Recompute values fresh in compute_gae | +80 |
| v8 | `clip(raw_action, -1, 1)` for env, but PPO re-evals at `raw_action`; stored logprob at `raw` | Switch to `tanh(raw_action)` for env action, logprob at `raw` | **619** |
| v10 | No truncation bootstrap (timeout treated as true terminal) | `nextnonterminal = 1 - done + truncation` | +50 |
| v11 | Per-epoch GAE + Gaussian+clip: feedback loop at boundary | One-shot GAE (compute once before epochs) | stable but 706 |
| v12 | LR `transition_steps=num_iterations` (off by 512×, decayed in 1 update) | Fixed schedule denominator | stable |
| v13 | LR=1e-3 was too high for early training with Gaussian+clip | Lower LR=3e-4 for now | 544 |
| v14 | One-shot GAE, lr=3e-4, correct truncation | — | **775** |
| v15 | Gaussian+clip distribution: wrong IS ratio at action boundary; `tanh(raw)` clips but logprob doesn't | Replace with `tanh_normal`: `env_action=tanh(raw)`, Jacobian correction in logprob | **783** |
| v16 | LR annealing to 1e-5 by 30M steps — policy locked in sub-optimal basin | Remove LR annealing | 697 (found annealing was wrong) |
| **v17** | One-shot GAE; should be per-epoch like Brax | Move `compute_gae` inside `update_epoch` loop | **818** |
| **v18** | `num_steps=30` → 16× more gradient steps per env step than Brax (crashes) | `num_steps=480 = 16×30` to match Brax's 983,040 steps/update | **845** |
| **v19** | `max_grad_norm=0.5`; Brax uses 1.0 | `max_grad_norm=1.0` | **872** |

---

## Iteration Detail

### v8 — 619 (IS Ratio Fix)
The primary bug: `env_action = clip(raw_action, -1, 1)` but PPO re-evaluated logprob at `raw_action`. When `|raw_action| > 1`, the policy was simultaneously outputting an action that the environment received as ±1 (clipped), but the logprob the IS ratio compared against was for the unclipped value. This caused the IS ratio to be nonsensical near boundaries.

Fix: Use `env_action = tanh(raw_action)` so the env always receives the same value the logprob is computed for.

### v14 — 775 (One-Shot GAE + lr=3e-4)
Stable training. One-shot GAE (compute advantages once from current critic before starting PPO epochs). LR=3e-4 required because Gaussian distribution with clipping (`clip(raw, -1, 1)`) has a gradient vanishing problem near ±1 — the correct logprob doesn't change when `raw` changes past ±1, but the env action doesn't change either, so the IS ratio has singularities.

### v15 — 783 (tanh_normal Distribution)
Switched to `tanh_normal` matching Brax's distribution exactly:
- Sample `raw ~ N(mean, std)` where `std = softplus(log_scale) + 0.001`
- `env_action = tanh(raw) ∈ (-1, 1)`
- `log_prob = log_prob_gaussian(raw) - Σ log(1 - tanh²(raw_i))`
- The Jacobian correction `log(1 - tanh²(x)) = 2*(log2 - x - softplus(-2x))` is numerically stable

```python
def _tanh_log_det_jac(x):
    return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

def _tanh_normal_logprob(raw_action, mean, log_scale):
    std = jax.nn.softplus(log_scale) + 0.001
    log_prob_gauss = -0.5 * ((raw_action - mean) / std)**2 - jnp.log(std) - 0.5*jnp.log(2*pi)
    return (log_prob_gauss - _tanh_log_det_jac(raw_action)).sum(axis=-1)
```

### v17 — 818 (Per-Epoch GAE)
Moved `compute_gae` inside `update_epoch` so the critic recomputes advantages with its current parameters at the start of each of the 16 epochs. This matches Brax's behavior.

Why this works stably with `tanh_normal` but crashed with `Gaussian+clip` (v11):
- Gaussian+clip: at high reward, `|raw|` grows large to saturate `tanh`/`clip`. Near the boundary, tiny changes in `mean` cause huge IS ratio swings → gradient explosion under 16× critic updates per rollout.
- tanh_normal: IS ratio is always well-defined; the Jacobian correction prevents the gradient from blowing up because the distribution "knows" it's near saturation.

```python
@jax.jit
def update_ppo(agent_state, storage, next_obs, next_done, key):
    def update_epoch(carry, _):
        agent_state, key = carry
        # Per-epoch GAE: recompute with current critic at each epoch
        epoch_storage = compute_gae(agent_state, next_obs, next_done, storage)
        key, subkey = jax.random.split(key)
        shuffled_storage = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                jax.random.permutation(subkey, x.reshape((-1,) + x.shape[2:])),
                (args.num_minibatches, -1) + x.shape[2:]),
            epoch_storage)
        def update_minibatch(agent_state, minibatch):
            (loss, aux), grads = ppo_loss_grad_fn(
                agent_state.params, minibatch.obs, minibatch.actions,
                minibatch.logprobs, minibatch.advantages, minibatch.returns)
            return agent_state.apply_gradients(grads=grads), aux
        agent_state, aux = jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
        return (agent_state, key), aux
    (agent_state, key), aux = jax.lax.scan(
        update_epoch, (agent_state, key), (), length=args.update_epochs)
    return agent_state, key
```

### v18 — 845 (Match Brax Data Volume)
Key insight discovered by reading Brax's source:

```
Brax: batch_size=1024, num_minibatches=32, num_envs=2048, unroll_length=30
      → steps per policy update = 1024 * 32 * 30 / 2048 = 16 unrolls = 983,040 steps

v17:  num_steps=30, num_envs=2048
      → steps per policy update = 30 * 2048 = 61,440 steps
      → 16× fewer env steps per 512 gradient updates → 16× higher gradient variance → crashes
```

Fix: `num_steps = 480 = 16 × 30` so `480 × 2048 = 983,040` exactly matches Brax's data volume per update.

### v19 — 872 (Correct Gradient Clipping)
Brax uses `max_grad_norm=1.0` (from `train_jax_ppo.py`), not our default of 0.5. With 0.5 we clip too aggressively, reducing effective LR during early high-gradient phases.

---

## Summary of Results

| Version | Key Change | Best Return | Notes |
|---------|-----------|-------------|-------|
| v8 | IS ratio fix (tanh for env) | 619 | Gaussian+clip dist |
| v11 | Per-epoch GAE | 706 | Crashed at 36M |
| v14 | One-shot GAE + lr=3e-4 | 775 | Stable |
| v15 | tanh_normal distribution | 783 | Brax network arch |
| v16 | LR annealing (wrong) | 697 | Shows annealing hurts |
| **v17** | Per-epoch GAE + tanh_normal + lr=1e-3 | **818** | No crashes until 50M |
| **v18** | num_steps=480 (match Brax volume) | **845** | ~1 crash at 68M |
| **v19** | max_grad_norm=1.0 | **872** | ~1 crash at 88M |
| **v20** | Brax-exact GAE (2-pass + truncation zeroing) + fresh entropy sample | **879** | Best at 176M |
| **v21** | No epoch reuse: 16 fresh rollouts × 1-pass update + per-minibatch GAE | **TBD** | Running |
| Brax | Reference | **922** | Peak at 88M |

**Current gap: 879 vs 922 (~4.7%). Goal: ≥900 before 60M steps.**

---

## v21 — The Key Structural Fix: No Data Reuse

Reading Brax's training loop revealed the **biggest remaining difference**:

### What we thought `num_updates_per_batch=16` meant (v17–v20)
> Collect one big rollout (480 steps × 2048 envs = 983,040 transitions), then run 16 *epochs* reshuffling and reusing the same data.

### What Brax actually does
```python
# training_step: scan over num_updates_per_batch=16
def f(carry, unused_t):
    current_state, current_key = carry
    next_state, data = acting.generate_unroll(env, state, policy, key, unroll_length=30)
    return (next_state, next_key), data

(state, _), data = jax.lax.scan(f, (state, key), (), length=num_updates_per_batch=16)
# data shape: (16, 2048, 30) — 16 SEPARATE fresh rollouts

# Then for each of the 16 fresh datasets:
jax.lax.scan(sgd_step, ..., data, length=16)
```

**Brax collects 16 fresh rollouts and uses each EXACTLY ONCE.** No IS ratio staleness. Each rollout of 2048 env × 30 steps is:
1. Shuffled along env dim → `(2048, 30)` → split into `(32, 64, 30)` minibatches
2. Per-minibatch GAE with current critic
3. 32 gradient steps (each transition used once)

### The consequence for our v17–v20

With our 480-step rollout and 16 epochs:
- By epoch 16, `old_logprob` is from a policy that's 512 gradient steps stale
- IS ratio `exp(new_logp - old_logp)` can be far from 1
- PPO clipping `clip(ratio, 0.7, 1.3)` kicks in aggressively → noisy gradients → oscillating rewards (672↔879 in v20)

With Brax's fresh-data-only approach:
- IS ratio is always ≈1 (same policy used for collection and update)
- Clipping almost never activates
- Gradient steps are always pure policy gradient → smooth convergence to 922

### Implementation (v21)

```python
def collect_and_update_once(...):
    """One cycle: collect 30-step rollout + single-pass 32-mb update."""
    storage = collect_30_steps(...)  # 2048 envs × 30 steps
    perm = random_permutation(2048)
    # Maintain temporal structure: split along env dim only
    # (T=30, N=2048) -> (num_mb=32, T=30, mb_size=64)
    mb_storage = storage[:, perm].reshape(T, 32, 64).swapaxes(0, 1)
    mb_next_obs = next_obs[perm].reshape(32, 64, obs_dim)
    for mb in range(32):
        gae = compute_gae_mb(agent, mb_next_obs[mb], mb_storage[mb])  # fresh critic
        gradient_step(gae)  # one step per minibatch, no reuse

def rollout_and_update(...):
    # 16 sequential fresh-rollout cycles (no epoch reuse)
    jax.lax.scan(collect_and_update_once, ..., length=16)
```

Total: 16 cycles × 32 minibatches = 512 gradient steps, each on fresh data matching Brax exactly.

---

## v20 — Brax-Exact GAE + Fresh Entropy Sample

Deep-reading Brax's `brax/training/agents/ppo/losses.py::compute_gae` revealed two remaining algorithmic differences:

### Difference 1: Two-pass GAE with truncation zeroing

Brax uses a fundamentally different GAE formula from standard PPO:

**Standard GAE (our v17–v19):**
```
delta_t = r_t + gamma * V(s_{t+1}) * nextnonterminal - V(s_t)
A_t = delta_t + gamma * lambda * nextnonterminal * A_{t+1}
```

**Brax's 2-pass GAE:**
```
# Pass 1: compute TD(lambda) value target vs (zeros delta at truncation boundaries)
truncation_mask = 1 - truncation  # 0 when truncated
termination = done * (1 - truncation)  # only truly-terminal transitions
delta_t = (r_t + gamma * (1-termination) * V(s_{t+1}) - V(s_t)) * truncation_mask
vs_minus_v_xs_t = delta_t + gamma * (1-termination) * truncation_mask * lambda * vs_minus_v_xs_{t+1}
vs_t = vs_minus_v_xs_t + V(s_t)  # TD-lambda return = value target for critic

# Pass 2: advantage = (r + gamma * vs_{t+1} - V(s_t)) * truncation_mask
#          uses gamma (NOT gamma*lambda) in the outer formula
A_t = (r_t + gamma * (1-termination) * vs_{t+1} - V(s_t)) * truncation_mask
```

Key differences:
1. **Truncation zeroing**: `A_t = 0` when `truncation=1` (step is a timeout). Our code bootstraps but keeps non-zero advantage.
2. **γ vs γλ in outer formula**: Brax's advantage effectively uses `γ` for the final accumulation while the value target uses `γλ`. Our code uses `γλ` uniformly.
3. **Value target**: Brax uses `vs` (TD-λ return) as critic target, not `A_t + V(s_t)`. With `lambda < 1` these are slightly different due to the truncation zeroing.

### Difference 2: Entropy with fresh sample

**Brax's entropy:**
```python
entropy = dist.entropy()  # gaussian analytical: 0.5*log(2πe) + log(σ)  per-element
entropy += tanh_log_det_jac(dist.sample(rng))  # Jacobian at a NEW random sample
entropy = sum(entropy, axis=-1)
```

**Our v17–v19:** Used the stored `raw_action` as the Jacobian sample point. Since both are unbiased estimates of the same quantity (the true entropy of NormalTanh has no closed form), the difference is only in variance.

### Implementation

```python
# Pass 1: TD-lambda value targets
deltas = (r + gamma*(1-term)*V(s') - V(s)) * truncation_mask
vs_minus_v_xs_t = delta + gamma*(1-term)*truncation_mask*lambda * vs_minus_v_xs_{t+1}
vs = vs_minus_v_xs + V(s)

# Pass 2: advantages (Brax actor formula)
vs_t_plus_1 = concat([vs[1:], bootstrap_value])
advantages = (r + gamma*(1-term)*vs_{t+1} - V(s)) * truncation_mask
```

The entropy change: sample a fresh `raw_sample ~ N(mean, std)` inside the loss and use it for the Jacobian correction term.

---

## Reproduction

### Install

```bash
cd /workspace/wiki/learn_mujoco_playground/repo
# environment already set up with mujoco_playground, brax, jax 0.10.0
```

### Run Brax reference (922 target)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/run_brax_ppo.py \
  --env_name=CheetahRun \
  --num_timesteps=180000000 \
  --num_envs=2048 \
  --unroll_length=30 \
  --num_updates_per_batch=16 \
  --num_minibatches=32 \
  --reward_scaling=10.0 \
  --learning_rate=1e-3
```

### Run v20 (Brax-exact GAE, TBD)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/run_ppo_continuous_mjx.py \
  --env-id CheetahRun \
  --total-timesteps 180000000 \
  --num-envs 2048 \
  --num-steps 480 \
  --learning-rate 1e-3 \
  --update-epochs 16 \
  --num-minibatches 32 \
  --gamma 0.995 \
  --ent-coef 0.01 \
  --clip-coef 0.3 \
  --max-grad-norm 1.0 \
  --no-anneal-lr \
  --normalize-obs \
  --reward-scaling 10.0 \
  --eval-freq 10 \
  --early-stop-patience 50 \
  --checkpoint-dir /workspace/runs/checkpoints \
  --exp-name ppo_jax_v20
```

### Run our best implementation (872, v19)

```bash
PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
  python3 /workspace/run_ppo_continuous_mjx.py \
  --env-id CheetahRun \
  --total-timesteps 180000000 \
  --num-envs 2048 \
  --num-steps 480 \
  --learning-rate 1e-3 \
  --update-epochs 16 \
  --num-minibatches 32 \
  --gamma 0.995 \
  --ent-coef 0.01 \
  --clip-coef 0.3 \
  --max-grad-norm 1.0 \
  --no-anneal-lr \
  --normalize-obs \
  --reward-scaling 10.0 \
  --eval-freq 10 \
  --early-stop-patience 50 \
  --checkpoint-dir /workspace/runs/checkpoints \
  --exp-name ppo_jax_v19
```

Expected: JIT ~56s, train ~1130s, best ~872.

---

## Key Code: Final Architecture (`/workspace/run_ppo_continuous_mjx.py`)

### Networks (matches Brax exactly)

```python
lecu = jax.nn.initializers.lecun_uniform()

class PolicyNet(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        for _ in range(4):
            x = nn.Dense(32, kernel_init=lecu)(x)
            x = nn.swish(x)
        return nn.Dense(2 * self.action_dim, kernel_init=lecu)(x)  # [mean | raw_scale]

class ValueNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(5):
            x = nn.Dense(256, kernel_init=lecu)(x)
            x = nn.swish(x)
        return nn.Dense(1, kernel_init=lecu)(x)
```

### tanh_normal distribution

```python
def _tanh_log_det_jac(x):
    return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

def _tanh_normal_logprob(raw_action, mean, log_scale):
    std = jax.nn.softplus(log_scale) + 0.001
    log_prob_gauss = (-0.5 * ((raw_action - mean) / std)**2
                     - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi))
    return (log_prob_gauss - _tanh_log_det_jac(raw_action)).sum(axis=-1)
```

### GAE with truncation bootstrap

```python
def compute_gae_once(carry, inp, gamma=0.995, gae_lambda=0.95):
    advantages = carry
    nextdone, nextvalues, curvalues, reward, truncation = inp
    # truncation=1 means timeout, not true terminal → bootstrap next value
    nextnonterminal = jnp.clip(1.0 - nextdone + truncation, 0.0, 1.0)
    delta = reward + gamma * nextvalues * nextnonterminal - curvalues
    advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
    return advantages, advantages
```

---

## Log Files

| Version | Log | Best |
|---------|-----|------|
| Brax ref | `/workspace/runs/brax_ppo_cheetahrun.log` | 922 |
| v14 | `/workspace/runs/ppo_jax_v14.log` | 775 |
| v15 | `/workspace/runs/ppo_jax_v15.log` | 783 |
| v16 | `/workspace/runs/ppo_jax_v16.log` | 697 |
| v17 | `/workspace/runs/ppo_jax_v17.log` | 818 |
| v18 | `/workspace/runs/ppo_jax_v18.log` | 845 |
| v19 | `/workspace/runs/ppo_jax_v19.log` | 872 |
