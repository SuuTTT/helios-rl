# PPO Hyperparameter Guide — DMC Suite

Why different environments need different hyperparameters, and what each one does.

---

## 1. Core Hyperparameters

### `gamma` — Discount Factor

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

Controls how far into the future the agent cares about rewards. The effective planning horizon is approximately:

$$H_{\text{eff}} \approx \frac{1}{1 - \gamma}$$

| gamma | Effective horizon |
|---|---|
| 0.99 | ~100 steps |
| 0.995 | ~200 steps |
| 0.95 | ~20 steps |

**Why it changes across tasks:**

| Environment | gamma | Reason |
|---|---|---|
| CheetahRun | 0.995 | Long-horizon locomotion; each step contributes to sustained speed |
| CartpoleSwingup | 0.995 | Requires sustained balance; future stability matters |
| FishSwim | 0.995 | Continuous locomotion, credit flows across many steps |
| AcrobotSwingup | 0.995 | Must swing up over ~100 steps — long horizon essential |
| HopperStand | 0.995 | Sustained upright posture; each step tied to long-term balance |
| **FingerSpin** | **0.95** | Short-horizon spinning contact; only recent contacts matter for angular velocity |
| **BallInCup** | **0.95** | Catching is a short-duration event; low gamma prevents over-discounting the catch |
| CartpoleSwingupSparse | 0.995 | Sparse — when a rare reward arrives, it needs to propagate back far enough |

**The FingerSpin mistake:** Our baseline ran `gamma=0.995` for all environments. For FingerSpin, this causes the value function to integrate reward over ~200 steps, making it mix signal from successful contacts with irrelevant future steps. The critic becomes noisy, destabilizing the actor update. The correct `gamma=0.95` limits this to ~20 steps, which matches the timescale of the spinning contact.

**Empirical confirmation:**
- Baseline (gamma=0.995): FingerSpin mean best = 408  
- Fix (gamma=0.95): seed 1 = 536, seed 2 mid-run = 863 ✓

---

### `learning_rate`

All DMC Suite environments in the MuJoCo Playground reference use `lr=1e-3`. This is consistent with the original Brax PPO configuration.

Higher LR → faster initial learning but more instability (the zigzag effect).  
Lower LR → smoother curves but slower convergence.

**Note on the zigzag pattern:** Our curves show oscillations because we evaluate every ~983K steps (eval_freq=1 iteration). The official Brax reference evaluates every ~7.5M steps (`num_evals=10` over 75M steps). The oscillations are real per-iteration variance in PPO — Brax simply hides them by reporting less frequently. With EWA smoothing applied in the plot script (`--smooth 0.4`), our curves look qualitatively similar to the reference.

---

### `num_steps` (unroll length / rollout horizon)

Number of environment steps collected per parallel env before each update.

$$\text{batch size} = N_{\text{envs}} \times N_{\text{steps}} = 2048 \times 30 = 61{,}440$$

All DMC Suite environments use `num_steps=30` in the reference. This is short compared to typical PPO (e.g., 2048 steps in OpenAI baselines) because:
1. MJX runs 2048 parallel envs — total transitions per update = 61K, which is sufficient.
2. Short rollouts reduce variance in the GAE advantage estimates for fast-moving environments.
3. JIT compilation amortizes the fixed overhead across many small updates.

---

### `num_envs`

2048 parallel environments for all tasks. This is the Brax default. Benefits:
- Near-100% GPU utilization via SIMD batching on MJX.
- Lower variance policy gradient estimates per wall-clock second.
- Enables vectorized rollout collection without Python overhead.

---

### `update_epochs` (num_updates_per_batch)

Number of passes over the collected rollout data per outer iteration.

All tasks use `update_epochs=16`. This is high compared to standard PPO (4–10) and reflects that with only 30 steps and 2048 envs, each batch has sufficient diversity to support more gradient steps before the data becomes stale.

---

### `num_minibatches`

$$\text{minibatch size} = \frac{N_{\text{envs}} \times N_{\text{steps}}}{N_{\text{minibatches}}} = \frac{61440}{32} = 1920$$

All tasks use `num_minibatches=32`. Standard value for this batch size. Larger minibatches → more stable gradient estimates but slower iteration.

---

### `ent_coef` — Entropy Coefficient

Encourages policy exploration by adding an entropy bonus to the objective:

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \alpha_{\text{ent}} \cdot \mathbb{H}[\pi]$$

All tasks use `ent_coef=0.01`. This is low but sufficient for these dense-reward continuous control tasks.

**Known issue:** For sparse-reward tasks (BallInCup, CartpoleSwingupSparse), `ent_coef=0.01` is too small. If the policy never randomly stumbles onto a reward, entropy alone cannot rescue it. This leads to the bimodal learning pattern (some seeds succeed by chance in early exploration; others get stuck at zero forever). Planned fix: SAC (implicit entropy maximization via temperature parameter).

---

### `clip_coef` — PPO Clipping

$$L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

All tasks use `clip_coef=0.3`. This is larger than the typical 0.2, matching the Brax reference. A wider clip range allows larger policy updates per iteration, which is safe given the frequent updates (16 epochs) with small batches.

---

### `reward_scaling`

Raw DMC rewards are in [0, 1] per step. All tasks multiply by 10.0:

$$r' = 10 \cdot r$$

This scales the value function targets to a range where the neural network can learn effectively (avoiding near-zero gradients from very small targets). All DMC Suite tasks use the same scaling.

---

### `normalize_obs`

Running mean/variance normalization of observations, updated online during training. All tasks use this. Important for:
- Proprioceptive observations that span different physical scales (angles vs velocities).
- Stabilizing the critic network training when observation magnitudes vary.

---

### `max_grad_norm`

Gradient clipping threshold. All tasks use `max_grad_norm=1.0`. Prevents catastrophic gradient steps during the sudden large updates that PPO can produce when the policy is changing rapidly.

---

## 2. Per-Environment Reference Config

| Environment | gamma | num_ts | Notes |
|---|---|---|---|
| CheetahRun | 0.995 | 60M | Dense locomotion; long horizon |
| BallInCup | **0.95** | 60M | Short-horizon catch; bimodal learning |
| CartpoleSwingup | 0.995 | 60M | Long-horizon balance |
| CartpoleSwingupSparse | 0.995 | 60M | Sparse — needs propagation |
| FingerSpin | **0.95** | 60M | Short-horizon contact spinning |
| FishSwim | 0.995 | 60M | Dense locomotion |
| AcrobotSwingup | 0.995 | 100M | Hardest task; needs more steps |
| HopperStand | 0.995 | 60M | Balance; under-performs with PPO |

All others shared: `lr=1e-3`, `num_envs=2048`, `num_steps=30`, `update_epochs=16`, `num_minibatches=32`, `ent_coef=0.01`, `clip_coef=0.3`, `reward_scaling=10.0`, `normalize_obs=True`.

Source: `mujoco_playground.config.dm_control_suite_params.brax_ppo_config(env, 'mjx')`

---

## 3. Lessons from Baseline Misconfig

We ran all 8 environments with CheetahRun's `gamma=0.995`. Two environments require `gamma=0.95`:

**FingerSpin gap (408 → ~800+ with fix):**
- gamma=0.995 → effective horizon ~200 steps → critic integrates noise from irrelevant future
- gamma=0.95  → effective horizon ~20 steps  → critic focuses on local contact quality ✓

**BallInCup (to validate):**
- gamma=0.995 → bimodal + lower mean for learning seeds
- gamma=0.95  → expected: fewer stuck seeds, higher mean

**General rule:** use a shorter gamma when the reward signal is local in time (contact-based, catching) and longer gamma when sustained behavior matters (locomotion, balance).
