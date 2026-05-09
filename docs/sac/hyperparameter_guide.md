# SAC Hyperparameter Guide — DMC Suite

Why different hyperparameters matter for SAC, what each one controls, and what we learned from experiments.

---

## 1. Core Hyperparameters

### `gamma` — Discount Factor

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

The effective planning horizon is:

$$H_{\text{eff}} \approx \frac{1}{1 - \gamma}$$

| gamma | Effective horizon |
|-------|-----------------|
| 0.99  | ~100 steps |
| 0.995 | ~200 steps |
| 0.95  | ~20 steps |

**SAC reference config uses `gamma=0.99`** for all DMC Suite tasks (vs PPO's 0.995 for most).

The slightly shorter horizon in SAC vs PPO is appropriate: off-policy replay means the value function is trained on a wide distribution of states, so longer horizons don't add variance the way they do in PPO's on-policy setting.

---

### `learning_rate` — Adam LR for Actor, Critic

SAC reference uses `lr=1e-3` for both actor and critic. Temperature uses `lr=3e-4`.

**Effect on stability:**
- `lr=1e-3` (standard): works well with `g/step=8`, small-medium networks
- Higher LR + more gradient steps = faster learning but collapse risk
- We tested `g/step=16` with 256×2 networks at `lr=1e-3`: initial fast learning (148 @ 1M) followed by monotonic decline (262 @ 10M) — policy collapse

**Rule:** If increasing `grad_updates_per_step`, lower `lr` proportionally or use larger networks.

---

### `grad_updates_per_step` — Gradient Updates per Env Step

This is the primary "data efficiency" knob in SAC. Default: **8**.

$$\text{total gradient steps} = \text{env steps} \times \text{grad\_updates\_per\_step} = 10\text{M} \times 8 = 80\text{M gradient steps}$$

| g/step | Observations |
|--------|-------------|
| 8 | Reference value; stable across seeds and network sizes |
| 16 | Fast early learning, but 256×2 collapses by 6–8M steps |
| 16 | With 512×2: untested (would be less prone to collapse due to larger network) |

**Why collapse happens with g/step=16:**
- Too many gradient steps per transition exhausts the information in each sample
- Q-value overestimation accumulates faster than the target network can correct
- The rising α (temperature) is a symptom: actor entropy increases as Q estimates become unreliable
- Result: policy drifts toward high-entropy noise

**Key lesson:** Do not increase `grad_updates_per_step` beyond 8 without also increasing network capacity or lowering `lr`.

---

### `batch_size` — Replay Sample Size

Default: **512** transitions per gradient step.

SAC gradients are computed on independent samples from the replay buffer — no temporal structure required. Larger batch = lower variance gradient, but:
- Memory: 512 × (obs + action + reward + next_obs + done) per step
- Speed: little impact since GPU batching saturates quickly

**Rule:** 512 is well-tuned for this problem scale. Increasing to 1024 is safe but rarely necessary.

---

### `min_replay_size` — Buffer Warmup

Default: **8,192** transitions.

The agent takes random actions until the buffer contains this many transitions, then begins gradient updates. This ensures early batches are reasonably diverse.

**Effect:** Warmup takes `8192 / (num_envs × 1) = 8192 / 128 = 64 env steps` ≈ negligible time.

Too small: early batches are near-identical (low diversity) → critic divergence.  
Too large: wastes time with random policy.

---

### `max_replay_size` — Buffer Capacity

Default: **4,194,304** (4M transitions).

At 10M total steps with 128 envs and 1 step/iter, we insert ~10M transitions but the buffer only holds 4M. The oldest data is discarded. This means:
- By mid-training, the buffer contains only the last 4M steps
- The effective data age is `4M / 128 envs ≈ 31K env steps` of history

Larger buffer = more off-policy data = potentially slower adaptation but more diversity.

---

### `tau` — Soft Target Update Rate

$$\bar{Q} \leftarrow (1 - \tau)\bar{Q} + \tau Q$$

Default: **0.005**. Applied after every gradient step.

This controls how quickly the target network tracks the online critic. Too fast (large τ) = target instability; too slow (small τ) = stale targets and slow learning.

0.005 is near-universal in SAC literature. No need to tune.

---

### `target_entropy` — SAC Entropy Target

$$\bar{H} = -0.5 \times |\mathcal{A}|$$

Default: half the (negative) action dimension. For HopperStand (4-dim): $\bar{H} = -2.0$.

The temperature α is automatically adjusted to make the policy's entropy match $\bar{H}$:
- If $H[\pi] > \bar{H}$: α decreases (less exploration pressure)
- If $H[\pi] < \bar{H}$: α increases (more exploration pressure)

**Alternative values tested:**
- `target_entropy=-4.0` (full negative action dim): more aggressive exploration — designed for sparse-reward tasks, not beneficial for HopperStand where random exploration is already sufficient

**Rule:** Use the default for dense-reward locomotion tasks. Use `target_entropy = -action_dim` (more negative) for sparse tasks (BallInCup, CartpoleSwingupSparse).

---

### `reward_scaling`

Default: **1.0** for all SAC DMC tasks (vs 10.0 for PPO).

SAC's critic directly learns Q-values in the original reward scale. Unlike PPO's value function which only needs to learn relative advantage, SAC Q-values are used in absolute form for the entropy trade-off. Scaling by 10× would require rescaling the temperature α accordingly — easier to leave at 1.0.

---

### `normalize_observations`

Default: **True** for all DMC Suite tasks.

Running mean/variance normalization stabilizes training when observations span different physical scales (joint angles in radians vs. velocities in m/s vs. contact forces in N).

**Implementation detail:** The official Brax SAC uses a fully XLA `RunningStatisticsState` that lives on GPU. Our custom implementation does the stat update on CPU (one transfer per iteration) and pushes normalized constants to JAX — a minor overhead (~100K step difference in throughput).

---

## 2. Network Architecture

### Hidden Layer Size

| Size | Parameters | g/step=8 stability | g/step=16 stability |
|------|-----------|-------------------|---------------------|
| 256×2 (official) | ~130K | ✓ stable | ✗ collapses |
| 512×2 (custom) | ~530K | ✓ stable | Not tested |

**Observed effect of 512×2 vs 256×2 (same g/step=8):**
- @1M: 42 vs 13 (3× better early learning)
- @2M: 222 vs 6 (37× better!)
- @5M: 506 vs 589 (official catches up)
- @10M: 645 vs 841 (official wins in final phase)

The larger network learns the standing policy faster but appears to over-fit the early replay buffer, while 256×2 continues improving late. This is consistent with the "capacity overshoot" phenomenon in off-policy RL with finite replay.

### LayerNorm

Applied to Q networks only (not actor). This reduces Q-value overestimation by normalizing intermediate activations, preventing the critic from assigning extreme values that mislead the actor.

**Rule:** Keep `q_network_layer_norm=True`. Removing it typically degrades performance on locomotion tasks.

---

## 3. Per-Environment Config (from MuJoCo Playground reference)

```python
from mujoco_playground.config import dm_control_suite_params
ref = dm_control_suite_params.brax_sac_config(env_id)
```

| Environment | num_timesteps | num_envs | g/step | batch | gamma | Notes |
|-------------|--------------|----------|--------|-------|-------|-------|
| BallInCup | 5M | 128 | 8 | 512 | 0.99 | Sparse; bimodal |
| CartpoleSwingupSparse | 5M | 128 | 8 | 512 | 0.99 | Sparse; bimodal |
| HopperStand | 10M | 128 | 8 | 512 | 0.99 | Dense; high variance |
| FingerSpin | 10M | 128 | 8 | 512 | 0.99 | Not yet benchmarked |
| CheetahRun | 10M | 128 | 8 | 512 | 0.99 | Dense locomotion |
| AcrobotSwingup | 10M | 128 | 8 | 512 | 0.99 | Sparse |

All share: `lr=1e-3`, `tau=0.005`, `min_replay_size=8192`, `max_replay_size=4194304`, `normalize_observations=True`, `reward_scaling=1.0`, `episode_length=1000`, `num_evals=10`.

---

## 4. Seed Variance Analysis (HopperStand, Official Brax SAC)

HopperStand exhibits extreme seed dependence:

| Seed | @1M | @5M | @10M | Notes |
|------|-----|-----|------|-------|
| 1 | 13 | 589 | **841** | Slow start, late surge |
| 2 | 13 | 26 | **90** | Failed to find standing |
| 3 | 4 | — | **922** | Best seen; fast convergence |

The variance is driven by whether the agent discovers the standing strategy before the replay buffer fills with failed episodes. Once a standing trajectory is in the buffer, the critic quickly propagates the value signal and the policy improves rapidly.

**Implication:** Comparing single seeds is unreliable. With official SAC:
- seed=1 gives 841 (our reference)
- seed=2 gives 90 (completely different)
- seed=3 gives 922 (exceeds our reference)

**For custom SAC (512×2, g/step=8, seed=1):** 653 — worse than official seed=1, but with a smoother, monotonically increasing curve (42→222→400→653). The 512×2 network appears to find a local optimum that's harder to escape.

---

## 5. Sparse Reward Environments (BallInCup, CartpoleSwingupSparse)

These require particular attention because SAC's replay buffer can become saturated with zero-reward transitions.

**BallInCup results (official, 5M steps):**
- Seed 1: 0 (never caught)
- Seed 2: **962** (caught at 1.1M, maintained)
- Seed 3: **965** (similar)
- Seed 4: **970** (similar)
- Seed 5: 0 (never caught)

**CartpoleSwingupSparse results (official, 5M steps):**
- Seed 1: **837** (solved at ~1.1M steps)
- Seed 2: 0 (never solved)
- Seed 3: 0 (never solved)
- Seed 4: **797** (solved at ~2.2M steps)
- Seed 5: **800** (solved at ~1.7M steps)

**Pattern:** Exactly 3/5 seeds solve each sparse-reward task. The bimodal distribution is structural: once the buffer is dominated by zero-reward transitions, the critic assigns near-zero Q-values everywhere, and the policy has no gradient signal to improve.

**Potential fixes (not yet implemented):**
- Higher `target_entropy` (more exploration)
- Prioritized experience replay (oversample the rare rewarded transitions)
- Curriculum (pre-warm buffer with a simpler policy or shaped reward)
- More seeds (accept the 60% solve rate)
