# SAC (Soft Actor-Critic) — Reverse Engineering & Design Doc

**Source:** Brax `brax.training.agents.sac`  
**Reference paper:** [Haarnoja et al. 2018, SAC v2 w/ auto-alpha: arxiv 1812.05905](https://arxiv.org/pdf/1812.05905.pdf)

---

## 1. Why SAC over PPO for Sparse/Hard Envs

PPO is on-policy: it discards experience after each update. This causes two problems:

1. **Exploration in sparse envs** — if the policy never stumbles onto a reward by chance in the current batch, no gradient signal arrives. With `ent_coef=0.01` in PPO, entropy is too weak to push the policy to explore aggressively. Result: seeds that don't catch the ball in the first few million steps stay stuck at 0 forever (bimodal learning).

2. **Sample efficiency** — on-policy PPO needs ~75M steps to converge on CheetahRun. SAC (off-policy with replay) typically gets similar performance in 5–10M steps on the same tasks.

SAC's advantages:
- **Replay buffer** — reuses transitions, 10–20× more sample efficient
- **Implicit max-entropy** — the temperature parameter α (automatically tuned) drives sustained exploration at the right scale
- **Twin critics** — reduces overestimation bias (key for continuous action spaces)
- **No clipping** — no PPO ratio clipping artefacts

**Confirmed SAC config from Playground reference:**

| Env group | `num_timesteps` |
|---|---|
| Default (BallInCup, FishSwim, CartpoleSwingup, etc.) | 5M |
| CheetahRun, Finger*, Hopper*, AcrobotSwingup, Swimmer*, HumanoidWalk, WalkerRun | 10M |

Compare: PPO needs 60–75M steps. SAC converges in 5–10M — **6–15× fewer environment steps**.

---

## 2. Algorithm Overview (SAC with automatic temperature tuning)

### 2.1 Three Learned Objects

| Component | Network | Purpose |
|---|---|---|
| Actor π | 2-layer MLP (256×256) + tanh-squashed Normal | Outputs actions |
| Twin Critics Q1, Q2 | 2-layer MLP (256×256) w/ LayerNorm | Estimate Q(s,a) |
| Temperature log α | Scalar | Exploration/exploitation trade-off |

### 2.2 Objectives

**Critic loss** (Bellman error with entropy-regularized bootstrap):
$$\mathcal{L}_Q = \frac{1}{2}\mathbb{E}\left[\left(Q(s,a) - \left(r + \gamma \cdot \underbrace{\left(\min_{i=1,2} Q_i(s',a') - \alpha \log\pi(a'|s')\right)}_{y}\right)\right)^2\right]$$

- Uses **target network** $\bar{Q}$ (EMA update: $\tau=0.005$) for stability
- `min` over twin critics reduces overestimation

**Actor loss** (maximize entropy-augmented Q):
$$\mathcal{L}_\pi = \mathbb{E}_{s,a\sim\pi}\left[\alpha \log\pi(a|s) - \min_i Q_i(s,a)\right]$$

**Alpha (temperature) loss** (auto-tuning to target entropy):
$$\mathcal{L}_\alpha = \mathbb{E}\left[\alpha \cdot (-\log\pi(a|s) - \bar{H})\right]$$

where $\bar{H} = -0.5 \cdot |\mathcal{A}|$ (target entropy = half the action dim).

### 2.3 Training Loop (per step)

```
for each env step:
    1. act: a = π(s) + noise (tanh-Normal sample)
    2. store (s, a, r, s', done) in replay buffer
    
    for k in range(grad_updates_per_step):  # k=8 in reference
        3. sample batch from replay buffer (size=512)
        4. update Q1, Q2 (critic_loss)
        5. update π (actor_loss)
        6. update α (alpha_loss)
        7. soft-update target: Q̄ ← (1-τ)Q̄ + τQ
```

Key difference vs PPO: **multiple gradient updates per env step** (`grad_updates_per_step=8`).

### 2.4 Action Distribution

**TanhNormal** — squashes a Gaussian through tanh to keep actions in [-1,1]:
$$a = \tanh(\mu + \sigma \epsilon), \quad \epsilon \sim \mathcal{N}(0,I)$$

Log-prob correction (change of variables):
$$\log\pi(a|s) = \sum_i \left[\log\mathcal{N}(\cdot) - \log(1 - \tanh^2(u_i) + \epsilon)\right]$$

This is numerically important — without it the entropy signal is wrong.

---

## 3. Brax SAC Architecture

### Files
```
brax/training/agents/sac/
├── train.py      — training loop, replay buffer, evaluation
├── losses.py     — alpha_loss, critic_loss, actor_loss
└── networks.py   — make_sac_networks (policy + twin-Q), make_inference_fn
```

### 3.1 `train.py` — Key Design Decisions

**State struct** (`TrainingState`):
```
policy_params, policy_optimizer_state
q_params, q_optimizer_state
target_q_params          ← separate EMA copy, no grad
alpha_params (log_alpha), alpha_optimizer_state
normalizer_params        ← running obs stats
gradient_steps, env_steps
```

**Replay buffer**: `brax.training.replay_buffers.UniformSamplingQueue`
- Uniform random sampling (not prioritized)
- Max size: configurable (default 4M transitions)
- Min prefill: 8192 transitions before any training
- Sharded across devices (multi-GPU)

**`training_step` flow**:
1. `get_experience` → run 1 env step × `num_envs` → insert into buffer → update normalizer
2. `replay_buffer.sample` → get `batch_size × grad_updates_per_step` transitions
3. Reshape to `(grad_updates_per_step, batch_size, ...)` 
4. `jax.lax.scan(sgd_step, ...)` → runs `grad_updates_per_step` gradient updates in one JIT call

**`sgd_step` update order** (per `grad_updates_per_step` step):
1. alpha update (Adam, lr=3e-4 fixed — separate from actor/critic lr)
2. critic update (using **old** alpha, not new)
3. actor update
4. soft target update: `Q̄ ← (1-0.005)Q̄ + 0.005·Q`

**Truncation handling**:
```python
q_error *= (1 - truncation)   # zero out bootstrap for truncated episodes
```
This is critical for environments with fixed episode lengths — without it, the terminal value is bootstrapped incorrectly.

### 3.2 `losses.py` — Critic Loss Detail

```python
next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob   # twin-Q min
target_q = r * reward_scaling + discount * discounting * next_v
q_error = q_old - target_q
q_error *= (1 - truncation)   # truncation masking
q_loss = 0.5 * mean(q_error²)
```

Note: `discount` in the transition already encodes episode termination (0 at true terminal, 1 otherwise). `discounting` is the γ hyperparameter.

### 3.3 `networks.py` — Architecture

**Policy network**: MLP(obs→256→256→2×action_dim) where output = [μ, log_std]
- Input preprocessed by normalizer
- Optional LayerNorm (not used by default for policy)
- Distribution: `NormalTanhDistribution` (tanh squash)
- `state_dependent_std=False` → log_std is a learned scalar per action dim

**Q network** (twin): MLP([obs,action]→256→256→2) — outputs 2 Q-values (twin critics)
- **LayerNorm enabled** (`q_network_layer_norm=True` in reference config)
- The `axis=-1` in `jnp.min(next_q, axis=-1)` takes the minimum across the 2 twin outputs

---

## 4. Reference Hyperparameters (from `dm_control_suite_params.brax_sac_config`)

| Param | Value | Notes |
|---|---|---|
| `num_timesteps` | 5M (10M for hard envs) | vs PPO's 60M |
| `num_envs` | 128 | vs PPO's 2048 — off-policy doesn't need as many |
| `batch_size` | 512 | Replay sample size per gradient step |
| `grad_updates_per_step` | 8 | Gradient updates per env step |
| `learning_rate` | 1e-3 | Actor + critic Adam LR |
| `alpha lr` | 3e-4 | Fixed (hardcoded in train.py) |
| `tau` | 0.005 | Target network soft update coefficient |
| `discounting` | 0.99 | Same γ for all envs (unlike PPO!) |
| `reward_scaling` | 1.0 | No scaling (unlike PPO's 10×) |
| `min_replay_size` | 8192 | Warmup transitions before training |
| `max_replay_size` | 4M (= `1048576 × 4`) | |
| `normalize_observations` | True | Running mean/var |
| `q_network_layer_norm` | True | Stabilizes critic training |
| `target_entropy` | `-0.5 × action_size` | Lower than typical (-action_size) |

**Notable: SAC uses the same `gamma=0.99` for ALL envs** — unlike PPO which needs per-env gamma (0.95 vs 0.995). This is because SAC's entropy regularization naturally handles the horizon problem that required per-env gamma tuning in PPO.

---

## 5. Our Implementation Plan (`run_sac_mjx.py`)

### 5.1 Architecture

We wrap Brax's SAC trainer directly (same as how our PPO script wraps Brax's internal env interface). This means:
- Zero custom math needed — Brax's losses and networks are battle-tested
- Full JAX JIT compilation on MJX environments
- Identical env wiring as our PPO script (registry + wrap_for_brax_training)

### 5.2 Design Choices

1. **Use `brax.training.agents.sac.train`** directly with mujoco_playground envs via `registry.load` + `wrapper.wrap_for_brax_training`
2. **CSV logging via `progress_fn` callback** — SAC calls this at each eval
3. **Per-env hyperparams** from `brax_sac_config` (automatic lookup)
4. **Override support** for gamma, lr, steps via CLI args

### 5.3 Expected Performance

| Environment | PPO (ours) | PPO (ref) | SAC (ref, 5–10M steps) |
|---|---|---|---|
| BallInCup | 755 (bimodal) | ~950 | ~950 |
| CartpoleSwingupSparse | 413 (bimodal) | — | expected >500 |
| FingerSpin | 408 → 600+ (w/ fix) | ~600 | ~600 |
| HopperStand | 84 (bimodal) | ~300 | expected ~300 |
| CheetahRun | 887 | ~900 | ~900 |

---

## 6. File Map

```
helios-rl/
├── scripts/
│   └── run_sac_mjx.py           ← SAC training script (our impl)
├── exp/
│   └── ppo/
│       └── csv/
│           └── ours_sac_*.csv   ← SAC results (when run)
└── docs/
    └── ppo/
        ├── sac_design_doc.md    ← this file
        └── ppo_hyperparameter_guide.md
```
