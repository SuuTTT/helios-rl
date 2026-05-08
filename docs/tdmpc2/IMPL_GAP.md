# TD-MPC2: Our Implementation vs Official Code — Gap Analysis

**Our script**: `helios-rl/scripts/train_tdmpc_hopper_v4.py`  
**Official code**: `tdmpc2/tdmpc2/` (PyTorch, upstream repo)  
**Reference results file**: `tdmpc2/results/tdmpc2/hopper-hop.csv`

Each gap is tagged **[PERF]** if it is likely to affect task performance or **[INFRA]**
if it is an engineering/tooling gap that is unlikely to change final scores.

---

## Gap 1: Reward and Q Loss — MSE vs Two-Hot Categorical [PERF, HIGH IMPACT]

### Official
Reward and Q targets are encoded via **two-hot encoding on a symlog-scaled bin grid**:
```python
# math.py
def two_hot(x, cfg):
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax)
    # distributes mass between two adjacent bins
    ...

reward_loss += math.soft_ce(rew_pred, rew_target, cfg).mean() * rho**t
value_loss  += math.soft_ce(q_pred,   td_target,  cfg).mean() * rho**t
```
- `symlog(x) = sign(x) * log(1 + |x|)` compresses large values and expands small ones.
- The scalar target becomes a soft probability distribution over `num_bins` bins.
- Loss is cross-entropy (log-softmax + dot product), not MSE.
- `num_bins` default: 101 bins, `vmin=-20, vmax=20` in symlog space.
- This is the **HL-Gauss / distributional regression** trick from Dreamer v3.

**Why it matters**: Cross-entropy on a two-hot target is equivalent to distributional
regression. It sidesteps the gradient-scale problem entirely — there is no need for
`rew_scale` because cross-entropy gradients do not shrink with small rewards.
The symlog transform also stabilises Q regression when Q values span many orders
of magnitude across tasks (e.g., 0.01–500).

### Ours
```python
rl = w * jnp.mean((pr - rew_scale * r_t) ** 2)          # MSE with manual scale
vl = w * jnp.mean(jnp.sum((qp - td[:, None]) ** 2, -1)) # MSE
```
Plain MSE + manual `rew_scale=10.0` hack.

### Fix required
Implement `symlog`, `two_hot(x, vmin, vmax, num_bins)`, `soft_ce`, replace both
loss terms. Remove `rew_scale` entirely once two-hot is in place.

---

## Gap 2: Policy — Deterministic vs Stochastic Gaussian [PERF, HIGH IMPACT]

### Official
Stochastic Gaussian policy with entropy bonus and tanh-squashing correction:
```python
# world_model.py — pi()
mean, log_std = self._pi(z).chunk(2, dim=-1)
log_std       = math.log_std(log_std, log_std_min, log_std_dif)  # clamp in tanh space
eps           = torch.randn_like(mean)
action        = mean + eps * log_std.exp()
mean, action, log_prob = math.squash(mean, action, log_prob)      # tanh + log-prob correction

# tdmpc2.py — update_pi()
qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
self.scale.update(qs[0])
qs = self.scale(qs)
rho = torch.pow(cfg.rho, torch.arange(len(qs)))
pi_loss = (-(cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
```
Key components:
- `log_std` is clipped/rescaled via `log_std_min, log_std_dif` (bounded exploration).
- `math.squash` applies tanh and corrects log-prob: `log_pi -= log(1 - tanh²(a) + ε)`.
- Policy loss includes **entropy bonus**: `entropy_coef × H[π]` added to Q.
- `RunningScale` normalises Q before policy gradient (5th–95th percentile range).
- **Separate optimizer** `pi_optim` with `eps=1e-5`, updated after world model step.

### Ours
```python
class Pi(nn.Module):
    def __call__(self, z): return jnp.tanh(NormMLP(self.hidden, self.action_dim)(z))

pl = -w * jnp.mean(jnp.min(q_net.apply(stop_grad(params["q"]), stop_grad(z_t), pi2), -1))
```
Deterministic tanh policy, no entropy, no Q normalisation, same optimizer as world model.

### Consequence
- No entropy bonus → policy collapses to argmax earlier, less exploration.
- No Q normalisation → policy gradient magnitude varies unpredictably as Q scale grows.
- Shared optimizer: policy gradient clips interfere with world model.
- Our `pi` metric shows strong policy learning (pi≈240) because the policy is much
  simpler (deterministic) — but this doesn't correspond to the full exploration benefit.

### Fix required
1. Change Pi output to `2×action_dim`, split into `mean` and `log_std`.
2. Add `math.squash` equivalent in JAX.
3. Add entropy coefficient: `pi_loss = -(entropy_coef * entropy + Q_avg)`.
4. Add `RunningScale` (running percentile normaliser on Q before pi update).
5. Separate `pi_optim` with its own optax state.

---

## Gap 3: Q Ensemble Size — 2 vs 5 [PERF, MEDIUM IMPACT]

### Official
```python
self._Qs = layers.Ensemble([
    layers.mlp(...) for _ in range(cfg.num_q)  # default num_q = 5
])
# random 2-subset for min/avg at each call:
qidx = torch.randperm(cfg.num_q)[:2]
Q = math.two_hot_inv(out[qidx], cfg)
```
5 Q-networks in a vectorised ensemble via `torch.vmap`. At each call, **2 are
randomly selected** (not always the same pair). This is the "random ensemble
distillation" approach: training all 5 but evaluating a random subset reduces
overestimation bias more robustly than fixed twin critics.

### Ours
2 hardcoded Q-heads. Always uses min of the same two. More prone to overestimation
and Q spikes (seen as `v=174` at 7M). With only 2 Q-networks there is no
anti-correlated diversity.

### Fix required
Parameterise `num_q`, increase to 5. Add random 2-subset selection at each call.
Use JAX `vmap` over a stacked parameter array for the ensemble.

---

## Gap 4: MPPI — Elite Selection vs Full Softmax [PERF, MEDIUM IMPACT]

### Official
```python
elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices  # top 64 of 512
elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
max_value = elite_value.max(0).values
score = torch.exp(cfg.temperature * (elite_value - max_value))
score = score / score.sum(0)
mean = (score * elite_actions).sum(1) / (score.sum(0) + 1e-9)
std  = sqrt(...variance of elite_actions...)
std  = std.clamp(cfg.min_std, cfg.max_std)
```
Key features:
- Selects `num_elites` top trajectories (default 64 from 512 samples) before weighting.
- Updates **both** mean and std from the elite distribution.
- `std` is updated per-iteration and clamped to `[min_std, max_std]` range.
- Larger default sample count: `num_samples=512` (vs our 256).
- More default iterations: `iterations=6` (we also use 6).
- Final action: **Gumbel-Softmax sample** from elite scores (stochastic, not argmax).

### Ours
```python
w = jax.nn.softmax((rets - rets.max()) / (temp + 1e-8))  # all 256 samples
new_mu = jnp.einsum("n,nha->ha", w, acts)                # no std update
```
All 256 samples enter softmax. Std is fixed at 0.5 throughout. No elite selection,
no variance update. Mean only (no Gumbel sampling).

### Consequence
Fixed std=0.5 does not adapt to convergence — in early training noise is too small
relative to the action space; in late training it may be too large. Elite selection
reduces contamination from very bad trajectories.

### Fix required
1. Add elite selection: `topk(rets, num_elites)`.
2. Add dynamic std update from elite variance.
3. Add `min_std / max_std` clamping on std.
4. Optional: Gumbel-softmax final action sample.

---

## Gap 5: MPPI Number of Pi Trajectories — 1 vs 24 [PERF, LOW-MEDIUM IMPACT]

### Official
```python
if cfg.num_pi_trajs > 0:  # default num_pi_trajs = 24
    pi_actions = torch.empty(horizon, num_pi_trajs, action_dim)
    _z = z.repeat(num_pi_trajs, 1)
    for t in range(horizon-1):
        pi_actions[t], _ = model.pi(_z, task)  # stochastic — samples
        _z = model.next(_z, pi_actions[t], task)
    pi_actions[-1], _ = model.pi(_z, task)
# These 24 trajectories fill the first 24 of 512 sample slots
actions[:, :num_pi_trajs] = pi_actions
```
24 **stochastic** pi trajectories (different noise samples each) injected at the
start of the sample set. Provides a strong baseline while still covering the
stochastic action distribution.

### Ours
1 deterministic pi trajectory, injected as the **last** sample. Also sets `mu[0]`.

### Consequence
With 24 stochastic pi-trajectories, the best of them reliably represents at least
the policy's expected performance. With 1 deterministic trajectory, there is only a
single baseline candidate — if the policy is slightly suboptimal at t=0, there is
no fallback.

---

## Gap 6: Policy Loss Q Normalisation — Missing RunningScale [PERF, MEDIUM IMPACT]

### Official
```python
qs = model.Q(zs, action, task, return_type='avg', detach=True)
self.scale.update(qs[0])   # update 5th-95th percentile running estimate
qs = self.scale(qs)         # divide Q by running scale before loss
pi_loss = (-(entropy_coef * entropy + qs)).mean()
```
`RunningScale` tracks the 5th–95th percentile range of Q values and divides the
policy gradient by it. This keeps the effective policy learning rate constant
regardless of Q magnitude. As Q grows (from ~5 to ~500 in scaled space over
training), the policy gradient would otherwise grow by 100× — causing instability.

### Ours
Raw Q values in policy loss with no normalisation. As Q grows over training,
policy gradients blow up and need to be absorbed by gradient clipping (`clip_norm=10`).
This is a coarser mechanism.

---

## Gap 7: Weight Initialisation [PERF, LOW IMPACT]

### Official
```python
# init.py
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

# world_model.py
self.apply(init.weight_init)
init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
```
- All linear layers: truncated normal with std=0.02 (very small init).
- Reward and Q **output** layers: **zeroed weights**. This means both heads start
  predicting exactly 0 — stable bootstrap early in training, avoids early overestimation.

### Ours
Default Flax/JAX init (Glorot uniform). Output layers start at non-zero random values.

### Consequence
Non-zero reward/Q at step 0 → first few thousand TD updates may be noisy. The
offset should wash out after warmup but zero-init would make the learning trajectory
smoother.

---

## Gap 8: UTD Ratio — 1:16 vs 1:1 [PERF, HIGH IMPACT — already known]

### Official
Single environment, 1 gradient update per env step → UTD = 1:1.
At 4M env steps: **4M gradient updates**.

### Ours
N=1024 envs, K=64 updates per global step → UTD = 64/1024 = **1:16**.
At 4M env steps: **250k gradient updates**.

Gradient deficit at 4M: 16×. This is the dominant gap (138 vs 449 MPPI at 4M steps).

See ITERATION_LOG.md gap analysis for the fix: N=256, K=256 → UTD=1:1 at same wall time.

---

## Gap 9: Discount Factor Computation [PERF, LOW IMPACT]

### Official
```python
def _get_discount(self, episode_length):
    frac = episode_length / cfg.discount_denom   # discount_denom=700 default
    return min(max((frac-1)/frac, cfg.discount_min), cfg.discount_max)
    # For episode_length=1000, discount_denom=700:
    # frac=1000/700=1.43; (1.43-1)/1.43=0.30/1.43≈0.30 → clamp to [0.95, 0.995]
    # → gamma ≈ 0.995 for episode_length=1000
```
Gamma scales with episode length. Longer episodes → higher discount.

### Ours
Fixed `GAMMA=0.99`. For episode_length=1000, official would use ≈0.995.
Lower gamma → shorter effective horizon → slightly less credit assignment.

---

## Gap 10: Activation Function — SiLU vs Mish [PERF, NEGLIGIBLE]

### Official
`nn.Mish(inplace=False)` as the default activation in all NormedLinear layers.

### Ours
`nn.silu` (Swish: `x * sigmoid(x)`). Mish is `x * tanh(softplus(x))`.

Both are smooth, non-monotonic activations. The difference in practice is negligible
(< 1% on most benchmarks). Not a meaningful gap.

---

## Gap 11: First-Step MPPI Reset Handling [PERF, NEGLIGIBLE]

### Official
```python
if not t0:
    mean[:-1] = self._prev_mean[1:]
# If t0: mean stays zeros → fresh start at episode beginning
```
At `t0=True` (first step of episode), mu is reset to zeros to avoid carrying
over stale plans from the previous episode.

### Ours
Always applies receding horizon shift including `pi_traj[-1]` — no reset at t0.
At episode start, the shift carries over the previous episode's last plan.

### Consequence
Tiny effect — only the first planning call per episode. The pi warm-start
(`mu[0] = pi_traj[0]`) mitigates the issue somewhat.

---

## Gap 12: Encoder Learning Rate Scale [INFRA]

### Official
```python
self.optim = torch.optim.Adam([
    {'params': model._encoder.parameters(), 'lr': cfg.lr * cfg.enc_lr_scale},
    {'params': model._dynamics.parameters()},  # lr = cfg.lr
    ...
])
```
`enc_lr_scale` (default 0.3 for state observations, 1.0 for pixels). Encoder
learns at 30% the speed of other networks for state-based tasks.

### Ours
Same LR for encoder as all other networks via `multi_transform` ('world' group).

---

## Gap 13: Q Dropout [INFRA]

### Official
```python
layers.mlp(..., dropout=cfg.dropout)  # dropout on first layer of each Q-head, default ~0.01
```

### Ours
No dropout anywhere.

---

## Summary Table

| Gap | Component | Severity | Lines to change |
|-----|-----------|----------|-----------------|
| **1** | Reward+Q loss: MSE → two-hot CE | 🔴 HIGH | `step_loss`, add `symlog`, `two_hot`, `soft_ce` |
| **2** | Policy: deterministic → stochastic Gaussian + entropy | 🔴 HIGH | `Pi` class, `step_loss` pi section, separate optimizer |
| **3** | Q ensemble: 2 → 5, random subset selection | 🟠 MED | `QEnsemble`, `step_loss`, MPPI |
| **4** | MPPI: full softmax → elite selection + dynamic std | 🟠 MED | `make_mppi_fn` |
| **5** | MPPI pi trajectories: 1 → 24 stochastic | 🟡 LOW | `make_mppi_fn` |
| **6** | Policy loss: no Q normalisation | 🟠 MED | add `RunningScale`, `update_pi` section |
| **7** | Weight init: glorot → trunc_normal + zero output heads | 🟡 LOW | module `__call__` or post-init |
| **8** | UTD: 1/16 → 1/1 | 🔴 HIGH | `N_ENVS`, `K_UPDATE` (already documented) |
| **9** | Gamma: 0.99 → episode-length-scaled | 🟢 NEGL | `GAMMA` config |
| **10** | Activation: SiLU → Mish | 🟢 NEGL | `NormMLP` |
| **11** | MPPI t0 reset | 🟢 NEGL | `plan()` |
| **12** | Encoder LR scale | ⚪ INFRA | optimizer setup |
| **13** | Q dropout | ⚪ INFRA | `QEnsemble` |

---

## Recommended Fix Priority

### Phase 1 — Biggest bang for effort (expected +100–150 MPPI at 4M)
1. **UTD fix**: N_ENVS=256, K_UPDATE=256 (no code change, just config).
2. **Two-hot reward/Q loss**: Removes the `rew_scale` hack; more robust regression.
3. **Stochastic policy + entropy**: Enables proper exploration; prevents early collapse.

### Phase 2 — Diminishing returns (+30–50 MPPI)
4. **RunningScale on Q before pi update**: Stabilises policy gradient as Q grows.
5. **Q ensemble 5 + random subset**: Reduces Q overestimation spikes (v≈174 problem).
6. **MPPI elite selection + dynamic std**: Better MPPI sample efficiency.

### Phase 3 — Fine-tuning (< +10 MPPI)
7. Zero-init output heads.
8. 24 stochastic pi trajectories in MPPI.
9. Encoder LR scale.
10. Gamma scaling.
