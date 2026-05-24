# TD-MPC-Glass HopperHop — Iteration 3 idea evaluation and roadmap

Date: 2026-05-14, after Phase-j seed 1 plateaued at MPPI≈380 (peak 380 at 2.25M)
and the video analysis revealed seed 3 / seed 4 are *knee-walking* not
*foot-hopping*. This doc evaluates 7 proposed directions, ranks them by
expected payoff per unit work, and proposes the order to try them.

The key thing we learned this round: **the bottleneck for stuck HopperHop seeds
is policy *technique* (knee-walk vs foot-hop), not Glass representation
quality.** Glass found valid K=4 partitions for seed 3 too, but the policy
converged to a low-energy crawl instead of a hop.

---

## Idea 1 — K-flicker penalty (penalise short clusters)

**Hypothesis.** Healthy gaits hold each cluster for ≥5 frames; bad gaits flicker
between clusters in 1-2 frames. From your video notes: seed 4 cycles K=0↔K=5
in tight oscillation, seed 1 holds each cluster for ~12-20 frames per hop
phase. So penalising "short clusters" should be a behavioural prior for healthy
gait without baking in the foot-vs-knee distinction explicitly.

**Implementation options.**

(a) **Cheapest** — increase the existing `lambda_temporal` in Glass loss. The
current term is `||Sᵀ c_src - sg(Sᵀ c_next)||²` averaged over the H-step
rollout (default `λ_temporal = 1e-3`). Raising to `1e-2` or `5e-2` strengthens
the "consecutive cluster labels should match" signal. **Cost: 1 CLI flag, 0
new lines of code. Risk: if too strong, will collapse partition.**

(b) **Sharper** — add a new policy-side penalty on the encoder/policy that
fires when the soft prototype assignment `c_t` differs from `c_{t-1}` *and*
the consistency loss says we're in the middle of a rollout (not a phase
boundary). Differentiable through soft assignments. **Cost: ~15 LOC in
`tdmpc_glass.py` loss_fn, similar to the smoothing term. ~30 min.**

(c) **Direct** — penalise discrete K-label changes per step, but the argmax is
non-differentiable. Would need a Gumbel-softmax surrogate. **Cost: nontrivial.
Skip unless (a)/(b) underperform.**

**Expected payoff.** Moderate-high. The visual is convincing and the
implementation is very cheap. Likely improves stuck seeds because the knee-walk
gait is *intrinsically flickery* — short cluster runs are how the bad gait
looks.

**Risk.** Could rigidify policy too early in training, preventing exploration.
Mitigation: use the same curriculum trick (off pre-warmup, on after).

---

## Idea 2 — K=4 may have a ceiling around 550-650; try hierarchical super-clusters

**Hypothesis.** Phase-f seed 1 peaked at 571. Blog §5.5 says K=4 seeds avg
403.7 (in original Phase 1 setup). DMC HopperHop max return is ~1000 — there
is room above 571. K=4 may itself be a structural ceiling for *this*
clustering depth; a finer partition (e.g. K=8 with hierarchical
super-clusters) could resolve sub-gait-phase distinctions and unlock higher
peaks.

**Implementation.** Add a second `assign_logits_super` of shape `(K, K_super)`
in `init_glass_params`. Compute `S_super = softmax(...)`. Apply 2D structural
entropy on the super-cluster partition (`S_super`) instead of (or in addition
to) the existing K-level partition. **Cost: ~30 LOC in
`src/helios/algorithms/tdmpc_glass.py`. ~45 min.**

**Expected payoff.** Low-medium. The blog already chose N=16, K=8 with 4
active because empirically 4 is what the DMC HopperHop gait needs. Going
hierarchical may not produce more *active* super-clusters — could just shuffle
parameters with no representational gain.

**Risk.** Doubles the prototype/cluster head parameter count. Could destabilise
training. Worth a one-seed test before commitment.

---

## Idea 3 — VLM → semantic cluster labels → symbolic reasoning

**Hypothesis.** Right now the cluster overlay tells us "K=3" but we have to
manually label that as "push-off" by watching the video. If a VLM (Claude,
GPT-4V, LLaVA) does this automatically per-seed, we get a stable symbolic
representation. Then we can impose **sequence constraints** during MPPI:
"plans that go push-off → push-off (no flight) are rejected".

**Implementation.** Heavy.
1. Render a short clip per cluster (e.g. 20 frames where K=3 was active).
2. Send to VLM with prompt "label each cluster with one of: stance / push-off
   / flight / landing / lying / recovery / other".
3. Build a per-seed lookup `cluster_id → symbol`.
4. Modify MPPI to score plans by both reward AND symbolic transition validity.

**Cost.** ~3-5 days for a working pipeline including VLM inference, label
stability, and MPPI constraint integration.

**Expected payoff.** Uncertain. The labels themselves don't change training —
they only constrain *planning*. Effective payoff = "MPPI gets better plans" =
"final MPPI > peak by some amount". If MPPI was already weak (it is, sometimes
< pi as we saw in Phase-j 1.5M), constraining it might just reject more good
plans without finding better ones.

**Risk.** High. Symbolic constraints are brittle. VLM labels may be unstable
across episodes for the same cluster.

**Recommendation: defer.** Try cheaper interventions first. Revisit if we plateau
again after Ideas 1-4.

---

## Idea 4 — MPPI in cluster space (planning with abstraction)

**Why MPPI is sometimes worse than pi.** Phase-j seed 1 at 1.5M had pi=326,
MPPI=272 — planner *underperformed* the reactive policy. Likely cause: MPPI
rolls H steps under the learned `dyn` network; small per-step prediction
errors compound over H=3 steps; if the dynamics model is inaccurate at
states the policy actually visits, MPPI's terminal Q estimate is unreliable.
The policy is reactive (no rollout) so it doesn't compound.

**Idea: plan in cluster space, not latent space.** Instead of MPPI sampling
H-step latent rollouts, sample H-step *cluster* rollouts (5-state HMM with
~10 transitions visible in Phase-f seed 1's videos), evaluate value of each
cluster sequence via a separately-learned `V_cluster(K_t, K_{t+1}, ...)`
head, pick the best cluster sequence, then constrain MPPI samples to follow
that high-level plan.

**Implementation.** Heavy. Two-level planning: cluster-MPC + action-MPC. New
value head, new sampling logic.

**Cost.** ~1 week.

**Expected payoff.** Medium. Addresses a real failure (MPPI < pi) but the
gain from "plan in abstract space" only materialises if the abstract value
function is *better than the latent value function at long horizons*. Plausible
but unproven for HopperHop.

**Recommendation: defer.** Test cheaper post-hoc constraints (Idea 3 in light
form: just whitelist legal cluster transitions) before doing two-level planning.

---

## Idea 5 — Why μ, why c, why P? Alternative Glass integrations

The blog §10.1 cites the lineage: μ (prototypes) from SwAV/Prototypical Networks,
c (soft cosine assignment) from MoCo/SimCLR, P (transition matrix via
c_src ⊗ c_next outer product) is just the natural extension to *sequential*
data, S (prototype→cluster coarsening) from VQ-VAE-2/Hi-SwAV.

**Reasonable alternatives you asked about:**

### 5a. "Use the TD-MPC codebook" — but there isn't really one.
The "codebook" the user is thinking of is probably one of two things in TD-MPC2:
- **Two-hot bins** (the 101-bin distributional Q/reward). These are a value
  discretisation, not a state codebook. Each bin is a *return value*, not a
  *state*. Not usable for behavioural clustering.
- **SimNorm groups** (the encoder's final activation, 8 groups of 64 codes).
  *This* is genuinely codebook-like — each sample's z lies on a product of 8
  simplices. Argmax within each group gives 8 discrete categorical variables
  (each ∈ 0..63), which is the closest thing to "free clusters" in current
  TD-MPC2.

### 5b. "Skip μ entirely, use SimNorm groups as the clustering"
Concrete proposal: replace
```
sim   = ẑ @ μ̂ᵀ          (N=16,)
c     = softmax(sim/T)
n_star = argmax(c)
```
with
```
group_argmaxes = [argmax(z[64*g:64*(g+1)]) for g in range(8)]   # one int per group
# treat the tuple (a_0, ..., a_7) as the cluster id
```
Now there are 64^8 possible "clusters" theoretically, but in practice the
encoder only visits a few combinations. **Pro: zero extra parameters, zero
extra losses, partition emerges naturally from the SimNorm geometry.** **Con:
the partition is not directly controllable, and the SE machinery would have to
be adapted to a higher-arity discrete space.**

**Cost.** ~1 day to prototype + a smoke run.

**Expected payoff.** Low-medium. The hypothesis is "Glass's μ/c/P pipeline is
redundant given SimNorm". Worth a small ablation but probably not the headline
fix.

### 5c. "Why is N=16?"
Heuristic: gives ~2 prototypes per active cluster (4 active out of 8 cluster
slots). At N=16 the structural-entropy graph is 16×16 (cheap), at N=64 it
would be 64×64 (still cheap), at N=512 it's the SimNorm dimension (= no
coarsening at all).

We never properly swept N. **Cheap experiment: try N=8, N=32, N=64 and see if
peak return moves.**

---

## Idea 6 — Drop Glass entirely; ablation on the smoothing term

**The most informative single experiment we could run right now.** From Phase-f:
smoothing flipped seed 1's policy from knee-walk to foot-hop. **Was Glass
necessary?** If we run `--algos tdmpc2` (no Glass) with
`--latent_action_smooth_coef 1e-3` and get the same surge, **Glass was passive
all along** — just adding compute without contributing.

**Implementation.** 0 LOC. Already supported (latent_action_smooth_coef is in
`make_update_fn` of tdmpc_glass.py — but the tdmpc2.py path doesn't have it
yet; one-line port). **Cost: 5 min code + 3h run.**

**Expected payoff.** **VERY HIGH** as a diagnostic. Three possible outcomes:
1. Smoothing alone produces the seed-1 surge → Glass is not needed; this
   becomes the new baseline; we can ditch all Glass complexity.
2. Smoothing alone falls back to Phase 1b's stuck seeds → Glass IS necessary;
   the win is real and we should keep iterating on the Glass+smoothing combo.
3. Smoothing alone gets a *different* shape (e.g. all seeds at 400) → tells
   us how Glass biases the basin distribution.

**Strongly recommend running this first** before any of the other ideas.

---

## Idea 7 — Hybrid abstraction + raw latent (curriculum on Glass loss weight)

**Concern.** Abstractions help early learning ("structure the latent space")
but may block late learning ("the partition is now an inductive bias the
policy can't unlearn"). Peak 571 on seed 1 may be exactly such a ceiling.

**Idea.** Schedule the Glass loss weights to **decay over training**. High
early (helps basin lock + technique discovery), zero late (lets the latent
fully refine for value/dynamics quality). E.g.
`λ_se(env_steps) = 5e-3 * max(0, 1 - env_steps / 2_000_000)` — full strength
for the first 1M, linear decay to 0 by 2M, then pure TD-MPC2 thereafter.

**Implementation.** Same pattern as the curriculum smoothing we just built
(re-JIT at threshold). The Glass `lambda_*` params would be the scheduled
ones. **Cost: ~30 min of code.**

**Expected payoff.** Medium. If the ceiling is at "Glass partition fixes
something the dynamics can't recover from", decaying Glass releases the
constraint. If the ceiling is downstream of Glass anyway, decaying does nothing.

**Risk.** The basin might *destabilise* late in training if Glass's pressure
disappears entirely. Better: floor at small but non-zero (e.g. 1/10th of
initial).

---

## Recommended implementation order

Ordered by `(expected payoff × diagnostic value) / implementation cost`. Items
1-3 are quick A/B tests; items 4-5 are deeper changes triggered by what items
1-3 tell us.

### 1. **Idea 6 — Plain TD-MPC2 + smoothing ablation** (highest priority)

`--algos tdmpc2 --latent_action_smooth_coef 1e-3` on local 4070 Ti, seeds 1+2.
Runs in 3-5 h. The most informative single experiment we can do right now.
Tells us whether Glass is doing anything beyond what smoothing alone does.

Needs ~5 LOC port of the latent-smooth term into `tdmpc2.py` (currently only
in `tdmpc_glass.py`).

### 2. **Idea 1a — Phase-k: curriculum smoothing + raised λ_temporal**

Take Phase-j's setup (curriculum smoothing) and ALSO raise
`--glass_lambda_temporal 1e-2` (10×) or `5e-2` (50×). Penalises rapid cluster
changes more strongly. Runs in parallel on remote 4060 (which is currently
idle since Phase-h finished).

Cost: 0 new code (existing flag). Test if your "K-flicker hurts" hypothesis
holds.

### 3. **Idea 7 — Phase-l: Glass loss decay curriculum**

After Phase-j seed 1 + 2 complete, if smoothing-only (item 1) didn't suffice,
run Phase-j knobs PLUS Glass loss decay (`λ_se`, `λ_balance`, `λ_temporal` all
scaled by `max(0.1, 1 - es / 2M)`). Implementation reuses the JIT-rebuild
pattern from `--latent_smooth_warmup_env_steps`.

### 4. **Idea 5c — N sweep**

Quick: just `--glass_num_prototypes 8` and `--glass_num_prototypes 32` on
remote 4060. Two single-seed runs to see if N matters at all. Probably doesn't,
but cheap to falsify.

### 5. **Idea 2 — hierarchical super-clusters**

Only if items 1-4 confirm there's a real K=4 ceiling. Add `K_super=2` layer.

### Defer indefinitely
- **Idea 3 (VLM symbolic reasoning)**: heavy, brittle, only constrains MPPI
  not training. Revisit after we have a stable >600 peak.
- **Idea 4 (cluster-space MPPI)**: heavy, addresses a real but secondary issue
  (MPPI < pi is fixable with a better dynamics model first).
- **Idea 5b (SimNorm groups as clusters)**: interesting research direction but
  not aligned with the immediate "all 5 seeds > 500" goal.

---

## What to run *right now* on the idle machines

- **Local 4070 Ti**: still running Phase-j seed 1 (at 2.25M, MPPI 380, climbing
  slowly). Let early-stop fire (~3.75M) before launching item 1.
- **Remote 4060**: idle since Phase-h. Available to launch item 2 (Phase-k =
  Phase-j knobs + raised λ_temporal) immediately. Single-seed test would
  tell us in ~3-5 h if K-flicker penalty helps.

If you want, I can launch Phase-k on remote now. Item 1 (TD-MPC2 + smoothing
ablation) goes on local once Phase-j ends.
