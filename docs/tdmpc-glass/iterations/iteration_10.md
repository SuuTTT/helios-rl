# TD-MPC-Glass Iteration 10 Plan

Created: 2026-05-27

## Goal

Move from fast probe discovery to clean evidence:

1. complete fair 5-seed confidence intervals for the strongest Glass handoff
   candidates;
2. complete a matching 5-seed internal TD-MPC2 K256 baseline;
3. design the next mechanism changes around the two strongest insights from
   Iterations 8-9:
   - MPPI can be worse than the actor policy;
   - current Glass may over-segment the latent, and stopping Glass later/earlier
     changes performance.
4. record a longer-horizon world-model replacement direction based on recent
   JEPA world-model work, but keep it out of the immediate queue until the
   current evidence and ablations are complete.

Primary target remains HopperHop G1: 5/5 seeds above 500.

G2 target: at least one benchmark-fair seed above 600, without reward shaping
or environment edits.

## Current Evidence Snapshot

Latest blog snapshot, generated 2026-05-27 07:34 UTC:

| Phase | Design | Countable runs | Mean best_any | 95% CI | G1 |
|---|---|---:|---:|---:|---:|
| `phasei9r` | Phase1b Glass, off at 1M | 5 | 523.9 | [426.3, 621.4] | 4/5 |
| `phasei9t` | Phase1b Glass, off at 1.5M | 3 | 469.9 | [164.2, 775.6] | 2/3 |
| `phasei9q` | Phase1b + temp-stability 0.01, off at 2M | 7 | 435.8 | [314.7, 556.9] | 3/7 |
| `phasei10c` | clean off-at-1M confirmation | 4 | 316.5 | [176.2, 456.8] | 0/4 |
| TD-MPC2 K256 | internal JAX TD-MPC2, ~official update rate | 3 | 389.3 | [82.7, 695.9] | 1/3 |
| TD-MPC2 K128 | internal JAX TD-MPC2, half update rate | 3 | 304.8 | [247.4, 362.2] | 0/3 |

Important caveat:
- `phasei9r` is the current leader, but its provenance is mixed from fast
  iteration.
- `phasei10c` is the clean confirmation of the same off-at-1M handoff idea,
  not necessarily a byte-for-byte repeat of every original `phasei9r` task.
- Do not claim "solved" until clean fills mature and the TD-MPC2 K256 baseline
  has 5 seeds.

### i9q/i9r/i10c Interpretation, 2026-05-28

Latest countable-only readout, including runs with `last_step >= 4M` or
`best_any >= 500`:

| Family | Countable seeds | Mean best_any | G1 | Interpretation |
|---|---:|---:|---:|---|
| `phasei9r_off1m` | 5 | ~532.5 | 4/5 | strong but mixed-provenance lead |
| `phasei9q_temp001_off2m` | 8 | ~400.7 | 2/8 | high-variance MPPI spike family |
| `phasei10c_clean_off1m` | 4 | ~321.8 | 0/4 | clean off-at-1M confirmation is weak |

`phasei9q` is not uniformly stronger than `phasei9r`. Its main value is that a
few seeds become very strong under `TEMP_STABILITY=0.01` plus Glass kept active
until 2M. The strongest examples are MPPI-led, such as seed 2 reaching
`604.5@3.50M` and seed 3/4 variants reaching G1/G2-level returns, while many
other i9q seeds remain in the 260-440 range. This suggests temp-stability plus
a longer Glass handoff can sometimes create planner-exploitable structure, but
it does not yet solve seed robustness.

The clean off-at-1M rerun is worse because the original `phasei9r` evidence was
found during fast iteration and auto-promotion. Its output tags use
`dbf5cea-dirty`, but queue logs show several actual launch SHAs across
`dbf5cea`, `87a4337`, and `df9bfb1`. `phasei10c_clean_off1m` is the cleaner
single-tag confirmation of the same handoff idea, and it did not reproduce the
4/5 G1 behavior. Treat original i9r as a useful lead, not a settled method.

Current conclusion:
- Stop spending more blind budget on clean off-at-1M repeats.
- Keep the in-flight fair-CI and K256 baseline fills running to close the
  comparison.
- Use newly freed capacity for non-clean Iteration 10 probes that either
  narrow the MPPI-policy gap or coarsen/restructure Glass.

## TD-MPC2 K256 Baseline Provenance

The current 3-seed TD-MPC2 K256 baseline is **not** from the official TD-MPC2
result files. It is our internal JAX TD-MPC2 run from `phaseaa`, the K_UPDATE
sweep.

Current countable K256 seeds:

| Seed | Source | best_any | Best step | Last step | Notes |
|---:|---|---:|---:|---:|---|
| 1 | `remote_mirror/ssh3_3060ti_new/HopperHop_phaseaa_codex_tdmpc2_k256/seed_1.csv` | 531.0 | 4.50M | 7.75M | countable early G1 |
| 2 | local `HopperHop_phaseaa_codex_tdmpc2_k256/seed_2.csv` | 331.4 | 9.75M | 10.00M | complete |
| 3 | `remote_mirror/ssh3_3060ti_new/HopperHop_phaseaa_codex_tdmpc2_k256/seed_3.csv` | 305.5 | 7.25M | 10.00M | complete |

Action taken:
- queued TD-MPC2 K256 fill seeds 4 and 5 through
  `scripts/run_phaseaa_codex_kupdate_sweep.sh`;
- each fill is one queue task;
- `SAVE_FULL_STATE=false` by default to avoid large checkpoint failures.

Queued tasks:

| Task | Seed | Env |
|---|---:|---|
| `t1cf7bc7` | 4 | `K_UPDATES=256 SEEDS=4 SAVE_FULL_STATE=false` |
| `tc0eaa6a` | 5 | `K_UPDATES=256 SEEDS=5 SAVE_FULL_STATE=false` |

## Current Queue Discipline

- One seed per queue task.
- For fair CI phases, cap auto-promotion at seeds 1-5.
- Count interrupted runs below 4M only if `best_any >= 500`.
- Keep logs for failed/partial runs, but separate them from final claims.
- No full-state checkpoint by default.
- Record worker failures; do not let destroyed-box runs silently enter final
  denominators.

## Direction 1: Fix the MPPI-Policy Gap

### Motivation

The result CSVs show a recurring issue: MPPI is sometimes worse than the actor
policy. This means the learned model/value/planner stack can mis-rank action
sequences even when the actor has learned a useful gait.

If Glass improves the actor basin but hurts planner calibration, MPPI-only
evaluation can understate good policies or, worse, planning can damage test-time
control.

### Hypotheses

1. MPPI is over-trusting a latent model that is locally coherent but not
   reward-calibrated.
2. Glass improves basin entry but can make the transition graph too coarse for
   accurate MPPI sequence ranking.
3. Actor policy can be a better controller than planner in some learned-model
   regimes.

### Probe Ideas

#### i10-a: Planner/Actor Arbitration

Important distinction:

| Variant | Affects eval return? | Affects replay/data collection? | Risk |
|---|---:|---:|---|
| `i10-a1` eval-only arbitration | yes | no | low |
| `i10-a2` behavior/data-collection arbitration | yes | yes | medium/high |
| `i10-a3` late behavior arbitration after basin entry | yes | yes, but only after a reward threshold | medium |

Eval-only arbitration computes or selects between `pi` and MPPI during
evaluation. It can reveal that the actor is already good and MPPI is hiding it,
but it does **not** create better replay transitions. It is mostly a reporting
and controller-selection diagnostic.

Data-collection arbitration is different: it uses the selected controller for
environment interaction, so the chosen action enters the replay buffer. This can
change learning because better actions produce better transitions, but it can
also bias exploration or reinforce a bad local gait if the early comparison is
noisy.

For Iteration 10, start with `i10-a1`, then only try data-collection arbitration
after diagnostics confirm that MPPI is systematically worse than `pi`.

##### i10-a1: Eval-only arbitration

Evaluate both `pi` and MPPI on recent evals. Use MPPI for reported/deployed eval
only if it beats `pi` by a margin; otherwise report/deploy `pi`.

Initial rule:

```text
if best_mppi_recent > best_pi_recent + margin:
    eval_controller = mppi
else:
    eval_controller = pi
```

Start with margins 0, 25, 50.

Success:
- raises final `best_any`;
- reduces cases where MPPI underperforms pi;
- does not hide poor MPPI calibration in diagnostics.
- does not alter replay or training dynamics.

Implementation status, 2026-05-27:

- Added runner flags:
  - `--controller_arbitration {none,eval_only}`;
  - `--arbitration_margin <float>`.
- `eval_only` writes an extra `eval_type=arb` row to each per-seed eval CSV.
- It also writes a sibling `seed_N_arbitration.csv` with:
  `step, seed, pi_reward, mppi_reward, mppi_minus_pi, selected, reward, margin`.
- Checkpoint payloads include the latest arbitration selector/reward.
- Data collection is unchanged; replay still uses the existing actor/noise path.
- Syntax and CLI parsing passed under `/root/venv`.
- Runtime smoke is intentionally deferred until an idle GPU is available, to
  avoid interfering with current running tasks.

Queue command template, one seed per task:

```bash
PROBE_ID=phasei10a1_arb_m25 \
SEEDS=1 \
K_UPDATE=128 \
TOTAL_STEPS=10000000 \
EARLY_STOP_PATIENCE=3000000 \
TEMP_STABILITY=0.01 \
GLASS_WARMUP=100000 \
GLASS_DECAY=1000000 \
LATENT_SMOOTH=0.001 \
LATENT_SMOOTH_WARMUP=250000 \
CONTROLLER_ARBITRATION=eval_only \
ARBITRATION_MARGIN=25 \
bash scripts/run_phasei9_glass_probe.sh
```

##### i10-a2: Behavior/data-collection arbitration

Use the same controller-selection rule for environment interaction, not only
evaluation. The selected action is what generates the replay transition.

This is the version that can improve learning, because better actions can create
better replay data:

```text
controller_for_collection =
    mppi if mppi_recent > pi_recent + margin
    else pi
```

Risks:
- early `pi`/MPPI comparisons are noisy before the agent has a gait;
- switching controllers can make the replay distribution nonstationary;
- using the actor too early may reduce exploration;
- using MPPI after it becomes miscalibrated can keep injecting bad transitions.

Do not queue this until `i10-a1` and `i10-c` identify a consistent gap.

##### i10-a3: Late behavior arbitration after basin entry

Safer behavior-arbitration variant:

```text
if best_any < 400:
    collect_with = default_tdmpc2_controller
else:
    collect_with = arbitration(pi, mppi, margin)
```

Motivation:
- before 400 reward, the run may not have found a real hopping basin, so the
  pi-vs-MPPI comparison is less meaningful;
- after 400 reward, avoiding the worse controller may help preserve and improve
  good basin transitions.

Initial thresholds:
- reward threshold: 350, 400, 450;
- margin: 25 or 50.

Success:
- improves G2 ceiling or late training stability;
- keeps G1 reliability no worse than `phasei9r`;
- reduces cases where MPPI corrupts an already good actor.

#### i10-b: MPPI Distillation Only On Positive Gap

Revisit MPPI-gated distillation, but smoke-test first. Distill only when:

```text
MPPI_reward - pi_reward > gap_threshold
```

Start thresholds:
- 50;
- 100.

Implementation requirement:
- run local smoke before queue;
- log gap statistics every eval;
- do not enqueue until one seed reaches at least the first eval.

#### i10-c: Planner Calibration Diagnostic

Add a diagnostic CSV column comparing:

- predicted MPPI objective;
- realized MPPI return;
- realized pi return.

This is not an optimization yet. It should tell us whether MPPI failures come
from model rollout error, Q terminal error, or action sequence optimization.

## Direction 2: Redesign Glass Hierarchy

### Motivation

Current Glass has multiple assignment levels:

1. SimNorm latent groups inside TD-MPC2: 512 dimensions split into 8 softmax
   groups.
2. Prototype assignment `c`: latent-to-prototype distribution over `mu`.
3. Cluster assignment `S`: prototype-to-cluster distribution.

This is effectively hierarchical clustering:

```text
observation -> SimNorm latent groups -> prototypes mu / assignment c -> clusters S
```

Videos from earlier phases suggest over-segmentation may harm control. The best
rollouts often did not need a rich 4-phase gait machine; they looked closer to a
coarse split such as "bad basin / hopping basin." Iterations 8-9 also suggest
that stopping Glass improves performance.

Core question:

> Instead of adding more fine-grained clustering, should Glass become coarser
> over training, or even collapse to a one-level structural entropy objective?

### Hypotheses

1. Two-level Glass (`z -> mu -> S`) is too expressive and over-segments contact
   dynamics.
2. A one-level SE objective directly on a small set of coarse assignments may be
   better for HopperHop.
3. A coarse-to-none schedule may be better than a fixed cluster hierarchy:
   strong early basin split, then remove the structure.
4. A multi-level hierarchy might still help if higher levels are optimized for
   coarse behavioural modes and lower levels are stopped or regularized.

### Probe Ideas

#### i10-d: One-Level Prototype SE

Remove `S` and treat prototypes directly as graph communities.

Current:

```text
z -> c over mu -> transition graph A over prototypes -> S clusters -> H2(A, S)
```

Probe:

```text
z -> c over mu -> transition graph A over prototypes -> H1/H2-style objective directly over prototype graph
```

Options:
- use fewer prototypes, e.g. N=4 or N=8;
- no prototype-to-cluster `S`;
- encourage a coarse transition graph, not fine segmentation.

Success:
- matches or beats `phasei9r` early G1 rate;
- lower pi/MPPI gap;
- cluster/video labels are simpler and less flickery.

#### i10-e: Coarse Assignment Directly From SimNorm Groups

Skip learned prototypes. Build graph nodes from coarse SimNorm group summaries.

Rationale:
- TD-MPC2 already imposes a product-of-simplexes latent via SimNorm.
- Adding prototype `mu` may duplicate that structure.
- Direct high-level SE on SimNorm groups may be simpler and less invasive.

Possible implementation:
- aggregate each of the 8 SimNorm groups;
- compute group-level transition similarity;
- apply SE pressure to group-transition structure.

This is risky because SimNorm groups are not guaranteed to correspond to
behaviour, but it tests whether prototypes are actually necessary.

#### i10-f: Coarse-To-None Glass Schedule

Keep current Glass early, but explicitly reduce structural resolution before
turning it off:

```text
0-100k: warmup
100k-500k: N=16, K=8
500k-1M: N=8, K=2 or K=4
1M+: Glass off
```

Reason:
- early fine structure may help exploration;
- later coarse structure preserves only basin-level separation;
- final off period lets TD-MPC2 optimize control.

Implementation caveat:
- dynamic shape changes are painful in JAX.
- First version should avoid changing array shapes by masking/merging clusters
  rather than reallocating parameters.

#### i10-g: High-Level SE Only

Use only a high-level two-cluster objective:

```text
bad/recovering basin vs hopping basin
```

This is inspired by video evidence where the best policies used very simple
cluster structure. It may be the most aligned with HopperHop.

Possible forms:
- `K=2`, `N=8`;
- stronger early entropy pressure;
- off at 1M.

Success:
- G1 improves by reducing low-return seeds;
- G2 may not improve directly, but it could give a better starting point for
  later actor/planner exploitation.

Implementation status, 2026-05-28:

- The generic probe launcher now exposes static hierarchy/coarsening flags:
  - `NUM_PROTOTYPES`;
  - `NUM_CLUSTERS`;
  - `NUM_SUPER_CLUSTERS`;
  - `LAMBDA_SUPER_SE`;
  - `LAMBDA_SUPER_BALANCE`.
- This does not implement dynamic coarse-to-none shape changes. It enables the
  first safe probes using existing static JAX shapes:
  - `phasei10g_k2_highse`: high-level `K=2`, `N=8`, off at 1M;
  - `phasei10g_k4_highse`: coarse `K=4`, `N=8`, off at 1M;
  - `phasei10h_hier2_super`: current `K=8`, plus `K_super=2` high-level SE.
- One seed per probe is enough for the first gate. Promote only if best_any
  crosses 380/500/600 under the current queue discipline.

## Longer-Horizon Direction: JEPA World Model Base

This direction is explicitly **not** a quick Iteration 10 queue item. It would
replace or substantially alter the TD-MPC2 latent world model, so it needs a
separate design/reimplementation track after the current G1 evidence fills and
Glass hierarchy ablations.

### Motivation

TD-MPC2 currently learns a compact latent state, dynamics model, reward model,
Q ensemble, and policy. Glass then adds structural pressure on top of that
latent. The Iteration 8-9 evidence suggests the current latent can support good
actor policies, but MPPI can still be worse than the actor. That points to a
world-model/planning calibration issue, not only a clustering issue.

Recent JEPA world-model papers are relevant because they learn compact latent
predictive models without pixel reconstruction:

| Paper | Relevance |
|---|---|
| LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels, arXiv:2603.19312 | Introduces LeWM, a stable end-to-end JEPA world model from raw pixels using next-embedding prediction plus a Gaussian latent regularizer; reports compact latent planning and fast planning compared with foundation-model world models. |
| V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning, arXiv:2603.14482 | Improves dense video representations with dense predictive loss and hierarchical/deep self-supervision; relevant for temporally grounded features, not directly an RL controller. |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, arXiv:2506.09985 | Uses self-supervised video pretraining and an action-conditioned variant for robot planning; relevant as a foundation-world-model reference point. |

Sources checked:
- https://arxiv.org/abs/2603.19312
- https://arxiv.org/abs/2603.14482
- https://arxiv.org/abs/2506.09985

### Why This Might Matter For TD-MPC-Glass

Current TD-MPC2 optimizes reward/Q/policy/dynamics jointly from online RL data.
Glass adds a structural entropy pressure to the same representation. A JEPA base
would change the representation objective:

```text
current:
  observation/proprio -> TD-MPC2 latent -> latent dynamics/reward/Q/policy
  plus Glass structural entropy on latent transitions

JEPA direction:
  observation or pixels -> JEPA embedding -> action-conditioned embedding predictor
  -> planner/policy/value heads
  plus optional Glass/SE on embedding transitions
```

Potential advantages:

- the latent may become more predictive and less tied to immediate reward/Q
  gradients;
- planning may improve if the embedding predictor is better calibrated;
- Glass could operate on a JEPA embedding graph that is already trained to
  predict future latent states;
- dense video features may help if we later move from proprioceptive HopperHop
  to pixels or richer observations.

Potential risks:

- TD-MPC2 is already fast because the latent is compact and the training loop is
  tightly JIT-compiled; a JEPA base could be much slower.
- Pixel JEPA is a different problem from current proprioceptive HopperHop.
- End-to-end JEPA training stability is itself a research problem.
- The reward/Q/policy heads would need careful integration with an embedding
  predictor that is not trained for reconstruction.
- We may spend weeks rebuilding the world model before solving the current G1
  reliability problem.

### Concrete Future Work

#### i10-h0: Literature/Design Note, No Code

Write a design note comparing TD-MPC2 latent dynamics to LeWM-style JEPA latent
prediction:

- inputs: proprio only vs pixels vs mixed;
- prediction target: next latent, future latent, masked dense video features;
- regularization: SimNorm, Gaussian latent regularizer, stopgrad/EMA;
- planner: MPPI over TD-MPC2 dynamics vs CEM/MPPI over JEPA predictor;
- where Glass/SE would attach.

Output:
- a short architecture diagram;
- estimated implementation cost;
- minimum viable smoke test.

#### i10-h1: Proprio-JEPA Smoke

Before pixels, test whether a JEPA-style latent predictor can replace or
auxiliary-train the TD-MPC2 dynamics on proprio observations:

```text
obs_t -> encoder -> z_t
(z_t, action_t) -> predictor -> z_hat_{t+1}
obs_{t+1} -> target encoder -> z_{t+1}
loss = embedding prediction + collapse regularizer
```

This avoids the full pixel pipeline and tests whether JEPA-style prediction is
compatible with MJX HopperHop.

Success:
- no representation collapse;
- predictor loss stable;
- reward/Q/policy training still runs;
- MPPI-policy gap narrows.

#### i10-h2: JEPA Auxiliary, Not Replacement

Add a JEPA next-embedding loss alongside TD-MPC2 dynamics, without removing the
existing TD-MPC2 model:

```text
L = L_TD-MPC2 + beta * L_JEPA + lambda * L_Glass
```

This is safer than a full replacement and can answer whether JEPA prediction
improves latent planning calibration.

#### i10-h3: Pixel JEPA World Model

Only after the proprio smoke passes, consider pixel inputs:

- render HopperHop observations/images;
- train a compact LeWM-style embedding predictor;
- plan in embedding space;
- then add reward/Q/policy heads.

This is a major project and should be treated as a separate paper-scale branch,
not a small Glass ablation.

### Decision Gate

Do not start JEPA implementation until:

1. TD-MPC2 K256 5-seed baseline is complete;
2. current Glass handoff 5-seed fills are analyzed;
3. MPPI-policy gap diagnostics identify whether the planner failure is a model
   prediction problem or an action-selection problem;
4. one-level/coarse Glass ablations are scoped.

If those steps still point to world-model calibration as the bottleneck, then
begin with i10-h0 and i10-h1.

## Initial Iteration 10 Task List

### Evidence Completion

- [x] Queue TD-MPC2 K256 seeds 4 and 5.
- [ ] Finish current promising Glass 5-seed fills.
- [ ] Recompute countable-only 95% CI after all fills mature.
- [ ] Update blog/table if `phasei10c`, `phasei9t`, or K256 changes conclusion.
- [x] Record that clean `phasei10c` does not reproduce the mixed-provenance
  `phasei9r` lead; do not add more blind clean off-at-1M repeats.

### MPPI/Policy Gap

- [ ] Add analysis script/table for pi-vs-MPPI gaps by phase and seed.
- [ ] Add diagnostics for predicted MPPI objective vs realized return.
- [x] Implement `i10-a1` eval-only arbitration probe.
- [ ] Runtime-smoke `i10-a1` on an idle GPU, then queue one seed.
- [ ] Only after diagnostics, implement `i10-a3` late behavior arbitration.
- [ ] Avoid `i10-a2` early behavior arbitration unless there is strong evidence
  that the collection controller is the bottleneck.
- [ ] Smoke-test MPPI-gated distillation before queueing.

### Glass Redesign

- [ ] Inspect current Glass code path and identify where `mu`, `c`, and `S`
  can be ablated cleanly.
- [ ] Design one-level prototype SE implementation.
- [ ] Design coarse `K=2` high-level SE implementation.
- [ ] Avoid dynamic-shape JAX changes in first implementation; prefer masking
  or static shapes.
- [x] Queue first static hierarchy/coarsening probes:
  `phasei10g`, `phasei10h`, `phasei10i`, with `phasei10j` and `phasei10k`
  pending.

### Queue Discipline, 2026-05-28

Current queue state at the decision point:

- 9 running tasks;
- 2 pending non-clean Iteration 10 hierarchy/coarsening tasks;
- auto-promotion daemon is active and will add follow-up seeds when a completed
  run crosses the configured thresholds.

Do not add more clean off-at-1M tasks. If a GPU frees up, let the pending
`phasei10j`/`phasei10k` probes start first. Add new probes only after at least
one of these produces a countable signal or fails quickly.

Next probe backlog, not queued yet:

| Probe | Direction | Env delta | Launch condition |
|---|---|---|---|
| `phasei10l_k4_temp001` | coarse Glass plus temp stability | `NUM_PROTOTYPES=8`, `NUM_CLUSTERS=4`, `TEMP_STABILITY=0.01`, `GLASS_DECAY=1000000` | queue if `phasei10g_k4_coarse` is not clearly worse by 2M |
| `phasei10m_k2_off1p5m` | high-level SE with later handoff | `NUM_PROTOTYPES=8`, `NUM_CLUSTERS=2`, `GLASS_DECAY=1500000` | queue if `phasei10g_k2_highse` reaches early basin but fades |
| `phasei10n_hier2_off1p5m` | hierarchy with later handoff | `NUM_PROTOTYPES=32`, `NUM_CLUSTERS=8`, `NUM_SUPER_CLUSTERS=2`, `GLASS_DECAY=1500000` | queue if `phasei10h_hier2_super` is stable but slow |
| `phasei10a1_arb_m25` | eval-only MPPI/policy arbitration | `CONTROLLER_ARBITRATION=eval_only`, `ARBITRATION_MARGIN=25` | queue when one GPU is idle and hierarchy probes have started |

Rationale: this keeps iteration fast but avoids mixing too many hypotheses
before the first hierarchy/coarsening probes have even reached 2-4M steps.

### Early-Spike Hypothesis Tests, 2026-05-30

Observation:

- Several runs enter a >500 reward basin extremely early:
  - `phasei10a1_arb_m25` seed 1: `538.8@0.75M`,
    `574.6@1.00M`, later `620.5@5.50M`;
  - `phasei10k_k2_temp001` seed 3 rerun: `515.4@0.75M`,
    `526.0@1.00M`, later `556.7@8.25M`;
  - `phasei10p_k2_temp0005` seed 1: `505.6@1.25M`,
    `537.1@1.50M`.
- The effect is not yet robust. Follow-up `phasei10a1` seeds 2-5 are weak so
  far, and `phasei10k` has mixed seeds.

Working explanation:

`K=2` Glass plus mild temp-stability appears to act as a coarse basin scaffold.
If early exploration touches a hopping basin, the stable two-way latent split
can preserve it and MPPI can amplify it. If the seed does not see that basin
early, the run remains ordinary. Eval arbitration exposes the better controller
when one exists, but does not by itself create better replay transitions.

Hypotheses and tests:

| Hypothesis | Prediction | Fast probes |
|---|---|---|
| H1: K2 scaffold matters | K2 beats K4 on >500-by-1M rate | `phasei10r` K2/temp0.01 vs `phasei10u` K4/temp0.01 |
| H2: temp-stability matters | temp0.005/0.01 beat temp0.0 and temp0.02 | `phasei10r/s/t` plus `phasei10v` no-temp |
| H3: MPPI amplifies a partial gait | high-NS MPPI variants spike more than low-NS | `phasei10r` NS=2048 vs `phasei10w` NS=512 |
| H4: basin-entry luck is exploration-limited | longer exploration raises spike frequency | `phasei10r` expl25k vs `phasei10x` expl500k |

Fast-test rule:

- Use `TOTAL_STEPS=2000000` and evaluate at the existing 250k interval.
- Primary metric: any eval reward >=500 by 1.0M.
- Secondary metric: best reward by 2.0M.
- Treat a family as promising only if >=2/5 seeds reach >500 by 1.0M, or if
  >=3/5 reach >500 by 2.0M.

Queued test batches:

| Probe | Purpose | Seeds | Priority |
|---|---|---:|---:|
| `phasei10q_arb_m25_fast2m` | reproduce eval-arbitration early spike | 6-10 | 4 |
| `phasei10r_k2_temp001_fast2m` | K2/temp0.01 without arbitration | 6-10 | 5 |
| `phasei10s_k2_temp0005_fast2m` | lower temp-stability | 3-7 | 5 |
| `phasei10t_k2_temp002_fast2m` | higher temp-stability | 1-5 | 6 |
| `phasei10u_k4_temp001_fast2m` | K4 scaffold control | 1-3 | 5 |
| `phasei10v_k2_notemp_fast2m` | no-temp control | 1-3 | 5 |
| `phasei10w_k2_temp001_ns512_fast2m` | lower-MPPI-sample control | 1-3 | 5 |
| `phasei10x_k2_temp001_expl500k_fast2m` | longer exploration control | 1-3 | 5 |

Do not infer from single seed spikes. The decision should be based on spike-rate
across these small batches.

### JEPA World Model Direction

- [ ] Write JEPA-vs-TD-MPC2 world-model design note.
- [ ] Decide whether a proprio-JEPA smoke can be implemented without touching
  the pixel pipeline.
- [ ] Defer pixel LeWM/V-JEPA-style implementation until current G1/G2 evidence
  identifies world-model calibration as the main blocker.

## Decision Gates

1. If clean `phasei10c` catches up to `phasei9r`, make early Glass off-at-1M
   the main method and focus on MPPI-policy gap for G2.
2. If `phasei10c` remains weak but `phasei9t` matures well, shift handoff later
   to 1.5M.
3. If TD-MPC2 K256 5-seed mean approaches `phasei9r`, require stronger evidence
   before claiming improvement.
4. If all clean reruns regress, treat the current `phasei9r` result as a useful
   lead but not a robust result; prioritize Glass hierarchy redesign.
