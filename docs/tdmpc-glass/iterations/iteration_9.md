# Iteration 9 — Fast one-seed phase probes for 5/5 G1

Date: 2026-05-22

Goal stays unchanged:
- **G1**: 5/5 HopperHop seeds >500 by verified `best_any = max(best_pi, best_mppi)`.
- **G2**: at least one fair seed >600.

## Why move on from Iteration 8

Iteration 8 was the implementation-and-triage iteration around three main
levers plus the measurement fix:

1. **Phase-eval**: best-pi / best-MPPI / best-any checkpointing.
2. **Phase-ar**: auto-restart on plateau.
3. **Phase-mpc-lite**: MPPI-gated planner distillation.
4. **Phase-g2**: Glass V2 temporal-stability loss.

Current read:
- Phase-eval is a permanent measurement fix.
- Phase-ar, as configured, does **not** look like a 5/5 solution. It produced
  a strong winner, but several seeds stayed far below 500 and restart rows did
  not appear as intended.
- Phase-mpc-lite has no positive early signal so far.
- Phase-g2 has the best scientific signal: seed 2 reached 570.6, while seed 1
  looked stuck. Seeds 3-5 are now being rerun through the standard queue after
  local disk/log issues.

So Iteration 8 is not "done" in the strict reporting sense until Phase-g2 and
Phase-mpc-lite finish, but it has enough signal to stop designing only around
those three phase families.

## Iteration 9 policy: one seed first

To iterate faster, each new phase starts as a **single sentinel-seed probe**.
Only probes that show a strong signal get promoted to a 5-seed sweep.

Default sentinel seed:
- **Seed 1** for robustness probes. It is repeatedly difficult under recent
  fair recipes, so rescuing it is more informative than hitting an easy seed.

Promotion rules:
- Promote to 5 seeds if `best_any >= 500` before 5M steps.
- Also promote if `best_any >= 400` by 3M with improving diagnostics
  (`standing_rate`, `full_reward_rate`, falls/time-to-hop).
- Kill or deprioritize if `best_any < 100` at 1M and diagnostics show no gait
  progress.

Important caveat:
- One seed cannot prove G1. It is only a cheap filter for mechanisms. The final
  claim still requires 5/5 seeds under one recipe.

## Code-version workflow

Queue tasks run the code that exists on the launcher box **when the task starts**,
not necessarily the code that existed when the task was added. The central queue
daemon rsyncs current `scripts/` and `src/` to remotes at launch time.

Therefore Iteration 9 uses this workflow:

1. **Design the probe in this doc first.**
   - Add phase name, hypothesis, exact code/flag changes, seed, output tag,
     pass/kill rule, and expected failure mode.
   - Do not implement or queue a probe until its design block exists.
2. **Prefer flag-only probes.**
   - If a probe can be represented by launcher/env flags, do not edit algorithm
     code. This keeps multiple queued probes safe.
3. **For code-changing probes, commit or at least record a code identity before
   queueing.**
   - Record `git rev-parse --short HEAD` in the design block.
   - Put the short SHA in the task label and `TDMPC_GLASS_OUTPUT_TAG`.
   - If the working tree is dirty, record that explicitly and avoid queueing
     multiple code-changing probes at the same time.
4. **One code-changing probe may be pending at a time.**
   - Safe sequence: design → implement → smoke → queue seed 1 → wait until the
     task is `running` → then begin the next code-changing probe.
   - Flag-only probes can be queued together if they use the same code SHA.
5. **Log the code identity at run start.**
   - Launchers should print:

```bash
echo "git_sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "code_sha_env=${CODE_SHA:-unset}"
```

This makes every CSV/log interpretable later even if `src/` changes after the
task starts.

## Probe design template

Use this template before implementation:

```
### Phase-i9x — short name

Status: designed | implemented | smoke-passed | queued | running | promoted | killed
Code identity: <short git sha or dirty-tree note>
Sentinel seed: 1
Target box: <box tag or any queue box>
Output tag: phasei9x_<short>_<sha>

Hypothesis:
- ...

Change:
- Code:
- Flags/env:

Run command / queue task:
- launcher:
- env:
- priority:

Pass rule:
- ...

Kill rule:
- ...

Risk / expected failure:
- ...

Readout:
- ...
```

## Candidate probes

### Phase-i9a — Restart actually fires, hard threshold

Status: designed
Code identity: TBD before implementation
Sentinel seed: 1
Target box: `ssh1_2080ti` if idle, otherwise any free queue box
Output tag: `phasei9a_restart300_<sha>`

Motivation:
- Phase-ar did not show restart rows, so the intended basin-entry mechanism may
  not have actually been exercised.

Probe:
- TD-MPC2, K=128, NS=2048, EXPL_UNTIL=500k.
- Restart check at 1M.
- Use a stricter threshold: restart if `best_any < 300` at 1M.
- Reset policy + Q + target Q; consider encoder reset if second attempt also
  fails by 1.5M.
- This is a code-changing probe unless current restart code already supports
  `best_any` thresholding and stricter reset semantics.

Pass condition:
- Seed 1 gets above 400 by 3M or above 500 by 5M.

Kill rule:
- If no restart row appears by the first eval after 1M when `best_any < 300`,
  stop and fix restart semantics before spending more GPU time.

### Phase-i9b — Glass V2 lower temp-stability coefficient

Status: running
Code identity: `dbf5cea-dirty` (dirty tree includes docs/launcher/queue ops
changes; algorithm source is current Phase-g2 code)
Sentinel seed: 1
Target box: any free queue box; expected first idle target is `ssh1_2080ti`
Output tag: `phasei9b_temp001_dbf5cea-dirty`

Motivation:
- Phase-g2 seed 2 hit 570.6, but seed 1 looked stuck. Coef 0.05 may be too
  strong for hard basins or may over-stabilize a wrong gait.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.01`.
- Flag-only probe if the current Glass V2 implementation already exposes this
  coefficient.

Run command / queue task:
- launcher: `scripts/run_phasei9b_temp001.sh`
- env: `SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.01 CODE_SHA=dbf5cea-dirty`
- priority: 7

Pass condition:
- Better seed-1 progress than Phase-g2 0.05 by 2M and `best_any >= 400` by 3M.

Kill rule:
- If `best_any < 100` at 1M and diagnostics show no standing/full-reward
  progress, deprioritize.

### Phase-i9c — Glass V2 delayed temp-stability

Status: designed
Code identity: TBD before implementation
Sentinel seed: 1
Target box: any free queue box after i9b starts or fails
Output tag: `phasei9c_delaytemp_<sha>`

Motivation:
- Temporal stability may help after the representation sees enough motion, but
  penalizing cluster changes too early could lock in crawling.

Probe:
- Same as Phase-g2, but apply temporal-stability only after 250k or 500k steps.
- Code-changing probe unless a delay/warmup flag already exists.

Pass condition:
- Seed 1 escapes early crawling and reaches `best_any >= 400` by 3M.

Kill rule:
- If it matches or underperforms Phase-g2 seed 1 through 2M, do not promote.

### Phase-i9d — Actor-first / MPPI-late

Status: designed
Code identity: TBD before implementation
Sentinel seed: 1
Target box: any free queue box after i9a/i9b readouts
Output tag: `phasei9d_actorfirst_<sha>`

Motivation:
- MPPI is often worse than pi. Early MPPI may reinforce model-favored but
  environment-bad contact timing.

Probe:
- TD-MPC2, K=128, NS=2048.
- During early training, evaluate/checkpoint pi as primary and delay MPPI
  action selection or reduce MPPI influence until after 1M.
- Code-changing probe unless MPPI action-selection delay is already exposed as
  a flag.

Pass condition:
- pi score climbs faster than recent TD-MPC2/Phase-ar seed 1 baselines.

Kill rule:
- If pi is still below 100 at 1M and diagnostics show no gait progress, stop
  rather than running to 10M.

### Phase-i9e — Glass V2 very-low temp-stability

Status: promoted from seed 1
Code identity: `dbf5cea-dirty` (flag-only probe; algorithm source unchanged
from Phase-g2/i9b)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9e_temp0005_dbf5cea-dirty`

Motivation:
- Phase-g2 0.05 produced one strong winner but stuck seed 1. Phase-i9b 0.01
  tests a lower coefficient; this probe tests whether the useful range is even
  weaker, preserving basin flexibility while still damping cluster flicker.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.005`.
- Standard Glass warmup 100k and latent smooth 0.001 after 250k.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9e_temp0005 SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.005 GLASS_WARMUP=100000 LATENT_SMOOTH=0.001 LATENT_SMOOTH_WARMUP=250000 CODE_SHA=dbf5cea-dirty`
- priority: 7
- queue id: `t57cb9bd` (rerun pending; first attempt on `ssh3_3060ti`
  failed during PTX temp-file creation under earlier disk pressure)

Pass rule:
- Better than Phase-g2 seed 1 by 2M and `best_any >= 400` by 3M.

Kill rule:
- If `best_any < 100` at 1M with no standing/full-reward progress,
  deprioritize.

Risk / expected failure:
- Coefficient may be too weak to change cluster stability, reproducing baseline
  Glass behavior.

Readout:
- Compare seed-1 CSV/diag against Phase-g2 0.05 and Phase-i9b 0.01.

### Phase-i9f — Glass V2 mid-low temp-stability

Status: implemented, queued
Code identity: `dbf5cea-dirty` (flag-only probe; algorithm source unchanged
from Phase-g2/i9b)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9f_temp002_dbf5cea-dirty`

Motivation:
- If 0.05 is too strong and 0.005/0.01 are too weak, 0.02 is a midpoint that
  may keep the positive temporal-stability signal without over-locking early
  representations.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.02`.
- Standard Glass warmup 100k and latent smooth 0.001 after 250k.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9f_temp002 SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.02 GLASS_WARMUP=100000 LATENT_SMOOTH=0.001 LATENT_SMOOTH_WARMUP=250000 CODE_SHA=dbf5cea-dirty`
- priority: 7

Pass rule:
- `best_any >= 400` by 3M or `best_any >= 500` by 5M on seed 1.

Kill rule:
- If it tracks Phase-g2 seed 1's stuck behavior through 2M, do not promote.

Risk / expected failure:
- Same over-stabilization failure as 0.05, but milder.

Readout:
- Compare early diagnostics and cluster metrics against i9b/i9e/i9g.

### Phase-i9g — Delayed Glass warmup with low temp-stability

Status: implemented, queued
Code identity: `dbf5cea-dirty` (flag-only probe; algorithm source unchanged
from Phase-g2/i9b)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9g_warm500k_temp001_dbf5cea-dirty`

Motivation:
- Temporal stability may help after the system has seen enough motion, but all
  Glass losses turning on at 100k could stabilize the wrong early crawl. This is
  a flag-only approximation to delayed temp-stability: delay all Glass updates
  until 500k while keeping temp-stability at 0.01.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.01`.
- `glass_warmup_env_steps=500000`.
- Latent smooth remains 0.001 after 250k.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9g_warm500k_temp001 SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.01 GLASS_WARMUP=500000 LATENT_SMOOTH=0.001 LATENT_SMOOTH_WARMUP=250000 CODE_SHA=dbf5cea-dirty`
- priority: 7
- queue id: `t134a300` (rerun pending; first attempt on `ssh3_3060ti`
  failed during PTX temp-file creation under earlier disk pressure)

Pass rule:
- Seed 1 avoids the Phase-g2 seed-1 collapse and reaches `best_any >= 400` by
  3M or `best_any >= 500` by 5M.

Kill rule:
- If no meaningful gait signal appears by 1M, deprioritize; delayed Glass is
  not rescuing basin entry.

Risk / expected failure:
- Delaying all Glass losses may remove the representation benefit entirely.

Readout:
- Compare 0.5M, 1M, 2M, and 3M diagnostics against i9b's normal-warmup 0.01.

### Phase-i9h — Delayed Glass warmup 250k with low temp-stability

Status: implemented, queued
Code identity: `dbf5cea-dirty` (flag-only probe; algorithm source unchanged)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9h_warm250k_temp001_dbf5cea-dirty`

Motivation:
- Phase-i9g delays Glass until 500k, which may be too late because it gives the
  policy almost the whole random-action phase without the representation bias.
  This probe uses a middle delay: let the encoder see 250k steps before Glass
  stabilizes clusters.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.01`.
- `glass_warmup_env_steps=250000`.
- Latent smooth remains 0.001 after 250k.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9h_warm250k_temp001 SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.01 GLASS_WARMUP=250000 GLASS_DECAY=0 LATENT_SMOOTH=0.001 LATENT_SMOOTH_WARMUP=250000 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `tafda394` (`ssh3_3060ti`, started 2026-05-22T08:26:55Z)

Pass rule:
- `best_any >= 400` by 3M or `best_any >= 500` by 5M on seed 1.

Kill rule:
- If it underperforms i9b/i9g through 2M, do not promote.

Risk / expected failure:
- Still may stabilize the wrong crawl if 250k is too early.

Readout:
- Compare 250k/500k/1M diagnostics against i9b and i9g.

### Phase-i9i — Early Glass then off

Status: implemented, queued
Code identity: `dbf5cea-dirty` (flag-only probe; algorithm source unchanged)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9i_glassoff2m_temp001_dbf5cea-dirty`

Motivation:
- Prior fair runs suggested "Glass off late" can preserve useful early
  representation shaping while avoiding late optimization drag. This probe
  combines that idea with low temporal stability: use Glass for basin formation,
  then turn it off at 2M.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.01`.
- `glass_warmup_env_steps=100000`.
- `glass_decay_steps=2000000`.
- Latent smooth remains 0.001 after 250k.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9i_glassoff2m_temp001 SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.01 GLASS_WARMUP=100000 GLASS_DECAY=2000000 LATENT_SMOOTH=0.001 LATENT_SMOOTH_WARMUP=250000 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `t7ddc8e0` (pending)

Pass rule:
- `best_any >= 400` by 3M or `best_any >= 500` by 5M on seed 1.

Kill rule:
- If the run is still below 100 at 1M with poor diagnostics, deprioritize.

Risk / expected failure:
- Turning Glass off may remove the only helpful signal before the gait has
  stabilized.

Readout:
- Compare pre/post-2M learning curve slope and diagnostics against i9b/i9f.

### Phase-i9j — No latent smoothing, low temp-stability

Status: implemented, queued
Code identity: `dbf5cea-dirty` (flag-only probe; algorithm source unchanged)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9j_nosmooth_temp001_dbf5cea-dirty`

Motivation:
- Latent action smoothing can improve mid-game representation smoothness, but
  it may also suppress contact-timing distinctions needed for hard seed basin
  entry. Test temporal stability without latent smoothing.

Probe:
- TD-MPC-Glass, K=128, NS=2048, EXPL_UNTIL=500k.
- `glass_lambda_temp_stability=0.01`.
- `latent_action_smooth_coef=0.0`.
- Standard Glass warmup 100k.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9j_nosmooth_temp001 SEEDS=1 K_UPDATE=128 TEMP_STABILITY=0.01 GLASS_WARMUP=100000 GLASS_DECAY=0 LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `tbfa94b2` (`ssh17637_gpu0`; process ended status 137 after
  reaching G1, so algorithmic readout is valid but task status is failed)

Pass rule:
- Faster seed-1 rise than i9b by 1M/2M, or `best_any >= 400` by 3M.

Kill rule:
- If diagnostics show unstable crawl and best_any remains below 100 at 1M,
  deprioritize.

Risk / expected failure:
- Removing smoothing may increase representation jitter and worsen the exact
  cluster oscillation problem.

Readout:
- Seed 1 reached `best_any=510.3@2.75M` by MPPI; pi also reached `503.5`.
  Diagnostics at the G1 eval were strong: pi full=0.53, stand=0.95; MPPI
  full=0.56, stand=0.95. This is the first Iteration 9 sentinel to exceed 500.
- Promoted to two more seeds:
  - `t5b6e3aa`: seed 2, pending.
  - `t95301e1`: seed 3, pending.

### Phase-i9k — Phase1b knobs plus K=128

Status: implemented, queued
Code identity: `dbf5cea-dirty` (launcher-only change; algorithm source unchanged)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9k_p1b_k128_dbf5cea-dirty`

Motivation:
- Iteration 8 closed with a negative result for the K=128/EXPL_UNTIL=500k/
  latent-smooth/temp-stability stack. Historical Phase1b-style Glass remains
  the best fair family by hit-rate, so test whether the useful part was the
  old Glass recipe rather than the newer exploration/smoothing stack.

Probe:
- TD-MPC-Glass, K=128, NS=2048.
- Restore Phase1b Glass knobs:
  `proto_temperature=0.7`, `assign_logits_init_scale=0.5`,
  `stopgrad_graph=true`.
- Restore short exploration: `EXPL_UNTIL=25000`.
- No latent smoothing and no temp-stability.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9k_p1b_k128 SEEDS=1 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.0 GLASS_WARMUP=100000 GLASS_DECAY=0 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `tf025984` (pending)

Pass rule:
- `best_any >= 400` by 3M or `best_any >= 500` by 5M on seed 1.

Kill rule:
- If it remains below 100 at 1M, deprioritize Phase1b+K without further seeds.

Risk / expected failure:
- K=128 may not be the missing ingredient; the original Phase1b hit-rate may
  simply be seed variance.

Readout:
- Compare seed 1 against original Phase1b seed 1 and Phase1b_10M seed 1.

### Phase-i9l — Phase1b knobs plus low temporal stability

Status: implemented, queued
Code identity: `dbf5cea-dirty` (launcher-only change; algorithm source unchanged)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9l_p1b_temp001_dbf5cea-dirty`

Motivation:
- Phase-g2's 0.05 temporal-stability coefficient produced one winner but
  generally over-locked weak seeds. Test the same idea inside the historically
  better Phase1b recipe with a much weaker coefficient.

Probe:
- TD-MPC-Glass, K=128, NS=2048.
- Phase1b Glass knobs and short exploration.
- `glass_lambda_temp_stability=0.01`.
- No latent smoothing.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9l_p1b_temp001 SEEDS=1 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.01 GLASS_WARMUP=100000 GLASS_DECAY=0 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `td905ee0` (pending)

Pass rule:
- Beat i9k on seed 1, or reach `best_any >= 400` by 3M.

Kill rule:
- If it looks like i9h/i9i/i9j below 100 at 1M, stop pursuing low-temp Glass
  variants until rollout diagnostics show why.

Risk / expected failure:
- Even weak temporal stability may preserve the wrong early crawl.

Readout:
- Compare against i9k to isolate the temp-stability term.

### Phase-i9m — Phase1b knobs with Glass off at 2M

Status: implemented, queued
Code identity: `dbf5cea-dirty` (launcher-only change; algorithm source unchanged)
Sentinel seed: 1
Target box: any free queue box
Output tag: `phasei9m_p1b_off2m_dbf5cea-dirty`

Motivation:
- Prior Glass-off-late runs were among the better fair variants, and Iteration
  8 suggests long-running Glass can drag weak seeds without rescuing them. Use
  Glass for early representation shaping, then remove the partition loss once
  the controller should be refining gait mechanics.

Probe:
- TD-MPC-Glass, K=128, NS=2048.
- Phase1b Glass knobs and short exploration.
- `glass_decay_steps=2000000`.
- No temp-stability and no latent smoothing.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9m_p1b_off2m SEEDS=1 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.0 GLASS_WARMUP=100000 GLASS_DECAY=2000000 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `t87b3131` (pending)

Pass rule:
- `best_any >= 400` by 3M or `best_any >= 500` by 5M on seed 1.

Kill rule:
- If it does not beat i9k by 3M, do not promote the off-at-2M variant.

Risk / expected failure:
- Glass-off timing may matter less than the initial basin-entry lottery.

Readout:
- Compare against i9k to isolate the off-late schedule.

### Phase-i9n — Phase1b K=128 on hard seed 4

Status: implemented, queued
Code identity: `dbf5cea-dirty` (launcher-only change; algorithm source unchanged)
Sentinel seed: 4
Target box: any free queue box
Output tag: `phasei9n_p1b_k128_s4_dbf5cea-dirty`

Motivation:
- Seed 1 is no longer the only useful sentinel for Phase1b-derived probes:
  historical Phase1b seed 1 was already a winner, while seed 4 is repeatedly
  hard (`235.4` original, `331.0` best 10M rerun). A recipe that cannot move
  seed 4 is unlikely to reach 5/5.

Probe:
- Same as i9k, but `SEEDS=4`.
- TD-MPC-Glass, K=128, NS=2048.
- Phase1b Glass knobs: `proto_temperature=0.7`,
  `assign_logits_init_scale=0.5`, `stopgrad_graph=true`.
- Short exploration, no temp-stability, no latent smoothing.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9n_p1b_k128_s4 SEEDS=4 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.0 GLASS_WARMUP=100000 GLASS_DECAY=0 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `t7bc1af5` (pending)
- note: first `ssh6_4060` attempt produced only a CSV header and no log;
  reset to pending on 2026-05-22.

Pass rule:
- Beat Phase1b_10M seed 4 (`331.0`) by 3M-5M; promote if `best_any >= 500`.

Kill rule:
- If still below 100 at 1M, Phase1b+K alone is not enough for hard seeds.

Risk / expected failure:
- May simply reproduce the old seed-4 basin lock.

Readout:
- Compare against original Phase1b seed 4 and Phase1b_10M seed 4.

### Phase-i9o — Phase1b K=128 off at 2M on hard seed 4

Status: implemented, queued
Code identity: `dbf5cea-dirty` (launcher-only change; algorithm source unchanged)
Sentinel seed: 4
Target box: any free queue box
Output tag: `phasei9o_p1b_off2m_s4_dbf5cea-dirty`

Motivation:
- If Phase1b's early representation bias is useful but continued Glass loss
  holds hard seeds in the wrong basin, turning Glass off at 2M should help seed
  4 more than the always-on i9n recipe.

Probe:
- Same as i9n, but `glass_decay_steps=2000000`.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9o_p1b_off2m_s4 SEEDS=4 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.0 GLASS_WARMUP=100000 GLASS_DECAY=2000000 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `t7909eb5` (pending)
- note: first `ssh6_4060` attempt produced only a CSV header and no log;
  reset to pending on 2026-05-22.

Pass rule:
- Beat i9n and Phase1b_10M seed 4; promote if `best_any >= 500`.

Kill rule:
- If it is below i9n by 3M, Glass-off timing is not rescuing hard seed 4.

Risk / expected failure:
- Off-at-2M may be too late if basin lock happens before 500k.

Readout:
- Compare against i9n to isolate the off-late schedule on the same seed.

### Phase-i9p — Phase1b K=128 with low temp-stability on hard seed 4

Status: implemented, queued
Code identity: `dbf5cea-dirty` (launcher-only change; algorithm source unchanged)
Sentinel seed: 4
Target box: any free queue box
Output tag: `phasei9p_p1b_temp001_s4_dbf5cea-dirty`

Motivation:
- This is the hard-seed counterpart to i9l. It tests whether weak temporal
  stability has any benefit under the Phase1b recipe when the seed is one of
  the known failures.

Probe:
- Same as i9n, plus `glass_lambda_temp_stability=0.01`.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- env: `PROBE_ID=phasei9p_p1b_temp001_s4 SEEDS=4 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.01 GLASS_WARMUP=100000 GLASS_DECAY=0 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- priority: 7
- queue id: `t9564862` (pending)
- note: first `ssh6_4060` attempt produced only a CSV header and no log;
  reset to pending on 2026-05-22.

Pass rule:
- Beat i9n; promote only if it gets seed 4 over 500 or produces much faster
  gait diagnostics than i9n.

Kill rule:
- If it follows the low-temp seed-1 failure mode, stop temp-stability variants
  until rollout/cluster diagnostics identify a better schedule.

Risk / expected failure:
- Temporal stability may still over-preserve an early crawl.

Readout:
- Compare against i9n to isolate low temp-stability on the same hard seed.

## Recent probe readout, 2026-05-22

Canonical best-any readout from current Iteration 9 CSVs:

| Probe | Seed | best_any | Step | Selector | Current action |
|---|---:|---:|---:|---|---|
| i9b temp 0.01 | 1 | 293.3 | 7.75M | mppi | do not promote |
| i9e temp 0.005 | 1 | 6.0 | 1.25M | mppi | do not promote |
| i9f temp 0.02 | 1 | 306.2 | 3.00M | mppi | do not promote |
| i9g warmup 500k | 1 | 3.6 | 2.50M | pi | do not promote |
| i9h warmup 250k | 1 | 3.9 | 1.75M | mppi | do not promote |
| i9i Glass off 2M | 1 | 301.1 | 6.00M | mppi | do not promote |
| **i9j no latent smoothing** | 1 | **510.3** | 2.75M | mppi | **promote +2 seeds** |
| i9k Phase1b K128 | 1 | 224.3 | 5.75M | mppi | do not promote |
| i9l Phase1b temp 0.01 | 1 | 484.6 | 3.00M | mppi | watch/near miss |
| i9m Phase1b off 2M | 1 | 373.5 | 8.00M | mppi | do not promote yet |
| i9n Phase1b K128 hard s4 | 4 | 462.4 | 1.25M | mppi | watch/near miss |
| i9o Phase1b off2M hard s4 | 4 | 256.0 | 0.50M | mppi | early/weak |

Interpretation:
- The strongest signal is not the Phase1b pivot yet; it is the simpler
  Phase-g2 low-temp recipe with **latent smoothing disabled**.
- Removing latent smoothing may be helping contact-timing discrimination. This
  directly contradicts the earlier assumption that smoothing is generally a
  safe mid-game stabilizer for HopperHop.
- i9l and i9n are close enough to keep watching, but they have not crossed the
  promotion threshold.


### Phase-i9g-promote — warm500k temp001 follow-up seeds

Status: running
Code identity: `dbf5cea-dirty`
Motivation:
- Phase-i9g seed 1 crossed G1 (`528.1@7.25M`) after the earlier invalid short attempt was rerun. This is the best current Glass V2 signal.
- Promote only enough to test breadth: seed 2 for another standard seed and seed 4 as the known hard seed.

Run command / queue task:
- launcher: `scripts/run_phasei9_glass_probe.sh`
- shared env: `K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=500000 TEMP_STABILITY=0.01 GLASS_WARMUP=500000 GLASS_DECAY=0 LATENT_SMOOTH=0.001 LATENT_SMOOTH_WARMUP=250000 CODE_SHA=dbf5cea-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp`
- seed 2 queue id: `t96fc8a5` (`local`, started 2026-05-23T06:39:42Z)
- seed 4 queue id: `tad38ccd` (`ssh17637_gpu1`, started 2026-05-23T06:39:42Z)

Pass rule:
- Promote to a full 5-seed recipe only if seed 4 reaches G1 or seed 2 reaches G1 with strong diagnostics.

Kill rule:
- If seed 4 stays below the Phase1b_10M seed-4 reference (`331.0`) by 3M, deprioritize this branch.


### Failed-promising reruns — 2026-05-23

Status: queued
Motivation:
- Several tasks marked `failed` ended with status 137 after producing useful partial CSVs. Treat these as interrupted promising runs, not algorithm failures.

Queued reruns with fresh output tags:
- `t584e5ee`: `phasei9j_nosmooth_temp001_rerun2`, seed 1. Prior partial: `510.3@2.75M`.
- `t9d91278`: `phasei9l_p1b_temp001_rerun2`, seed 1. Prior partial: `484.6@3.00M`.
- `t2b01be1`: `phasei9n_p1b_k128_s4_rerun2`, seed 4. Prior partial: `476.1@5.25M`.
- `t07a268c`: `phasei9o_p1b_off2m_s4_rerun2`, seed 4. Prior partial: `387.1@3.50M`.

Operational note:
- These use new `PROBE_ID` values to preserve the partial failed-run CSVs.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35` is set to reduce memory pressure on small remotes.


### Promising-family extra seeds — 2026-05-23

Status: queued/running
Motivation:
- User requested more probes for recipes with previous seed results above roughly 385.
- Focused on families with partial/finished signals: i9j (`510.3` seed 1), i9m (`410.4` seed 1), i9l (`484.6` seed 1), and hard-seed i9n/i9o (`476.1`/`387.1` seed 4 partials).

Queued tasks:
- `td9a5d92`: `phasei9j_nosmooth_temp001_s4`, seed 4, running on `ssh9_2060_gpu1`.
- `t10afa9e`: `phasei9j_nosmooth_temp001_s5`, seed 5, running on `ssh9_2060_gpu2`.
- `tcc342a0`: `phasei9m_p1b_off2m_s2`, seed 2, running on `ssh9_2060_gpu3`.
- `t221b3c3`: `phasei9m_p1b_off2m_s4`, seed 4, pending.
- `td71aa6a`: `phasei9l_p1b_temp001_s2`, seed 2, pending.
- `t691f3fb`: `phasei9l_p1b_temp001_s4`, seed 4, pending.

Fleet note:
- Added `ssh9.vast.ai:17647` as `ssh9_2060_gpu0..3` after bootstrapping `/root/helios-rl`, `/root/venv`, `/.uv/python_install`, and `/root/mujoco_playground_repo`.
- Smoke check passed: JAX sees four CUDA devices and queued tasks started on all four slots.

## Current readout, 2026-05-23

Queue state at readout time:
- `10` running, `6` done, `10` failed, `0` pending before adding the next batch.
- Failed training tasks were logged here before queue cleanup.

Best observed Iteration 9 CSV results:

| Probe | Seed | Status | best_any | Step | Selector | Readout |
|---|---:|---|---:|---:|---|---|
| i9g warm500k temp0.01 | 1 | done | **576.8** | 9.75M | mppi | strongest finished G1 signal |
| i9j no latent smoothing temp0.01 | 1 | failed 137 | **510.3** | 2.75M | mppi | strong partial, interrupted |
| i9l Phase1b temp0.01 | 1 | failed 137 | 484.6 | 3.00M | mppi | near G1, interrupted |
| i9n Phase1b K128 hard seed | 4 | failed 137 | 476.1 | 5.25M | mppi | best hard-seed partial |
| i9m Phase1b off at 2M | 1 | done | 410.4 | 9.50M | mppi | moderate, completed |
| i9o Phase1b off at 2M hard seed | 4 | failed 137 | 387.1 | 3.50M | mppi | moderate hard-seed partial |
| i9p Phase1b temp0.01 hard seed | 4 | done | 364.0 | 3.25M | mppi | not enough |
| i9f temp0.02 | 1 | partial | 306.2 | 3.00M | mppi | weak |
| i9i Glass off at 2M temp0.01 | 1 | failed 137 | 301.1 | 6.00M | mppi | weak/moderate |
| i9b temp0.01 | 1 | done | 293.3 | 7.75M | mppi | weak |
| i9h warm250k temp0.01 | 1 | done | 32.9 | 7.00M | pi | clear failure |

Follow-up seed readout so far:
- i9g seed 2 finished at `260.3@3.75M`; seed 4 is running and has only reached `286.4@3.00M`. The seed-1 G1 hit is real, but not yet robust.
- i9j seed 2 failed at `8.4@3.00M`, seed 3 finished at `250.0@3.25M`, and seeds 4/5 are early low on `ssh9`. The no-smoothing branch is high variance and should not be promoted blindly.
- i9l rerun seed 1 failed at `265.9@1.00M`; seeds 2/4 are early low. The first near-G1 result may be interruption-sensitive or seed-path dependent.
- i9m seed 1 completed at `410.4`; seeds 2/4 are still too early to judge.
- i9n rerun seed 4 is early low despite the first partial reaching `476.1`; keep it running only to confirm whether this is a delayed gait or a collapsed repeat.

What is working:
- Longer Glass warmup with weak temp-stability (`GLASS_WARMUP=500000`, `TEMP_STABILITY=0.01`) can reach G1 on seed 1.
- Phase1b-style MPPI capacity (`K_UPDATE=128`, `MPPI_NS=2048`) still produces the best hard-seed partials, especially i9n seed 4.
- Turning off latent smoothing can unlock a fast seed-1 gait, but the follow-up seeds show it is not a robust recipe by itself.

What is not working:
- Shorter warmup at 250k is a clear miss on seed 1.
- Lower temp-stability alone (`0.005`, `0.02`) and Glass-off-at-2M without the Phase1b package remain below the bar.
- Phase1b K128 without the right schedule is not sufficient on seed 1 (`226.1@6.00M`).
- Current branches do not yet support a 5/5 G1 claim; the best family has one G1 seed and one weak completed follow-up.

Failure modes:
- Most failed training tasks ended with queue `status=137`, consistent with external SIGKILL / memory pressure / container interruption rather than a Python algorithm exception. Treat those CSVs as valid partial evidence but not completed trials.
- The earlier `ssh6_4060` attempts for i9n/i9o/i9p produced header-only CSVs and no useful run body; those remain invalid and should not be counted.
- Several reruns with the same recipe underperform the first interrupted run, so we should distinguish "interrupted but promising" from "repeatably promising."

Queue cleanup:
- After recording the above failures, failed dashboard rows should be deleted to keep the active queue readable. The partial CSVs remain under `exp/tdmpc_glass` and `remote_mirror`.

Next queue batch:
- Complete coverage for the current best family: i9g seeds 3 and 5.
- Complete coverage for the moderate but finished Phase1b-off family: i9m seeds 3 and 5.
- Add missing breadth for the near-G1 Phase1b-temp family: i9l seeds 3 and 5.
- Do not add more i9j yet; seeds 2/3/4/5 are already weak or early weak, so the current queue is enough to test whether seed 1 was a fluke.

Queued follow-up tasks:
- `t0cdd1c7`: `phasei9g_warm500k_temp001_s3`, seed 3.
- `t8cb43bb`: `phasei9g_warm500k_temp001_s5`, seed 5.
- `t081335b`: `phasei9m_p1b_off2m_s3`, seed 3.
- `t1a499f7`: `phasei9m_p1b_off2m_s5`, seed 5.
- `t4785dd6`: `phasei9l_p1b_temp001_s3`, seed 3.
- `t584ee77`: `phasei9l_p1b_temp001_s5`, seed 5.

## Overnight backlog, 2026-05-23

Reason:
- User will be away for roughly 8 hours; keep GPUs fed through the central queue.
- The two families with at least two seeds above 380 are already queued to five
  seeds: i9m has seeds 1/2/4 running or done plus seeds 3/5 pending; i9l has
  seeds 1/2/4 running or done plus seeds 3/5 pending.
- Add lower-priority probes behind those so idle boxes pick up useful work.

Readout before adding backlog:
- `phasei9l_p1b_temp001`: 2/3 over 380 (`s1=484.6`, `s4=453.2`; `s2` weak so far).
- `phasei9m_p1b_off2m`: 2/3 over 380 (`s1=410.4`, `s4=511.8`; `s2` mid/weak so far).
- `phasei9g` seed 4 failed after only reaching `286.4@3.00M`; do not retry now.

Queued backlog tasks:
- `td00ea4b`: `phasei9n_p1b_k128_s2`, seed 2, priority 8.
- `t392fd59`: `phasei9n_p1b_k128_s3`, seed 3, priority 8.
- `t5ccd417`: `phasei9n_p1b_k128_s5`, seed 5, priority 8.
- `t25c11f2`: `phasei9q_p1b_temp001_off2m_s1`, seed 1, priority 8.
- `t6214896`: `phasei9q_p1b_temp001_off2m_s2`, seed 2, priority 8.
- `tafb2f62`: `phasei9q_p1b_temp001_off2m_s4`, seed 4, priority 8.
- `t7899c7e`: `phasei9r_p1b_off1m_s1`, seed 1, priority 9.
- `ta90d285`: `phasei9r_p1b_off1m_s4`, seed 4, priority 9.

Probe definitions:
- i9n-fill: Phase1b K128 recipe, filling seeds 2/3/5 around the prior hard-seed
  `476.1` partial. Seed 1 exists as i9k and was weak, so this is breadth rather
  than immediate promotion.
- i9q: hybrid of the two current >380 families, combining weak temp-stability
  (`TEMP_STABILITY=0.01`) with Glass off at 2M (`GLASS_DECAY=2000000`).
- i9r: earlier off schedule (`GLASS_DECAY=1000000`) to test whether i9m works
  because late Glass is drag, not because Glass must stay active until 2M.

## Latest review, 2026-05-24

Queue state at review time:
- `11` done, `11` running, `6` failed, `2` pending.
- Dashboard, queue daemon, and remote mirror are alive.
- Local disk is healthy (`49%` used on `/root/helios-rl`).

Current family readout:

| Family | Seeds observed | >380 | >=500 | Current best | Review |
|---|---:|---:|---:|---:|---|
| i9m Phase1b off at 2M | 5 | **3** | **2** | `525.6` s5 | best robust branch so far |
| i9n Phase1b K128 | 4 plus seed-4 rerun | 3 early/partial | 1 | `563.2` s2 | very promising but not stable yet |
| i9j no latent smoothing | 5 | 2 | 1 | `510.3` s1 | high variance; useful insight, not recipe |
| i9l Phase1b temp0.01 | 5 | 2 | 0 | `484.6` s1 | near-G1 but below target |
| i9g warm500k temp0.01 | 5 | 1 | 1 | `576.8` s1 | seed-1 spike only |
| i9q temp0.01 + off2M | 3 early | 0 | 0 | `246.0` s2 | no early sign yet |
| i9p temp0.01 hard seed | 1 | 0 | 0 | `364.0` s4 | not enough |
| i9h warm250k temp0.01 | 1 | 0 | 0 | `32.9` s1 | clear failure |

Important seed-level results:
- i9m: `s1=410.4`, `s2=255.4`, `s3=263.7`, `s4=524.1`, `s5=525.6`.
- i9l: `s1=484.6`, `s2=255.9`, `s3=320.0`, `s4=455.9`, `s5=374.9` so far.
- i9j: `s1=510.3`, `s2=8.4`, `s3=250.0`, `s4=487.5`, `s5=358.5`.
- i9g: `s1=576.8`, `s2=260.3`, `s3=281.7`, `s4=286.4`, `s5=99.0`.
- i9n: `s2=563.2@1.25M`, `s3=468.2@1.00M`, original `s4=476.1`, `s5=218.9@1.00M`; the completed seed-4 rerun only reached `345.0`, so the hard-seed result is not yet repeatable.

Conclusion:
- The best current candidate for a 5-seed push is **i9m Phase1b off at 2M**. It
  has three seeds above 380 and two seeds above 500, including hard seed 4. This
  is the first Iteration 9 family that looks like more than a one-seed spike.
- i9n may be the next breakthrough if early s2/s3 progress holds, but it needs
  completion before promotion because its seed-4 rerun did not reproduce the
  earlier 476 partial.
- i9l is useful but probably not sufficient: it repeatedly gets into the
  450-485 band without crossing 500.
- i9j and i9g are diagnostic: they prove particular schedules can make one seed
  jump, but the variance across follow-up seeds is too large for the 5/5 goal.

What worked:
- Phase1b capacity plus scheduled Glass removal is the strongest pattern. The
  i9m result suggests Glass is useful early but can become late-training drag.
- High MPPI capacity remains important: the strongest branches all use
  `K_UPDATE=128` and `MPPI_NS=2048`.
- Removing latent smoothing can help some seeds, but it is not reliable enough
  as the main recipe.

What failed or looks weak:
- Delayed warmup alone (`i9g`) is not robust despite the best single score.
- Warmup at 250k (`i9h`) is a clear miss.
- Temp-stability alone (`i9b/e/f`) and temp-stability with Phase1b (`i9l/p`) do
  not currently produce the 5/5 profile.
- i9q's hybrid of temp-stability and off-at-2M has no positive early signal yet,
  though the running seeds are still young.

Operational notes:
- Failed tasks in this batch are still mostly queue/process failures after
  partial CSVs, not clear algorithm exceptions. Keep the CSV evidence but do not
  count failed runs as completed seeds.
- Keep all remaining running tasks alive. The pending i9r off-at-1M probes are
  still useful because they test whether i9m should turn Glass off earlier than
  2M.

Next decision:
- If i9m seed 3 or the continuing i9m/i9n runs add one more >500 or lift weak
  seeds past 380, promote the Phase1b-off schedule into Iteration 10 as the main
  candidate and run a clean 5-seed confirmation with fixed code SHA and no dirty
  launcher/source changes.
- If i9m stalls at 3/5 over 380, the next design should tune the off schedule
  around 1M-2M rather than adding more auxiliary losses.

## Queue discipline for Iteration 9

- One seed per task, always through the central queue.
- No direct local manual scripts except smoke tests.
- No `--save_full_state` by default.
- Use short, explicit output tags: `phasei9a_*`, `phasei9b_*`, etc.
- Include `CODE_SHA=<short_sha>` in task env when available.
- Include the short SHA in the task label.
- For code-changing probes, do not edit `src/` again until the queued sentinel
  seed has actually started, unless the older task is cancelled/retried later.
- Before promoting any probe to 5 seeds, inspect CSV + diag + one rollout video.
- Automatic seed promotion rule:
  - If one completed seed reaches `best_any > 380`, launch one more seed.
  - If one completed seed reaches `best_any > 500`, launch two more seeds.
  - If one completed seed reaches `best_any > 600`, launch five more seeds.
  - If a run fails for an infrastructure-fixable reason but has usable eval
    rows, lower the bar by 100 for promotion (`>280`, `>400`, `>500`). This
    keeps SIGKILL/interruption partials useful without counting header-only
    failures.
  - The queue daemon now checks finished tasks and appends follow-up seed tasks
    automatically, guarded by `auto_promoted` metadata to avoid duplicates.

## Off-schedule probes, 2026-05-24

Reason:
- i9m suggests Phase1b Glass is useful early but may become drag late.
- i9r tests off-at-1M. Add midpoint and later-off probes behind the active
  queue to map the useful decay window.

Queued:
- `tb8fd255`: i9s, Phase1b off at 1.5M, seed 1.
- `t6d15d04`: i9t, Phase1b off at 1.5M, hard seed 4.
- `te511ed4`: i9u, Phase1b off at 3.0M, hard seed 4.

## Latest review, 2026-05-25

Queue state at readout time:
- `12` running, `8` pending, `8` done, `11` failed.
- Dashboard, queue daemon, and remote mirror are alive.
- `ssh6_4060` appears idle in the dashboard only because no queue task is
  assigned; direct SSH to `ssh6.vast.ai:11115` returns connection refused, so
  it should be treated as unavailable fleet capacity until the Vast instance is
  fixed or replaced.

Promising-phase dashboard caveat:
- The dashboard groups all `phasei9m_*` result directories into canonical
  `phasei9m` and counts every discovered CSV with eval rows.
- Therefore `phasei9m max 525.6 · mean 368.7 · G1 3/10` is **not** a completed
  3/10 seed success rate. It includes failed/interrupted partial CSVs from both
  the original `p1b_off2m` runs and the fresh `sleep8h` reruns.
- Treat partial failed rows as signal for promotion/debugging, but do not count
  them as completed seeds for a G1 claim.

Current promising-family dashboard readout:

| Family | Dashboard max | Mean | G1 count | Active queue | Interpretation |
|---|---:|---:|---:|---|---|
| i9m Phase1b off at 2M | 525.6 | 368.7 | 3/10 | 2 running, 5 pending | still interesting, but G1 hits are partial/failed |
| i9n Phase1b K128 | 579.8 | 419.8 | 2/6 | 3 running, 3 pending | best current breadth candidate if running seeds mature |
| i9r Phase1b off at 1M | 540.2 | 497.4 | 1/2 | 2 running | very strong early off-schedule signal |
| i9q temp0.01 + off2M | 558.8 | 394.9 | 1/4 | 2 running | one strong auto-promote; still sparse |
| phaseq_knee baseline | 557.2 | 387.9 | 4/11 | none | useful non-Glass reference |
| phasear_restart_K128 | 584.6 | 395.2 | 1/5 | none | high max, not robust |
| phasez TD-MPC2 baseline | 535.4 | 432.4 | 1/4 | none | baseline can still hit G1 |
| phaseg2 temp0.05 | 570.6 | 359.0 | 1/5 | none | one winner, not robust |

`phasei9m` detailed readout:

| Variant | Seed | Queue status | best_any | Best step | Last step | Selector | Notes |
|---|---:|---|---:|---:|---:|---|---|
| `phasei9m_p1b_off2m` | 1 | no matched active task | 410.4 | 9.50M | 10.00M | mppi | completed local historical run |
| `phasei9m_p1b_off2m_s2` | 2 | no matched active task | 255.4 | 3.25M | 6.25M | mppi | weak/partial mirror row |
| `phasei9m_p1b_off2m_s3` | 3 | failed | 263.7 | 3.25M | 6.50M | pi | interrupted/failed |
| `phasei9m_p1b_off2m_s4` | 4 | failed | 524.1 | 7.75M | 8.00M | mppi | G1 partial, not completed |
| `phasei9m_p1b_off2m_s5` | 5 | failed | 525.6 | 2.25M | 3.00M | mppi | G1 partial, not completed |
| `phasei9m_sleep8h_20260524_s1` | 1 | failed | 340.1 | 3.00M | 5.25M | mppi | fresh-tag rerun, partial |
| `phasei9m_sleep8h_20260524_s2` | 2 | failed | 220.2 | 2.00M | 2.25M | mppi | fresh-tag rerun, weak partial |
| `phasei9m_sleep8h_20260524_s3` | 3 | failed | 519.2 | 3.25M | 3.50M | mppi | fresh-tag G1 partial |
| `phasei9m_sleep8h_20260524_s4` | 4 | running | 368.1 | 2.25M | 2.25M | mppi | still running on `ssh6_3080` |
| `phasei9m_sleep8h_20260524_s5` | 5 | running | 254.9 | 1.00M | 1.25M | mppi | still running on `ssh17637_gpu0` |

Pending `phasei9m` auto-promotes:
- `t9c3cb49`: seed 6 from `sleep8h_s1`, triggered by fixable failed-run bar
  (`best_any=340.1`).
- `t1230ed6`, `t4cb9313`, `t738f024`, `te91c729`: seeds 7-10 from
  `sleep8h_s3`, triggered by partial G1 (`best_any=519.2`).

Other latest seed-level readouts:
- i9n: `s2=579.8@3.75M`, `s3=515.3@3.25M`, `s4=476.1@6.50M` on the first
  hard-seed path, `s4_rerun2=345.0@10.00M`, `s5=335.7@7.75M`, and fresh
  `sleep8h_s1=261.3@1.50M` still early. This is the current best candidate if
  the running/pending seeds continue improving.
- i9r: `s1=540.2@5.00M`, `s4=454.6@5.75M`; off-at-1M is now a serious
  candidate because it has both a G1 seed and a high hard-seed partial.
- i9q: `auto_s3=558.8@7.00M`, `s4=405.7@1.50M`, `s1=347.5@5.50M`,
  `s2=267.6@4.75M`; the hybrid temp-stability/off-at-2M branch has one strong
  seed but needs more breadth.
- i9s/i9t off-at-1.5M are still running and not mature enough for a decision.

Updated interpretation:
- Do not use the dashboard G1 denominator as a final success metric while many
  failed/interrupted tasks have usable CSVs. For final claims, require complete
  or deliberately accepted fixed-budget trials under one clean code SHA.
- i9m remains useful mechanistic evidence for "early Glass, then remove it",
  but the apparent `3/10` G1 is inflated by partial failed rows.
- i9n and i9r are now at least as important as i9m. i9n tests whether the
  Phase1b K128 recipe has enough breadth, while i9r tests whether Glass should
  turn off earlier than 2M.
- Next clean confirmation should likely compare three schedules under the same
  recipe and code SHA: Glass off at 1M (`i9r`), 2M (`i9m`), and always-on K128
  (`i9n`).

## Decision rule

If no one-seed probe beats the current hard-seed baseline, finish the pending
Iteration 8 Phase-g2 readout and pivot from "new losses" to "basin-entry
mechanism": restart semantics, exploration schedule, and actor/planner coupling.

## Queue update, 2026-05-26

Fleet cleanup:
- `ssh4_8080` / Vast contract `37565664` was destroyed after repeated PJRT
  pthread creation failures. Do not rent similar JAX workers unless
  `pids.max >= 512` or `max`.
- `ssh4_3060_bar` / contract `37907664` was destroyed after SSH banner
  timeouts and broken `rsync` during setup.
- The dashboard, queue daemon, and stream registry were updated to remove those
  workers.

Run-length support:
- `scripts/run_phasei9_glass_probe.sh` now accepts `TOTAL_STEPS` and
  `EARLY_STOP_PATIENCE`.
- Default remains `10M` steps with `3M` early-stop patience.
- We are not using 12M long confirmations right now. The near-term goal is fair
  5-seed confidence intervals against TD-MPC2, so promising recipes should get
  seeds 1-5 under the standard 10M budget.

Fair-CI queue correction:
- Removed the pending `phasei10d_off1m_long12m` tasks before launch.
- Capped automatic promotion to seeds 1-5 so the daemon does not keep expanding
  promising families to 10 seeds while the current comparison target is 5-seed
  95% CI.
- Fill priority is: `phasei10c` clean off-at-1M, `phasei9r` off-at-1M,
  `phasei9t` off-at-1.5M, and `phasei9q` temp0.01/off-at-2M, because these are
  the current Glass families with mean or early best above the TD-MPC2 baseline
  region.
- Added standard-budget CI fill tasks:
  - `phasei10c_off1m_clean5`: seed 5 added; pending seeds 3-4 reprioritized.
  - `phasei9r_p1b_off1m_fairci`: replacement seed 1 and new seed 5.
  - `phasei9t_p1b_off1p5m_fairci`: seeds 1 and 5.
  - `phasei9q_p1b_temp001_off2m_fairci`: replacement seeds 1 and 5 because the
    prior seed 5 ran on the destroyed `ssh4_8080` worker.
