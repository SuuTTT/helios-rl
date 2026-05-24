# TD-MPC-Glass HopperHop — Live Dashboard

**Two goals** (iter 6):
- **G1 — Consistency**: all 5 seeds > MPPI 500 (we're currently at 2-of-5: s3=523, s8=525)
- **G2 — Ceiling**: can we break MPPI 600? (Phase-t s2 hit 612 with knee penalty)

Phase-1b baseline finals reference: `[526, 526, 294, 227, 562]`, mean 427, 2/5 > 500.

**For resuming after a session restart, see [dashboard_resume.md](dashboard_resume.md).**

Refresh with: `bash scripts/iter5_dashboard.sh` (terminal, all 6 boxes auto-detected)

Or web dashboard (live boxes + Plotly curves + click-to-render rollouts):

```
bash scripts/launch_web_dashboard.sh          # blocks, Ctrl-C to stop
# or background:
nohup setsid bash scripts/launch_web_dashboard.sh > /tmp/web_dashboard.log 2>&1 & disown
```

Then open `http://localhost:5055`. Auto-refreshes boxes every 30 s and curves every 60 s. The "Render Rollout" section lists every checkpoint that has `best_mppi.pkl` on disk; click → background `render_glass_rollout.py` job with progress bar, MP4 plays inline when done.

## Hardware fleet (current)

| Box | GPU | VRAM | sps | Stability | Currently running |
|---|---|---|---|---|---|
| Local | RTX 4070 Ti | 12GB | ~540 | high | **Phase-z s1-5 (iter 6 Q1, vanilla TD-MPC2 baseline)** |
| 78.83.187.54:17637 GPU0 | RTX 3060 Lap | 6GB | ~250 | flaky | **Phase-q s1 / Phase-z s4-5 (iter 6 Q2)** |
| 78.83.187.54:17637 GPU1 | RTX 3060 Lap | 6GB | ~250 | flaky | **Phase-q s2 / Phase-y s4-5** |
| ssh6:11115 | RTX 4060 | 8GB | ~540 | high | **Phase-x s8=525 (DONE), Phase-q s3-5 queued** |
| **ssh1:34217** | **RTX 2080 Ti** | **22GB** | **~384** | TBD | **Phase-q s6 (queued, G2 ceiling)** |
| **ssh3:15229** | **RTX 3070** | **8GB** | **~445** | TBD | **Phase-q s7 (queued, G2 ceiling)** |
| **ssh6:16779** | **RTX 3080** | **10GB** | **~329** | TBD | **Phase-q s8 (queued, G2 ceiling)** |
| ~~ssh3:11271 (3060Ti)~~ | RTX 3060 Ti | 8GB | ~120 | DESTROYED 2026-05-18 | killed |
| ~~ssh9:16233 (3090)~~ | RTX 3090 | 24GB | — | BLOCKED | driver 535 incompatible |

## Top-line current results (iter 5 + iter 6 in flight)

| Phase | Seed | Best MPPI | Status |
|---|---|---|---|
| **Phase-x s3** (Path 9) | local | **523.5** ✅ | DONE |
| **Phase-x s8** (Path 9) | 4060 | **524.8** ✅ | running, plateau near s3's level @ 8.75M, expected to end ~10M |
| Phase-y s3 (Path 10) | 3060Ti | **461.8** | DONE |
| Phase-x s1 v1 (Path 9) | dead box | 453 | OOM-truncated @ 4.5M, archived |
| Phase-x s1 v2 (Path 9) | dead box | 380.9 | OOM-killed @ 5.25M |
| Phase-x s6 (Path 9) | local | 287.3 | DONE |
| ns1024 s5 (NS=1024 variant) | 2x3060 | 272.7 (v1) | restarted, fresh CSV |
| Phase-x s9 (Path 9) | local | 234.3 | DONE |
| Phase-v s3 (Path 7) | 4060 | 232.0 | DONE |
| Phase-y s2 (Path 10) | 3060Ti | 211.1 | DONE |
| Phase-v s1 (Path 7) | local | 218.0 | DONE |
| Phase-y s4 v1 (Path 10) | 2x3060 | 382.9 (lost to relaunch) | rerunning |
| Phase-y s1 (Path 10) | 3060Ti | 185.7 (overwritten) | data lost |
| **Phase-x s4** (Path 9) | 4060 | ~15 | stuck-seed DONE |
| **Phase-v s2** (Path 7) | local | 19.9 | stuck-seed DONE |
| **Phase-x s2** (Path 9) | dead box | 5.8 | OOM-killed, stuck |

**Phase-x (Path 9 NS=2048) seed stats so far** (5+ completed): variance is wide.
`{523, 501 climbing, 453 partial, 287, 234, 15}` — 2/5 winners, 1/5 stuck, 2/5 mid.

## Phase legend (full iter 1–6)

See `docs/tdmpc-glass/iterations/iteration_6_plan.md §0` for the full ledger of all 23+
phases. Quick "what works" summary:

| Intervention | Best | Notes |
|---|---|---|
| Knee penalty (Phase-t) | **612** | best — 2/3 > 500 — benchmark-unfair |
| Glass OFF after 2M (Phase-o) | 577 | hybrid |
| Smoothing 1e-3 (Phase-f/j) | 572 / 518 | reliable |
| EXPL_UNTIL=500k (Phase-p) | 538 | helps winners |
| NS=2048 MPPI (Phase-x) | 523 | reliable winner — Path 9 |
| Hierarchical Glass (Phase-y) | 462 | close — Path 10 |

## What to run NEXT (suggestion)

### Currently running (let them finish, ~5-10h)
- **Phase-z** vanilla TD-MPC2 5-seed baseline — Q1 of iter 6
- **Phase-q** knee penalty no-Glass 5-seed — Q2 of iter 6
- Phase-x s7, s8 — finish out the Path 9 CI sweep
- Phase-y s4 (rerunning) — finish out Path 10 sweep

### Next priorities once boxes free up

1. **Path 4 (Phase-s): BC from Phase-x s3 (523 winner)** — TOP priority.
   Per iter 5 §5.3, stuck-seed problem is exploration-bound. BC pre-training
   transplants the winner's policy into other seeds. Implementation: collect
   ~5 episodes of inference from `phasex_local/seed_3/checkpoints/best_mppi.pkl`,
   pre-train pi on (obs, action) pairs for ~10k updates, then continue normal training.

2. **Reward-shaping for Path 9 hybrid** — combine Phase-x (NS=2048) with knee penalty
   to push past 612 ceiling. Benchmark-unfair but cleanly shows what the algorithm
   class can do physically.

3. **Stuck-seed checkpoint resume** — fix `run_benchmark.py` to load + APPEND CSV on
   restart instead of overwriting. Eliminates the trajectory-loss problem we hit
   ~5 times in iter 5.

### Do NOT run (per ledger §0.2)

- Path P / Pa (intrinsic cluster entropy) — falsified
- Path 7 (Phase-v cluster-id obs) — falsified
- Phase 1c (act_noise anneal) — falsified
- Phase-i (smooth=1e-4), Phase-k (λ_temporal=0.05) — too weak/aggressive

## Iter-6 interpretation pending

We expect Phase-z baseline (5 seeds) result tomorrow. If vanilla TD-MPC2 +
NS=2048 produces 2-of-5 > 500 (matching Phase-x), then **Glass is neutral**
(it doesn't help or hurt). If it produces fewer winners, **Glass adds value**.
If it matches or beats Phase-x, **drop Glass entirely** (the whole research
project's premise weakens).

Phase-q (knee+no-glass) tells us the physical ceiling. If 3+ of 5 break 600,
then the answer to "can HopperHop be solved consistently" is YES; the only
question is whether algorithm-internal tricks (Path 4 BC, etc.) can match
that ceiling without modifying rewards.

## Core problem statement (from iter 6 plan §2)

> HopperHop has at least two gait basins. Random initial conditions during the
> 500k EXPL_UNTIL phase determine which basin a seed lands in. Stuck-basin seeds
> plateau at MPPI 0-300 regardless of any algorithm intervention. The 5-of-5 > 500
> goal is fundamentally a basin-lottery problem, not a representation/planning
> problem.

## Backup hygiene reminder

CSVs that got overwritten in iter 5 (lessons learned):
- Phase-y s4 v1 = 382.9 (lost — watcher relaunch without backup, before watcher v2)
- Phase-y s1 = 185.7 (lost — watcher's queue-relaunch wiped seed_1 before backup)
- All others have `_v1_<timestamp>.csv` backups

Watcher v2+ now backs up CSV before any relaunch. Same protection should be
added to launcher scripts (use `tee -a` not `tee`).
