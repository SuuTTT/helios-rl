# Idea Queue: Research Inbox to GPU Probes

Date: 2026-05-25

This is the first version of a research idea queue for TD-MPC-Glass. It sits
above the existing task queue:

```text
idea_queue.json
  idea: "try off-schedule curriculum"
    probe specs: launcher + env + pass/kill rule
      ↓ enqueue
central_queue.json
  GPU task: one seed on one box
      ↓ daemon
VastAI/local worker
      ↓ CSV/diag/checkpoint/video
dashboard + analysis
      ↓ evidence back onto idea
idea status: promoted / retired / needs-code / confirmed
```

The idea queue is deliberately file-backed and inspectable. It does not require
a database before the workflow is stable.

## Files

- `scripts/queues/idea_queue.json`: research ideas and probe specs.
- `scripts/idea_queue.py`: CLI for add/list/claim/probe/enqueue/evidence.
- `scripts/queues/central_queue.json`: existing GPU task queue.
- `docs/tdmpc-glass/iterations/`: human-readable iteration records.

## States

Idea states:
- `new`: captured but not triaged.
- `triage`: an agent is converting it into a testable design.
- `probe_designed`: at least one launcher/env probe exists.
- `queued`: probes are in `central_queue.json`.
- `running`: queue tasks are active.
- `needs_code`: idea requires source edits before more probes.
- `promoted`: evidence justifies more seeds or a clean confirmation.
- `retired`: evidence is weak or falsified.
- `confirmed`: clean fixed-SHA run meets the target.

Probe states:
- `designed`: launcher/env exists in idea queue only.
- `queued`: copied into central queue.
- `running`: central queue task is running.
- `done`: central queue task completed.
- `failed`: infra or algorithm failure.

## Add An Idea

```bash
cd /root/helios-rl

/root/venv/bin/python3 scripts/idea_queue.py add \
  --title "Compare Glass off schedules under Phase1b K128" \
  --goal "Find a 5/5 HopperHop G1 recipe" \
  --hypothesis "Glass helps early representation but becomes late-training drag; off-at-1M may beat off-at-2M and always-on." \
  --metric "5/5 seeds best_any >= 500; at least one seed >600 is G2" \
  --tags "hopperhop,glass,off-schedule,iteration9" \
  --priority 6 \
  --owner "human"
```

List ideas:

```bash
/root/venv/bin/python3 scripts/idea_queue.py list
```

Claim an idea for an agent:

```bash
/root/venv/bin/python3 scripts/idea_queue.py claim i123abc --agent codex-ec2
```

## Add Probe Specs

A probe spec is a queue task template: launcher, env, priority, pass rule, and
kill rule.

```bash
/root/venv/bin/python3 scripts/idea_queue.py add-probe i123abc \
  --label "i10a off1m clean seed 1" \
  --launcher "scripts/run_phasei9_glass_probe.sh" \
  --env "PROBE_ID=phasei10a_off1m_clean_s1 SEEDS=1 K_UPDATE=128 MPPI_NS=2048 EXPL_UNTIL=25000 TEMP_STABILITY=0.0 GLASS_WARMUP=100000 GLASS_DECAY=1000000 PROTO_TEMP=0.7 ASSIGN_SCALE=0.5 STOPGRAD=true LATENT_SMOOTH=0.0 LATENT_SMOOTH_WARMUP=0 CODE_SHA=$(git rev-parse --short HEAD)-dirty XLA_FLAGS=--xla_gpu_autotune_level=0 TMPDIR=/root/helios-rl/tmp" \
  --priority 8 \
  --pass-rule "best_any >= 500 by 5M or >= 380 by 3M with improving diag" \
  --kill-rule "best_any <100 at 1M and no standing/full-reward progress"
```

Enqueue all unqueued probes for that idea:

```bash
/root/venv/bin/python3 scripts/idea_queue.py enqueue i123abc
```

This appends pending tasks to `scripts/queues/central_queue.json` with
`idea_id` and `idea_probe_id` metadata.

## Add Evidence

After a run finishes or a dashboard readout changes the decision:

```bash
/root/venv/bin/python3 scripts/idea_queue.py evidence i123abc \
  --summary "i9r off-at-1M reached 540.2 on s1 and 454.6 on hard s4; promote clean off-schedule comparison." \
  --path "docs/tdmpc-glass/iterations/iteration_9.md" \
  --decision "promote" \
  --by codex
```

## Agent Contract

An autonomous or semi-autonomous research agent should do this loop:

1. Claim the highest-priority `new` or `triage` idea.
2. Read current iteration docs and dashboard summaries.
3. Convert the idea into one to three probe specs.
4. Prefer flag-only probes. If code changes are needed, mark `needs_code`.
5. Smoke test code-changing probes before enqueueing.
6. Enqueue only one sentinel seed per probe unless evidence already meets the
   promotion rules.
7. Monitor central queue results.
8. Add evidence to the idea.
9. Promote, retire, or design the next probe.

The agent should not silently modify `src/` for multiple queued ideas at once.
If a task will run against dirty code, record the dirty SHA in `CODE_SHA`.

## Safety Rules

- One queue master only. Do not run multiple `task_queue_daemon.py` processes
  on different machines.
- One code-changing idea at a time.
- Use fresh `PROBE_ID` values so old CSV history does not pollute dashboards.
- One seed per central task.
- Failed/partial CSVs may trigger follow-up probes but do not count as complete
  success for G1.
- Store large artifacts outside GitHub.

## Storage and Crash-Resistance Stack

Use the split from `storageAWS.md`:

| Data | Home | Reason |
|---|---|---|
| Code/docs/configs | GitHub | versioned, small |
| Master queue state | EC2 EBS + S3 snapshots | must survive local/Vast crashes |
| Metrics | W&B | scalar history and comparisons |
| Best/final checkpoints | Hugging Face or S3 | shareable, larger binaries |
| Frequent resume checkpoints | S3/B2/local worker disk | cheap crash recovery |
| CSV/diag mirror | EC2 EBS + S3 | dashboard + analysis |

Recommended master cron on EC2:

```bash
*/15 * * * * cd /root/helios-rl && S3_URI=s3://YOUR-BUCKET/control bash scripts/backup_control_plane.sh >> exp/tdmpc_glass/logs/daemons/backup.log 2>&1
```

Without S3 configured, the same script writes local archives under
`exp/tdmpc_glass/control_backups/`.

Recommended worker habit:

```text
local worker disk: latest crash checkpoint every 10-30 min
S3/B2: periodic resume checkpoint and CSV/diag upload
HF/W&B: selected best/final artifacts only
```

## Next Upgrades

The current implementation is intentionally simple. The next useful upgrades:

- Dashboard panel for idea queue status and linked central tasks.
- A `research_agent.py` runner that claims one idea and invokes Codex/OpenAI
  with a bounded prompt containing docs, current queue state, and code diff.
- S3 snapshot script for `central_queue.json`, `idea_queue.json`, docs, logs,
  and `remote_mirror`.
- Fleet registry file so the queue daemon does not hard-code Vast hosts.
- Automatic stale-running detector after a budget shutdown.
