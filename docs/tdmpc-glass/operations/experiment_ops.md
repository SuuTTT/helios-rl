# Experiment Ops — Dashboard & Queue Guide

How to push a run, monitor it live, and iterate on algo changes across the
helios-rl GPU fleet. Written for teammates who are new to the stack.

---

## Architecture in 60 seconds

```
central_queue.json          ← the task ledger (plain JSON)
         │
         ▼
scripts/task_queue_daemon.py ← daemon: claims idle boxes, syncs code, launches tasks
         │
         ├──► remote box: nohup bash <launcher>.sh (writes seed_N.csv + _diag.csv)
         ├──► local box: subprocess bash <launcher>.sh (same task/log contract)
         │
iter5_stream_remotes.sh     ← daemon: rsync mirror every 5 min → remote_mirror/<box>/
         │
         ▼
web_dashboard.py (Flask)    ← http://localhost:5055
         ├── Box Fleet  (live GPU/CPU, running phase·seed·MPPI, ETA)
         ├── Run Inspector  (per-task card: SPS, patience, behaviour diag)
         ├── Task Queue  (add / delete / retry / reprioritize tasks)
         └── Learning Curves  (Plotly of all active CSVs)
```

**Queue file**: `scripts/queues/central_queue.json` — hand-edit or use the API.
**Launchers**: `scripts/run_phase*.sh` — one script per experiment variant.
**Output CSV**: `exp/tdmpc_glass/HopperHop_<TAG>/seed_N.csv` (local) or mirrored
from remotes under `exp/tdmpc_glass/remote_mirror/<box>/`.

---

## 1. Pre-requisites (one-time per machine)

### SSH key

All remote boxes accept `root@` login with the coder SSH key:

```bash
ssh-add /home/coder/.ssh/id_ed25519
# Verify access to a specific box
ssh -p 11115 -i /home/coder/.ssh/id_ed25519 root@ssh6.vast.ai echo ok
```

### Python environment

```bash
source /root/venv/bin/activate
export PYTHONPATH=/root/helios-rl/src:/root/mujoco_playground_repo
```

The `run_benchmark.py` launcher scripts set these automatically.

---

## 2. Box fleet

| Tag | SSH | GPU | VRAM | XLA_MEM | Notes |
|---|---|---|---|---|---|
| `local` | — | 4070 Ti | 12 GB | 0.85 | local machine |
| `ssh1_2080ti` | `ssh1.vast.ai:34217` | 2080 Ti | 22 GB | 0.75 | |
| `ssh1_a4000` | `ssh1.vast.ai:24456` | A4000 | 16 GB | 0.75 | Python 3.11 + CUDA12 JAX validated on 2026-05-31 |
| `ssh6_3080` | `ssh6.vast.ai:16779` | 3080 | 10 GB | 0.75 | |
| `ssh9_2060_gpu0` | `ssh9.vast.ai:17647` | 2060 (slot 0) | 6 GB | 0.35 | 4-GPU box |
| `ssh9_2060_gpu1` | `ssh9.vast.ai:17647` | 2060 (slot 1) | 6 GB | 0.35 | 4-GPU box |
| `ssh9_2060_gpu2` | `ssh9.vast.ai:17647` | 2060 (slot 2) | 6 GB | 0.35 | 4-GPU box |
| `ssh9_2060_gpu3` | `ssh9.vast.ai:17647` | 2060 (slot 3) | 6 GB | 0.35 | 4-GPU box |

Removed workers:

| Tag | Contract | Reason |
|---|---:|---|
| `ssh4_8080` | `37565664` | destroyed after repeated PJRT pthread creation failures; cgroup `pids.max=256` is too low for this JAX workload |
| `ssh4_3060_bar` | `37907664` | destroyed after repeated SSH banner timeouts and broken `rsync` during setup |
| `ssh17637_gpu0` / `ssh17637_gpu1` | unknown | user destroyed on 2026-05-27; removed from dashboard and central queue dispatch |

Fleet retirement rule:

- If any Vast GPU slot is unreachable or disconnected for 24 consecutive hours,
  stop assigning new tasks to it, mark any still-`running` queue entries on that
  slot as failed with `failure_reason=box_disconnected_gt_24h`, sync whatever
  logs are available, and destroy the instance from Vast.
- Do not leave disconnected boxes in `scripts/task_queue_daemon.py` or
  `scripts/web_dashboard.py`; stale fleet entries make the queue look busy and
  can hide real idle capacity.

**Tip**: the dashboard Box Fleet section shows live GPU%, mem, CPU%, running seed,
best MPPI, ETA. Check there before SSHing.

---

## 3. Confirm the stack is alive

```bash
# Dashboard
pgrep -fa web_dashboard.py         # should print a PID + path

# Remote mirror sync
pgrep -fa iter5_stream_remotes.sh

# Central queue daemon
pgrep -fa task_queue_daemon.py
```

If any are dead, see section 8 (Restart playbook).

Open the dashboard: **http://localhost:5055**

---

## 4. Pushing a run

### Path A — via the dashboard UI (easiest)

1. Open http://localhost:5055 → scroll to **Task Queue**.
2. Fill in the **Add task** form:
   - **Label**: human-readable name, e.g. `phaseac Glass K128 seed 1`
   - **Launcher**: path relative to repo root, e.g. `scripts/run_phaseac_codex_glass_5seed.sh`
   - **Env vars**: space-separated overrides, e.g. `SEEDS=1 K_UPDATE=128 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65`
   - **Priority**: lower number = higher priority. Default 10. Use 5 for urgent runs.
3. Click **Add**. The task appears as `pending`.

The central queue daemon picks it up within about 60 seconds when a box is free.

### Path B — via curl (scriptable)

```bash
curl -s -X POST http://localhost:5055/api/queue \
  -H 'Content-Type: application/json' \
  -d '{
    "label": "phaseac Glass K128 seed 1",
    "launcher": "scripts/run_phaseac_codex_glass_5seed.sh",
    "env": "SEEDS=1 K_UPDATE=128 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65",
    "priority": 10
  }'
```

Returns `{"ok": true, "id": "t<hex>"}`.

### Path C — direct SSH launch (bypass queue, immediate)

Use when you want a run to start right now on a specific box you know is free:

```bash
ssh -p 11115 -i /home/coder/.ssh/id_ed25519 root@ssh6.vast.ai \
  "cd /root/helios-rl && \
   SEEDS=1 K_UPDATE=128 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
   nohup setsid bash scripts/run_phaseac_codex_glass_5seed.sh \
   > /tmp/fleet_manual_s1.log 2>&1 < /dev/null & disown; echo launched"
```

This bypasses the queue daemon — the dashboard will still see the process via the
box probe and show it in Box Fleet, but it won't track it as a queue task.

### Task lifecycle

```
pending → (daemon: box goes idle) → running → done
                                              ↓
                                           failed  (use retry button to re-queue)
```

To retry a failed/done task: Task Queue → click **retry** on the task row.
To bump priority: use the ▲/▼ arrows on the task row.
To cancel: click **×**.

### Auto-promotion discipline

The queue daemon auto-checks finished `scripts/run_phasei9_glass_probe.sh`
single-seed tasks with parseable eval rows:

- `best_any > 380`: append one more seed from the same recipe.
- `best_any > 500`: append two more seeds from the same recipe.
- `best_any > 600`: append five more seeds from the same recipe.
- If the run fails from an infrastructure-fixable interruption but has eval
  rows, lower the trigger bars by 100 (`>280`, `>400`, `>500`).

The daemon records `auto_promoted` metadata on the completed task so a restart
does not duplicate the generated follow-up seeds.

---

## 5. Writing a launcher script

Copy the template and adjust:

```bash
cp scripts/run_phaseab_codex_tdmpc2_5seed.sh scripts/run_myphase_5seed.sh
```

**Required structure:**

```bash
#!/usr/bin/env bash
set -u; set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}

# Accept overrides from task env field
SEEDS=${SEEDS:-"1 2 3 4 5"}
MY_HPARAM=${MY_HPARAM:-0.001}

# REQUIRED: set output tag so data doesn't clobber other phases
export TDMPC_GLASS_OUTPUT_TAG="myphase_${MY_HPARAM}"
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"

echo "[myphase] start $(date -u +%FT%TZ) seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 \
    --tasks HopperHop \
    --total_steps 10000000 \
    --seed "$seed" \
    --k_update 128 \
    --mppi_n_samples 2048 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
done
```

**Rules:**
- **One seed per task** — queue `SEEDS=1`, `SEEDS=2`, … as separate tasks so
  each gets an ETA and can be retried individually.
- **`TDMPC_GLASS_OUTPUT_TAG`** — always set this. It determines where CSVs land.
  Without it, every run writes to the untagged `HopperHop/` directory and will
  overwrite each other.
- **`tee -a`** (append) not `tee` — so a restart doesn't truncate the log.
- **`--no_plot`** — always. Remote boxes have no display.
- **`--save_full_state`** — default off for queue sweeps. Full-state checkpoints
  can grow to around 1 GB after replay/env state fills, so enable it only for a
  specific resume/debug run with enough disk headroom.
- **Code sync** — remote queue launches sync both `scripts/` and `src/` before
  starting. After changing daemon sync/launch behavior, restart
  `scripts/task_queue_daemon.py`.

---

## 6. Key `run_benchmark.py` flags

| Flag | Default | Notes |
|---|---|---|
| `--algos` | `tdmpc-glass` | `tdmpc2`, `tdmpc-glass`, `ppo`, `sac` |
| `--tasks` | `HopperHop` | any MuJoCo Playground task |
| `--total_steps` | 1 000 000 | env steps; 10M recommended for full runs |
| `--seed` | 1 | single seed per invocation |
| `--k_update` | 64 | gradient updates per collection batch; **128** is current best |
| `--mppi_n_samples` | 512 | MPPI sample count; **2048** for better planning |
| `--expl_until` | 25 000 | random exploration steps at start |
| `--latent_action_smooth_coef` | 0.0 | latent smoothing; 1e-3 is the active default |
| `--latent_smooth_warmup_env_steps` | 0 | steps before smoothing kicks in |
| `--early_stop_patience` | 0 (off) | stop N steps after best MPPI; 3M recommended |
| `--resume_checkpoint` | — | path to `*.pkl`; full resume needs `*_full.pkl` |
| `--save_full_state` | false | saves replay buffer + env state for exact resume; avoid by default in queue sweeps due disk cost |
| `--glass_*` | various | Glass-specific knobs (see `--help`) |

---

## 7. Monitoring a run

### Box Fleet panel

Shows every box with:
- **GPU%**, **mem**, **CPU%** — live from SSH probe
- **phase·seed** — the `TDMPC_GLASS_OUTPUT_TAG` + seed from the process env
- **best MPPI** — from the local (or mirrored) CSV
- **last MPPI** — most recent eval
- **SPS** — steps/s (derived from elapsed time + last_step)
- **ETA** — SPS-based: patience-aware remaining steps / SPS

Color coding: green = best ≥ 500; yellow = 300–499; red = < 300.

### Run Inspector panel

Expandable card per running task. Shows three sections:

**System** — GPU%, mem, CPU%, reachable, wall-clock runtime.

**Training Progress** — SPS, last step, last/best MPPI, patience left (color
coded: green > 1.5M, yellow 0.5–1.5M, red < 0.5M), ETA.

**Behaviour Diag (last eval)** — read from `seed_N_diag.csv`:
- `standing_rate` — fraction of eval episode the hopper is standing; < 20% = stuck.
- `falls/ep` — knee touches per episode; 0 = stable hopping.
- `time-to-hop` — env steps from episode start to first full reward; < 60 = hopping fast.
- `full-rew rate` — fraction of steps earning the full reward.

**Artifacts** — full launch command, log path, output dir, checkpoint path. The
**tail log** button fetches the last 60 lines from `/tmp/tqd_<id>.log` on the
remote box.

### Learning Curves panel

Plotly view of every `HopperHop_<phase>/seed_*.csv`. Filters:
- **MPPI only** toggle — hide `pi` eval rows
- **Only running** toggle — hide completed seeds
- **phase contains** text box — narrow by phase name substring

### Terminal alternative

```bash
bash scripts/iter5_dashboard.sh
```

Prints per-box: running procs, GPU/CPU util, CSV best/last MPPI. Runs once and
exits — wrap in `watch` for live updates.

---

## 8. Output paths

```
exp/tdmpc_glass/
  HopperHop_<TAG>/              ← local runs
    seed_N.csv                  ← step,reward,eval_type,seed (eval_type: pi|mppi)
    seed_N_diag.csv             ← step,eval_type,seed,full_reward_rate,standing_rate,fall_count,time_to_first_full
    seed_N/checkpoints/
      best_mppi.pkl             ← model weights at best MPPI
      best_pi.pkl               ← model weights at best deterministic actor eval
      best_any.pkl              ← best of pi or MPPI
      latest_eval.pkl           ← latest eval checkpoint
      latest_full.pkl           ← full state for exact resume (only with --save_full_state)
  remote_mirror/<box>/          ← rsync'd copies of remote runs (updated every 5 min)
    HopperHop_<TAG>/seed_N.csv
  rollout_videos/<job_id>.mp4   ← dashboard-triggered render outputs
  logs/<TAG>/HopperHop_seed_N.log
  glass_diag/HopperHop_<TAG>/seed_N/step_<N>.npz  ← Glass transition matrices
```

---

## 9. Rendering a rollout

In the dashboard → **Render Rollout** section:
1. Select checkpoint from the dropdown (auto-discovered from checkpoints/).
2. Choose camera (`cam0` for HopperHop — do not use the default free camera).
3. Choose episode length (short=250, medium=500, long=1000, extra=2000 steps).
4. Click **Render** → watch progress bar.
5. Video appears inline when done (also saved to `exp/tdmpc_glass/rollout_videos/`).

Via API:

```bash
curl -s -X POST http://localhost:5055/api/queue/render \
  -H 'Content-Type: application/json' \
  -d '{
    "ckpt": "exp/tdmpc_glass/HopperHop_phaseab_codex_tdmpc2_k128/seed_1/checkpoints/best_mppi.pkl",
    "env_id": "HopperHop",
    "camera": "cam0",
    "n_episodes": 1,
    "episode_length": 500
  }'
```

---

## 10. Restart playbook

If any daemon died:

```bash
# 1. Remote CSV mirror (rsync every 5 min)
nohup setsid /root/helios-rl/scripts/iter5_stream_remotes.sh \
  > /root/helios-rl/exp/tdmpc_glass/logs/daemons/stream.log 2>&1 < /dev/null & disown

# 2. Central queue (poll boxes every 60s, launch next pending task)
nohup setsid /root/venv/bin/python3 /root/helios-rl/scripts/task_queue_daemon.py \
  >> /root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log 2>&1 < /dev/null & disown

# 3. Web dashboard
nohup setsid /root/venv/bin/python3 -u /root/helios-rl/scripts/web_dashboard.py \
  > /tmp/web_dashboard.log 2>&1 < /dev/null & disown
```

All three survive session close (PPID=1). Verify: `ps -o pid,ppid,cmd <PID>`.

Logs:
```
/root/helios-rl/exp/tdmpc_glass/logs/daemons/stream.log
/root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log
/tmp/web_dashboard.log
```

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Box shows `best —` / `last —` | Mirror outdated or run hasn't done first eval (~250k steps) | `pgrep -fa iter5_stream_remotes`; restart if dead. Otherwise just wait. |
| Box marked "unreachable" | SSH timeout / vast.ai box rebooted | `ssh -p <port> root@<host> echo ok` to confirm. If gone, comment box row in `BOXES` in `web_dashboard.py` and `task_queue_daemon.py`. |
| Task stuck as "running" after dashboard restart | Orphaned task — render tasks are auto-reset on startup; training tasks need a manual retry | Click **retry** in Task Queue. |
| Queue keeps re-launching same seed | Launcher exits non-zero before GPU warms up, box still probes as idle | Check `/root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log` and the box's `/tmp/tqd_<id>.log` |
| Remote task fails with `unexpected keyword argument` after code changes | Remote `src/` is stale or daemon sync logic was not restarted | Restart `task_queue_daemon.py`; verify it syncs both `scripts/` and `src/`. |
| Local queued task immediately becomes done with stale log | Old `/tmp/tqd_<id>.log` ownership or local launcher issue | Remove stale `/tmp/tqd_<id>.log`, restart `task_queue_daemon.py`, then retry the task. |
| Task is `done` after 1-5 min with no eval rows | Launcher/JIT failed before first eval, or box disappeared before logs were mirrored | Tail `/tmp/tqd_<id>.log`; if CSV has only the header, retry the task. Current daemon marks future nonzero launcher statuses as `failed` instead of `done`. |
| PTX/JAX compile error says `Could not open output file .../tmpxft_*` | Remote `TMPDIR` or overlay was full/unwritable during JIT | Check `df -h /root/helios-rl/tmp /tmp`; clean caches/checkpoints, ensure `TMPDIR=/root/helios-rl/tmp`, then retry. |
| `ptxas fatal: Internal error: writing file` | Disk/cache write failure, usually disk pressure | `df -h`; delete low-value full checkpoints/caches; rerun through queue after confirming free space. |
| "no active csvs" in stream monitor | CSV is header-only (first eval not yet written) | Wait ~250k env steps for first eval row. Or verify `TDMPC_GLASS_OUTPUT_TAG` is set correctly in the launcher. |
| MJX Warp-901 crash after ~1M steps | `act_noise` > 0.30 triggers graph-capture bug | Do not set `act_noise` above 0.30 on HopperHop. |
| Run Inspector shows `standing_rate —` | `_diag.csv` not found (run pre-dates diag logging, or wrong output tag) | Only runs from 2026-05-19 onward write diag CSVs. |

---

## 12. Queue REST API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/queue` | All tasks sorted by priority; includes ETA fields |
| `POST` | `/api/queue` | Add task `{label, launcher, env, priority}` |
| `DELETE` | `/api/queue/<id>` | Force-delete any task |
| `POST` | `/api/queue/<id>/priority` | Adjust priority `{delta: ±1}` |
| `POST` | `/api/queue/<id>/retry` | Reset running/failed/done → pending |
| `GET` | `/api/queue/<id>/log` | Last 60 lines from `/tmp/tqd_<id>.log` on the box |
| `POST` | `/api/queue/render` | Queue a render task |
| `GET` | `/api/boxes` | Live GPU/CPU stats for all boxes |
| `GET` | `/api/curves` | All discovered CSV paths |
| `GET` | `/api/checkpoints` | All discovered checkpoint `.pkl` files |
