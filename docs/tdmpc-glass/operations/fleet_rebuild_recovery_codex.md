# TD-MPC-Glass Fleet Rebuild and Recovery Policy

Created: 2026-05-24

This note is for rebuilding the VastAI fleet after budget exhaustion, crashes, or
master migration. It complements `experiment_ops.md` and `hardware_req.md`.

## 1. Current Status Snapshot

Checked from the current master `/root/helios-rl` after the VastAI budget event:

| Component | Status | Action |
|---|---|---|
| Local dashboard `web_dashboard.py` | stopped | restart only after queue cleanup |
| Queue daemon `task_queue_daemon.py` | stopped | keep stopped until stale running tasks are reset |
| Local TD-MPC run | not running | local slot is free |
| Central queue | 23 tasks: 12 running, 7 failed, 4 done | 12 running entries are stale unless verified |

Remote SSH reachability:

| Tag | Last Known GPU | SSH Status | Run Process Status | Keep? |
|---|---:|---|---|---|
| `ssh6_4060` | RTX 4060 8 GB | refused | gone | recycle/re-rent |
| `ssh17637_gpu0` | 3060 Laptop 6 GB | reachable | no run process | marginal |
| `ssh17637_gpu1` | 3060 Laptop 6 GB | reachable | no run process | marginal |
| `ssh1_2080ti` | 2080 Ti 22 GB | reachable | no run process | keep if cheap |
| `ssh3_3070` | RTX 3070 8 GB | reachable | no run process | keep if cheap |
| `ssh6_3080` | RTX 3080 10 GB | refused | gone | recycle/re-rent |
| `ssh3_3060ti` | RTX 3060 Ti 8 GB | reachable | no run process | keep only if cheap |
| `ssh4_8080` | RTX 2060 12 GB | destroyed | PJRT pthread failures | reject future low-`pids.max` boxes |
| `ssh9_2060_gpu0-3` | 4x RTX 2060 6 GB | reachable | no run process | marginal, memory constrained |

Immediate interpretation: the queue ledger is stale. Do not restart the queue
daemon until stale `running` tasks have been either marked failed/retried or
deleted.

## 2. Fleet Selection Standard

Rent for effective throughput, not nominal GPU class. The useful metric is:

```
effective_value = stable_sps / hourly_price
```

Use a box only if it also passes the stability and storage gates below.

### Hard Gates

| Gate | Minimum | Preferred |
|---|---:|---:|
| Driver | 535+ for CUDA 12, 580+ for CUDA 13 | 580.126+ |
| VRAM | 8 GB | 10-16 GB |
| Disk free | 50 GB | 100 GB |
| Host RAM | 16 GB | 32 GB |
| Python | 3.11 | 3.12 |
| JAX smoke | `jax.default_backend() == "gpu"` | same plus HopperHop smoke |
| Stability | no SSH drops in 10 min | survives 30 min smoke |

Reject immediately:

- Driver too old for the chosen JAX wheel.
- `jax.devices()` returns CPU only.
- Disk under 50 GB.
- 6 GB cards unless price is extremely low and task uses conservative memory.
- Containers with `pids.max < 512`; `ssh4_8080` had `pids.max=256` and failed with PJRT pthread creation errors.

### Price/SPS Bars

Approximate known throughput from current runs:

| Class | Observed SPS | Good Price | Max Price | Notes |
|---|---:|---:|---:|---|
| 4070 Ti 12 GB local | ~540 | owned | owned | reference |
| RTX 4060 8 GB | ~540 | <= $0.25/hr | $0.30/hr | best rented class so far |
| RTX 3070/3080 | unknown/currently useful | <= $0.25/hr | $0.35/hr | keep if stable |
| RTX 2080 Ti 22 GB | unknown | <= $0.20/hr | $0.30/hr | large VRAM, older arch |
| RTX 3060 Ti 8 GB | ~100 | <= $0.08/hr | $0.12/hr | slow but stable fallback |
| RTX 2060 12 GB | usable only if cgroup/thread limits pass | <= $0.08/hr | $0.12/hr | reject if `pids.max < 512` |
| 6 GB multi-2060 / laptop 3060 | 200-250 but OOM risk | <= $0.05/hr/GPU | $0.08/hr/GPU | only low-memory probes |

Decision bar:

- **Core fleet**: value >= 1800 SPS per $/hr and VRAM >= 8 GB.
- **Probe fleet**: value >= 1000 SPS per $/hr and stable for 30 min.
- **Do not rent/keep**: value < 800 SPS per $/hr, frequent SSH loss, or repeated OOM.

Example: RTX 4060 at 540 SPS and $0.25/hr gives 2160 SPS/$/hr, so it is a good
core fleet target.

## 3. Target Fleet Shape

For fast iteration:

| Role | Count | Hardware | Workload |
|---|---:|---|---|
| Master/control | 1 | local or cheap CPU VM with 100+ GB disk | dashboard, queue, docs, mirrors |
| Core trainers | 3-5 | 4060/4070/3070/3080, 8+ GB | promising full probes |
| Cheap scouts | 2-4 | 2060/3060 Ti/6 GB cards | one-seed smoke, low-memory probes |
| Storage node | 1 | 200 GB-1 TB disk | checkpoints, videos, full run archive |

Do not let storage compete with training. A cheap CPU instance, object storage,
or external service should handle artifacts.

## 4. Artifact Storage Strategy

### What Must Be Stored

| Artifact | Keep Locally | Remote/Cloud | Notes |
|---|---:|---:|---|
| `seed_*.csv`, `_diag.csv` | yes | yes | source of truth for analysis |
| best checkpoints `best_any/pi/mppi.pkl` | yes | yes | small enough, important |
| `latest_eval.pkl` | yes | optional | useful for inspection |
| full-state checkpoints | no by default | selective | can be ~1 GB each |
| rollout videos | no by default | yes | store only best/interesting runs |
| logs | recent only | compressed archive | keep for debugging failed probes |

### Recommended Backends

- **GitHub**: code, docs, queue templates, small CSV summaries. Do not commit
  checkpoints or videos.
- **Weights & Biases artifacts**: run metadata, CSVs, model checkpoints, selected
  videos. Good fit because the project already uses `wandb`.
- **Hugging Face datasets/models**: publish curated CSVs, checkpoint bundles,
  and videos for reproducibility. Better for shareable artifacts than raw
  constantly-changing scratch files.
- **Rclone to S3/R2/Backblaze**: best for cheap bulk backup of `exp/`.
- **Dedicated storage Vast/CPU node**: acceptable only if it is cheaper than
  object storage and has stable uptime.

Practical split:

- Push code/docs to GitHub every research checkpoint.
- Sync `exp/tdmpc_glass/**/seed_*.csv`, `_diag.csv`, and best checkpoints to
  W&B/HF/object storage every 10-30 min.
- Sync full-state checkpoints only for selected high-value runs.

## 5. Crash Recovery Procedure

Use this after Vast kills instances or a master session dies.

1. Stop local daemons if any are half-alive:

```bash
pkill -f scripts/task_queue_daemon.py
pkill -f scripts/web_dashboard.py
```

2. Probe fleet reachability:

```bash
ssh -p PORT root@HOST 'hostname; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; pgrep -af "run_phasei9|run_benchmark.py" || true'
```

3. For each queue task marked `running`, verify the box still has the process.

4. If the process is gone:

- Pull any remote CSVs/logs before releasing the box.
- Mark the task `failed` if it has useful partial data.
- Use `retry` only after the recipe is still worth running.
- Delete stale low-value tasks rather than blindly retrying the whole old queue.

5. Restart daemons only after stale state is cleaned:

```bash
nohup setsid /root/venv/bin/python3 -u /root/helios-rl/scripts/web_dashboard.py \
  > /tmp/web_dashboard.log 2>&1 < /dev/null &

nohup setsid /root/venv/bin/python3 /root/helios-rl/scripts/task_queue_daemon.py \
  >> /root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log 2>&1 < /dev/null &
```

## 6. Automatic Backup Standard

Minimum viable backup:

```bash
git add docs scripts requirements-*.txt
git commit -m "checkpoint fleet docs and queue workflow"
git push
```

Recommended automated backup daemon:

- Every 10 min: rsync remote CSV/log/checkpoint metadata into local
  `exp/tdmpc_glass/remote_mirror/`.
- Every 30 min: upload CSVs, diag CSVs, queue JSON, and daemon logs to object
  storage/W&B.
- Every 2-4 hours: create a Git commit if docs/scripts changed.
- On every completed high-value seed: upload `best_any.pkl`, `best_pi.pkl`,
  `best_mppi.pkl`, and the CSV pair.

Never auto-commit large binary outputs to Git. Use `.gitignore` and artifact
storage.

## 7. Master Migration

The current master responsibilities are:

- central queue file: `scripts/queues/central_queue.json`
- dashboard: `scripts/web_dashboard.py`
- queue daemon: `scripts/task_queue_daemon.py`
- local mirrored results: `exp/tdmpc_glass/remote_mirror/`
- docs/analysis state: `docs/tdmpc-glass/`
- SSH key: `/home/coder/.ssh/id_ed25519` or configured `SSH_IDENTITY_FILE`

Migration checklist:

1. On old master, stop daemons to freeze queue mutation.
2. Commit and push code/docs.
3. Sync state to new master:

```bash
rsync -az /root/helios-rl/scripts/queues/ new:/root/helios-rl/scripts/queues/
rsync -az /root/helios-rl/docs/tdmpc-glass/ new:/root/helios-rl/docs/tdmpc-glass/
rsync -az --exclude='**/*.pkl' /root/helios-rl/exp/tdmpc_glass/remote_mirror/ new:/root/helios-rl/exp/tdmpc_glass/remote_mirror/
```

4. Install the same Python/JAX stack on the new master if it will train locally.
5. Copy/deploy SSH key or configure a new key across live remotes.
6. Start dashboard first, inspect queue, then start daemon.
7. Do not start two queue daemons against the same fleet.

## 8. Better Fleet Utilization

The fleet should not only run GPU seeds.

Good secondary jobs:

- Independent research agents on cheap CPU/GPU boxes:
  - analyze failed seeds,
  - summarize logs,
  - propose probes,
  - write iteration docs,
  - triage dashboard anomalies.
- CPU-heavy analysis:
  - CSV aggregation,
  - bootstrap confidence intervals,
  - rollout video encoding,
  - checkpoint pruning,
  - artifact upload.
- Environment throughput experiments:
  - EnvPool-style vectorized CPU env tests are useful for CPU-based baselines,
    but HopperHop here is MJX/JAX-GPU, so do not expect EnvPool to accelerate
    the current main path directly.
- Render workers:
  - run rollout rendering on otherwise idle GPUs, with low priority.

Rule: only run secondary jobs through a separate low-priority queue or on boxes
that are not part of active training. The training queue remains the priority.

## 9. Rebuild Plan After This Crash

1. Keep queue daemon stopped.
2. Pull partial artifacts from reachable boxes.
3. Mark stale `running` tasks as failed or retryable based on whether CSVs have
   useful eval rows.
4. Release refused/dead boxes from Vast.
5. Re-rent core fleet with the price/SPS bars above:
   - first target: RTX 4060/4070 8-12 GB, driver 580+, <= $0.25/hr,
   - second target: 3070/3080 if <= $0.30/hr,
   - cheap scout: 2060/3060Ti only if <= $0.10/hr and stable.
6. Add a storage backend before launching many more full-state runs.
7. Relaunch only the highest-signal Iteration 9 probes:
   - `phasei9n`: Phase1b K=128 variance/fill,
   - `phasei9m`: Phase1b off after 2M,
   - `phasei9g`: warmup 500k + temp stability,
   - `phasei9q/r/s/t/u`: Glass handoff timing ladder.

## 10. Open Engineering Tasks

- Add a dashboard action to mark stale running tasks failed after remote process
  is gone.
- Add automatic artifact upload after each seed completes.
- Add a `fleet.json` registry separate from hard-coded `BOXES`.
- Add per-box benchmark calibration: smoke for 250k env steps, record SPS,
  price, driver, pids limit, and stability score.
- Add a master migration script that packages queue/docs/mirror metadata.
