# gpu-fleet — reusable multi-box GPU experiment runner

Three lightweight processes that together let you queue arbitrary training runs across
a fleet of local and remote GPU machines, watch them from a browser, and get ETA
estimates — all with zero cloud dependencies.

```
fleet_dashboard.py   Flask UI (port 5055) — Box Fleet, Task Queue, Metrics
fleet_daemon.py      Poll loop — claims idle boxes, SSH-launches tasks
config.yaml          Single source of truth for boxes, paths, and ports
```

The queue is a plain JSON file; the launcher is any shell script; the metric viewer
reads any CSV with a `step` column.  No Kubernetes, no Ray, no special agent code.

---

## 5-minute quickstart

### 1 — Install

```bash
pip install flask pyyaml pandas plotly    # all the Python deps
```

### 2 — Configure

```bash
cp tools/gpu-fleet/config.yaml.example tools/gpu-fleet/config.yaml
```

Edit `config.yaml`:
- Set `fleet.repo` to the **absolute path** of your repo (must be identical on all remote boxes).
- Add your boxes under `boxes:` — see the example for single-GPU, dual-GPU, and local entries.
- Set `ssh.key` to the private key whose public half is in `authorized_keys` on every remote.
- Set `dashboard.exp_dir` to wherever your experiment CSVs land.

### 3 — SSH key setup (once per remote box)

The daemon SSHes as `ssh.user` (default `root`) using `ssh.key`.  To verify:

```bash
ssh -p <port> -i <key> root@<host> "echo ok"
```

If that works, you're done.  If not, add the public key:

```bash
ssh-copy-id -i <key>.pub -p <port> root@<host>
```

### 4 — Start the stack

Run both processes as the user who owns the SSH key (avoids key permission issues):

```bash
cd /path/to/repo

# Dashboard (port from config, default 5055)
pkill -f fleet_dashboard.py; sleep 1
FLEET_CONFIG=tools/gpu-fleet/config.yaml \
nohup python3 tools/gpu-fleet/fleet_dashboard.py \
  >> tools/gpu-fleet/logs/dashboard.log 2>&1 &

# Daemon
pkill -f fleet_daemon.py; sleep 1
FLEET_CONFIG=tools/gpu-fleet/config.yaml \
nohup python3 tools/gpu-fleet/fleet_daemon.py \
  >> tools/gpu-fleet/logs/daemon.log 2>&1 &
```

Open **http://localhost:5055**.

### 5 — Queue an experiment

Via the dashboard "add task" form, or with curl:

```bash
curl -s -X POST http://localhost:5055/api/queue \
  -H 'Content-Type: application/json' \
  -d '{"label":"my-exp seed 1","launcher":"scripts/run_myexp.sh",
       "env":"SEED=1 LR=3e-4","priority":10}'
```

The daemon picks it up on the next poll (default 60 s), rsyncs your `scripts/` directory
to the remote, then SSH-launches the task.

---

## How it works

### Task lifecycle

```
pending → (daemon claims idle box) → running → (box becomes idle) → done
                                                                   ↓
                                              failed   (manual, via retry button)
```

Retrying resets `status=pending`, `box=null`, `started_at=null` so the daemon
re-runs it on the next free box.

### Idle detection

| Box config         | Check method                                |
|--------------------|---------------------------------------------|
| `idle_check: ps`   | SSH, `ps \| grep <proc_pattern> \| wc -l`  |
| `idle_check: nvidia-smi` | SSH, `nvidia-smi mem -i <gpu_idx>` ≤ 100 MiB |
| `tag: local`       | `pgrep -f <proc_pattern>` locally           |

Use `nvidia-smi` mode for multi-GPU boxes where two tasks share the same host:port
but different `gpu_idx` — it distinguishes per-GPU utilisation.

### Auto-rsync

Before launching on a remote box, the daemon rsyncs `scripts/` from the local repo
to the same path on the remote.  This means you never need to `git push` just to run
a new launcher script — edit locally, queue, done.

### ETA estimation

- **Running tasks**: `started_at + avg_duration_of_same_launcher` (from `done` history).
- **Pending tasks**: scheduling simulation — min-heap of box free-times assigns each
  pending task to the earliest-free box; ETA = assigned start + avg duration.
- Fallback when no history exists: 4 hours.
- Box Fleet shows ETA from the queue for each box's running task.

---

## Writing launcher scripts

Follow the pattern in `launchers/example_launcher.sh`:

```bash
#!/usr/bin/env bash
set -u; set +e
REPO=${REPO:-/root/my-repo}
cd "$REPO" || exit 1
source /root/venv/bin/activate
SEEDS=${SEEDS:-1}   # ← one seed per task for clean parallelism
for seed in $SEEDS; do
  python3 scripts/train.py --seed "$seed" ...
done
```

**Rules:**
1. **One seed per task** — queue `SEEDS=1`, `SEEDS=2`, … as separate tasks.  Looping
   over multiple seeds in one task blocks the slot until all finish and makes retry
   re-run from seed 1.
2. **`tee -a` not `tee`** — append to logs so a restart doesn't truncate the history.
3. **Accept config from env** — `LR=${LR:-3e-4}`.  The task's `env` field injects
   overrides, and the daemon also injects `XLA_PYTHON_CLIENT_MEM_FRACTION` per-box.
4. **Use an output tag** — write CSVs to `exp/<TAG>/seed_N.csv` so experiments don't
   clobber each other.  Set the tag from an env var: `TAG=${TAG:-myexp}`.

---

## CSV format for the metrics viewer

The dashboard scans `dashboard.exp_dir` recursively.  Any CSV file that contains a
column named `dashboard.step_col` (default `step`) is eligible.  Discovery groups
CSVs by their **first directory component** after `exp_dir`:

```
exp/
  phaseab_tdmpc2/       ← tag "phaseab_tdmpc2"
    seed_1.csv          ← series "seed_1"
    seed_2.csv
  phaseac_glass/        ← tag "phaseac_glass"
    seed_1.csv
```

Any additional numeric columns become selectable metrics (`reward`, `loss`, `accuracy`, …).

---

## Config reference

```yaml
fleet:
  name: "my-fleet"          # shown in dashboard title
  repo: /abs/path/to/repo   # must match on all boxes

ssh:
  key: /home/user/.ssh/id_ed25519
  user: root
  connect_timeout: 10       # seconds

queue:
  file: tools/gpu-fleet/queue.json   # relative to fleet.repo
  poll_seconds: 60

dashboard:
  port: 5055
  exp_dir: exp              # relative to fleet.repo
  step_col: step            # x-axis column in CSVs
  log_dir: tools/gpu-fleet/logs

proc_pattern: "run_benchmark"   # grep pattern for idle detection
xla_mem_default: "0.65"        # injected unless task env overrides it

boxes:
  - tag: local              # unique id used in queue JSON + UI
    host: null              # null = no SSH
    port: null
    gpu_idx: 0
    label: "Local 4070 Ti"
    xla_mem: "0.85"

  - tag: remote1
    host: ssh6.vast.ai
    port: 11115
    gpu_idx: 0
    label: "4060 8GB"
    xla_mem: "0.65"
    idle_check: ps          # default; use nvidia-smi for dual-GPU boxes
```

---

## Queue REST API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/queue` | All tasks, sorted by priority; includes ETA fields |
| POST | `/api/queue` | Add task `{label, launcher, env, priority}` |
| DELETE | `/api/queue/<id>` | Force-delete any task |
| POST | `/api/queue/<id>/priority` | Adjust priority `{delta: ±1}` |
| POST | `/api/queue/<id>/retry` | Reset running/failed/done → pending |
| GET  | `/api/boxes` | Live GPU/CPU stats for all boxes |
| GET  | `/api/metrics` | Discovered tags and metric columns |
| GET  | `/api/metrics/series?tag=X&metric=Y` | Time series data for plotting |

---

## Adapting for a new project

1. Copy `tools/gpu-fleet/` into the new repo.
2. Edit `config.yaml.example` → `config.yaml` with the new box list and paths.
3. Write launcher scripts following the pattern in `launchers/example_launcher.sh`.
4. Make sure the training script writes CSVs with a `step` column to `exp/<tag>/seed_N.csv`.
5. Start the stack with `FLEET_CONFIG=tools/gpu-fleet/config.yaml`.

No code changes are needed in `fleet_daemon.py` or `fleet_dashboard.py` for the basic case.
Add project-specific box probing or metric parsing by subclassing or patching the relevant
functions (all clearly marked).

---

## Troubleshooting

**All boxes show unreachable**
The dashboard process may be running stale code (started before you edited `config.yaml`).
Kill and restart both processes.

**Tasks complete instantly without running**
The launcher script is missing on the remote.  This is now prevented by the auto-rsync
in the daemon, but for the first run you can pre-sync manually:
```bash
rsync -az -e "ssh -p <port> -i <key>" scripts/ root@<host>:/path/to/repo/scripts/
```

**`pgrep` shows box idle immediately after launch**
The process starts asynchronously; the daemon checks again after `poll_seconds`.  If the
task transitions running→done in the very first poll, check `/tmp/fleet_<task_id>.log`
on the remote for the error.

**Dual-GPU box: both GPUs always show busy / idle together**
Set `idle_check: nvidia-smi` and separate `gpu_idx: 0` / `gpu_idx: 1` entries in config.

**SSH key permission denied**
The private key file must be `chmod 600` and owned by the running user.  If running as
root, coder's key is at `/home/coder/.ssh/id_ed25519` — pass it explicitly via
`SSH_IDENTITY_FILE=/home/coder/.ssh/id_ed25519`.

---

## Files

```
tools/gpu-fleet/
├── README.md                    ← you are here
├── config.yaml.example          ← copy to config.yaml and edit
├── fleet_daemon.py              ← task queue daemon (config-driven)
├── fleet_dashboard.py           ← Flask dashboard (config-driven)
├── queue.json                   ← auto-created; the live queue state
├── logs/                        ← daemon + dashboard log files
│   ├── daemon.log
│   └── dashboard.log
└── launchers/
    └── example_launcher.sh      ← template for new experiment launchers
```
