# Dashboard & Task Queue — Launch Guide

## How it works (one paragraph)

Three daemons run in the background. **`web_dashboard.py`** is a Flask app (port 5055) that
serves the UI: it SSH-probes every box every 30 s for GPU/CPU stats, reads local CSV mirrors for
learning curves, and exposes a REST API for the task queue. **`task_queue_daemon.py`** polls
`scripts/queues/central_queue.json` every 60 s — when a box goes idle it atomically claims the
highest-priority pending task and SSH-launches it there. **`iter5_stream_remotes.sh`** rsyncs
remote experiment CSVs into `exp/tdmpc_glass/remote_mirror/` every 5 min so the dashboard curves
stay fresh without opening a browser-to-remote connection.

SSH authentication: all three scripts connect as `root@<remote-host>` using
**coder's private key** (`/home/coder/.ssh/id_ed25519`), which is the key whose public half was
deployed to each vast.ai box. The key is passed explicitly via `-i` on every SSH call; root's own
`~/.ssh/` keys are untouched.

---

## One-time SSH setup (do once per controlling machine)

**Key facts:**
- Remote boxes accept `root@<host>` login
- The authorized key on remotes is **coder's public key** (`/home/coder/.ssh/id_ed25519`)
- When SSHing as `coder` locally, the key is used automatically
- When SSHing as `root` locally, you must point at coder's key explicitly

Add the vast.ai hosts to **root's** SSH config (so `ssh -p <port> <host>` works from root too):

```bash
cat >> /root/.ssh/config << 'EOF'

# vast.ai fleet — authenticate with coder's key
Host ssh6.vast.ai ssh1.vast.ai ssh3.vast.ai 78.83.187.54
    IdentityFile /home/coder/.ssh/id_ed25519
    User root
    StrictHostKeyChecking no
EOF
chmod 600 /root/.ssh/config   # required — SSH ignores config if permissions are loose
```

Verify:
```bash
ssh -p 11115 ssh6.vast.ai "echo ok"   # from root shell
# or as coder:
ssh -p 11115 root@ssh6.vast.ai "echo ok"
```

> This config is host-specific. Root's own default key is still used for GitHub, GCP, etc.

---

## Starting the stack

**Run as `coder`** — coder's SSH key authenticates to remote boxes automatically, and Claude
Code (which also runs as coder) can manage these processes (pkill, etc.).

```bash
# Switch to coder first if you're in a root shell:
su - coder
cd /root/helios-rl

# 1. Web dashboard (Flask, port 5055)
pkill -f web_dashboard.py 2>/dev/null; sleep 1
nohup /root/venv/bin/python3 scripts/web_dashboard.py \
  >> exp/tdmpc_glass/logs/daemons/dashboard.log 2>&1 &

# 2. Central task queue daemon
pkill -f task_queue_daemon.py 2>/dev/null; sleep 1
nohup /root/venv/bin/python3 scripts/task_queue_daemon.py \
  >> exp/tdmpc_glass/logs/daemons/tqd.log 2>&1 &

# 3. Remote CSV mirror (keeps learning curves fresh)
pkill -f iter5_stream_remotes.sh 2>/dev/null; sleep 1
nohup setsid bash scripts/iter5_stream_remotes.sh \
  >> exp/tdmpc_glass/logs/daemons/stream.log 2>&1 < /dev/null & disown
```

Or from a root shell in one shot:
```bash
su - coder -c "cd /root/helios-rl && pkill -f web_dashboard.py; sleep 1; nohup /root/venv/bin/python3 scripts/web_dashboard.py >> exp/tdmpc_glass/logs/daemons/dashboard.log 2>&1 &"
su - coder -c "cd /root/helios-rl && pkill -f task_queue_daemon.py; sleep 1; nohup /root/venv/bin/python3 scripts/task_queue_daemon.py >> exp/tdmpc_glass/logs/daemons/tqd.log 2>&1 &"
su - coder -c "cd /root/helios-rl && pkill -f iter5_stream_remotes.sh; sleep 1; nohup setsid bash scripts/iter5_stream_remotes.sh >> exp/tdmpc_glass/logs/daemons/stream.log 2>&1 < /dev/null & disown"
```

Open the dashboard: **http://localhost:5055**
(or the cloudflare tunnel URL if port-forwarding via vast.ai's web proxy)

---

## Checking what's alive

```bash
pgrep -fa web_dashboard.py
pgrep -fa task_queue_daemon.py
pgrep -fa iter5_stream_remotes.sh
```

All three should print a PID + path. If any are missing, re-run just that block above.

Live logs:
```
tail -f /root/helios-rl/exp/tdmpc_glass/logs/daemons/dashboard.log
tail -f /root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log
tail -f /root/helios-rl/exp/tdmpc_glass/logs/daemons/stream.log
```

---

## Dashboard sections

| Section | What it shows | Refresh |
|---|---|---|
| **Box Fleet** | GPU/CPU util, running phase·seed·best MPPI for every box | 30 s auto / manual |
| **Task Queue** | All tasks in `central_queue.json` — status, priority, box | 10 s auto |
| **Learning Curves** | Plotly of every phase's seed CSVs; 95% CI mode; multi-phase compare | 60 s auto |
| **Render Rollout** | Click to queue a rollout render (runs locally, priority 1) | 90 s auto |

---

## Task queue cheat sheet

**Add a training task** — via UI "add task" form, or curl:
```bash
curl -s -X POST http://localhost:5055/api/queue \
  -H 'Content-Type: application/json' \
  -d '{"label":"phaseab seed 1","launcher":"scripts/run_phase1b_10m.sh","env":"SEEDS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75","priority":10}'
```

**Priority**: lower number runs first. Default = 10. Use 1 for urgent (renders are always 1).

**Task statuses**:
- `pending` — waiting for a free box; can delete or reorder via UI
- `running` — claimed by a box; daemon launched it via SSH
- `done` — box became idle again after the task ran
- `failed` — marked failed (e.g. after manual retry of a crashed task)

**Retry a crashed task**: click `↺ retry` in the UI (or curl):
```bash
curl -s -X POST http://localhost:5055/api/queue/<task-id>/retry
```

This resets status → `pending`, clears `box`/`started_at`/`ended_at`. The daemon will re-launch
it on the next idle box. Use this when a task OOM'd or the box was killed mid-run.

**Box-offline behaviour**: if a box goes unreachable (SSH fails), the daemon leaves its task as
`running` — the box may reconnect with the process still alive. If you know it died, hit retry.

---

## Queue file location

```
scripts/queues/central_queue.json
```

Edit directly with any text editor if needed. The daemon uses an `fcntl` file lock so edits are
safe to make while the daemon is running (write to a tmp file and `mv` in, or just edit in-place
and the lock will protect concurrent access).

**Render tasks** (type=`render`) are handled locally by the dashboard process, not SSHed out.
The SSH daemon skips them automatically.

---

## Sync scripts to remote boxes (rsync approach)

The task queue daemon **auto-rsyncs `scripts/`** to each remote box immediately before launching a
task on it (`rsync_scripts()` in `task_queue_daemon.py`). This keeps launchers in sync without
requiring a git push to every box.

If you add a new launcher script and need it on all boxes **now** (before the daemon picks up a
task), run manually:

```bash
for HOST_PORT in "ssh6.vast.ai:11115" "ssh6.vast.ai:16779" "78.83.187.54:17637" \
                 "ssh1.vast.ai:34217" "ssh3.vast.ai:11271"; do
  HOST="${HOST_PORT%%:*}"; PORT="${HOST_PORT##*:}"
  rsync -az --delete \
    -e "ssh -p $PORT -i /home/coder/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
    /root/helios-rl/scripts/ root@${HOST}:/root/helios-rl/scripts/ &
done
wait && echo "done"
```

---

## Troubleshooting

### All boxes show "unreachable" in Box Fleet

**Cause**: Dashboard process started before the SSH key fix was saved, so it runs old code without
`-i /home/coder/.ssh/id_ed25519`. New code is not automatically picked up.

**Fix**: Kill and restart the dashboard. Use the start commands above (note `pkill` + `sleep 1`).

---

### Tasks stay "running" after remote box became idle

**Cause A — Box went offline then came back**: SSH probe failed while box was reconnecting, daemon
left task as `running`. The task may still be running on the box.
- Check: `ssh -p <port> root@<host> "ps -eo cmd | grep run_benchmark"`
- If dead: click **↺ retry** in UI (or `curl .../api/queue/<id>/retry`) to reset to pending.

**Cause B — Smoke test tasks finished but daemon hasn't polled yet**: Wait one POLL cycle (60 s)
or restart the daemon. After smoke, click **✕** to delete the entries.

---

### Tasks complete instantly without running (job log missing or says "no such file")

**Cause**: Launcher script not present on the remote box — SSH exits immediately with error,
process appears to end, daemon marks box idle → task done within the next poll.

**Fix**: The daemon now auto-rsyncs scripts before each remote launch. For existing tasks that
already failed this way, click **↺ retry** so they re-run with the rsync step.

**Verify** by checking the task log on the remote:
```bash
ssh -p <port> -i /home/coder/.ssh/id_ed25519 root@<host> "cat /tmp/tqd_<task-id>.log"
```

---

### `grep -c` exit-code trap (historical, already fixed)

`ps | grep -c '[r]un_benchmark'` exits with code 1 when count=0. Python's
`subprocess.check_output` raises `CalledProcessError`, which the exception handler catches and
returns `False` (box always appears busy). Fixed by switching to `grep ... | wc -l` +
`subprocess.run(..., check=False)` — `wc -l` always exits 0.

---

### Two daemon processes running simultaneously

After a `pkill` + restart, verify only one instance is alive:
```bash
pgrep -fa task_queue_daemon.py
```
If two PIDs appear, kill them both explicitly:
```bash
kill <pid1> <pid2>
```
Then restart once.

---

### Local box never picks up tasks

**Cause**: `BOXES` list didn't include `"local"` entry (pre-fix), or idle check uses `pgrep`
which returns exit 0 when processes match. The `is_box_idle` local branch returns `True` only when
`pgrep` exit code is 1 (no match = idle).

**Current behaviour**: local box is probed via `pgrep -f run_benchmark` locally; if idle, the
daemon spawns the task via `bash -c "..."` without SSH.

---

## Deploying to Google Cloud (future)

The only machine-specific value is the SSH key path. Override via env var before starting:

```bash
export SSH_IDENTITY_FILE=/path/to/gcp_service_key
```

Both `web_dashboard.py` and `task_queue_daemon.py` read:
```python
SSH_KEY = os.environ.get("SSH_IDENTITY_FILE", "/home/coder/.ssh/id_ed25519")
```

On GCP, run the stack as a non-root service user with its own SSH key deployed to the target
boxes. Use systemd units rather than `nohup` for process supervision.

---

## Box registry

Boxes are listed in `BOXES` at the top of both `web_dashboard.py` and `task_queue_daemon.py`.
Keep them in sync. Format: `(tag, port, host, gpu_idx, label)`.

Current fleet (2026-05-20):

| Tag | Host | Port | GPU |
|---|---|---|---|
| local | — | — | 4070 Ti 12 GB |
| ssh6_4060 | ssh6.vast.ai | 11115 | RTX 4060 8 GB |
| ssh17637_gpu0 | 78.83.187.54 | 17637 | 3060 Laptop 6 GB (GPU 0) |
| ssh17637_gpu1 | 78.83.187.54 | 17637 | 3060 Laptop 6 GB (GPU 1) |
| ssh1_2080ti | ssh1.vast.ai | 34217 | RTX 2080 Ti |
| ssh6_3080 | ssh6.vast.ai | 16779 | RTX 3080 |
