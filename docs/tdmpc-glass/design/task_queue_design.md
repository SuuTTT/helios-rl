# Central Task Queue — Design Doc

Date: 2026-05-20

## Problem

The per-box `.queue` files require manually knowing which box is free.
DIRECT launches bypass the queue entirely, leaving boxes idle when jobs
finish. There is no UI for queue visibility or priority editing.

## Design

### Single shared queue

All pending work lives in one file:

```
scripts/queues/central_queue.json
```

Any idle box in the fleet claims the next highest-priority task.
No per-box assignment needed at enqueue time.

### Task schema

```json
{
  "id":          "8-char hex",
  "label":       "human-readable description",
  "launcher":    "scripts/run_phaseaa_codex_kupdate_sweep.sh",
  "env":         "K_UPDATES=128 SEEDS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65",
  "priority":    10,
  "status":      "pending | running | done | failed",
  "box":         null,
  "created_at":  "2026-05-20T08:00:00Z",
  "started_at":  null,
  "ended_at":    null
}
```

Priority: lower number = runs first. Default = 10. Use 1 for urgent.

### Daemon — `scripts/task_queue_daemon.py`

- Polls every 60 s.
- For each box: SSH probe → is `run_benchmark` running?
- If idle: claim the lowest-`priority` pending task (atomic read-modify-write
  on the JSON file via `fcntl` lock).
- SSH launch: `env nohup setsid bash launcher > /tmp/tqd_<id>.log 2>&1 < /dev/null & disown`.
- If a task is `running` but its assigned box becomes idle again → mark `done`.
- No per-box queue files needed (old files kept for reference, daemon ignores them).

### Web API (Flask, port 5055)

| Method | Path | Action |
|--------|------|--------|
| GET | `/api/queue` | Return all tasks sorted by priority |
| POST | `/api/queue` | Add task (`label`, `launcher`, `env`, `priority`) |
| DELETE | `/api/queue/<id>` | Remove pending task |
| POST | `/api/queue/<id>/priority` | Bump priority (`{"delta": -1}` moves up) |

### Web UI

New **"Task Queue"** section in the dashboard (between Fleet and Curves):
- Table: priority · label · status · box · launcher · env · created · actions
- Status colour: pending=blue, running=green, done=grey, failed=red
- Actions: ↑ / ↓ (reorder pending), 🗑 (delete pending only)
- Collapse-able "Add task" form
- Auto-refreshes every 10 s

### Box-idle detection

Same SSH probe as existing dashboard `/api/boxes`:

```
ssh -p <port> root@<host> "ps -eo cmd | grep run_benchmark | grep -v grep | wc -l"
```

Single-GPU boxes: any run_benchmark process = busy.
Dual-GPU (ssh17637): check `nvidia-smi -i <idx> --query-gpu=memory.used` > 100 MiB.

### Quoting fix

Remote command is built as a Python string (no local shell expansion) and
passed as a **single list element** to `subprocess.Popen`. The remote bash
receives `SEEDS="1 2 3"` verbatim and interprets the quotes correctly.
This avoids the word-splitting bug in the old bash daemon.

### Migration

Old per-box `.queue` files are left in place. The new daemon reads only
`central_queue.json`. The old `iter6_auto_queue.sh` daemon can be stopped
once the Python daemon is running.
