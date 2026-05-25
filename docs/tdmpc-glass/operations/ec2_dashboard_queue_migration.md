# EC2 Dashboard and Queue Master Migration

Date: 2026-05-25

Purpose: move the TD-MPC-Glass control plane from the current master machine to
an Amazon EC2 instance. The EC2 box should run the dashboard, central queue
daemon, mirror streamer, docs, queue ledger, and result mirrors. VastAI boxes
remain disposable GPU workers.

## Target EC2 Shape

Recommended starting point:
- Instance: `t3.large`, `t3.xlarge`, `m6i.large`, or similar.
- Disk: 200-500 GB gp3 EBS mounted at `/root/helios-rl` or `/data/helios-rl`.
- OS: Ubuntu 22.04 or 24.04.
- Network: public IPv4 or stable Elastic IP.
- Security group:
  - SSH `22/tcp` from your IP only.
  - Dashboard `5056/tcp` from your IP only, or keep closed and use SSH tunnel.

Prefer EBS over instance store for the master. The master holds the queue
ledger and result mirror; losing it creates avoidable recovery work.

## One-Master Rule

Never run two queue daemons against the same fleet.

Before starting EC2 queue services:
1. Stop the queue daemon on the old master.
2. Stop or ignore the old dashboard.
3. Copy queue/mirror state to EC2.
4. Start dashboard on EC2 first for inspection.
5. Start the queue daemon only after verifying the queue state.

If both masters run `scripts/task_queue_daemon.py`, they can double-claim tasks
and launch duplicate experiments.

## Freeze Old Master

On the current master:

```bash
cd /root/helios-rl

pgrep -af 'task_queue_daemon.py|web_dashboard.py|iter5_stream_remotes.sh|run_benchmark.py'

pkill -f '/root/helios-rl/scripts/task_queue_daemon.py' || true
pkill -f '/root/helios-rl/scripts/web_dashboard.py' || true
pkill -f '/root/helios-rl/scripts/iter5_stream_remotes.sh' || true
```

Back up the queue before any migration mutation:

```bash
cd /root/helios-rl
cp scripts/queues/central_queue.json \
  scripts/queues/central_queue.json.bak_pre_ec2_$(date -u +%Y%m%d_%H%M%S)
```

If VastAI budget shutdown killed the fleet, reset stale `running` rows to
`pending` before copying, but keep historical `done` and `failed` rows:

```bash
cd /root/helios-rl
python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

p = Path("scripts/queues/central_queue.json")
q = json.loads(p.read_text())
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
for t in q:
    if t.get("status") == "running":
        t.setdefault("recovery_history", []).append({
            "at": now,
            "from_status": "running",
            "from_box": t.get("box"),
            "reason": "master migration / stale running reset",
        })
        t["status"] = "pending"
        t.pop("box", None)
        t.pop("started_at", None)
        t.pop("eta_iso", None)
        t.pop("elapsed_s", None)
p.write_text(json.dumps(q, indent=2))
PY
```

## Bootstrap EC2

On EC2:

```bash
sudo apt-get update
sudo apt-get install -y git rsync curl tmux htop ripgrep python3-venv python3-pip

mkdir -p /root/.ssh
chmod 700 /root/.ssh
```

Install the SSH private key used to reach VastAI workers:

```bash
install -m 600 id_ed25519 /root/.ssh/id_ed25519
ssh-keyscan github.com >> /root/.ssh/known_hosts
```

Clone or rsync the repo:

```bash
git clone <repo-url> /root/helios-rl
cd /root/helios-rl
```

Create the Python environment:

```bash
python3 -m venv /root/venv
/root/venv/bin/pip install --upgrade pip
/root/venv/bin/pip install flask
```

The dashboard and queue daemon need only the lightweight control-plane
dependencies. GPU/JAX training dependencies are needed on workers, not on EC2.

## Copy State From Old Master

From EC2, pull queue, docs, logs, and mirrored CSVs:

```bash
OLD=root@<old-master-ip>

rsync -az $OLD:/root/helios-rl/scripts/queues/ /root/helios-rl/scripts/queues/
rsync -az $OLD:/root/helios-rl/docs/tdmpc-glass/ /root/helios-rl/docs/tdmpc-glass/
rsync -az $OLD:/root/helios-rl/exp/tdmpc_glass/remote_mirror/ \
  /root/helios-rl/exp/tdmpc_glass/remote_mirror/
rsync -az $OLD:/root/helios-rl/exp/tdmpc_glass/logs/ \
  /root/helios-rl/exp/tdmpc_glass/logs/
```

Optional larger copy if disk allows:

```bash
rsync -az --exclude='*/checkpoints/full_state*' \
  $OLD:/root/helios-rl/exp/tdmpc_glass/ /root/helios-rl/exp/tdmpc_glass/
```

Do not blindly copy old full checkpoints unless needed; they are large and the
dashboard mostly needs CSVs, diag CSVs, queue JSON, and selected checkpoints.

## Verify Worker Reachability

On EC2:

```bash
cd /root/helios-rl

for spec in \
  "11115 root@ssh6.vast.ai" \
  "17637 root@78.83.187.54" \
  "34217 root@ssh1.vast.ai" \
  "15229 root@ssh3.vast.ai" \
  "16779 root@ssh6.vast.ai" \
  "11271 root@ssh3.vast.ai" \
  "15665 root@ssh4.vast.ai" \
  "17647 root@ssh9.vast.ai"
do
  set -- $spec
  port=$1
  host=$2
  echo "== $host:$port =="
  ssh -i /root/.ssh/id_ed25519 -o BatchMode=yes \
    -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
    -p "$port" "$host" 'hostname; nvidia-smi -L | head -5' || true
done
```

Any unreachable worker should either be removed from `BOXES` temporarily or
left as unreachable. The queue daemon treats unreachable remotes as busy and
will not assign tasks there.

## Start Services On EC2

Start dashboard first:

```bash
cd /root/helios-rl
mkdir -p exp/tdmpc_glass/logs/daemons

nohup setsid env DASHBOARD_PORT=5056 /root/venv/bin/python3 -u \
  /root/helios-rl/scripts/web_dashboard.py \
  > /tmp/web_dashboard_5056.log 2>&1 < /dev/null &
```

Inspect:

```bash
curl -fsS http://127.0.0.1:5056/api/queue | python3 -m json.tool | head
```

Start the mirror streamer:

```bash
nohup setsid bash /root/helios-rl/scripts/iter5_stream_remotes.sh \
  > /root/helios-rl/exp/tdmpc_glass/logs/daemons/stream.log \
  2>&1 < /dev/null &
```

Start the queue daemon last:

```bash
nohup setsid /root/venv/bin/python3 /root/helios-rl/scripts/task_queue_daemon.py \
  > /root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log \
  2>&1 < /dev/null &
```

Verify:

```bash
pgrep -af 'web_dashboard.py|task_queue_daemon.py|iter5_stream_remotes.sh'
tail -f /root/helios-rl/exp/tdmpc_glass/logs/daemons/tqd.log
```

Access dashboard:

```bash
ssh -L 5056:127.0.0.1:5056 root@<ec2-ip>
```

Open `http://127.0.0.1:5056`.

## Optional systemd Services

For a longer-lived EC2 master, prefer systemd over ad-hoc `nohup`.

Example dashboard unit:

```ini
[Unit]
Description=TD-MPC-Glass dashboard
After=network-online.target

[Service]
WorkingDirectory=/root/helios-rl
Environment=DASHBOARD_PORT=5056
ExecStart=/root/venv/bin/python3 -u /root/helios-rl/scripts/web_dashboard.py
Restart=always
RestartSec=5
StandardOutput=append:/tmp/web_dashboard_5056.log
StandardError=append:/tmp/web_dashboard_5056.log

[Install]
WantedBy=multi-user.target
```

Create analogous units for:
- `/root/helios-rl/scripts/task_queue_daemon.py`
- `/root/helios-rl/scripts/iter5_stream_remotes.sh`

Then:

```bash
systemctl daemon-reload
systemctl enable --now tdmpc-glass-dashboard
systemctl enable --now tdmpc-glass-stream
systemctl enable --now tdmpc-glass-queue
```

Keep queue service disabled until after queue inspection during migrations.

## Backups

Minimum daily backup from EC2:

```bash
cd /root/helios-rl
tar -czf /root/tdmpc_glass_control_$(date -u +%Y%m%d_%H%M%S).tgz \
  scripts/queues \
  docs/tdmpc-glass \
  exp/tdmpc_glass/logs \
  exp/tdmpc_glass/remote_mirror
```

Recommended persistent backups:
- GitHub: code, docs, queue snapshots, small summaries.
- S3: `remote_mirror`, daemon logs, queue backups, selected rollout videos.
- Hugging Face or W&B artifacts: selected best checkpoints and rollout videos.

Do not commit large checkpoints to GitHub.

## Recovery Checklist After Budget Shutdown

1. Stop queue/dashboard/streamer on the master.
2. Back up `scripts/queues/central_queue.json`.
3. Reset stale `running` tasks to `pending`.
4. Re-rent or replace VastAI workers.
5. Verify each worker by SSH and `nvidia-smi`.
6. Start dashboard.
7. Start streamer.
8. Start queue daemon.
9. Confirm queue transitions from pending to running only on reachable boxes.
10. Watch `tqd.log` for rsync/launch errors.

## Files To Keep In Sync

Control-plane state:
- `scripts/queues/central_queue.json`
- `scripts/queues/*.bak_*`
- `docs/tdmpc-glass/`
- `exp/tdmpc_glass/remote_mirror/`
- `exp/tdmpc_glass/logs/daemons/`

Code launched on workers:
- `scripts/`
- `src/`

The queue daemon rsyncs `scripts/` and `src/` at task launch. That means EC2
becomes the source of truth for worker code once the daemon moves there.

