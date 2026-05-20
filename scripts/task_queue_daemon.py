#!/usr/bin/env python3
"""Central task queue daemon.

Polls every POLL_SECONDS. When a box is idle, claims the highest-priority
pending task from central_queue.json and SSH-launches it there.
Marks tasks done when their assigned box becomes free again.

Usage:
    nohup python3 scripts/task_queue_daemon.py \
        >> exp/tdmpc_glass/logs/daemons/tqd.log 2>&1 &
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/helios-rl")
SSH_KEY = os.environ.get("SSH_IDENTITY_FILE", "/home/coder/.ssh/id_ed25519")
QUEUE_FILE = REPO / "scripts" / "queues" / "central_queue.json"
POLL_SECONDS = 60
LOG_DIR = REPO / "exp" / "tdmpc_glass" / "logs" / "daemons"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Box registry — mirrors BOXES in web_dashboard.py
# (tag, port, host, gpu_idx)  port=None/host=None means local execution
BOXES = [
    ("local",         None,    None,             0),
    ("ssh6_4060",     11115,  "ssh6.vast.ai",   0),
    ("ssh17637_gpu0", 17637,  "78.83.187.54",   0),
    ("ssh17637_gpu1", 17637,  "78.83.187.54",   1),
    ("ssh1_2080ti",   34217,  "ssh1.vast.ai",   0),
    ("ssh3_3070",     15229,  "ssh3.vast.ai",   0),
    ("ssh6_3080",     16779,  "ssh6.vast.ai",   0),
    ("ssh3_3060ti",   11271,  "ssh3.vast.ai",   0),
]

# Per-box XLA_MEM override used when env doesn't already specify it.
DEFAULT_MEM = {
    "local":         "0.85",
    "ssh6_4060":     "0.65",
    "ssh17637_gpu0": "0.35",
    "ssh17637_gpu1": "0.35",
    "ssh1_2080ti":   "0.75",
    "ssh3_3070":     "0.55",
    "ssh6_3080":     "0.65",
    "ssh3_3060ti":   "0.55",
}
CUDA_MASK = {
    "ssh17637_gpu0": "CUDA_VISIBLE_DEVICES=0",
    "ssh17637_gpu1": "CUDA_VISIBLE_DEVICES=1",
}


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def log(msg: str):
    print(f"[tqd] {ts()} {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Queue I/O with file lock ──────────────────────────────────────────────────

def load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_queue(tasks: list[dict]):
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tasks, indent=2))
    tmp.replace(QUEUE_FILE)


def with_queue_lock(fn):
    """Run fn(tasks) → tasks atomically with an exclusive file lock."""
    lock_path = QUEUE_FILE.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tasks = load_queue()
            result = fn(tasks)
            if result is not None:
                save_queue(result)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ── Box idle check ────────────────────────────────────────────────────────────

def is_box_idle(tag: str, port: int, host: str, gpu_idx: int) -> bool:
    """Return True if no run_benchmark process is running on this slot."""
    if tag == "local":
        try:
            res = subprocess.run(
                ["pgrep", "-f", "run_benchmark"],
                capture_output=True, timeout=5,
            )
            return res.returncode != 0  # returncode 1 = no match = idle
        except Exception:
            return True
    elif tag.startswith("ssh17637"):
        # Dual-GPU box: check GPU memory on the specific CUDA index.
        cmd = ["ssh", "-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
               f"root@{host}",
               f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_idx} 2>/dev/null"]
        try:
            out = subprocess.check_output(cmd, timeout=12, stderr=subprocess.DEVNULL).decode().strip()
            return int(out) <= 100
        except Exception:
            return False  # SSH unreachable → treat as busy
    else:
        cmd = ["ssh", "-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
               f"root@{host}",
               "ps -eo cmd | grep '[r]un_benchmark' | wc -l"]
        try:
            res = subprocess.run(cmd, timeout=12, capture_output=True)
            return int(res.stdout.decode().strip()) == 0
        except Exception:
            return False


# ── Task launch ───────────────────────────────────────────────────────────────

def rsync_scripts(port: int, host: str):
    """Rsync scripts/ to remote box so launcher scripts are present."""
    cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -p {port} -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15",
        str(REPO / "scripts") + "/",
        f"root@{host}:/root/helios-rl/scripts/",
    ]
    try:
        subprocess.run(cmd, timeout=60, capture_output=True, check=True)
    except Exception as e:
        log(f"rsync to {host}:{port} failed: {e}")


def launch_task(task: dict, tag: str, port: int, host: str):
    """Launch a task on the given box. Fire-and-forget."""
    env = task["env"]
    # Inject CUDA mask for dual-GPU boxes if not already in env.
    mask = CUDA_MASK.get(tag, "")
    if mask and "CUDA_VISIBLE_DEVICES" not in env:
        env = f"{mask} {env}"
    # Inject default mem fraction if not already set.
    mem_key = "XLA_PYTHON_CLIENT_MEM_FRACTION"
    if mem_key not in env:
        env = f"{env} {mem_key}={DEFAULT_MEM.get(tag, '0.65')}"

    log(f"{tag} → launching task {task['id']}: {task['label']}")

    if tag == "local":
        log_local = f"/tmp/tqd_{task['id']}.log"
        shell_cmd = (
            f"cd {REPO} ; "
            f"{env} nohup setsid bash {task['launcher']} "
            f"> {log_local} 2>&1 < /dev/null & disown"
        )
        try:
            subprocess.Popen(["bash", "-c", shell_cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"local launch error: {e}")
        return

    # Remote: rsync scripts first, then SSH-launch.
    rsync_scripts(port, host)

    log_remote = f"/tmp/tqd_{task['id']}.log"
    # Build the remote command as a Python string — passed as a SINGLE argument
    # to ssh so the local shell never word-splits the env vars.
    remote_cmd = (
        f"cd /root/helios-rl ; "
        f"{env} nohup setsid bash {task['launcher']} "
        f"> {log_remote} 2>&1 < /dev/null & disown ; sleep 1"
    )
    ssh_cmd = [
        "ssh", "-f", "-n", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        f"root@{host}",
        remote_cmd,
    ]
    try:
        subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"{tag} launch error: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def poll_once():
    tasks = load_queue()
    if not tasks:
        return

    # Check which running tasks have finished (box became idle again).
    running = [t for t in tasks if t["status"] == "running"]
    idle_boxes: set[str] = set()

    for tag, port, host, gpu_idx in BOXES:
        busy = not is_box_idle(tag, port, host, gpu_idx)
        if not busy:
            idle_boxes.add(tag)

    # Mark done: running tasks whose assigned box is now idle.
    changed = False
    for t in running:
        if t.get("box") in idle_boxes:
            log(f"task {t['id']} ({t['label']}) done on {t['box']}")
            t["status"] = "done"
            t["ended_at"] = now_iso()
            changed = True

    if changed:
        save_queue(tasks)
        tasks = load_queue()  # reload after save

    # Assign pending tasks to idle boxes that have no running task assigned.
    busy_boxes = {t["box"] for t in tasks if t["status"] == "running" and t.get("box")}
    free_boxes = [b for b in BOXES if b[0] in idle_boxes and b[0] not in busy_boxes]
    pending = sorted(
        [t for t in tasks if t["status"] == "pending" and t.get("type") != "render"],
        key=lambda t: (t["priority"], t["created_at"])
    )

    for box_entry in free_boxes:
        if not pending:
            break
        tag, port, host, gpu_idx = box_entry
        task = pending.pop(0)

        def claim(tasks, _task=task, _tag=tag):
            for t in tasks:
                if t["id"] == _task["id"] and t["status"] == "pending":
                    t["status"] = "running"
                    t["box"] = _tag
                    t["started_at"] = now_iso()
            return tasks

        with_queue_lock(claim)
        # Re-read to get the claimed task's env/launcher.
        tasks = load_queue()
        claimed = next((t for t in tasks if t["id"] == task["id"]), None)
        if claimed and claimed["status"] == "running":
            launch_task(claimed, tag, port, host)
        else:
            log(f"task {task['id']} already claimed by another process, skipping")


def main():
    log(f"start — polling {len(BOXES)} boxes every {POLL_SECONDS}s")
    log(f"queue: {QUEUE_FILE}")
    while True:
        try:
            poll_once()
        except Exception as e:
            log(f"poll error: {e}")
        # Show summary every cycle.
        tasks = load_queue()
        pending = sum(1 for t in tasks if t["status"] == "pending")
        running = sum(1 for t in tasks if t["status"] == "running")
        if pending or running:
            log(f"queue: {pending} pending, {running} running, "
                f"{sum(1 for t in tasks if t['status']=='done')} done")
        else:
            log("queue: all idle or empty")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
