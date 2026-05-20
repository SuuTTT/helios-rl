#!/usr/bin/env python3
"""gpu-fleet task queue daemon.

Polls every poll_seconds. When a GPU slot is idle, claims the highest-priority
pending task and launches it there via SSH (or locally). Auto-rsyncs the repo's
scripts/ directory to the remote before launching so launchers are always in sync.

Usage:
    FLEET_CONFIG=/path/to/config.yaml \\
    nohup python3 tools/gpu-fleet/fleet_daemon.py \\
        >> tools/gpu-fleet/logs/daemon.log 2>&1 &
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
CONFIG_PATH = Path(os.environ.get("FLEET_CONFIG", _DEFAULT_CONFIG))

def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = _load_cfg()
REPO       = Path(CFG["fleet"]["repo"])
SSH_KEY    = os.environ.get("SSH_IDENTITY_FILE",
                CFG.get("ssh", {}).get("key", "/home/coder/.ssh/id_ed25519"))
SSH_USER   = CFG.get("ssh", {}).get("user", "root")
SSH_TO     = int(CFG.get("ssh", {}).get("connect_timeout", 10))
QUEUE_FILE = REPO / CFG.get("queue", {}).get("file", "tools/gpu-fleet/queue.json")
POLL_S     = int(CFG.get("queue", {}).get("poll_seconds", 60))
LOG_DIR    = REPO / CFG.get("dashboard", {}).get("log_dir", "tools/gpu-fleet/logs")
PROC_PAT   = CFG.get("proc_pattern", "run_benchmark")
XLA_DEFAULT = CFG.get("xla_mem_default", "0.65")

LOG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Box registry: list of (tag, port, host, gpu_idx, idle_check)
BOXES = [
    (b["tag"], b.get("port"), b.get("host"), b.get("gpu_idx", 0),
     b.get("idle_check", "ps"))
    for b in CFG.get("boxes", [])
]
DEFAULT_MEM  = {b["tag"]: b.get("xla_mem", XLA_DEFAULT) for b in CFG.get("boxes", [])}
CUDA_VISIBLE = {
    b["tag"]: f"CUDA_VISIBLE_DEVICES={b['gpu_idx']}"
    for b in CFG.get("boxes", [])
    if b.get("idle_check") == "nvidia-smi"
}


# ── Logging ───────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")

def log(msg: str):
    print(f"[fleet] {ts()} {msg}", flush=True)


# ── Queue I/O ─────────────────────────────────────────────────────────────────

def load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_queue(tasks: list[dict]):
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tasks, indent=2))
    tmp.replace(QUEUE_FILE)

def with_queue_lock(fn):
    """Run fn(tasks) → tasks atomically under an exclusive file lock."""
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


# ── Idle detection ────────────────────────────────────────────────────────────

def is_idle(tag: str, port, host: str, gpu_idx: int, idle_check: str) -> bool:
    if tag == "local":
        try:
            r = subprocess.run(["pgrep", "-f", PROC_PAT], capture_output=True, timeout=5)
            return r.returncode != 0
        except Exception:
            return True

    ssh_base = ["ssh", "-p", str(port), "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={SSH_TO}",
                "-o", "BatchMode=yes",
                f"{SSH_USER}@{host}"]

    if idle_check == "nvidia-smi":
        cmd = ssh_base + [
            f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            f" -i {gpu_idx} 2>/dev/null"
        ]
        try:
            out = subprocess.check_output(cmd, timeout=15, stderr=subprocess.DEVNULL)
            return int(out.decode().strip()) <= 100
        except Exception:
            return False
    else:
        cmd = ssh_base + [f"ps -eo cmd | grep '{PROC_PAT}' | grep -v grep | wc -l"]
        try:
            r = subprocess.run(cmd, timeout=15, capture_output=True)
            return int(r.stdout.decode().strip()) == 0
        except Exception:
            return False


# ── Launch ────────────────────────────────────────────────────────────────────

def rsync_scripts(port, host: str):
    cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -p {port} -i {SSH_KEY} -o StrictHostKeyChecking=no"
              f" -o ConnectTimeout={SSH_TO}",
        str(REPO / "scripts") + "/",
        f"{SSH_USER}@{host}:{REPO}/scripts/",
    ]
    try:
        subprocess.run(cmd, timeout=90, capture_output=True, check=True)
    except Exception as e:
        log(f"rsync to {host}:{port} failed: {e}")


def launch(task: dict, tag: str, port, host: str):
    env = task["env"]
    mask = CUDA_VISIBLE.get(tag, "")
    if mask and "CUDA_VISIBLE_DEVICES" not in env:
        env = f"{mask} {env}"
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env:
        env = f"{env} XLA_PYTHON_CLIENT_MEM_FRACTION={DEFAULT_MEM.get(tag, XLA_DEFAULT)}"

    log(f"{tag} → {task['id']}: {task['label']}")
    log_path = f"/tmp/fleet_{task['id']}.log"

    if tag == "local":
        shell = (f"cd {REPO} ; {env} nohup setsid bash {task['launcher']}"
                 f" > {log_path} 2>&1 < /dev/null & disown")
        try:
            subprocess.Popen(["bash", "-c", shell],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"local launch error: {e}")
        return

    rsync_scripts(port, host)
    remote = (f"cd {REPO} ; {env} nohup setsid bash {task['launcher']}"
              f" > {log_path} 2>&1 < /dev/null & disown ; sleep 1")
    ssh_cmd = ["ssh", "-f", "-n", "-p", str(port), "-i", SSH_KEY,
               "-o", "StrictHostKeyChecking=no", "-o", f"ConnectTimeout={SSH_TO}",
               f"{SSH_USER}@{host}", remote]
    try:
        subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"{tag} launch error: {e}")


# ── Main poll loop ────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def poll_once():
    tasks = load_queue()
    if not tasks:
        return

    idle_boxes: set[str] = set()
    for tag, port, host, gpu_idx, idle_check in BOXES:
        if is_idle(tag, port, host, gpu_idx, idle_check):
            idle_boxes.add(tag)

    changed = False
    for t in tasks:
        if t["status"] == "running" and t.get("box") in idle_boxes:
            log(f"done: {t['id']} ({t['label']}) on {t['box']}")
            t["status"] = "done"
            t["ended_at"] = now_iso()
            changed = True

    if changed:
        save_queue(tasks)
        tasks = load_queue()

    busy_boxes = {t["box"] for t in tasks if t["status"] == "running" and t.get("box")}
    free_boxes = [b for b in BOXES if b[0] in idle_boxes and b[0] not in busy_boxes]
    pending = sorted(
        [t for t in tasks if t["status"] == "pending" and t.get("type") != "local-only"],
        key=lambda t: (t.get("priority", 10), t.get("created_at", ""))
    )

    for box_entry in free_boxes:
        if not pending:
            break
        tag, port, host, gpu_idx, idle_check = box_entry
        task = pending.pop(0)

        def claim(tasks, _t=task, _tag=tag):
            for t in tasks:
                if t["id"] == _t["id"] and t["status"] == "pending":
                    t["status"] = "running"
                    t["box"] = _tag
                    t["started_at"] = now_iso()
            return tasks

        with_queue_lock(claim)
        tasks = load_queue()
        claimed = next((t for t in tasks if t["id"] == task["id"]), None)
        if claimed and claimed["status"] == "running":
            launch(claimed, tag, port, host)
        else:
            log(f"task {task['id']} already claimed, skipping")


def main():
    log(f"start — {len(BOXES)} boxes, poll every {POLL_S}s")
    log(f"config: {CONFIG_PATH}")
    log(f"queue:  {QUEUE_FILE}")
    while True:
        try:
            poll_once()
        except Exception as e:
            log(f"poll error: {e}")
        tasks = load_queue()
        n_p = sum(1 for t in tasks if t["status"] == "pending")
        n_r = sum(1 for t in tasks if t["status"] == "running")
        n_d = sum(1 for t in tasks if t["status"] == "done")
        if n_p or n_r:
            log(f"queue: {n_p} pending  {n_r} running  {n_d} done")
        else:
            log("queue: idle")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
