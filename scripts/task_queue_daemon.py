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
import re
import shlex
import subprocess
import sys
import time
import uuid
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
    ("ssh17637_gpu0", 17637,  "78.83.187.54",   0),
    ("ssh17637_gpu1", 17637,  "78.83.187.54",   1),
    ("ssh1_2080ti",   34217,  "ssh1.vast.ai",   0),
    ("ssh3_3070",     15229,  "ssh3.vast.ai",   0),
    ("ssh6_3080",     16779,  "ssh6.vast.ai",   0),
    ("ssh5_3060_bar", 27233,  "ssh5.vast.ai",   0),
    ("ssh9_2060_gpu0", 17647, "ssh9.vast.ai",   0),
    ("ssh9_2060_gpu1", 17647, "ssh9.vast.ai",   1),
    ("ssh9_2060_gpu2", 17647, "ssh9.vast.ai",   2),
    ("ssh9_2060_gpu3", 17647, "ssh9.vast.ai",   3),
]

# Per-box XLA_MEM override used when env doesn't already specify it.
DEFAULT_MEM = {
    "local":         "0.85",
    "ssh17637_gpu0": "0.35",
    "ssh17637_gpu1": "0.35",
    "ssh1_2080ti":   "0.75",
    "ssh3_3070":     "0.55",
    "ssh6_3080":     "0.65",
    "ssh5_3060_bar":  "0.65",
    "ssh9_2060_gpu0": "0.35",
    "ssh9_2060_gpu1": "0.35",
    "ssh9_2060_gpu2": "0.35",
    "ssh9_2060_gpu3": "0.35",
}
CUDA_MASK = {
    "ssh17637_gpu0": "CUDA_VISIBLE_DEVICES=0",
    "ssh17637_gpu1": "CUDA_VISIBLE_DEVICES=1",
    "ssh9_2060_gpu0": "CUDA_VISIBLE_DEVICES=0",
    "ssh9_2060_gpu1": "CUDA_VISIBLE_DEVICES=1",
    "ssh9_2060_gpu2": "CUDA_VISIBLE_DEVICES=2",
    "ssh9_2060_gpu3": "CUDA_VISIBLE_DEVICES=3",
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
    elif tag in CUDA_MASK:
        # Multi-GPU box: check GPU memory on the specific CUDA index.
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

def rsync_code(port: int, host: str):
    """Rsync launcher and source code needed by queued experiment tasks."""
    mkdir_cmd = [
        "ssh", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        "-o", "BatchMode=yes", f"root@{host}",
        "mkdir -p /root/helios-rl/scripts /root/helios-rl/src",
    ]
    try:
        subprocess.run(mkdir_cmd, timeout=30, capture_output=True, check=True)
    except Exception as e:
        log(f"mkdir remote code dirs on {host}:{port} failed: {e}")
    for rel in ("scripts", "src"):
        cmd = [
            "rsync", "-az", "--delete",
            "-e", f"ssh -p {port} -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15",
            str(REPO / rel) + "/",
            f"root@{host}:/root/helios-rl/{rel}/",
        ]
        try:
            subprocess.run(cmd, timeout=90, capture_output=True, check=True)
        except Exception as e:
            log(f"rsync {rel}/ to {host}:{port} failed: {e}")


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
    if tag == "ssh4_8080":
        # ssh4's Vast image has a tight cgroup pids.max=256. JAX/XLA can abort
        # during compile if it tries to create the default host thread pool.
        env_map = parse_env(env)
        for key in (
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env_map.setdefault(key, "1")
        xla_flags = env_map.get("XLA_FLAGS", "--xla_gpu_autotune_level=0")
        for flag in ("--xla_cpu_multi_thread_eigen=false", "intra_op_parallelism_threads=1"):
            if flag not in xla_flags:
                xla_flags = f"{xla_flags} {flag}"
        env_map["XLA_FLAGS"] = xla_flags
        env = format_env(env_map)

    log(f"{tag} → launching task {task['id']}: {task['label']}")

    if tag == "local":
        log_local = f"/tmp/tqd_{task['id']}.log"
        try:
            try:
                Path(log_local).unlink(missing_ok=True)
            except Exception:
                pass
            proc_env = os.environ.copy()
            for item in shlex.split(env):
                if "=" in item:
                    k, v = item.split("=", 1)
                    proc_env[k] = v
            fh = open(log_local, "w")
            subprocess.Popen(
                ["bash", task["launcher"]],
                cwd=REPO,
                env=proc_env,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as e:
            log(f"local launch error: {e}")
        return

    # Remote: sync launcher + algorithm source first, then SSH-launch.
    rsync_code(port, host)

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


def task_log_tail(task: dict, n: int = 80) -> str:
    """Best-effort tail of a queue task's launcher log."""
    task_id = task.get("id", "")
    box = task.get("box", "")
    log_path = f"/tmp/tqd_{task_id}.log"
    if box == "local":
        try:
            return subprocess.check_output(["tail", "-n", str(n), log_path], timeout=5).decode(errors="replace")
        except Exception:
            return ""
    box_info = next((b for b in BOXES if b[0] == box), None)
    if not box_info:
        return ""
    _tag, port, host, _gpu_idx = box_info
    cmd = [
        "ssh", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
        "-o", "BatchMode=yes", f"root@{host}",
        f"tail -n {int(n)} {shlex.quote(log_path)} 2>/dev/null || true",
    ]
    try:
        return subprocess.check_output(cmd, timeout=12).decode(errors="replace")
    except Exception:
        return ""


def task_eval_log(task: dict) -> str:
    """Best-effort eval-line extraction from a task log."""
    task_id = task.get("id", "")
    box = task.get("box", "")
    log_path = f"/tmp/tqd_{task_id}.log"
    pattern = r"step=.*pi_reward=.*MPPI="
    if box == "local":
        try:
            return subprocess.check_output(["grep", "-E", pattern, log_path], timeout=8).decode(errors="replace")
        except Exception:
            return ""
    box_info = next((b for b in BOXES if b[0] == box), None)
    if not box_info:
        return ""
    _tag, port, host, _gpu_idx = box_info
    cmd = [
        "ssh", "-p", str(port), "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
        "-o", "BatchMode=yes", f"root@{host}",
        f"grep -E {shlex.quote(pattern)} {shlex.quote(log_path)} 2>/dev/null || true",
    ]
    try:
        return subprocess.check_output(cmd, timeout=15).decode(errors="replace")
    except Exception:
        return ""


def infer_finished_status(task: dict) -> str:
    """Return done/failed from the launcher log when possible."""
    tail = task_log_tail(task)
    if "No such file or directory" in tail:
        return "failed"
    if "ERROR in " in tail:
        return "failed"
    if "done status=" in tail and "done status=0" not in tail:
        return "failed"
    if "all done status=" in tail and "all done status=0" not in tail:
        return "failed"
    return "done"


def parse_env(env: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in shlex.split(env or ""):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def format_env(env_map: dict[str, str]) -> str:
    return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_map.items())


def normalized_family_env(env: str) -> tuple[tuple[str, str], ...]:
    skip = {"SEEDS", "PROBE_ID", "CUDA_VISIBLE_DEVICES"}
    return tuple(sorted((k, v) for k, v in parse_env(env).items() if k not in skip))


MIN_FAILED_PROMOTION_STEPS = 4_000_000


def best_any_and_last_step_from_log(task: dict) -> tuple[float | None, int]:
    text = task_eval_log(task)
    best = None
    last_step = -1
    for line in text.splitlines():
        sm = re.search(r"step=\s*([0-9,]+)", line)
        if sm:
            try:
                last_step = max(last_step, int(sm.group(1).replace(",", "")))
            except ValueError:
                pass
        m = re.search(r"pi_reward=\s*([-+]?\d+(?:\.\d+)?)\s+MPPI=\s*([-+]?\d+(?:\.\d+)?)", line)
        if not m:
            continue
        val = max(float(m.group(1)), float(m.group(2)))
        best = val if best is None else max(best, val)
    return best, last_step


def existing_family_seeds(tasks: list[dict], family_key: tuple[tuple[str, str], ...]) -> set[int]:
    seeds: set[int] = set()
    for task in tasks:
        if normalized_family_env(task.get("env", "")) != family_key:
            continue
        for token in parse_env(task.get("env", "")).get("SEEDS", "").split():
            try:
                seeds.add(int(token))
            except ValueError:
                pass
    return seeds


def auto_promote_task(tasks: list[dict], task: dict, status: str) -> None:
    """Append follow-up seed tasks according to Iteration 9 promotion thresholds."""
    if task.get("type") == "render" or task.get("auto_promoted"):
        return
    launcher = task.get("launcher", "")
    env = task.get("env", "")
    if "run_phasei9_glass_probe.sh" not in launcher or "SEEDS=" not in env:
        return
    best, last_step = best_any_and_last_step_from_log(task)
    if best is None:
        return
    if status == "failed" and last_step < MIN_FAILED_PROMOTION_STEPS:
        task["auto_promoted"] = {
            "best_any": best,
            "status": status,
            "added": 0,
            "at": now_iso(),
            "skip_reason": f"failed_before_{MIN_FAILED_PROMOTION_STEPS}_steps",
            "last_step": last_step,
        }
        log(f"skip auto-promote {task['id']}: failed at {last_step} < {MIN_FAILED_PROMOTION_STEPS}")
        return

    # Failed-but-informative runs get a 100-point lower bar. This covers SIGKILL
    # or infra failures after useful eval rows, without promoting header-only runs.
    discount = 100.0 if status == "failed" else 0.0
    if best >= 600.0 - discount:
        add_count = 5
    elif best >= 500.0 - discount:
        add_count = 2
    elif best >= 380.0 - discount:
        add_count = 1
    else:
        return

    family_key = normalized_family_env(env)
    existing = existing_family_seeds(tasks, family_key)
    env_map = parse_env(env)
    base_probe = env_map.get("PROBE_ID", task.get("id", "probe"))
    priority = int(task.get("priority", 7))
    added = 0
    for seed in range(1, 11):
        if seed in existing:
            continue
        new_env = dict(env_map)
        new_env["SEEDS"] = str(seed)
        new_env["PROBE_ID"] = f"{base_probe}_auto_s{seed}"
        new_task = {
            "id": "t" + uuid.uuid4().hex[:7],
            "label": f"auto-promote {base_probe} seed {seed} from {task['id']} best_any={best:.1f}",
            "launcher": launcher,
            "env": format_env(new_env),
            "priority": priority,
            "status": "pending",
            "box": None,
            "created_at": now_iso(),
            "started_at": None,
            "ended_at": None,
            "auto_parent": task["id"],
            "auto_reason": f"best_any={best:.1f} status={status} add_count={add_count}",
        }
        tasks.append(new_task)
        existing.add(seed)
        added += 1
        if added >= add_count:
            break
    task["auto_promoted"] = {
        "best_any": best,
        "status": status,
        "added": added,
        "at": now_iso(),
    }
    if added:
        log(f"auto-promoted {task['id']} best_any={best:.1f} status={status}: added {added} seed task(s)")


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
            status = infer_finished_status(t)
            log(f"task {t['id']} ({t['label']}) {status} on {t['box']}")
            t["status"] = status
            t["ended_at"] = now_iso()
            auto_promote_task(tasks, t, status)
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
