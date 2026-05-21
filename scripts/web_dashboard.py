"""TD-MPC-Glass live web dashboard.

Serves a single HTML page on http://localhost:5055 showing:
  - per-box status (running process, GPU/CPU util) — polled via SSH
  - learning curves of every active HopperHop_phase*/seed_*.csv (from local mirror)
  - per-seed video render trigger (click → background render_glass_rollout.py
    job with progress bar)

Run:
  /root/venv/bin/python3 scripts/web_dashboard.py
or via:
  bash scripts/launch_web_dashboard.sh

Stop with Ctrl-C. Single-process Flask; for development / single-user only.
"""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort

REPO = Path("/root/helios-rl")
LOCAL_EXP = REPO / "exp" / "tdmpc_glass"
MIRROR = LOCAL_EXP / "remote_mirror"
VIDEO_OUT = REPO / "exp" / "tdmpc_glass" / "rollout_videos"
VIDEO_OUT.mkdir(parents=True, exist_ok=True)
EXISTING_VIDEOS_ROOT = REPO / "exp" / "tdmpc_glass" / "videos"
QUEUE_DIR = REPO / "scripts" / "queues"
CENTRAL_QUEUE_FILE = QUEUE_DIR / "central_queue.json"

# SSH key: remotes accept root@ login with coder's key (coder's pubkey was deployed to remote authorized_keys)
SSH_KEY = os.environ.get("SSH_IDENTITY_FILE", "/home/coder/.ssh/id_ed25519")
SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]

# ─── Box registry. Mirrors scripts/iter5_dashboard.sh BOXES. ────────────
BOXES = [
    # (tag, port, host, gpu_idx, label)
    ("local",         None,    None,             0, "Local 4070 Ti (12GB)"),
    ("ssh6_4060",     11115,   "ssh6.vast.ai",   0, "ssh6:11115 4060 (8GB)"),
    ("ssh17637_gpu0", 17637,   "78.83.187.54",   0, "78.83.187.54 GPU0 3060Lap (6GB)"),
    ("ssh17637_gpu1", 17637,   "78.83.187.54",   1, "78.83.187.54 GPU1 3060Lap (6GB)"),
    ("ssh1_2080ti",   34217,   "ssh1.vast.ai",   0, "ssh1:34217 2080 Ti (22GB)"),
    ("ssh3_3070",     15229,   "ssh3.vast.ai",   0, "ssh3:15229 3070 (8GB)"),
    ("ssh6_3080",     16779,   "ssh6.vast.ai",   0, "ssh6:16779 3080 (10GB)"),
    ("ssh3_3060ti",   11271,   "ssh3.vast.ai",   0, "ssh3:11271 3060Ti (8GB)"),
]

# Render workers scan the whole fleet dynamically. Keep local last so training
# on the dashboard host is disturbed only when it is truly idle.
RENDER_MEM_FRACTION = {
    "ssh17637_gpu0": "0.75",
    "ssh17637_gpu1": "0.75",
    "ssh6_4060": "0.70",
    "ssh3_3060ti": "0.55",
    "ssh3_3070": "0.55",
    "ssh6_3080": "0.65",
    "ssh1_2080ti": "0.75",
    "local": "0.70",
}
RENDER_POLL_SECONDS = 30
RENDER_DISPATCH_LOCK = threading.Lock()

# Box probe cache — populated by api_boxes(), consumed by _compute_queue_etas().
_BOX_CACHE: dict = {}
_BOX_CACHE_LOCK = threading.Lock()

SPS_RENDER_APPROX = 200  # rollout steps/sec for render ETA estimate

# Keyed by CANONICAL phase name (after canonical_phase() normalization).
PHASE_NOTES: dict[str, str] = {
    # Iter 1 baselines
    "pre_phase1":   "Very early Glass baseline runs",
    "phase1b":      "Glass baseline: [438,526,294,187,562] mean=401",
    "phase1c":      "Act-noise anneal 0.30->0.10; FALSIFIED: hurts winners",
    "phase2":       "Early iteration 2 runs",
    # Iter 2 (phases d-h)
    "phased":       "H=5+noise=0.40 (Warp-901) or H=5-only (plateau 199); FALSIFIED",
    "phasee":       "Q-reset (impl bug: zeroed all opt state); FALSIFIED",
    "phasef":       "Smooth=1e-3; s1=571 first>500 ever; s2-5 plateau 255-284",
    "phaseg":       "Consistency coef=1.0; best 482 (s2), none>500",
    "phaseh":       "Smooth=1e-3 + ccoef=1.0; peak 490, plateau",
    "phasei":       "Smooth=1e-4 (too weak); peak 312",
    # Iter 3 (phases j-o)
    "phasej":       "Curriculum smooth (off<250k); s2=518 winner; 12% hit rate",
    "phasek":       "Smooth + lambda_temporal=0.05 (over-reg); peak 292",
    "phasel":       "tdmpc2 + smooth, Glass zeroed; peak 289; Glass IS needed",
    "phasem":       "Python-cond smooth graph; s4 K=3->K=4 rescued",
    "phasen":       "Proto temp=0.4 sharper; basin still K=3; FALSIFIED",
    "phaseo":       "Glass OFF after 2M hybrid; s3=577 winner; 1/3>500",
    # Iter 4 (paths 1,5,7,9,10,P,Pa)
    "phasep":       "EXPL_UNTIL=500k random explore; s4=538 slow-burn; 1/3>500",
    "phaset":       "Knee penalty reward shaping; 2/3>500, max=612 G2 (benchmark-unfair)",
    "phasev":       "Cluster soft-dist as pi/q obs (Path 7); peak 232; FALSIFIED (drift)",
    "phasex":       "NS=2048 MPPI; s3=523, s8~500; ~40% G1 hit rate",
    "phasex_ns1024":"NS=1024 compromise for 6GB GPU; lower ceiling than 2048",
    "phasey":       "Hierarchical Glass K_super=4 (Path 10); peak 462; 0/3>500",
    "phasePP":      "Cluster entropy intrinsic (static 0.1); FALSIFIED: peak 91 collapsed",
    "phasePa":      "Cluster entropy intrinsic (decayed [500k,3M]->0); FALSIFIED: peak 25",
    # Iter 6
    "phasez":       "Vanilla tdmpc2 baseline (NS=512, no Glass, no shaping)",
    "phaseq_knee":  "Knee penalty + NS=2048 no Glass; 4/12>500 (33%), max=557",
    "phaser1_soft": "Soft stand bonus 0.1 annealed->0@3M; 1/5>500 (20%), max=553",
    "phaser2_gait": "Gait fall penalty + action smooth; 1/4>500, max=510",
    "phaserstack":  "ALL shaping stacked; COLLAPSED (MPPI~5-16, all seeds)",
    "phaserstack_nosmooth": "Stack ablation A: drop action_smooth",
    "phaserstack_nosoft":   "Stack ablation B: drop soft_stand_bonus",
    # Iter 7 - benchmark-fair K_UPDATE sweep
    "phaseaa_codex_tdmpc2_k64":  "K_UPDATE=64 tdmpc2; ~0.25 updates/env-step",
    "phaseaa_codex_tdmpc2_k128": "K_UPDATE=128 tdmpc2; ~0.5 updates/env-step",
    "phaseaa_codex_tdmpc2_k256": "K_UPDATE=256 tdmpc2; ~1.0 update/env-step (official rate)",
    "phaseab_codex_tdmpc2_5seed":"K_UPDATE winner, 5-seed vanilla tdmpc2",
    "phaseac_codex_glass_5seed": "K_UPDATE winner, 5-seed Glass vs tdmpc2 comparison",
    "phasead_codex_explmix":     "Fair expl-mix: random action prob 1->0 over 2M steps",
    # Smoke tests
    "smoke":        "Smoke test (hardware validation only)",
}
RESERVED_RENDER_TARGETS: set[str] = set()


def parse_etime_seconds(etime: str) -> int | None:
    """Parse `ps -o etime` formats: MM:SS, HH:MM:SS, D-HH:MM:SS. Returns seconds or None."""
    if not etime:
        return None
    try:
        days = 0
        if "-" in etime:
            d, etime = etime.split("-", 1)
            days = int(d)
        parts = [int(p) for p in etime.split(":")]
        if len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 3:
            h, m, s = parts
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None

# Remote shell snippet — returns one-line ASCII parsable by parse_box_status.
# For each running run_benchmark PID, also reads TDMPC_GLASS_OUTPUT_TAG from
# /proc/<pid>/environ so the host can pin the proc to the right phase CSV.
REMOTE_PROBE = r'''
gpu_idx="$1"
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null | head -1)
cpu=$(top -bn1 2>/dev/null | grep -E "^%Cpu" | head -1 | awk '{printf "%.0f", 100-$8}')
printf "GPU=%s\nCPU=%s\n" "$gpu" "$cpu"
ps -eo pid,etime,cmd --no-headers 2>/dev/null | grep -E "run_benchmark" | grep -v grep \
  | awk '{cmd=""; for(i=3;i<=NF;i++) cmd=cmd" "$i; printf "%s\t%s\t%s\n", $1, $2, cmd}' \
  | while IFS=$'\t' read -r pid etime cmd; do
    tag=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
          | awk -F= '$1=="TDMPC_GLASS_OUTPUT_TAG"{print $2; exit}')
    cuda=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
          | awk -F= '$1=="CUDA_VISIBLE_DEVICES"{print $2; exit}')
    printf "PROC|%s|%s|%s|%s|%s\n" "$pid" "$etime" "$tag" "$cuda" "$cmd"
  done
'''


def parse_box_status(raw: str):
    """Parse REMOTE_PROBE output. Proc lines look like:
       PROC|<pid>|<etime>|<output_tag>|<full_cmd_line>
    """
    gpu_util = mem_used = mem_total = None
    cpu_util = None
    procs = []
    for line in raw.splitlines():
        line = line.rstrip()
        if line.startswith("GPU="):
            v = line[4:].strip()
            parts = [x.strip() for x in v.split(",")]
            if len(parts) >= 3:
                try:
                    gpu_util = int(parts[0].split()[0])
                    mem_used = int(parts[1].split()[0])
                    mem_total = int(parts[2].split()[0])
                except Exception:
                    pass
        elif line.startswith("CPU="):
            try:
                cpu_util = int(line[4:].strip())
            except Exception:
                pass
        elif line.startswith("PROC|"):
            try:
                _, pid, etime, output_tag, cuda, cmd = line.split("|", 5)
                m_seed = re.search(r"--seed\s+(\S+)", cmd)
                m_ns = re.search(r"--mppi_n_samples\s+(\S+)", cmd)
                m_algo = re.search(r"--algos\s+(\S+)", cmd)
                tags = []
                if "--knee_penalty_coef" in cmd:
                    tags.append("knee")
                if "--use_cluster_obs" in cmd:
                    tags.append("cobs")
                if "--glass_num_super_clusters" in cmd:
                    tags.append("hier")
                procs.append({
                    "pid": pid.strip(),
                    "etime": etime.strip(),
                    "seed": m_seed.group(1) if m_seed else "?",
                    "ns": m_ns.group(1) if m_ns else "512",
                    "algo": m_algo.group(1) if m_algo else "?",
                    "tag": "+".join(tags) if tags else "",
                    "output_tag": (output_tag or "").strip(),
                    "cuda_visible": (cuda or "").strip(),
                })
            except Exception:
                pass
    return {
        "gpu_util": gpu_util, "mem_used": mem_used, "mem_total": mem_total,
        "cpu_util": cpu_util, "procs": procs,
    }


def probe_box(tag, port, host, gpu_idx):
    """SSH to box, run REMOTE_PROBE, parse output. Local case: bash subprocess."""
    try:
        if tag == "local":
            res = subprocess.run(
                ["bash", "-c", REMOTE_PROBE, "_", str(gpu_idx)],
                capture_output=True, text=True, timeout=10,
            )
        else:
            res = subprocess.run(
                ["ssh", "-p", str(port), "-i", SSH_KEY,
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=8",
                 "-o", "BatchMode=yes",
                 f"root@{host}", "bash", "-s", "--", str(gpu_idx)],
                input=REMOTE_PROBE, capture_output=True, text=True, timeout=15,
            )
        status = parse_box_status(res.stdout)
        status["reachable"] = (res.returncode == 0)
        return status
    except subprocess.TimeoutExpired:
        return {"reachable": False, "error": "ssh-timeout", "procs": []}
    except Exception as e:
        return {"reachable": False, "error": str(e), "procs": []}


def render_candidates():
    """Return render-capable boxes in dynamic priority order."""
    remotes = [b for b in BOXES if b[0] != "local"]
    local = [b for b in BOXES if b[0] == "local"]
    return remotes + local


def is_render_target_idle(tag: str, port, host, gpu_idx: int) -> tuple[bool, str]:
    """Return whether a render target's specific GPU looks idle enough to use."""
    try:
        if tag in RESERVED_RENDER_TARGETS:
            return False, f"{tag}: reserved for another render"
        if tag == "local":
            mem_cmd = ["nvidia-smi", "--query-gpu=memory.used",
                       "--format=csv,noheader,nounits", "-i", str(gpu_idx)]
            proc_cmd = "ps -eo cmd | grep -E 'render_glass_rollout|run_benchmark' | grep -v grep | wc -l"
            mem_res = subprocess.run(mem_cmd, capture_output=True, text=True, timeout=5)
            proc_res = subprocess.run(["bash", "-c", proc_cmd], capture_output=True, text=True, timeout=5)
        else:
            remote = (
                f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_idx}; "
                "ps -eo cmd | grep -E 'render_glass_rollout' | grep -v grep | wc -l"
            )
            res = subprocess.run(
                ["ssh", "-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=6", "-o", "BatchMode=yes",
                 f"root@{host}", remote],
                capture_output=True, text=True, timeout=10,
            )
            lines = [x.strip() for x in res.stdout.splitlines() if x.strip()]
            if res.returncode != 0 or len(lines) < 2:
                return False, f"{tag}: unreachable"
            mem_res = type("R", (), {"stdout": lines[0], "returncode": 0})()
            proc_res = type("R", (), {"stdout": lines[1], "returncode": 0})()
        mem = int(str(mem_res.stdout).strip().splitlines()[0])
        render_procs = int(str(proc_res.stdout).strip().splitlines()[0])
        if render_procs > 0:
            return False, f"{tag}: render already running"
        if mem > 250:
            return False, f"{tag}: gpu{gpu_idx} mem {mem} MiB"
        return True, f"{tag}: gpu{gpu_idx} idle ({mem} MiB)"
    except Exception as e:
        return False, f"{tag}: {e}"


def choose_render_target() -> dict:
    reasons = []
    with RENDER_DISPATCH_LOCK:
        for tag, port, host, gpu_idx, _label in render_candidates():
            idle, reason = is_render_target_idle(tag, port, host, gpu_idx)
            reasons.append(reason)
            if idle:
                RESERVED_RENDER_TARGETS.add(tag)
                return {
                    "tag": tag, "port": port, "host": host, "gpu_idx": gpu_idx,
                    "mem_frac": RENDER_MEM_FRACTION.get(tag, "0.60"),
                    "reasons": reasons,
                }
    return {"tag": None, "reasons": reasons}


def release_render_target(target: dict | None):
    tag = (target or {}).get("tag")
    if not tag:
        return
    with RENDER_DISPATCH_LOCK:
        RESERVED_RENDER_TARGETS.discard(tag)


def launch_next_queue_line(tag: str) -> str:
    """Best-effort one-shot queue launch for a just-freed render target."""
    if tag == "local":
        return "local has no queue file"
    qf = QUEUE_DIR / f"{tag}.queue"
    if not qf.exists():
        return f"{tag}: no queue file"
    try:
        lines = qf.read_text().splitlines()
        picked_idx = None
        picked = None
        for i, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            picked_idx = i
            picked = line
            break
        if picked_idx is None or picked is None:
            return f"{tag}: queue empty"
        parts = picked.split("|", 3)
        if len(parts) != 4:
            return f"{tag}: malformed queue line"
        port, host, launcher, envvars = parts
        lines[picked_idx] = f"# DONE {time.strftime('%H:%M:%SZ', time.gmtime())} {picked}"
        qf.write_text("\n".join(lines) + "\n")
        cmd = [
            "ssh", "-f", "-n", "-p", port, "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            f"root@{host}",
            f"cd /root/helios-rl ; {envvars} nohup setsid bash {launcher} "
            f"> /tmp/dashboard_queue_{tag}.log 2>&1 < /dev/null & disown ; sleep 1",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return f"{tag}: queue launch returned {res.returncode}: {res.stderr.strip()}"
        return f"{tag}: launched queued task {envvars} bash {launcher}"
    except Exception as e:
        return f"{tag}: queue kick failed: {e}"


def infer_render_error_type(log_lines: list[str]) -> str:
    text = "\n".join(log_lines[-80:]).lower()
    if "out of memory" in text or "resource_exhausted" in text or "failed to allocate" in text:
        return "OOM"
    if "cuda" in text or "jaxruntimeerror" in text:
        return "CUDA/JAX"
    if "no such file" in text or "checkpoint" in text or "pickle" in text:
        return "checkpoint"
    if "timeout" in text or "ssh" in text:
        return "remote/ssh"
    if "exception:" in text or "traceback" in text:
        return "exception"
    return "failed"


# ─── Phase name normalization ─────────────────────────────────────────────

# Strips device/run-variant suffixes so seeds from the same logical experiment
# but run on different hardware are grouped under one canonical phase name.
# Examples: phasex_local → phasex, phasem_remote → phasem, phasep_remote_3m → phasep
_CANON_DEVICE_RE = re.compile(
    r"(?:_(?:local|remote|baseline))*"   # _local, _remote, _baseline
    r"(?:_(?:\d*x)?\d+(?:ti|lap)?)*"    # _4060, _3060ti, _2x3060, _2080ti
    r"(?:_gpu\d+)*"                       # _gpu0, _gpu1
    r"(?:_3m)?"                           # _3m after _remote
    r"$",
    re.IGNORECASE,
)
_CANON_VERSION_RE = re.compile(
    r"_v\d+(?:_[a-z0-9]+)*$",
    re.IGNORECASE,
)


def canonical_phase(phase: str) -> str:
    """Strip device/hardware and run-variant suffixes to get canonical family name.

    Preserved: _ns1024, _nosmooth, _nosoft, _codex_*, _knee, _soft, _gait, _stack.
    """
    result = _CANON_DEVICE_RE.sub("", phase)
    result = _CANON_VERSION_RE.sub("", result)
    return result or phase


# ─── CSV discovery ────────────────────────────────────────────────────────

def discover_csvs():
    """Walk LOCAL_EXP (incl. remote_mirror) for HopperHop_phase*/seed_*.csv.

    Deduplicates by (phase, seed): if the same phase+seed appears in both the
    local exp tree and one or more remote mirrors, the one with the latest mtime
    AND the largest file size wins (size is a proxy for "more eval rows logged").
    """
    by_key: dict[tuple, dict] = {}
    for csv_path in LOCAL_EXP.rglob("HopperHop_phase*/seed_*.csv"):
        name = csv_path.name
        if re.search(r"_v\d+_|_partial_|_died_|_final_|_done_|_diag\.csv$", name):
            continue
        try:
            st = csv_path.stat()
        except OSError:
            continue
        if st.st_size < 30:
            continue  # < 30 bytes = header only or smaller
        if time.time() - st.st_mtime > 7 * 86400:
            continue
        phase_dir = csv_path.parent.name.replace("HopperHop_", "")
        seed = csv_path.stem.replace("seed_", "")
        rel = csv_path.relative_to(LOCAL_EXP)
        box = "local"
        if str(rel).startswith("remote_mirror/"):
            box = str(rel).split("/")[1]
        key = (phase_dir, seed)
        cand = {"phase": phase_dir, "seed": seed, "box": box,
                "path": str(csv_path), "mtime": st.st_mtime, "size": st.st_size}
        prev = by_key.get(key)
        if prev is None or (cand["size"], cand["mtime"]) > (prev["size"], prev["mtime"]):
            by_key[key] = cand
    found = list(by_key.values())
    found.sort(key=lambda r: (r["phase"], int(r["seed"]) if r["seed"].isdigit() else 99999))
    return found


def read_curve(csv_path):
    """Read CSV → list of {step, reward, eval_type}. Returns [] on parse error."""
    points = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(float(row.get("step", 0)))
                    rew = float(row.get("reward", 0))
                    et = row.get("eval_type", "")
                    points.append({"step": step, "reward": rew, "eval_type": et})
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return points


def best_and_last(csv_path):
    """Return (best_mppi, best_step, last_mppi, last_step) for a phase-seed CSV.
    All -1.0 if no mppi rows present."""
    best = -1.0
    best_step = -1
    last = -1.0
    last_step = -1
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("eval_type") != "mppi":
                    continue
                try:
                    r = float(row.get("reward", 0))
                    s = int(float(row.get("step", 0)))
                except (ValueError, TypeError):
                    continue
                if r > best:
                    best = r
                    best_step = s
                last = r
                last_step = s
    except Exception:
        pass
    return best, best_step, last, last_step


def read_diag_last_mppi(diag_path: str) -> dict | None:
    """Return the last mppi row of a _diag.csv as a dict, or None on any error."""
    result = None
    try:
        with open(diag_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("eval_type") == "mppi":
                    result = {
                        "standing_rate": float(row.get("standing_rate", 0)),
                        "fall_count": float(row.get("fall_count", 0)),
                        "ttf": float(row.get("time_to_first_full", 1000)),
                        "full_reward_rate": float(row.get("full_reward_rate", 0)),
                    }
    except Exception:
        pass
    return result


def find_active_csv_for(box: str, seed: str):
    """Locate the per-box CSV for a given seed that's actively being written.

    Strategy: look at the deduped CSV list (discover_csvs already filters to
    last 7 days). Prefer entries whose mirror box matches; fall back to the
    local exp tree. Return the freshest match, or None.
    """
    if not seed:
        return None
    matches = [c for c in discover_csvs()
               if c["seed"] == str(seed) and (c["box"] == box or box == "local")]
    if not matches:
        return None
    # most recently modified wins
    matches.sort(key=lambda r: r["mtime"], reverse=True)
    return matches[0]


# ─── Render jobs ──────────────────────────────────────────────────────────

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def find_best_ckpt(phase: str, seed: str):
    """Locate best_mppi.pkl for a phase+seed under exp/tdmpc_glass/."""
    candidates = list(LOCAL_EXP.rglob(f"HopperHop_{phase}/seed_{seed}/checkpoints/best_mppi.pkl"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


_VIDEO_NAME_PAT = re.compile(r"seed_?(\d+)", re.I)


def discover_existing_videos():
    """Scan exp/tdmpc_glass/videos/<phase>/*.mp4 + rollout_videos/<job>.mp4.

    Maps to {(phase, seed): [{label, url, mtime}, ...]} for in-place display
    on the dashboard. Phase comes from the directory name. Seed is extracted
    from the filename (e.g., 'seed_1_best_mppi_small.mp4' or 'seed3_x.mp4').
    """
    by_key: dict[tuple, list] = {}
    if EXISTING_VIDEOS_ROOT.exists():
        for mp4 in EXISTING_VIDEOS_ROOT.rglob("*.mp4"):
            try:
                rel = mp4.relative_to(EXISTING_VIDEOS_ROOT)
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) < 2:
                continue
            phase = parts[0]
            m = _VIDEO_NAME_PAT.search(mp4.stem)
            if not m:
                continue
            seed = m.group(1)
            try:
                st = mp4.stat()
            except OSError:
                continue
            by_key.setdefault((phase, seed), []).append({
                "label": mp4.stem,
                "url": "/exp_videos/" + "/".join(parts),
                "mtime": st.st_mtime,
                "size_mb": round(st.st_size / 1e6, 1),
                "source": "archive",
            })
    # Surface dashboard-produced rollout_videos across dashboard restarts via
    # sidecar metadata written when each render completes.
    for meta_path in VIDEO_OUT.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
            mp4 = VIDEO_OUT / f"{meta_path.stem}.mp4"
            if not mp4.exists():
                continue
            st = mp4.stat()
            phase = str(meta.get("phase") or "")
            seed = str(meta.get("seed") or "")
            if not phase or not seed:
                continue
            by_key.setdefault((phase, seed), []).append({
                "label": f"job-{meta_path.stem}",
                "url": f"/videos/{mp4.name}",
                "mtime": st.st_mtime,
                "size_mb": round(st.st_size / 1e6, 1),
                "source": "dashboard",
            })
        except Exception:
            continue
    for lst in by_key.values():
        lst.sort(key=lambda v: v["mtime"], reverse=True)
    return by_key


def render_worker(job_id: str, ckpt: str, env_id: str, camera: str,
                   n_episodes: int, episode_length: int):
    """Spawn render_glass_rollout.py, stream stdout → progress.

    Rendering is CUDA-heavy and used to run on the dashboard host, where it
    collided with training. Prefer an idle remote GPU, copy the checkpoint to a
    temp directory, render there, then pull the mp4 back into VIDEO_OUT.
    """
    out_mp4 = VIDEO_OUT / f"{job_id}.mp4"
    target = None
    while True:
        target = choose_render_target()
        if target and target.get("tag"):
            break
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "queued"
            JOBS[job_id]["progress"] = 0.0
            JOBS[job_id]["log"].append(
                f"waiting for an idle render GPU; checking again in {RENDER_POLL_SECONDS}s"
            )
            for reason in (target or {}).get("reasons", []):
                JOBS[job_id]["log"].append(f"probe: {reason}")
        time.sleep(RENDER_POLL_SECONDS)
    base_cmd = [
        "/root/venv/bin/python3", "-u", "scripts/render_glass_rollout.py",
        "--ckpt", ckpt, "--env_id", env_id,
        "--out", str(out_mp4), "--camera", camera,
        "--n_episodes", str(n_episodes),
        "--episode_length", str(episode_length),
    ]
    env = {
        **os.environ,
        "MUJOCO_GL": "egl",
        "PYTHONPATH": "/root/helios-rl/src:/root/mujoco_playground_repo",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": target["mem_frac"],
        "CUDA_VISIBLE_DEVICES": str(target["gpu_idx"]),
    }
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["render_host"] = target["tag"]
        JOBS[job_id]["log"].extend([f"render target: {target['tag']}",
                                    *[f"probe: {r}" for r in target.get("reasons", [])]])
    try:
        cleanup_cmd = None
        if target["tag"] == "local":
            cmd = base_cmd
            run_cwd = str(REPO)
            popen_env = env
            with JOBS_LOCK:
                JOBS[job_id]["cmd"] = " ".join(cmd)
        else:
            remote_dir = f"/tmp/helios_render_{job_id}"
            remote_ckpt = f"{remote_dir}/ckpt.pkl"
            remote_out = f"{remote_dir}/{job_id}.mp4"
            ssh_base = [
                "ssh", "-p", str(target["port"]), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=15", f"root@{target['host']}",
            ]
            rsync_base = ["rsync", "-a", "-e",
                          f"ssh -p {target['port']} -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15"]
            subprocess.run(ssh_base + [f"mkdir -p {shlex.quote(remote_dir)}"],
                           check=True, capture_output=True, text=True, timeout=30)
            subprocess.run(rsync_base + [ckpt, f"root@{target['host']}:{remote_ckpt}"],
                           check=True, capture_output=True, text=True, timeout=120)
            remote_cmd = [
                "cd /root/helios-rl &&",
                f"MUJOCO_GL=egl",
                "PYTHONPATH=/root/helios-rl/src:/root/mujoco_playground_repo",
                "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                f"XLA_PYTHON_CLIENT_MEM_FRACTION={shlex.quote(target['mem_frac'])}",
                f"CUDA_VISIBLE_DEVICES={target['gpu_idx']}",
                "/root/venv/bin/python3 -u scripts/render_glass_rollout.py",
                "--ckpt", shlex.quote(remote_ckpt),
                "--env_id", shlex.quote(env_id),
                "--out", shlex.quote(remote_out),
                "--camera", shlex.quote(camera),
                "--n_episodes", str(n_episodes),
                "--episode_length", str(episode_length),
            ]
            cmd = ssh_base + [" ".join(remote_cmd)]
            run_cwd = str(REPO)
            popen_env = os.environ.copy()
            cleanup_cmd = ssh_base + [f"rm -rf {shlex.quote(remote_dir)}"]
            with JOBS_LOCK:
                JOBS[job_id]["cmd"] = " ".join(cmd)
        proc = subprocess.Popen(
            cmd, cwd=run_cwd, env=popen_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        eps_done = 0
        for line in proc.stdout:
            line = line.rstrip()
            with JOBS_LOCK:
                JOBS[job_id]["log"].append(line)
                # render_glass_rollout prints "  episode N: steps=K return=R" per ep,
                # then "wrote <path> (N frames)" at end.
                if re.search(r"episode \d+:.*return=", line):
                    eps_done += 1
                    # account for both rollout + render passes (~2x the work)
                    JOBS[job_id]["progress"] = min(eps_done / (2 * n_episodes), 0.95)
                elif m := re.search(r"rollout \d+: step=(\d+)/(\d+)", line):
                    done = int(m.group(1))
                    total = max(int(m.group(2)), 1)
                    JOBS[job_id]["progress"] = min(0.5 * done / total, 0.49)
                elif "wrote " in line and "frames" in line:
                    JOBS[job_id]["progress"] = 0.99
        proc.wait()
        if target["tag"] != "local" and proc.returncode == 0:
            remote_out_ref = f"root@{target['host']}:/tmp/helios_render_{job_id}/{job_id}.mp4"
            subprocess.run(
                ["rsync", "-a", "-e",
                 f"ssh -p {target['port']} -i {SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15",
                 remote_out_ref, str(out_mp4)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        if cleanup_cmd:
            subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = 1.0
            ok = (proc.returncode == 0 and out_mp4.exists())
            JOBS[job_id]["status"] = "done" if ok else "failed"
            JOBS[job_id]["video"] = f"/videos/{job_id}.mp4" if out_mp4.exists() else None
            if not ok:
                JOBS[job_id]["error_type"] = infer_render_error_type(JOBS[job_id].get("log", []))
            if ok:
                (VIDEO_OUT / f"{job_id}.json").write_text(json.dumps({
                    "job_id": job_id,
                    "phase": JOBS[job_id].get("phase"),
                    "seed": JOBS[job_id].get("seed"),
                    "render_host": target.get("tag"),
                    "created_at": time.time(),
                }))
                JOBS[job_id]["log"].append(launch_next_queue_line(target["tag"]))
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["log"].append(f"EXCEPTION: {e}")
            JOBS[job_id]["error_type"] = infer_render_error_type(JOBS[job_id].get("log", []))
            JOBS[job_id]["progress"] = 1.0
    finally:
        release_render_target(target)


# ─── Flask app ───────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return INDEX_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/boxes")
def api_boxes():
    boxes = []
    threads, results = [], {}

    def worker(entry):
        tag, port, host, gpu_idx, label = entry
        results[tag] = probe_box(tag, port, host, gpu_idx)

    for entry in BOXES:
        t = threading.Thread(target=worker, args=(entry,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=20)
    # Compute CSV index once so we annotate every proc against the same snapshot.
    csv_index = discover_csvs()
    # Index by (phase, seed) and (box, phase, seed) for fast lookups.
    by_phase_seed = {(c["phase"], c["seed"]): c for c in csv_index}
    by_box_phase_seed = {(c["box"], c["phase"], c["seed"]): c for c in csv_index}
    for tag, port, host, gpu_idx, label in BOXES:
        info = results.get(tag, {"reachable": False, "error": "no-result", "procs": []})
        # If this is a slot on a multi-GPU box, drop procs that aren't pinned to
        # this CUDA index. (Single-GPU boxes have cuda_visible='' for both, which
        # we keep — no filtering.)
        if any(p.get("cuda_visible") for p in info.get("procs", [])):
            info["procs"] = [p for p in info["procs"]
                             if p.get("cuda_visible", "") == str(gpu_idx)
                             or not p.get("cuda_visible")]
        # Dedupe: if two procs share (seed, output_tag), keep the longer-running
        # one and surface dup_count so the UI can flag it.
        deduped = {}
        for p in info.get("procs", []):
            key = (p.get("seed"), p.get("output_tag"))
            prev = deduped.get(key)
            if prev is None or len(p.get("etime", "")) > len(prev.get("etime", "")):
                if prev is not None:
                    p["dup_count"] = prev.get("dup_count", 1) + 1
                else:
                    p["dup_count"] = 1
                deduped[key] = p
            else:
                prev["dup_count"] = prev.get("dup_count", 1) + 1
        info["procs"] = list(deduped.values())
        for p in info.get("procs", []):
            seed = p.get("seed", "?")
            phase_from_env = p.get("output_tag") or ""
            picked = None
            # Best-effort phase resolution: prefer TDMPC_GLASS_OUTPUT_TAG from
            # /proc/<pid>/environ, fall back to "latest CSV for this seed on this box".
            if phase_from_env:
                picked = (by_box_phase_seed.get((tag, phase_from_env, seed))
                          or by_phase_seed.get((phase_from_env, seed)))
                # Even if no CSV yet, we still want to surface the phase name.
                if picked is None:
                    p["phase"] = phase_from_env
                    p["best_mppi"] = p["last_mppi"] = None
                    p["best_step"] = p["last_step"] = None
                    continue
            if picked is None:
                same_seed = [c for c in csv_index if c["seed"] == seed]
                on_box = [c for c in same_seed if c["box"] == tag]
                cand = on_box or same_seed
                cand.sort(key=lambda r: r["mtime"], reverse=True)
                picked = cand[0] if cand else None
            if picked:
                best, best_step, last, last_step = best_and_last(picked["path"])
                p["phase"] = picked["phase"]
                p["best_mppi"] = round(best, 1) if best >= 0 else None
                p["best_step"] = best_step if best_step >= 0 else None
                p["last_mppi"] = round(last, 1) if last >= 0 else None
                p["last_step"] = last_step if last_step >= 0 else None
                diag_path = picked["path"].replace(".csv", "_diag.csv")
                p["diag"] = read_diag_last_mppi(diag_path)
            else:
                p["phase"] = None
                p["best_mppi"] = p["last_mppi"] = None
                p["best_step"] = p["last_step"] = None
                p["diag"] = None
            # Approximate live SPS = last_step / (etime - JIT_warmup). Underestimates
            # at short runs while JIT dominates; settles to true sps by ~1M env steps.
            et = parse_etime_seconds(p.get("etime", ""))
            last_step = p.get("last_step")
            if et and et > 60 and last_step and last_step > 0:
                JIT_WARMUP_S = 60  # rough; varies 35-160s by box
                effective = max(et - JIT_WARMUP_S, 1)
                p["sps_avg"] = int(last_step / effective)
            else:
                p["sps_avg"] = None
        boxes.append({"tag": tag, "label": label, "host": host, "port": port,
                      "gpu_idx": gpu_idx, **info})
    # active_keys = set of (phase, seed) tuples currently running anywhere
    active = sorted({(p["phase"], p["seed"]) for b in boxes for p in b.get("procs", [])
                    if p.get("phase") and p.get("seed")})
    # Cache box data so ETA computation can use live SPS without a separate probe.
    with _BOX_CACHE_LOCK:
        for b in boxes:
            _BOX_CACHE[b["tag"]] = b
    return jsonify({"boxes": boxes, "active": [{"phase": p, "seed": s} for p, s in active],
                    "ts": time.time()})


@app.route("/api/curves")
def api_curves():
    csvs = discover_csvs()
    out = []
    phase_filter = request.args.get("phase")
    for c in csvs:
        if phase_filter and phase_filter not in c["phase"]:
            continue
        pts = read_curve(c["path"])
        # downsample to 200 points max to keep payload small
        if len(pts) > 200:
            step = len(pts) // 200
            pts = pts[::step]
        out.append({**c, "points": pts})
    return jsonify({"curves": out, "ts": time.time()})


@app.route("/api/checkpoints")
def api_checkpoints():
    """List (phase, seed) tuples for which a best_mppi.pkl exists locally,
    annotated with best/last MPPI reward and sorted by best DESC."""
    out = []
    # Build CSV index once, keyed by (phase, seed) → path
    by_key = {}
    for c in discover_csvs():
        by_key[(c["phase"], c["seed"])] = c["path"]
    seen: dict[tuple, dict] = {}  # (phase, seed) → checkpoint dict
    videos_by_key = discover_existing_videos()
    for pkl in LOCAL_EXP.rglob("HopperHop_*/seed_*/checkpoints/best_mppi.pkl"):
        phase = pkl.parents[2].name.replace("HopperHop_", "")
        seed = pkl.parents[1].name.replace("seed_", "")
        key = (phase, seed)
        try:
            st = pkl.stat()
        except OSError:
            continue
        prev = seen.get(key)
        if prev is not None and prev["mtime"] >= st.st_mtime:
            continue  # keep older one if it's somehow more recent
        csv_path = by_key.get(key)
        best, best_step, last, last_step = (-1, -1, -1, -1)
        if csv_path:
            best, best_step, last, last_step = best_and_last(csv_path)
        # Pre-existing rendered videos (archive + new) for this checkpoint
        archive_videos = videos_by_key.get(key, [])
        job_videos = []
        with JOBS_LOCK:
            for jid, j in JOBS.items():
                if (j.get("phase"), j.get("seed")) == key and j.get("video"):
                    job_videos.append({
                        "label": f"job-{jid}",
                        "url": j["video"],
                        "mtime": j.get("started_at", 0),
                        "source": "dashboard",
                    })
        seen[key] = {
            "phase": phase, "seed": seed,
            "ckpt": str(pkl), "mtime": st.st_mtime,
            "size_mb": round(st.st_size / 1e6, 1),
            "best_mppi": round(best, 1) if best >= 0 else None,
            "best_step": best_step if best_step >= 0 else None,
            "last_mppi": round(last, 1) if last >= 0 else None,
            "videos": archive_videos + job_videos,
        }
    out = list(seen.values())
    # Sort: known reward DESC, then unknown by phase
    out.sort(key=lambda r: (-(r["best_mppi"] if r["best_mppi"] is not None else -1.0),
                            r["phase"]))
    return jsonify({"checkpoints": out})


@app.route("/api/render", methods=["POST"])
def api_render():
    data = request.get_json(force=True, silent=True) or {}
    phase = data.get("phase")
    seed = data.get("seed")
    env_id = data.get("env_id", "HopperHop")
    camera = data.get("camera", "cam0")
    n_episodes = int(data.get("n_episodes", 2))
    episode_length = int(data.get("episode_length", 1000))
    if not phase or seed is None:
        return jsonify({"error": "phase + seed required"}), 400
    ckpt = find_best_ckpt(phase, str(seed))
    if not ckpt:
        return jsonify({"error": f"no best_mppi.pkl for {phase}/seed_{seed}"}), 404
    # If a render for this (phase, seed) is already in flight, return its job_id
    # rather than starting a duplicate.
    with JOBS_LOCK:
        for jid, j in JOBS.items():
            if (j.get("phase") == phase and j.get("seed") == str(seed)
                    and j.get("status") in ("queued", "running")):
                return jsonify({"job_id": jid, "existing": True})
    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "phase": phase, "seed": str(seed), "env_id": env_id, "camera": camera,
            "n_episodes": n_episodes, "episode_length": episode_length,
            "ckpt": str(ckpt),
            "status": "queued", "progress": 0.0, "log": [], "video": None,
            "started_at": time.time(),
        }
    threading.Thread(target=render_worker,
                     args=(job_id, str(ckpt), env_id, camera, n_episodes, episode_length),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/render/<job_id>")
def api_render_status(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            abort(404)
        # send a copy minus the huge log; only last 30 lines
        return jsonify({**j, "log": j["log"][-30:]})


@app.route("/videos/<path:fn>")
def serve_video(fn):
    return send_from_directory(str(VIDEO_OUT), fn, conditional=True)


@app.route("/exp_videos/<path:rel>")
def serve_exp_video(rel):
    # Restrict to mp4s under EXISTING_VIDEOS_ROOT only.
    safe = (EXISTING_VIDEOS_ROOT / rel).resolve()
    try:
        safe.relative_to(EXISTING_VIDEOS_ROOT.resolve())
    except ValueError:
        abort(403)
    if not safe.exists() or safe.suffix != ".mp4":
        abort(404)
    return send_from_directory(str(safe.parent), safe.name, conditional=True)


@app.route("/api/jobs")
def api_jobs():
    """Active and recent (last hour) render jobs, so the UI can rehydrate after
    a page refresh and surface any jobs another tab kicked off."""
    cutoff = time.time() - 3600
    with JOBS_LOCK:
        items = []
        for jid, j in JOBS.items():
            if j.get("started_at", 0) < cutoff and j.get("status") in ("done", "failed"):
                continue
            items.append({
                "job_id": jid,
                "phase": j.get("phase"),
                "seed": j.get("seed"),
                "status": j.get("status"),
                "progress": j.get("progress"),
                "video": j.get("video"),
                "render_host": j.get("render_host"),
                "error_type": j.get("error_type"),
                "started_at": j.get("started_at"),
                "log_tail": j.get("log", [])[-8:],
            })
    items.sort(key=lambda r: -(r.get("started_at") or 0))
    return jsonify({"jobs": items})


@app.route("/api/phases")
def api_phases():
    """Per-CANONICAL-phase aggregated MPPI stats.

    Seeds from phasex_local, phasex_4060, phasex_2x3060, etc. are all merged
    under canonical name "phasex" so the browser shows one aggregated row.
    """
    csvs = discover_csvs()
    # Group by canonical phase name; track raw variant names for tooltip
    by_canon: dict[str, dict] = {}
    for c in csvs:
        canon = canonical_phase(c["phase"])
        entry = by_canon.setdefault(canon, {"paths": [], "variants": set()})
        entry["paths"].append(c["path"])
        if c["phase"] != canon:
            entry["variants"].add(c["phase"])
    out = []
    for canon, info in sorted(by_canon.items()):
        paths = info["paths"]
        variants = sorted(info["variants"])
        bests = []
        for path in paths:
            best, _, _, _ = best_and_last(path)
            if best >= 0:
                bests.append(best)
        n = len(bests)
        mean_b = sum(bests) / n if n else None
        std_b = (sum((b - mean_b) ** 2 for b in bests) / n) ** 0.5 if n > 1 else None
        mx = max(bests) if bests else None
        n_g1 = sum(1 for b in bests if b >= 500)
        out.append({
            "phase": canon,
            "variants": variants,
            "n_seeds": len(paths),
            "n_with_data": n,
            "mean_best": round(mean_b, 1) if mean_b is not None else None,
            "std_best": round(std_b, 1) if std_b is not None else None,
            "max_best": round(mx, 1) if mx is not None else None,
            "n_g1": n_g1,
            "notes": PHASE_NOTES.get(canon, ""),
        })
    out.sort(key=lambda r: -(r["mean_best"] or -1))
    return jsonify({"phases": out})


def _phase_matches(canon: str, tokens: list[str]) -> bool:
    """True if canon phase matches any of the filter tokens (substring)."""
    if not tokens:
        return True
    return any(t in canon for t in tokens)


@app.route("/api/phase_ci")
def api_phase_ci():
    """Per-CANONICAL-phase 95% CI curves.

    Query params:
      phase — comma-separated filter tokens; each token is a substring of the
              canonical phase name. Empty = top-10 by max MPPI.
              E.g. ?phase=phasex,phaset  shows two separate CI bands.
    """
    csvs = discover_csvs()
    raw_filter = request.args.get("phase", "").strip()
    tokens = [t.strip() for t in raw_filter.split(",") if t.strip()] if raw_filter else []

    # Group seed curves by CANONICAL phase name (merges device variants)
    by_canon: dict[str, list] = {}
    for c in csvs:
        canon = canonical_phase(c["phase"])
        if tokens and not _phase_matches(canon, tokens):
            continue
        pts = [p for p in read_curve(c["path"]) if p["eval_type"] == "mppi"]
        if pts:
            by_canon.setdefault(canon, []).append(pts)

    # When no filter, cap at top-10 by max reward
    if not tokens and len(by_canon) > 10:
        def _max_reward(curves):
            return max(p["reward"] for pts in curves for p in pts)
        ranked = sorted(by_canon.items(), key=lambda kv: -_max_reward(kv[1]))
        by_canon = dict(ranked[:10])

    out = []
    for canon, seed_curves in sorted(by_canon.items()):
        all_steps = sorted({p["step"] for pts in seed_curves for p in pts})
        if len(all_steps) > 150:
            stride = max(1, len(all_steps) // 150)
            all_steps = all_steps[::stride]
        rows = []
        for qs in all_steps:
            vals = []
            for pts in seed_curves:
                v = next((p["reward"] for p in reversed(pts) if p["step"] <= qs), None)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            if n > 1:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                se = math.sqrt(var / n)
                margin = 1.96 * se
            else:
                margin = 0.0
            rows.append({"step": qs, "mean": round(mean, 2),
                         "lower": round(mean - margin, 2),
                         "upper": round(mean + margin, 2), "n": n})
        if rows:
            out.append({"phase": canon, "n_seeds": len(seed_curves), "rows": rows,
                        "notes": PHASE_NOTES.get(canon, "")})
    return jsonify({"ci_curves": out})


# ─── Central task queue API ────────────────────────────────────────────────

def _load_central_queue() -> list[dict]:
    if not CENTRAL_QUEUE_FILE.exists():
        return []
    try:
        return json.loads(CENTRAL_QUEUE_FILE.read_text())
    except Exception:
        return []


def _save_central_queue(tasks: list[dict]):
    CENTRAL_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CENTRAL_QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tasks, indent=2))
    tmp.replace(CENTRAL_QUEUE_FILE)


def _with_queue_lock(fn):
    lock_path = CENTRAL_QUEUE_FILE.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tasks = _load_central_queue()
            result = fn(tasks)
            if result is not None:
                _save_central_queue(result)
            return tasks if result is None else result
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


DEFAULT_TASK_DURATION_S = 14400  # 4-hour fallback when no SPS or history

# Max training steps and early-stop patience (in env steps).
_MAX_STEPS = 10_000_000
_PATIENCE_STEPS = 3_000_000


def _sps_remaining_s(task: dict) -> float | None:
    """Return remaining seconds for a running training task using cached SPS data.

    Formula: remaining_steps = min(MAX, last + max(0, PATIENCE - (last-best))) - last
    Falls back to None if no SPS data is available.
    """
    box_tag = task.get("box")
    if not box_tag:
        return None
    with _BOX_CACHE_LOCK:
        box_data = _BOX_CACHE.get(box_tag, {})
    procs = box_data.get("procs", [])
    if not procs:
        return None
    # Try to match by seed extracted from task env.
    seed_m = re.search(r"SEEDS?=(\S+)", task.get("env", ""))
    seed = seed_m.group(1) if seed_m else None
    matched = [p for p in procs if seed and str(p.get("seed")) == seed]
    proc = (matched or procs)[0]
    sps = proc.get("sps_avg")
    if not sps or sps <= 0:
        return None
    last_step = proc.get("last_step") or 0
    best_step = proc.get("best_step") or 0
    steps_since_best = max(0, last_step - best_step)
    patience_left = max(0, _PATIENCE_STEPS - steps_since_best)
    effective_target = min(_MAX_STEPS, last_step + patience_left)
    remaining_steps = max(0, effective_target - last_step)
    return remaining_steps / sps


def _compute_queue_etas(tasks: list[dict]) -> tuple[list[dict], str | None]:
    """Add elapsed/remaining/eta fields to each task. Returns (annotated, queue_eta_iso)."""
    import heapq
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)

    def parse_iso(s):
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def to_iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Per-launcher average duration from done tasks (fallback when no live SPS).
    launcher_durs: dict[str, list[float]] = {}
    for t in tasks:
        if t["status"] == "done" and t.get("started_at") and t.get("ended_at"):
            s, e = parse_iso(t["started_at"]), parse_iso(t["ended_at"])
            if s and e and e > s:
                launcher_durs.setdefault(t.get("launcher", ""), []).append((e - s).total_seconds())
    avg_dur: dict[str, float] = {l: sum(ds) / len(ds) for l, ds in launcher_durs.items()}

    def est_remaining_s(task) -> float:
        """Estimate remaining seconds for a task (running or pending)."""
        if task.get("type") == "render":
            rp = task.get("render_params") or {}
            total_s = (int(rp.get("n_episodes", 1)) * int(rp.get("episode_length", 250))
                       / SPS_RENDER_APPROX)
            if task.get("started_at"):
                s = parse_iso(task["started_at"])
                if s:
                    return max(0.0, total_s - (now - s).total_seconds())
            return total_s
        # Try live SPS for running tasks.
        if task["status"] == "running":
            sps_rem = _sps_remaining_s(task)
            if sps_rem is not None:
                return sps_rem
        return avg_dur.get(task.get("launcher", ""), DEFAULT_TASK_DURATION_S)

    # Estimate total duration for scheduling pending tasks (use DEFAULT for pending).
    def est_total_dur(task) -> float:
        if task.get("type") == "render":
            rp = task.get("render_params") or {}
            return (int(rp.get("n_episodes", 1)) * int(rp.get("episode_length", 250))
                    / SPS_RENDER_APPROX)
        return avg_dur.get(task.get("launcher", ""), DEFAULT_TASK_DURATION_S)

    # Box free-at times: running tasks → now + remaining; idle boxes → now.
    all_tags = [b[0] for b in BOXES]
    box_free: dict[str, datetime] = {tag: now for tag in all_tags}
    for t in tasks:
        if t["status"] == "running" and t.get("box") and t.get("started_at"):
            rem = est_remaining_s(t)
            box_free[t["box"]] = max(now + timedelta(seconds=rem), now)

    # Simulate pending task scheduling with a min-heap of (free_time, box_tag).
    heap = [(ts, tag) for tag, ts in box_free.items()]
    heapq.heapify(heap)
    pending_sorted = sorted(
        [t for t in tasks if t["status"] == "pending" and t.get("type") != "render"],
        key=lambda t: (t.get("priority", 10), t.get("created_at", ""))
    )
    sched: dict[str, tuple[datetime, datetime]] = {}
    for t in pending_sorted:
        if not heap:
            break
        free_ts, tag = heapq.heappop(heap)
        start = max(free_ts, now)
        finish = start + timedelta(seconds=est_total_dur(t))
        sched[t["id"]] = (start, finish)
        heapq.heappush(heap, (finish, tag))

    # Annotate tasks with ETA fields.
    result = []
    for t in tasks:
        t = dict(t)
        if t["status"] == "running" and t.get("started_at"):
            s = parse_iso(t["started_at"])
            if s:
                elapsed = (now - s).total_seconds()
                rem = est_remaining_s(t)
                eta = now + timedelta(seconds=rem)
                t["elapsed_s"] = int(elapsed)
                t["remaining_s"] = int(rem)
                t["eta_iso"] = to_iso(eta)
                # For SPS-based runs, also surface current step info for the UI.
                box_tag = t.get("box", "")
                with _BOX_CACHE_LOCK:
                    box_data = _BOX_CACHE.get(box_tag, {})
                procs = box_data.get("procs", [])
                if procs:
                    seed_m = re.search(r"SEEDS?=(\S+)", t.get("env", ""))
                    seed = seed_m.group(1) if seed_m else None
                    matched = [p for p in procs if seed and str(p.get("seed")) == seed]
                    proc = (matched or procs)[0]
                    t["_live_sps"] = proc.get("sps_avg")
                    t["_live_last_step"] = proc.get("last_step")
                    t["_live_best_reward"] = proc.get("best_mppi")
        elif t["status"] == "pending" and t["id"] in sched:
            start, finish = sched[t["id"]]
            t["estimated_start_iso"] = to_iso(start)
            t["eta_iso"] = to_iso(finish)
        result.append(t)

    active_etas = [t["eta_iso"] for t in result
                   if t["status"] in ("running", "pending") and t.get("eta_iso")]
    queue_eta = max(active_etas) if active_etas else None
    return result, queue_eta


@app.route("/api/queue")
def api_queue_get():
    tasks = _load_central_queue()
    tasks_sorted = sorted(tasks, key=lambda t: (t.get("priority", 10), t.get("created_at", "")))
    annotated, queue_eta = _compute_queue_etas(tasks_sorted)
    return jsonify({"tasks": annotated, "queue_eta": queue_eta})


@app.route("/api/queue", methods=["POST"])
def api_queue_add():
    body = request.get_json(force=True)
    label = (body.get("label") or "").strip()
    launcher = (body.get("launcher") or "").strip()
    env = (body.get("env") or "").strip()
    priority = int(body.get("priority", 10))
    if not label or not launcher:
        return jsonify({"error": "label and launcher are required"}), 400
    from datetime import datetime, timezone
    task = {
        "id": "t" + uuid.uuid4().hex[:7],
        "label": label,
        "launcher": launcher,
        "env": env,
        "priority": priority,
        "status": "pending",
        "box": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started_at": None,
        "ended_at": None,
    }
    def add(tasks):
        tasks.append(task)
        return tasks
    _with_queue_lock(add)
    return jsonify({"ok": True, "id": task["id"]})


@app.route("/api/queue/<task_id>", methods=["DELETE"])
def api_queue_delete(task_id):
    removed = [False]
    def delete(tasks):
        nonlocal removed
        new = [t for t in tasks if t["id"] != task_id]
        removed[0] = len(new) < len(tasks)
        return new
    _with_queue_lock(delete)
    if not removed[0]:
        return jsonify({"error": "task not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/queue/<task_id>/priority", methods=["POST"])
def api_queue_priority(task_id):
    body = request.get_json(force=True)
    delta = int(body.get("delta", -1))  # negative = higher priority (lower number)
    def bump(tasks):
        for t in tasks:
            if t["id"] == task_id and t["status"] == "pending":
                t["priority"] = max(1, t["priority"] + delta)
        return tasks
    _with_queue_lock(bump)
    return jsonify({"ok": True})


@app.route("/api/queue/<task_id>/retry", methods=["POST"])
def api_queue_retry(task_id):
    """Reset a running/failed/done task back to pending so the daemon re-runs it."""
    from datetime import datetime, timezone
    def retry(tasks):
        for t in tasks:
            if t["id"] == task_id and t["status"] in ("running", "failed", "done"):
                t["status"] = "pending"
                t["box"] = None
                t["started_at"] = None
                t["ended_at"] = None
        return tasks
    _with_queue_lock(retry)
    return jsonify({"ok": True})


@app.route("/api/queue/<task_id>/log")
def api_queue_task_log(task_id):
    """Return last 60 lines of the task's remote log (/tmp/fleet_<id>.log)."""
    tasks = _load_central_queue()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "task not found"}), 404
    box_tag = task.get("box")
    log_path = f"/tmp/fleet_{task_id}.log"
    box_info = next((b for b in BOXES if b[0] == box_tag), None)
    try:
        if box_tag == "local":
            r = subprocess.run(["tail", "-n", "60", log_path],
                               capture_output=True, text=True, timeout=5)
            lines = r.stdout.splitlines()
        elif box_info:
            tag, port, host, gpu_idx, label = box_info
            r = subprocess.run(
                ["ssh", "-p", str(port), "-i", SSH_KEY,
                 "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                 "-o", "BatchMode=yes", f"root@{host}",
                 f"tail -n 60 {log_path} 2>/dev/null || echo '(log not found at {log_path})'"],
                capture_output=True, text=True, timeout=15,
            )
            lines = r.stdout.splitlines()
        else:
            lines = [f"(box '{box_tag}' not in registry)"]
    except Exception as e:
        lines = [f"(error fetching log: {e})"]
    return jsonify({"lines": lines, "log_path": log_path, "box": box_tag})


@app.route("/api/queue/render", methods=["POST"])
def api_queue_render_add():
    """Add a render task to the central queue (type=render, priority=1).

    The local _render_queue_worker thread will pick it up and run render_worker.
    """
    from datetime import datetime, timezone
    body = request.get_json(force=True)
    phase = (body.get("phase") or "").strip()
    seed = str(body.get("seed", ""))
    env_id = body.get("env_id", "HopperHop")
    camera = body.get("camera", "cam0")
    n_episodes = int(body.get("n_episodes", 1))
    episode_length = int(body.get("episode_length", 250))
    if not phase or not seed:
        return jsonify({"error": "phase + seed required"}), 400
    ckpt = find_best_ckpt(phase, seed)
    if not ckpt:
        return jsonify({"error": f"no best_mppi.pkl for {phase}/seed_{seed}"}), 404
    task = {
        "id": "t" + uuid.uuid4().hex[:7],
        "label": f"render {phase} s{seed} ({n_episodes}×{episode_length}steps)",
        "launcher": "render_glass_rollout.py",
        "env": "",
        "type": "render",
        "render_params": {
            "ckpt": str(ckpt),
            "env_id": env_id,
            "camera": camera,
            "n_episodes": n_episodes,
            "episode_length": episode_length,
            "phase": phase,
            "seed": seed,
        },
        "priority": 1,
        "status": "pending",
        "box": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started_at": None,
        "ended_at": None,
    }
    def add(tasks):
        tasks.append(task)
        return tasks
    _with_queue_lock(add)
    return jsonify({"ok": True, "id": task["id"]})


def _render_queue_worker():
    """Background thread: claims pending render tasks from central_queue and runs them locally."""
    from datetime import datetime, timezone
    # On startup reset any render tasks stuck as "running" (orphaned from a previous dashboard run).
    def _reset_stuck_renders(tasks):
        for t in tasks:
            if t.get("type") == "render" and t["status"] == "running":
                t["status"] = "pending"
                t["box"] = None
                t["started_at"] = None
        return tasks
    _with_queue_lock(_reset_stuck_renders)
    while True:
        time.sleep(10)
        try:
            tasks = _load_central_queue()
            pending = [t for t in tasks if t.get("type") == "render" and t["status"] == "pending"]
            if not pending:
                continue
            task = sorted(pending, key=lambda t: (t.get("priority", 10), t.get("created_at", "")))[0]

            def claim(tasks, _id=task["id"]):
                for t in tasks:
                    if t["id"] == _id and t["status"] == "pending":
                        t["status"] = "running"
                        t["box"] = "local_render"
                        t["started_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                return tasks

            _with_queue_lock(claim)
            tasks2 = _load_central_queue()
            claimed = next((t for t in tasks2 if t["id"] == task["id"] and t["status"] == "running"), None)
            if not claimed:
                continue

            rp = claimed.get("render_params", {})
            ckpt = rp.get("ckpt", "")
            env_id = rp.get("env_id", "HopperHop")
            camera = rp.get("camera", "cam0")
            n_eps = int(rp.get("n_episodes", 1))
            ep_len = int(rp.get("episode_length", 250))
            phase = rp.get("phase", "")
            seed = rp.get("seed", "")

            if not ckpt:
                def fail_no_ckpt(tasks, _id=task["id"]):
                    for t in tasks:
                        if t["id"] == _id:
                            t["status"] = "failed"
                            t["ended_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    return tasks
                _with_queue_lock(fail_no_ckpt)
                continue

            def _do_render(tid=task["id"], ckpt=ckpt, env_id=env_id, camera=camera,
                           n_eps=n_eps, ep_len=ep_len, phase=phase, seed=seed):
                job_id = uuid.uuid4().hex[:10]
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "phase": phase, "seed": seed, "env_id": env_id, "camera": camera,
                        "n_episodes": n_eps, "episode_length": ep_len,
                        "ckpt": ckpt, "status": "queued", "progress": 0.0, "log": [],
                        "video": None, "started_at": time.time(), "queue_task_id": tid,
                    }
                render_worker(job_id, ckpt, env_id, camera, n_eps, ep_len)
                with JOBS_LOCK:
                    final_status = JOBS.get(job_id, {}).get("status", "failed")

                def finish(tasks, _id=tid, _status=final_status):
                    for t in tasks:
                        if t["id"] == _id:
                            t["status"] = _status
                            t["ended_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    return tasks
                _with_queue_lock(finish)

            threading.Thread(target=_do_render, daemon=True).start()
        except Exception:
            pass


threading.Thread(target=_render_queue_worker, daemon=True).start()


# ─── HTML (Plotly via CDN; vanilla JS fetch) ─────────────────────────────

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>TD-MPC-Glass Live Dashboard</title>
<style>
  :root { --bg:#0e0f12; --panel:#161922; --fg:#dbe1eb; --muted:#7e8ba0; --accent:#4ec9b0; --warn:#e0a44c; --bad:#e15c5c; --good:#7dd87b; --line:#2a2f3b; }
  *{box-sizing:border-box}
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg);font-size:13px}
  header{padding:12px 18px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:18px}
  header h1{margin:0;font-size:16px;font-weight:600}
  header .meta{color:var(--muted);font-size:12px}
  .container{padding:14px 18px;max-width:1600px;margin:0 auto}
  section{background:var(--panel);border:1px solid var(--line);border-radius:6px;margin-bottom:14px;padding:12px 16px}
  section h2{font-size:13px;font-weight:600;margin:0 0 10px 0;color:var(--accent);letter-spacing:.04em;text-transform:uppercase;display:flex;align-items:center;gap:10px}
  .refresh-btn{font-size:11px;padding:2px 8px;letter-spacing:0;text-transform:none;font-weight:400;background:#2a3346;color:var(--fg);border:1px solid var(--line);border-radius:3px;cursor:pointer;margin-left:auto}
  .refresh-btn:hover{background:#36405a}
  select{background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:2px 6px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
  th{color:var(--muted);font-weight:500}
  .box-good{color:var(--good)} .box-bad{color:var(--bad)} .box-warn{color:var(--warn)}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .util-bar{display:inline-block;width:60px;height:9px;background:var(--line);border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:4px}
  .util-bar>span{display:block;height:100%;background:var(--accent)}
  .util-bar.hot>span{background:var(--warn)}
  button{background:#2a3346;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px}
  button:hover{background:#36405a}
  button:disabled{opacity:.5;cursor:not-allowed}
  .chip{display:inline-block;padding:1px 7px;border-radius:10px;background:#28313f;color:#bdc7d5;font-size:11px;margin-right:4px}
  .chip.knee{background:#54391c;color:#e0a44c}
  .chip.hier{background:#1c4254;color:#4ec9b0}
  #curves{width:100%;height:520px}
  .video-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .video-card{background:#1b1f2a;border:1px solid var(--line);border-radius:5px;padding:8px 10px;min-width:260px}
  progress{width:160px;height:6px}
  .small{font-size:11px;color:var(--muted)}
  .pill{padding:0 6px;border-radius:8px;font-size:10px;margin-left:4px}
  .pill.green{background:#1f3d22;color:#7dd87b}
  .pill.gray{background:#262a35;color:#9099a8}
  .phase-browser{background:#1b1f2a;border:1px solid var(--line);border-radius:5px;padding:8px 10px;max-height:300px;overflow-y:auto}
  .phase-info-bar{background:#1b1f2a;border:1px solid var(--line);border-radius:5px;padding:6px 12px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
  .sortable-th{cursor:pointer;user-select:none}
  .sortable-th:hover{color:var(--accent)}
  .filter-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
  /* Run Inspector */
  .ri-card{background:#1b1f2a;border:1px solid var(--line);border-radius:5px;margin-bottom:8px;overflow:hidden}
  .ri-header{display:flex;align-items:center;gap:10px;padding:8px 12px;cursor:pointer;user-select:none}
  .ri-header:hover{background:#222a3b}
  .ri-title{font-weight:600;flex:1}
  .ri-meta{font-size:11px;color:var(--muted)}
  .ri-body{padding:10px 14px;border-top:1px solid var(--line);display:none}
  .ri-body.open{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .ri-section-title{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:5px;font-weight:600}
  .ri-row{display:flex;justify-content:space-between;font-size:12px;padding:2px 0;border-bottom:1px solid #1e2535}
  .ri-key{color:var(--muted)}
  .ri-val{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .ri-log{background:#0e0f12;border:1px solid var(--line);border-radius:3px;padding:6px 8px;font-size:10px;font-family:ui-monospace,monospace;max-height:120px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin-top:6px;line-height:1.4}
  .ri-span-full{grid-column:1/-1}
  .ri-path{color:#64b5f6;font-size:11px;font-family:ui-monospace,monospace;word-break:break-all}
  .ri-chevron{transition:transform .15s;font-size:14px;color:var(--muted)}
  .ri-chevron.open{transform:rotate(90deg)}
</style></head><body>
<header>
  <h1>TD-MPC-Glass Live Dashboard</h1>
  <span class="meta">refresh every 30s · <span id="ts">—</span></span>
  <span class="meta" style="margin-left:auto">G1 = 5/5 &gt; 500 · G2 = break 600</span>
</header>

<div class="container">

  <section><h2>Box Fleet <button class="refresh-btn" onclick="loadBoxes()">&#x21bb; refresh</button></h2>
    <table id="boxes"><thead>
      <tr><th>Tag</th><th>Label</th><th>GPU</th><th>Mem</th><th>CPU</th><th>SPS</th><th>Running (phase · seed · best · last)</th></tr>
    </thead><tbody></tbody></table>
  </section>

  <section id="queue-section"><h2>Task Queue
    <button class="refresh-btn" onclick="loadQueue()">&#x21bb; refresh</button>
    <button class="refresh-btn" id="add-task-btn" onclick="toggleAddTask()" style="margin-left:4px">+ add task</button>
    <span id="queue-eta-hdr" class="small" style="opacity:.65;margin-left:10px;font-weight:normal"></span>
  </h2>
  <div id="add-task-form" style="display:none;background:#1b1f2a;border:1px solid var(--line);border-radius:5px;padding:10px 14px;margin-bottom:10px">
    <div style="display:grid;grid-template-columns:1fr 2fr 2fr 80px;gap:8px;align-items:end">
      <div><label class="small">Priority (lower=first)<br>
        <input id="at-priority" type="number" value="10" min="1" max="99"
          style="width:100%;background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:3px 6px"></label></div>
      <div><label class="small">Label<br>
        <input id="at-label" type="text" placeholder="e.g. phaseab seed 1"
          style="width:100%;background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:3px 6px"></label></div>
      <div><label class="small">Launcher script<br>
        <input id="at-launcher" type="text" placeholder="scripts/run_phase1b_10m.sh"
          style="width:100%;background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:3px 6px"></label></div>
      <div><button onclick="addTask()" style="width:100%;padding:6px 0">Add</button></div>
    </div>
    <div style="margin-top:6px"><label class="small">Env vars (optional)<br>
      <input id="at-env" type="text" placeholder="SEEDS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75"
        style="width:100%;background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:3px 6px"></label></div>
    <div id="at-error" class="small box-bad" style="display:none;margin-top:6px"></div>
  </div>
  <table id="queue-table"><thead>
    <tr><th style="width:50px">Pri</th><th>Label</th><th style="width:80px">Status</th><th style="width:90px">Box</th><th style="width:170px">ETA / Progress</th><th style="width:90px">Actions</th></tr>
  </thead><tbody></tbody></table>
  <div id="queue-empty" class="small" style="display:none;padding:6px;color:var(--muted)">Queue is empty.</div>
  </section>

  <section id="run-inspector-section"><h2>Run Inspector
    <button class="refresh-btn" onclick="loadRunInspector()">&#x21bb; refresh</button>
    <span class="small" id="ri-count" style="opacity:.65;font-weight:normal;margin-left:6px"></span>
  </h2>
  <div id="run-inspector-cards"></div>
  </section>

  <section><h2>Learning Curves <span class="small" id="curves-count"></span>
    <button class="refresh-btn" onclick="loadCurves()">&#x21bb; refresh</button></h2>
    <div class="small filter-row">
      <label><input type="checkbox" id="only-mppi" checked> only MPPI</label>
      <label><input type="checkbox" id="only-running"> running only</label>
      <span>View:
        <label style="margin-left:4px"><input type="radio" name="curve-mode" id="mode-seeds" value="seeds" checked> seeds</label>
        <label style="margin-left:6px"><input type="radio" name="curve-mode" id="mode-ci" value="ci"> 95% CI</label>
      </span>
      <label>Phase: <input id="phase-filter" list="phase-list" type="text"
        style="background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:2px 6px;width:200px"
        placeholder="type to filter…"></label>
      <datalist id="phase-list"></datalist>
      <button onclick="loadCurves()">apply</button>
      <button class="refresh-btn" id="phases-btn" onclick="togglePhaseBrowser()">📊 phases</button>
    </div>
    <div id="phase-info" style="display:none;margin-bottom:6px"></div>
    <div id="phase-browser" style="display:none;margin-bottom:8px"></div>
    <div id="curves"></div>
  </section>

  <section><h2>Render Rollout
    <button class="refresh-btn" onclick="loadCheckpoints()">&#x21bb; refresh</button></h2>
    <div class="small" style="margin-bottom:8px">
      Length:
      <select id="render-length">
        <option value="1|250" selected>default (1 × 250 steps)</option>
        <option value="1|500">long (1 × 500 steps)</option>
      </select>
      <span class="small">camera: cam0 · 320×240 · queued via task queue (priority 1)</span>
    </div>
    <div id="ckpts" class="video-row"></div>
    <div id="jobs" style="margin-top:14px"></div>
  </section>

</div>

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
const $ = sel => document.querySelector(sel);
// Active (phase, seed) set, refreshed by loadBoxes(); consumed by loadCurves().
let ACTIVE_KEYS = new Set();

function fmtMppi(v){
  if (v==null) return '<span class="small">—</span>';
  const cls = v>=500 ? 'box-good' : (v>=300 ? 'box-warn' : '');
  return `<span class="mono ${cls}">${v.toFixed ? v.toFixed(1) : v}</span>`;
}
function fmtStep(s){ if (s==null) return ''; return `<span class="small">@${(s/1e6).toFixed(2)}M</span>`; }

// ── Phase browser + CI mode ────────────────────────────────────────────
let PHASE_DATA = [];
// Stable color palette — indexed by canonical phase name so same phase always same color.
const PHASE_COLORS = ['#4ec9b0','#e0a44c','#7dd87b','#e15c5c','#9b7de8','#64b5f6','#f48fb1','#a5d6a7','#ffcc80','#80deea'];
const _phaseColorCache = {};
function phaseColor(name) {
  if (!_phaseColorCache[name]) {
    const keys = Object.keys(_phaseColorCache);
    _phaseColorCache[name] = PHASE_COLORS[keys.length % PHASE_COLORS.length];
  }
  return _phaseColorCache[name];
}
// seed colors for per-seed view — also use canonical phase color family but vary opacity
function seedColor(phase, seed) {
  const col = phaseColor(phase);
  return col; // same hue; CI band uses 16% opacity fill already
}

// Multi-phase selection for CI comparison
const SELECTED_CI_PHASES = new Set();

function togglePhaseBrowser() {
  const panel = document.getElementById('phase-browser');
  const btn = document.getElementById('phases-btn');
  const open = panel.style.display !== 'none';
  panel.style.display = open ? 'none' : '';
  btn.textContent = open ? '📊 phases' : '📊 phases ▲';
  if (!open) loadPhases();
}

function loadPhases() {
  fetch('/api/phases').then(r=>r.json()).then(j=>{
    PHASE_DATA = j.phases;
    // Preload color cache so colors stay stable across sorts
    PHASE_DATA.forEach(p => phaseColor(p.phase));
    const dl = document.getElementById('phase-list');
    if (dl) { dl.innerHTML = ''; PHASE_DATA.forEach(p=>{ const o=document.createElement('option'); o.value=p.phase; dl.appendChild(o); }); }
    renderPhaseBrowser(PHASE_DATA, 'mean_best', false);
    updatePhaseInfo(document.getElementById('phase-filter')?.value.trim() || '');
  });
}

let _pbSortKey = 'mean_best', _pbSortAsc = false;
function renderPhaseBrowser(phases, sortKey, sortAsc) {
  _pbSortKey = sortKey; _pbSortAsc = sortAsc;
  const panel = document.getElementById('phase-browser');
  if (!panel || panel.style.display === 'none') return;
  const ciMode = document.getElementById('mode-ci')?.checked;
  const sorted = [...phases].sort((a,b)=>{
    const av = (sortKey==='phase') ? a.phase : (a[sortKey]??-999);
    const bv = (sortKey==='phase') ? b.phase : (b[sortKey]??-999);
    if (typeof av==='string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av-bv : bv-av;
  });
  function th(col,label){ const arr=col===_pbSortKey?(_pbSortAsc?'↑':'↓'):''; return `<th class="sortable-th" onclick="renderPhaseBrowser(PHASE_DATA,'${col}',${col===_pbSortKey?!sortAsc:false})">${label}${arr?` ${arr}`:''}</th>`; }
  const cbHeader = ciMode ? '<th title="Select for multi-phase CI">&#x2713;</th>' : '';
  const rows = sorted.map(p=>{
    const col = phaseColor(p.phase);
    const dot = `<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${col};margin-right:5px;vertical-align:middle"></span>`;
    const mean = p.mean_best!=null ? p.mean_best.toFixed(1) : '—';
    const std  = p.std_best!=null  ? `±${p.std_best.toFixed(1)}` : '';
    const max  = p.max_best!=null  ? p.max_best.toFixed(1) : '—';
    const mCls = (p.mean_best??0)>=500?'box-good':((p.mean_best??0)>=300?'box-warn':'');
    const g1Cls= p.n_g1>0?'box-good':'';
    const varTip = p.variants&&p.variants.length ? ` title="Merged from: ${p.variants.join(', ')}"` : '';
    const varBadge = p.variants&&p.variants.length ? `<span class="small" style="color:var(--muted)" title="${p.variants.join(', ')}"> +${p.variants.length}</span>` : '';
    const cbCell = ciMode ? `<td><input type="checkbox" class="phase-cb" data-phase="${p.phase}" ${SELECTED_CI_PHASES.has(p.phase)?'checked':''}
        onclick="toggleCiPhase('${p.phase}',this.checked)"></td>` : '';
    const notes = p.notes||'';
    return `<tr${varTip}>
      ${cbCell}
      <td class="mono" style="cursor:pointer" onclick="setPhaseFilter('${p.phase}')">${dot}${p.phase}${varBadge}</td>
      <td>${p.n_with_data}/${p.n_seeds}</td>
      <td class="mono ${mCls}">${mean} <span style="color:var(--muted)">${std}</span></td>
      <td class="mono">${max}</td>
      <td class="${g1Cls}">${p.n_g1}/${p.n_with_data}</td>
      <td class="small" style="max-width:280px;white-space:normal">${notes}</td></tr>`;
  }).join('');
  const compareBtn = ciMode ? `<div style="padding:6px 0">
    <button onclick="compareSelected()" id="compare-btn"
      ${SELECTED_CI_PHASES.size===0?'disabled':''}>
      Compare ${SELECTED_CI_PHASES.size} selected in CI</button>
    <button class="refresh-btn" onclick="clearCiSelection()" style="margin-left:6px">clear</button>
  </div>` : '';
  panel.innerHTML = `<div class="phase-browser">
    ${compareBtn}
    <table style="width:100%;font-size:12px"><thead><tr>
      ${cbHeader}${th('phase','Phase')}${th('n_with_data','Seeds')}
      ${th('mean_best','Mean')}${th('max_best','Max')}${th('n_g1','G1')}
      <th>Notes</th></tr></thead>
    <tbody>${rows}</tbody></table>
  </div>`;
}

function toggleCiPhase(phase, checked) {
  if (checked) SELECTED_CI_PHASES.add(phase);
  else SELECTED_CI_PHASES.delete(phase);
  const btn = document.getElementById('compare-btn');
  if (btn) { btn.disabled = SELECTED_CI_PHASES.size === 0; btn.textContent = `Compare ${SELECTED_CI_PHASES.size} selected in CI`; }
}

function compareSelected() {
  if (!SELECTED_CI_PHASES.size) return;
  const pf = document.getElementById('phase-filter');
  if (pf) pf.value = [...SELECTED_CI_PHASES].join(',');
  document.getElementById('mode-ci').checked = true;
  loadCurves();
}

function clearCiSelection() {
  SELECTED_CI_PHASES.clear();
  renderPhaseBrowser(PHASE_DATA, _pbSortKey, _pbSortAsc);
}

function setPhaseFilter(phase) {
  const pf = document.getElementById('phase-filter'); if (pf) pf.value = phase;
  loadCurves();
}

function updatePhaseInfo(filterText) {
  const panel = document.getElementById('phase-info');
  if (!panel) return;
  if (!filterText || !PHASE_DATA.length) { panel.style.display='none'; return; }
  // support comma-separated tokens — match any
  const tokens = filterText.split(',').map(t=>t.trim()).filter(Boolean);
  const matching = PHASE_DATA.filter(p=> tokens.some(t=>p.phase.includes(t)));
  if (!matching.length) { panel.style.display='none'; return; }
  panel.style.display = '';
  panel.innerHTML = `<div class="phase-info-bar small">${matching.map(p=>{
    const col = phaseColor(p.phase);
    const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${col};margin-right:4px;vertical-align:middle"></span>`;
    const mean = p.mean_best!=null ? p.mean_best.toFixed(1) : '—';
    const std  = p.std_best!=null  ? `±${p.std_best.toFixed(1)}` : '';
    const max  = p.max_best!=null  ? p.max_best.toFixed(1) : '—';
    const mCls = (p.mean_best??0)>=500?'box-good':((p.mean_best??0)>=300?'box-warn':'');
    const varStr = p.variants&&p.variants.length ? ` <span style="color:var(--muted)" title="${p.variants.join(', ')}">(+${p.variants.length} device variants merged)</span>` : '';
    return `<span>${dot}<b class="mono">${p.phase}</b>${varStr} · ${p.n_with_data} seeds · `+
           `<span class="${mCls}">${mean}${std}</span> · max ${max} · G1 ${p.n_g1}/${p.n_with_data}`+
           `${p.notes?` <span style="color:var(--muted)">· ${p.notes}</span>`:''}</span>`;
  }).join('<span style="color:var(--line);margin:0 4px">│</span>')}</div>`;
}

function loadPhaseCI(phaseFilter) {
  const url = '/api/phase_ci' + (phaseFilter ? '?phase='+encodeURIComponent(phaseFilter) : '');
  fetch(url).then(r=>r.json()).then(j=>{
    const traces = [];
    j.ci_curves.forEach(ci=>{
      // Same canonical phase always gets same color (stable across filter changes)
      const col = phaseColor(ci.phase);
      const steps = ci.rows.map(r=>r.step);
      const upper = ci.rows.map(r=>r.upper);
      const lower = ci.rows.map(r=>r.lower);
      const mean  = ci.rows.map(r=>r.mean);
      const label = `${ci.phase} (n=${ci.n_seeds})${ci.notes?' · '+ci.notes:''}`;
      // CI shaded band: lower anchor → upper filled with same color at ~18% opacity
      traces.push({x:steps, y:lower, type:'scatter', mode:'lines', showlegend:false,
                   line:{color:'transparent'}, hoverinfo:'skip', legendgroup:ci.phase});
      traces.push({x:steps, y:upper, type:'scatter', mode:'lines', fill:'tonexty',
                   fillcolor:col+'2e', line:{color:'transparent'}, showlegend:false,
                   hoverinfo:'skip', legendgroup:ci.phase});
      // Mean line — same color, full opacity
      traces.push({x:steps, y:mean, type:'scatter', mode:'lines', name:label,
                   line:{color:col, width:2.5}, legendgroup:ci.phase,
                   hovertemplate:`<b>${ci.phase}</b><br>step %{x:,d}<br>mean %{y:.1f}<extra></extra>`});
    });
    $('#curves-count').textContent = `(${j.ci_curves.length} phase${j.ci_curves.length===1?'':'s'}, 95% CI bands)`;
    Plotly.react('curves', traces, {
      paper_bgcolor:'#161922', plot_bgcolor:'#161922',
      font:{color:'#dbe1eb', size:11},
      xaxis:{title:'env step', gridcolor:'#2a2f3b'},
      yaxis:{title:'mean MPPI ± 95% CI', gridcolor:'#2a2f3b'},
      shapes:[
        {type:'line',x0:0,x1:1,xref:'paper',y0:500,y1:500,line:{color:'#7dd87b',dash:'dot',width:1}},
        {type:'line',x0:0,x1:1,xref:'paper',y0:600,y1:600,line:{color:'#4ec9b0',dash:'dot',width:1}},
      ],
      margin:{l:50,r:20,t:20,b:40}, legend:{font:{size:10},x:1.02,y:1}, showlegend:true,
    }, {displaylogo:false, responsive:true});
  });
}

// ── Per-seed curve view ────────────────────────────────────────────────
function loadCurves(){
  const phaseFilter = $('#phase-filter').value.trim();
  const ciMode = document.getElementById('mode-ci')?.checked;

  // Update phase info panel (lazy-load phase data if needed)
  if (PHASE_DATA.length) updatePhaseInfo(phaseFilter);
  else fetch('/api/phases').then(r=>r.json()).then(j=>{ PHASE_DATA=j.phases; updatePhaseInfo(phaseFilter);
    const dl=document.getElementById('phase-list'); if(dl){ dl.innerHTML=''; PHASE_DATA.forEach(p=>{const o=document.createElement('option');o.value=p.phase;dl.appendChild(o);}); }
  });

  // Refresh browser table so checkboxes appear/disappear with mode switch
  if (PHASE_DATA.length) renderPhaseBrowser(PHASE_DATA, _pbSortKey, _pbSortAsc);
  if (ciMode) { loadPhaseCI(phaseFilter); return; }

  const url = '/api/curves' + (phaseFilter ? '?phase='+encodeURIComponent(phaseFilter) : '');
  fetch(url).then(r=>r.json()).then(j=>{
    const onlyMppi = $('#only-mppi').checked;
    const onlyRunning = $('#only-running') && $('#only-running').checked;
    const filtered = onlyRunning
      ? j.curves.filter(c => ACTIVE_KEYS.has(`${c.phase}|${c.seed}`))
      : j.curves;
    $('#curves-count').textContent =
      `(${filtered.length}/${j.curves.length} traces${onlyRunning ? ', running only' : ''})`;
    const traces = [];
    filtered.forEach(c=>{
      let pts = c.points;
      if (onlyMppi) pts = pts.filter(p=>p.eval_type==='mppi');
      if (!pts.length) return;
      const best = Math.max(...pts.map(p=>p.reward));
      const isRunning = ACTIVE_KEYS.has(`${c.phase}|${c.seed}`);
      // Use stable phase color (same hue as CI view); dim if not winner
      const baseCol = phaseColor(c.phase);
      const color = best>=500 ? baseCol : (best>=300 ? baseCol+'bb' : baseCol+'66');
      traces.push({
        x: pts.map(p=>p.step), y: pts.map(p=>p.reward),
        type:'scattergl', mode:'lines',
        name: `${c.phase} s${c.seed} (${best.toFixed(0)})${isRunning ? ' ●' : ''}`,
        line:{width: isRunning ? 2.2 : 1.2, color},
        hovertemplate: `<b>${c.phase}</b> s${c.seed}${isRunning?' (running)':''}<br>step %{x:,d} → %{y:.1f}<extra></extra>`,
      });
    });
    Plotly.react('curves', traces, {
      paper_bgcolor:'#161922', plot_bgcolor:'#161922',
      font:{color:'#dbe1eb', size:11},
      xaxis:{title:'env step', gridcolor:'#2a2f3b'},
      yaxis:{title:'reward', gridcolor:'#2a2f3b'},
      shapes:[
        {type:'line', x0:0, x1:1, xref:'paper', y0:500, y1:500, line:{color:'#7dd87b', dash:'dot', width:1}},
        {type:'line', x0:0, x1:1, xref:'paper', y0:600, y1:600, line:{color:'#4ec9b0', dash:'dot', width:1}},
      ],
      margin:{l:50, r:20, t:20, b:40}, legend:{font:{size:10}, x:1.02, y:1},
      showlegend:true,
    }, {displaylogo:false, responsive:true});
  });
}

// Track which (phase, seed) keys currently have an in-flight render job so we
// don't reset their button to "Render rollout" on the next loadCheckpoints().
const ACTIVE_RENDER_KEYS = new Set();
const LAST_RENDER_FAILURES = new Map();
function renderKey(phase, seed){ return `${phase}|${seed}`; }

function loadCheckpoints(){
  fetch('/api/checkpoints').then(r=>r.json()).then(j=>{
    const root = $('#ckpts'); root.innerHTML = '';
    if (!j.checkpoints.length) { root.innerHTML = '<span class="small">no checkpoints found yet</span>'; return; }
    j.checkpoints.sort((a,b)=> (b.best_mppi ?? -1) - (a.best_mppi ?? -1));
    j.checkpoints.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'video-card';
      const v = c.best_mppi;
      const badgeCls = v==null ? 'gray' : (v>=500 ? 'green' : 'gray');
      const badgeText = v==null ? '— MPPI' : `MPPI ${v.toFixed(1)}`;
      const key = renderKey(c.phase, c.seed);
      const busy = ACTIVE_RENDER_KEYS.has(key);
      const failed = LAST_RENDER_FAILURES.get(key);
      const btnLabel = busy ? 'rendering…' : (failed ? 'Render failed · retry' : (c.videos && c.videos.length ? 'Re-render' : 'Render rollout'));
      const failHTML = failed ? `<div class="box-bad small mono" style="margin-top:6px;max-height:58px;overflow:auto">${failed}</div>` : '';
      const videosHTML = (c.videos||[]).map(v => `
        <div style="margin-top:6px">
          <div class="small" style="opacity:.7">${v.source==='archive'?'archived':'rendered'} · ${v.label}</div>
          <video src="${v.url}" controls preload="metadata" style="width:100%;border-radius:4px;background:#000"></video>
        </div>
      `).join('');
      card.innerHTML = `
        <div><b>${c.phase}</b> · seed ${c.seed}
          <span class="pill ${badgeCls}">${badgeText}</span>
        </div>
        <div class="small">last ${c.last_mppi==null?'—':c.last_mppi.toFixed(1)} · ${c.size_mb} MB · ${new Date(c.mtime*1000).toLocaleString()}</div>
        <button data-phase="${c.phase}" data-seed="${c.seed}" ${busy?'disabled':''}>${btnLabel}</button>
        ${failHTML}
        ${videosHTML}
      `;
      root.appendChild(card);
    });
    root.querySelectorAll('button').forEach(btn=>{
      btn.addEventListener('click', () => startRender(btn.dataset.phase, btn.dataset.seed, btn));
    });
  });
}

function loadJobs(){
  fetch('/api/jobs').then(r=>r.json()).then(j=>{
    ACTIVE_RENDER_KEYS.clear();
    j.jobs.forEach(job=>{
      if (job.status==='queued' || job.status==='running')
        ACTIVE_RENDER_KEYS.add(renderKey(job.phase, job.seed));
    });
    const root = $('#jobs');
    // Reuse any existing card; otherwise add a new one. Don't blow away the
    // panel on each tick — that loses scroll position and rebuilds <video>s.
    const have = new Set();
    j.jobs.forEach(job=>{
      have.add(job.job_id);
      if (!document.getElementById('job-'+job.job_id)){
        addJobCard(job.job_id, job.phase, job.seed);
        // start polling for ongoing jobs we just discovered
        if (job.status==='queued' || job.status==='running')
          pollJob(job.job_id, null);
      }
      // For done/failed jobs we just discovered, populate once.
      if (job.status==='done' || job.status==='failed'){
        const stEl = document.getElementById('st-'+job.job_id);
        const pgEl = document.getElementById('pg-'+job.job_id);
        const vidEl = document.getElementById('vid-'+job.job_id);
        const logEl = document.getElementById('log-'+job.job_id);
        const hostEl = document.getElementById('host-'+job.job_id);
        if (stEl){ stEl.textContent = job.status;
                   stEl.className = 'pill ' + (job.status==='done'?'green':'gray'); }
        if (pgEl) pgEl.value = 100;
        if (hostEl && job.render_host) hostEl.textContent = `on ${job.render_host}`;
        if (logEl && job.log_tail) logEl.textContent = job.log_tail.join('\n');
        if (vidEl && job.video && !vidEl.innerHTML)
          vidEl.innerHTML = `<video src="${job.video}" controls preload="metadata" style="width:100%;margin-top:6px;border-radius:4px"></video>`;
        if (job.status==='failed')
          LAST_RENDER_FAILURES.set(renderKey(job.phase, job.seed), `${job.error_type || 'failed'}\n${(job.log_tail||[]).join('\n')}`);
        if (job.status==='done')
          LAST_RENDER_FAILURES.delete(renderKey(job.phase, job.seed));
        if (vidEl && job.status==='failed' && !vidEl.innerHTML)
          vidEl.innerHTML = `<span class="box-bad small">render failed — see log above</span>`;
      }
    });
    // Drop any orphan job cards whose jobs the server forgot.
    document.querySelectorAll('#jobs .video-card').forEach(card=>{
      const id = card.id.replace(/^job-/,'');
      if (!have.has(id)) card.remove();
    });
  });
}

function startRender(phase, seed, btn){
  btn.disabled = true; btn.textContent = 'queuing…';
  const lenSel = document.getElementById('render-length');
  const [nEps, epLen] = (lenSel ? lenSel.value : '1|250').split('|').map(Number);
  fetch('/api/queue/render', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({phase, seed, env_id:'HopperHop', camera:'cam0',
                          n_episodes:nEps, episode_length:epLen})
  }).then(r=>r.json()).then(j=>{
    if (j.error) {
      btn.textContent='Render rollout'; btn.disabled=false;
      alert(j.error); return;
    }
    btn.textContent = 'queued (task ' + j.id + ')';
    loadQueue();
    // Also start polling jobs so progress shows up if render starts immediately
    loadJobs();
  }).catch(()=>{ btn.textContent='Render rollout'; btn.disabled=false; });
}

function addJobCard(jobId, phase, seed){
  const root = $('#jobs');
  const card = document.createElement('div'); card.id = 'job-'+jobId;
  card.className = 'video-card'; card.style.marginBottom='8px';
  card.innerHTML = `
    <div><b>${phase}</b> s${seed} <span class="pill gray" id="st-${jobId}">queued</span> <span class="small" id="host-${jobId}"></span></div>
    <progress id="pg-${jobId}" max="100" value="0"></progress>
    <div class="small mono" id="log-${jobId}" style="margin-top:4px;max-height:60px;overflow:auto"></div>
    <div id="vid-${jobId}"></div>
  `;
  root.appendChild(card);
}

function pollJob(jobId, btn){
  fetch('/api/render/'+jobId).then(r=>r.json()).then(j=>{
    const stEl = document.getElementById('st-'+jobId);
    const pgEl = document.getElementById('pg-'+jobId);
    const logEl = document.getElementById('log-'+jobId);
    const vidEl = document.getElementById('vid-'+jobId);
    const hostEl = document.getElementById('host-'+jobId);
    if (!stEl) return;  // card was removed
    pgEl.value = (j.progress||0)*100;
    stEl.textContent = j.status;
    stEl.className = 'pill ' + (j.status==='done' ? 'green' : 'gray');
    if (hostEl && j.render_host) hostEl.textContent = `on ${j.render_host}`;
    logEl.textContent = (j.log||[]).slice(-8).join('\n');
    const key = renderKey(j.phase, j.seed);
    if (j.status==='done' && j.video){
      if (!vidEl.innerHTML)
        vidEl.innerHTML = `<video src="${j.video}" controls preload="metadata" style="width:100%;margin-top:6px;border-radius:4px"></video>`;
      ACTIVE_RENDER_KEYS.delete(key);
      LAST_RENDER_FAILURES.delete(key);
      if (btn) { btn.disabled=false; btn.textContent='Re-render'; }
      // refresh checkpoint cards so the new video shows there too
      loadCheckpoints();
      return;
    }
    if (j.status==='failed'){
      LAST_RENDER_FAILURES.set(key, `${j.error_type || 'failed'}\n${(j.log||[]).slice(-8).join('\n')}`);
      vidEl.innerHTML = `<span class="box-bad small">render failed — see log above</span>`;
      ACTIVE_RENDER_KEYS.delete(key);
      if (btn) { btn.disabled=false; btn.textContent='Render failed · retry'; }
      loadCheckpoints();
      return;
    }
    setTimeout(()=>pollJob(jobId, btn), 1500);
  });
}

// ── Task queue UI ─────────────────────────────────────────────────────────
const STATUS_STYLE = {
  pending: 'color:#64b5f6',
  running: 'color:#7dd87b;font-weight:600',
  done:    'color:#7e8ba0',
  failed:  'color:#e15c5c',
};

let LAST_QUEUE_TASKS = [];

function fmtDur(s) {
  if (s == null) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function fmtEta(isoStr, short) {
  if (!isoStr) return '';
  const d = new Date(isoStr), now = new Date();
  const diffMs = d - now;
  const timeStr = d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  const isToday = d.toDateString() === now.toDateString();
  const label = isToday ? timeStr : 'tmrw ' + timeStr;
  if (diffMs < 0) return `<span style="opacity:.5">overdue</span>`;
  const h = Math.floor(diffMs / 3600000), m = Math.floor((diffMs % 3600000) / 60000);
  const rel = h > 0 ? `${h}h ${m}m` : `${m}m`;
  return short ? `~${rel}` : `~${rel} · ${label}`;
}

function loadQueue(){
  fetch('/api/queue').then(r=>r.json()).then(j=>{
    LAST_QUEUE_TASKS = j.tasks || [];
    loadRunInspector();
    // Update queue-level ETA in section header
    const hdr = document.getElementById('queue-eta-hdr');
    if (hdr) {
      const running = LAST_QUEUE_TASKS.filter(t=>t.status==='running').length;
      const pending = LAST_QUEUE_TASKS.filter(t=>t.status==='pending').length;
      if (j.queue_eta && (running || pending)) {
        const etaStr = fmtEta(j.queue_eta, false);
        hdr.innerHTML = `all done in ${etaStr}`;
      } else {
        hdr.textContent = '';
      }
    }
    const tbody = document.querySelector('#queue-table tbody');
    const empty = document.getElementById('queue-empty');
    tbody.innerHTML = '';
    if (!j.tasks.length){ empty.style.display=''; return; }
    empty.style.display='none';
    j.tasks.forEach(t=>{
      const style = STATUS_STYLE[t.status] || '';
      const isPending = t.status === 'pending';
      const canRetry = t.status === 'running' || t.status === 'failed' || t.status === 'done';
      const deleteBtn = `<button title="force delete" onclick="deleteTask('${t.id}')" style="padding:2px 6px;color:var(--bad)">✕</button>`;
      let actions = '';
      if (isPending) {
        actions = `
          <button title="increase priority" onclick="movePriority('${t.id}',-1)" style="padding:2px 6px">↑</button>
          <button title="decrease priority" onclick="movePriority('${t.id}',1)"  style="padding:2px 6px">↓</button>
          ${deleteBtn}
        `;
      } else if (canRetry) {
        actions = `<button title="re-queue this task" onclick="retryTask('${t.id}')" style="padding:2px 8px;color:var(--accent)">↺ retry</button> ${deleteBtn}`;
      }
      const envDisplay = t.env && !t.type ? `<div class="small mono" style="opacity:.6;margin-top:1px">${t.env}</div>` : '';
      const typeTag = t.type === 'render' ? '<span class="chip" style="background:#1c4254;color:#4ec9b0;margin-left:4px">render</span>' : '';
      const boxStr = t.box ? `<span class="mono" style="font-size:11px">${t.box}</span>` : '<span class="small">—</span>';
      // ETA column
      let etaCell = '<span class="small" style="opacity:.4">—</span>';
      if (t.status === 'running' && t.eta_iso) {
        const bar = t.estimated_duration_s > 0
          ? Math.min(100, Math.round(t.elapsed_s / t.estimated_duration_s * 100)) : 0;
        etaCell = `<div style="font-size:11px;line-height:1.4">
          <div style="color:#7dd87b">⏱ ${fmtDur(t.elapsed_s)} elapsed</div>
          <div style="color:#64b5f6">→ ${fmtEta(t.eta_iso, false)}</div>
          <div style="background:#1a2234;border-radius:2px;height:3px;margin-top:3px;width:100%">
            <div style="background:#7dd87b;height:3px;border-radius:2px;width:${bar}%"></div></div>
        </div>`;
      } else if (t.status === 'pending' && t.eta_iso) {
        etaCell = `<div style="font-size:11px;line-height:1.4;color:var(--muted)">
          <div>starts ${fmtEta(t.estimated_start_iso, true)}</div>
          <div>done ${fmtEta(t.eta_iso, false)}</div>
        </div>`;
      } else if (t.status === 'done' && t.started_at && t.ended_at) {
        const s = new Date(t.started_at), e = new Date(t.ended_at);
        etaCell = `<span class="small" style="opacity:.5">${fmtDur((e-s)/1000)}</span>`;
      }
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono" style="text-align:center">${t.priority}</td>
        <td><span title="${t.launcher}&#10;${t.env||''}">${t.label}</span>${typeTag}${envDisplay}</td>
        <td><span style="${style}">${t.status}</span></td>
        <td>${boxStr}</td>
        <td>${etaCell}</td>
        <td style="white-space:nowrap">${actions}</td>
      `;
      tbody.appendChild(tr);
    });
  });
}
function toggleAddTask(){
  const f = document.getElementById('add-task-form');
  f.style.display = f.style.display === 'none' ? '' : 'none';
}
function addTask(){
  const label = document.getElementById('at-label').value.trim();
  const launcher = document.getElementById('at-launcher').value.trim();
  const env = document.getElementById('at-env').value.trim();
  const priority = parseInt(document.getElementById('at-priority').value) || 10;
  const errEl = document.getElementById('at-error');
  errEl.style.display = 'none';
  if (!label || !launcher){ errEl.textContent='Label and launcher are required'; errEl.style.display=''; return; }
  fetch('/api/queue', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({label, launcher, env, priority})
  }).then(r=>r.json()).then(j=>{
    if (j.error){ errEl.textContent=j.error; errEl.style.display=''; return; }
    document.getElementById('at-label').value = '';
    document.getElementById('at-env').value = '';
    loadQueue();
  });
}
function deleteTask(id){
  fetch('/api/queue/'+id, {method:'DELETE'}).then(()=>loadQueue());
}
function movePriority(id, delta){
  fetch('/api/queue/'+id+'/priority', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({delta})
  }).then(()=>loadQueue());
}
function retryTask(id){
  fetch('/api/queue/'+id+'/retry', {method:'POST'}).then(()=>loadQueue());
}

// ── Run Inspector ─────────────────────────────────────────────────────────
const RI_OPEN = new Set();       // task ids with expanded card
const RI_LOG_CACHE = {};         // task_id → {lines, loaded}
const LAST_BOX_DATA = {};        // tag → box entry (updated by loadBoxes)

function loadBoxes(){
  fetch('/api/boxes').then(r=>r.json()).then(j=>{
    $('#ts').textContent = new Date(j.ts*1000).toLocaleTimeString();
    ACTIVE_KEYS = new Set((j.active||[]).map(a=>`${a.phase}|${a.seed}`));
    j.boxes.forEach(b => { LAST_BOX_DATA[b.tag] = b; });
    const tbody = $('#boxes tbody'); tbody.innerHTML = '';
    j.boxes.forEach(b=>{
      const tr = document.createElement('tr');
      const ok = b.reachable;
      const gpuUtil = b.gpu_util ?? null;
      const memPct = b.mem_total ? (b.mem_used/b.mem_total*100) : null;
      const hotG = gpuUtil!=null && gpuUtil>=90;
      const hotM = memPct!=null && memPct>=80;
      const queueTask = LAST_QUEUE_TASKS.find(t => t.status === 'running' && t.box === b.tag);
      const queueEtaHtml = queueTask && queueTask.eta_iso
        ? `<div class="small" style="color:#64b5f6;margin-top:2px">⏱ ${fmtDur(queueTask.elapsed_s)} · done ${fmtEta(queueTask.eta_iso, false)} <span style="opacity:.5">${queueTask.label}</span></div>`
        : '';
      const procHTML = (b.procs||[]).map(p=>{
        const tag = (p.tag||'').split('+').filter(Boolean).map(t=>`<span class="chip ${t}">${t}</span>`).join('');
        const phaseStr = p.phase ? `<b>${p.phase}</b>` : '<span class="small">(no csv yet)</span>';
        const dupChip = (p.dup_count && p.dup_count > 1) ? `<span class="chip" style="background:#54391c;color:#e0a44c" title="more than one process found for this seed+phase — likely zombie">${p.dup_count}× DUP</span>` : '';
        return `<div class="mono" style="line-height:1.5">
          ${phaseStr} · s${p.seed} · best ${fmtMppi(p.best_mppi)}${fmtStep(p.best_step)} · last ${fmtMppi(p.last_mppi)}${fmtStep(p.last_step)} ${dupChip}
          <div class="small" style="opacity:.7">PID ${p.pid} · ${p.etime} · ${p.algo} NS=${p.ns} ${tag}</div>
        </div>`;
      }).join('') || '<span class="small">(idle)</span>';
      const spsVals = (b.procs||[]).map(p=>p.sps_avg).filter(v=>v!=null);
      const sps = spsVals.length ? Math.max(...spsVals) : null;
      tr.innerHTML = `
        <td class="mono ${ok?'':'box-bad'}">${b.tag}</td>
        <td>${b.label}${ok?'':'<span class="small box-bad"> · unreachable</span>'}</td>
        <td>${gpuUtil==null?'—':`<span class="util-bar ${hotG?'hot':''}"><span style="width:${gpuUtil}%"></span></span>${gpuUtil}%`}</td>
        <td>${memPct==null?'—':`<span class="util-bar ${hotM?'hot':''}"><span style="width:${memPct}%"></span></span>${b.mem_used}/${b.mem_total} MiB`}</td>
        <td>${b.cpu_util==null?'—':b.cpu_util+'%'}</td>
        <td class="mono">${sps==null?'<span class="small">—</span>':sps+'/s'}</td>
        <td>${procHTML}${queueEtaHtml}</td>
      `;
      tbody.appendChild(tr);
    });
    if ($('#only-running') && $('#only-running').checked) loadCurves();
    loadRunInspector();
  });
}

function riCardHtml(task) {
  const box = LAST_BOX_DATA[task.box] || {};
  const procs = (box.procs || []);
  // Try to match proc by seed.
  const seedM = (task.env||'').match(/SEEDS?=(\S+)/);
  const taskSeed = seedM ? seedM[1] : null;
  const matchedProcs = taskSeed ? procs.filter(p=>String(p.seed)===taskSeed) : [];
  const proc = matchedProcs[0] || procs[0] || null;

  // System section
  const gpuUtil = box.gpu_util != null ? `${box.gpu_util}%` : '—';
  const memStr = box.mem_total ? `${box.mem_used}/${box.mem_total} MiB` : '—';
  const cpuUtil = box.cpu_util != null ? `${box.cpu_util}%` : '—';
  const reachable = box.reachable != null ? (box.reachable ? 'yes' : 'no') : '—';
  const runtime = task.started_at ? fmtDur(task.elapsed_s) : '—';

  // Training section
  const sps = proc ? (proc.sps_avg != null ? proc.sps_avg + '/s' : '—') : '—';
  const lastStep = proc && proc.last_step != null ? (proc.last_step/1e6).toFixed(2)+'M' : '—';
  const bestMppi = proc && proc.best_mppi != null ? proc.best_mppi.toFixed(1) : '—';
  const bestStep = proc && proc.best_step != null ? (proc.best_step/1e6).toFixed(2)+'M' : '—';
  const lastMppi = proc && proc.last_mppi != null ? proc.last_mppi.toFixed(1) : '—';

  // Patience remaining (from live SPS data)
  let patienceHtml = '—';
  if (proc && proc.last_step != null && proc.best_step != null) {
    const stepsSinceBest = Math.max(0, proc.last_step - proc.best_step);
    const patienceLeft = Math.max(0, 3000000 - stepsSinceBest);
    const patiencePct = Math.round((1 - patienceLeft/3000000)*100);
    const patienceM = (patienceLeft/1e6).toFixed(1);
    const col = patienceLeft < 500000 ? 'var(--bad)' : (patienceLeft < 1500000 ? 'var(--warn)' : 'var(--good)');
    patienceHtml = `<span style="color:${col}">${patienceM}M left</span>`;
  }

  // Diag section (from _diag.csv last mppi row)
  const diag = proc ? (proc.diag || null) : null;
  const standPct = diag ? (diag.standing_rate * 100).toFixed(1) + '%' : '—';
  const fallsPerEp = diag ? diag.fall_count.toFixed(1) + '/ep' : '—';
  const ttfSteps = diag ? diag.ttf.toFixed(0) + ' steps' : '—';
  const fullRateStr = diag ? (diag.full_reward_rate * 100).toFixed(1) + '%' : '—';
  // Color-code standing rate: <20% bad, 20-50% warn, >50% good
  const standCol = diag
    ? (diag.standing_rate < 0.20 ? 'var(--bad)' : diag.standing_rate < 0.50 ? 'var(--warn)' : 'var(--good)')
    : '';

  // ETA
  const etaStr = task.eta_iso ? fmtEta(task.eta_iso, false) : '—';

  // Artifacts section
  const envDisplay = (task.env||'').trim() || '(none)';
  const cmd = `${envDisplay} bash ${task.launcher||''}`;
  const logPath = `/tmp/fleet_${task.id}.log (on ${task.box||'?'})`;
  // Guess output dir from seed + output_tag (from proc if available)
  const outputTag = proc ? (proc.output_tag||'') : '';
  const seedStr = taskSeed || (proc ? proc.seed : '?');
  const outputDir = outputTag ? `exp/tdmpc_glass/HopperHop_${outputTag}/seed_${seedStr}/` : '—';
  const ckptPath = outputTag ? `${outputDir}checkpoints/best_mppi.pkl` : '—';

  const isOpen = RI_OPEN.has(task.id);
  return `
  <div class="ri-card" id="ri-${task.id}">
    <div class="ri-header" onclick="toggleRI('${task.id}')">
      <span class="ri-chevron ${isOpen?'open':''}">▶</span>
      <span class="ri-title">${task.label}</span>
      <span class="ri-meta">${task.box||'?'} · elapsed ${runtime} · ETA ${etaStr}</span>
      <span class="pill ${task.status==='running'?'green':'gray'}">${task.status}</span>
    </div>
    <div class="ri-body ${isOpen?'open':''}" id="ri-body-${task.id}">
      <div>
        <div class="ri-section-title">System (${task.box||'?'})</div>
        <div class="ri-row"><span class="ri-key">GPU util</span><span class="ri-val">${gpuUtil}</span></div>
        <div class="ri-row"><span class="ri-key">GPU mem</span><span class="ri-val">${memStr}</span></div>
        <div class="ri-row"><span class="ri-key">CPU util</span><span class="ri-val">${cpuUtil}</span></div>
        <div class="ri-row"><span class="ri-key">Reachable</span><span class="ri-val">${reachable}</span></div>
        <div class="ri-row"><span class="ri-key">Runtime</span><span class="ri-val">${runtime}</span></div>
      </div>
      <div>
        <div class="ri-section-title">Training Progress</div>
        <div class="ri-row"><span class="ri-key">SPS</span><span class="ri-val">${sps}</span></div>
        <div class="ri-row"><span class="ri-key">Last step</span><span class="ri-val">${lastStep}</span></div>
        <div class="ri-row"><span class="ri-key">Last MPPI</span><span class="ri-val">${lastMppi}</span></div>
        <div class="ri-row"><span class="ri-key">Best MPPI</span><span class="ri-val box-good">${bestMppi}</span></div>
        <div class="ri-row"><span class="ri-key">Best @step</span><span class="ri-val">${bestStep}</span></div>
        <div class="ri-row"><span class="ri-key">Patience left</span><span class="ri-val">${patienceHtml}</span></div>
        <div class="ri-row"><span class="ri-key">ETA</span><span class="ri-val">${etaStr}</span></div>
        <div class="ri-section-title" style="margin-top:8px">Behaviour Diag (last eval)</div>
        <div class="ri-row"><span class="ri-key">Standing rate</span><span class="ri-val" style="color:${standCol}">${standPct}</span></div>
        <div class="ri-row"><span class="ri-key">Falls/ep</span><span class="ri-val">${fallsPerEp}</span></div>
        <div class="ri-row"><span class="ri-key">Time-to-hop</span><span class="ri-val">${ttfSteps}</span></div>
        <div class="ri-row"><span class="ri-key">Full-rew rate</span><span class="ri-val">${fullRateStr}</span></div>
      </div>
      <div>
        <div class="ri-section-title">Artifacts</div>
        <div class="ri-row"><span class="ri-key">Command</span></div>
        <div class="ri-path" style="margin-bottom:4px;font-size:10px">${cmd}</div>
        <div class="ri-row"><span class="ri-key">Log path</span></div>
        <div class="ri-path">${logPath}</div>
        <div class="ri-row" style="margin-top:4px"><span class="ri-key">Output dir</span></div>
        <div class="ri-path">${outputDir}</div>
        <div class="ri-row" style="margin-top:4px"><span class="ri-key">Checkpoint</span></div>
        <div class="ri-path">${ckptPath}</div>
        <button class="refresh-btn" onclick="loadRILog('${task.id}')" style="margin-top:8px">📋 tail log</button>
      </div>
      <div class="ri-span-full" id="ri-log-${task.id}" style="display:${RI_LOG_CACHE[task.id]?'':'none'}">
        <div class="ri-section-title">Log tail</div>
        <div class="ri-log" id="ri-loglines-${task.id}">${(RI_LOG_CACHE[task.id]||[]).join('\n')}</div>
      </div>
    </div>
  </div>`;
}

function loadRunInspector() {
  const tasks = LAST_QUEUE_TASKS.filter(t => t.status === 'running');
  const container = document.getElementById('run-inspector-cards');
  const count = document.getElementById('ri-count');
  if (!container) return;
  if (!tasks.length) {
    container.innerHTML = '<div class="small" style="color:var(--muted);padding:4px">No running tasks.</div>';
    if (count) count.textContent = '';
    return;
  }
  if (count) count.textContent = `(${tasks.length} running)`;
  container.innerHTML = tasks.map(riCardHtml).join('');
}

function toggleRI(id) {
  const body = document.getElementById('ri-body-'+id);
  const chevron = document.querySelector(`#ri-${id} .ri-chevron`);
  if (!body) return;
  const isOpen = body.classList.contains('open');
  if (isOpen) { body.classList.remove('open'); chevron.classList.remove('open'); RI_OPEN.delete(id); }
  else { body.classList.add('open'); chevron.classList.add('open'); RI_OPEN.add(id); }
}

function loadRILog(taskId) {
  const logDiv = document.getElementById('ri-log-'+taskId);
  const logLines = document.getElementById('ri-loglines-'+taskId);
  if (!logDiv || !logLines) return;
  logDiv.style.display = '';
  logLines.textContent = 'loading…';
  fetch('/api/queue/'+taskId+'/log').then(r=>r.json()).then(j=>{
    const lines = j.lines || ['(empty)'];
    RI_LOG_CACHE[taskId] = lines;
    logLines.textContent = lines.join('\n');
    logLines.scrollTop = logLines.scrollHeight;
  }).catch(()=>{ logLines.textContent = '(fetch error)'; });
}

// initial + periodic refresh
loadBoxes(); loadCurves(); loadJobs(); loadCheckpoints(); loadQueue();
// Preload phase data for autocomplete + info bar
fetch('/api/phases').then(r=>r.json()).then(j=>{
  PHASE_DATA=j.phases;
  const dl=document.getElementById('phase-list');
  if(dl){ dl.innerHTML=''; PHASE_DATA.forEach(p=>{const o=document.createElement('option');o.value=p.phase;dl.appendChild(o);}); }
});
setInterval(loadBoxes, 30000);
setInterval(loadCurves, 60000);
setInterval(loadJobs, 4000);
setInterval(loadCheckpoints, 90000);
setInterval(loadQueue, 10000);

// Wire up filter inputs
['only-mppi','only-running'].forEach(id=>{
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', loadCurves);
});
document.querySelectorAll('input[name="curve-mode"]').forEach(el=>{
  el.addEventListener('change', loadCurves);
});
const pf = document.getElementById('phase-filter');
if (pf) {
  pf.addEventListener('keydown', e=>{ if (e.key==='Enter') loadCurves(); });
  pf.addEventListener('input', ()=>{ const v=pf.value.trim(); if(PHASE_DATA.length) updatePhaseInfo(v); });
}
</script>
</body></html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5055))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    print(f"[web_dashboard] serving on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
