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
import json
import os
import re
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
                ["ssh", "-p", str(port),
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
    # Also surface rollout_videos/ produced by this dashboard, keyed via JOBS.
    # Those are already accessible via /videos/<id>.mp4 from render jobs.
    for lst in by_key.values():
        lst.sort(key=lambda v: v["mtime"], reverse=True)
    return by_key


def render_worker(job_id: str, ckpt: str, env_id: str, camera: str,
                   n_episodes: int, episode_length: int):
    """Spawn render_glass_rollout.py, stream stdout → progress."""
    out_mp4 = VIDEO_OUT / f"{job_id}.mp4"
    cmd = [
        "/root/venv/bin/python3", "-u", "scripts/render_glass_rollout.py",
        "--ckpt", ckpt, "--env_id", env_id,
        "--out", str(out_mp4), "--camera", camera,
        "--n_episodes", str(n_episodes),
        "--episode_length", str(episode_length),
    ]
    env = {**os.environ, "MUJOCO_GL": "egl",
           "PYTHONPATH": "/root/helios-rl/src:/root/mujoco_playground_repo"}
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["cmd"] = " ".join(cmd)
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), env=env,
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
                elif "wrote " in line and "frames" in line:
                    JOBS[job_id]["progress"] = 0.99
        proc.wait()
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = 1.0
            ok = (proc.returncode == 0 and out_mp4.exists())
            JOBS[job_id]["status"] = "done" if ok else "failed"
            JOBS[job_id]["video"] = f"/videos/{job_id}.mp4" if out_mp4.exists() else None
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["log"].append(f"EXCEPTION: {e}")


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
            else:
                p["phase"] = None
                p["best_mppi"] = p["last_mppi"] = None
                p["best_step"] = p["last_step"] = None
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
    episode_length = int(data.get("episode_length", 200))
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
                "started_at": j.get("started_at"),
                "log_tail": j.get("log", [])[-3:],
            })
    items.sort(key=lambda r: -(r.get("started_at") or 0))
    return jsonify({"jobs": items})


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

  <section><h2>Learning Curves <span class="small" id="curves-count"></span>
    <button class="refresh-btn" onclick="loadCurves()">&#x21bb; refresh</button></h2>
    <div class="small" style="margin-bottom:8px">
      Filter:
      <label><input type="checkbox" id="only-mppi" checked> only MPPI evals</label>
      &nbsp;&nbsp;
      <label><input type="checkbox" id="only-running"> only currently-running seeds</label>
      &nbsp;&nbsp;
      <label>Phase contains: <input id="phase-filter" type="text" style="background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:2px 6px;width:160px"></label>
      <button onclick="loadCurves()">apply</button>
    </div>
    <div id="curves"></div>
  </section>

  <section><h2>Render Rollout
    <button class="refresh-btn" onclick="loadCheckpoints()">&#x21bb; refresh</button></h2>
    <div class="small" style="margin-bottom:8px">
      Length:
      <select id="render-length">
        <option value="2|200">short (2 × 200 steps)</option>
        <option value="1|500">medium (1 × 500 steps)</option>
        <option value="1|1000" selected>long (1 × 1000 steps)</option>
        <option value="3|1000">extra long (3 × 1000 steps)</option>
      </select>
      <span class="small">camera: cam0 · 320×240</span>
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

function loadBoxes(){
  fetch('/api/boxes').then(r=>r.json()).then(j=>{
    $('#ts').textContent = new Date(j.ts*1000).toLocaleTimeString();
    ACTIVE_KEYS = new Set((j.active||[]).map(a=>`${a.phase}|${a.seed}`));
    const tbody = $('#boxes tbody'); tbody.innerHTML = '';
    j.boxes.forEach(b=>{
      const tr = document.createElement('tr');
      const ok = b.reachable;
      const gpuUtil = b.gpu_util ?? null;
      const memPct = b.mem_total ? (b.mem_used/b.mem_total*100) : null;
      const hotG = gpuUtil!=null && gpuUtil>=90;
      const hotM = memPct!=null && memPct>=80;
      const procHTML = (b.procs||[]).map(p=>{
        const tag = (p.tag||'').split('+').filter(Boolean).map(t=>`<span class="chip ${t}">${t}</span>`).join('');
        const phaseStr = p.phase ? `<b>${p.phase}</b>` : '<span class="small">(no csv yet)</span>';
        const dupChip = (p.dup_count && p.dup_count > 1) ? `<span class="chip" style="background:#54391c;color:#e0a44c" title="more than one process found for this seed+phase — likely zombie">${p.dup_count}× DUP</span>` : '';
        return `<div class="mono" style="line-height:1.5">
          ${phaseStr} · s${p.seed} · best ${fmtMppi(p.best_mppi)}${fmtStep(p.best_step)} · last ${fmtMppi(p.last_mppi)}${fmtStep(p.last_step)} ${dupChip}
          <div class="small" style="opacity:.7">PID ${p.pid} · ${p.etime} · ${p.algo} NS=${p.ns} ${tag}</div>
        </div>`;
      }).join('') || '<span class="small">(idle)</span>';
      // SPS: take max across procs (mostly there's one), or '—' if none reported
      const spsVals = (b.procs||[]).map(p=>p.sps_avg).filter(v=>v!=null);
      const sps = spsVals.length ? Math.max(...spsVals) : null;
      tr.innerHTML = `
        <td class="mono ${ok?'':'box-bad'}">${b.tag}</td>
        <td>${b.label}${ok?'':'<span class="small box-bad"> · unreachable</span>'}</td>
        <td>${gpuUtil==null?'—':`<span class="util-bar ${hotG?'hot':''}"><span style="width:${gpuUtil}%"></span></span>${gpuUtil}%`}</td>
        <td>${memPct==null?'—':`<span class="util-bar ${hotM?'hot':''}"><span style="width:${memPct}%"></span></span>${b.mem_used}/${b.mem_total} MiB`}</td>
        <td>${b.cpu_util==null?'—':b.cpu_util+'%'}</td>
        <td class="mono">${sps==null?'<span class="small">—</span>':sps+'/s'}</td>
        <td>${procHTML}</td>
      `;
      tbody.appendChild(tr);
    });
    // re-render curves if the running-only filter is active
    if ($('#only-running') && $('#only-running').checked) loadCurves();
  });
}

function loadCurves(){
  const phaseFilter = $('#phase-filter').value.trim();
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
      const color = best>=500 ? '#7dd87b' : (best>=300 ? '#e0a44c' : '#7e8ba0');
      traces.push({
        x: pts.map(p=>p.step), y: pts.map(p=>p.reward),
        type:'scattergl', mode:'lines',
        name: `${c.phase} s${c.seed} (${best.toFixed(0)})${isRunning ? ' ●' : ''}`,
        line:{width: isRunning ? 2 : 1.2, color, dash: isRunning ? 'solid' : 'solid'},
        hovertemplate: `${c.phase} s${c.seed}${isRunning?' (running)':''}<br>step %{x:,d} → %{y:.1f}<extra></extra>`,
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
      const btnLabel = busy ? 'rendering…' : (c.videos && c.videos.length ? 'Re-render' : 'Render rollout');
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
        if (stEl){ stEl.textContent = job.status;
                   stEl.className = 'pill ' + (job.status==='done'?'green':'gray'); }
        if (pgEl) pgEl.value = 100;
        if (vidEl && job.video && !vidEl.innerHTML)
          vidEl.innerHTML = `<video src="${job.video}" controls preload="metadata" style="width:100%;margin-top:6px;border-radius:4px"></video>`;
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
  btn.disabled = true; btn.textContent = 'queued…';
  ACTIVE_RENDER_KEYS.add(renderKey(phase, seed));
  const lenSel = document.getElementById('render-length');
  const [nEps, epLen] = (lenSel ? lenSel.value : '1|1000').split('|').map(Number);
  fetch('/api/render', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({phase, seed, env_id:'HopperHop', camera:'cam0',
                          n_episodes:nEps, episode_length:epLen})
  }).then(r=>r.json()).then(j=>{
    if (j.error) {
      ACTIVE_RENDER_KEYS.delete(renderKey(phase, seed));
      btn.textContent='Render rollout'; btn.disabled=false;
      alert(j.error); return;
    }
    if (!document.getElementById('job-'+j.job_id)) addJobCard(j.job_id, phase, seed);
    pollJob(j.job_id, btn);
  });
}

function addJobCard(jobId, phase, seed){
  const root = $('#jobs');
  const card = document.createElement('div'); card.id = 'job-'+jobId;
  card.className = 'video-card'; card.style.marginBottom='8px';
  card.innerHTML = `
    <div><b>${phase}</b> s${seed} <span class="pill gray" id="st-${jobId}">queued</span></div>
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
    if (!stEl) return;  // card was removed
    pgEl.value = (j.progress||0)*100;
    stEl.textContent = j.status;
    stEl.className = 'pill ' + (j.status==='done' ? 'green' : 'gray');
    logEl.textContent = (j.log||[]).slice(-3).join('\n');
    const key = renderKey(j.phase, j.seed);
    if (j.status==='done' && j.video){
      if (!vidEl.innerHTML)
        vidEl.innerHTML = `<video src="${j.video}" controls preload="metadata" style="width:100%;margin-top:6px;border-radius:4px"></video>`;
      ACTIVE_RENDER_KEYS.delete(key);
      if (btn) { btn.disabled=false; btn.textContent='Re-render'; }
      // refresh checkpoint cards so the new video shows there too
      loadCheckpoints();
      return;
    }
    if (j.status==='failed'){
      vidEl.innerHTML = `<span class="box-bad small">render failed — see log</span>`;
      ACTIVE_RENDER_KEYS.delete(key);
      if (btn) { btn.disabled=false; btn.textContent='Render rollout'; }
      return;
    }
    setTimeout(()=>pollJob(jobId, btn), 1500);
  });
}

// initial + periodic refresh
loadBoxes(); loadCurves(); loadJobs(); loadCheckpoints();
setInterval(loadBoxes, 30000);
setInterval(loadCurves, 60000);
setInterval(loadJobs, 4000);          // job state — fast enough that the active set
                                       // stays accurate during a 30-90 s render
setInterval(loadCheckpoints, 90000);

// Wire up checkbox/text filter inputs so they re-render immediately.
['only-mppi','only-running'].forEach(id=>{
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', loadCurves);
});
const pf = document.getElementById('phase-filter');
if (pf) pf.addEventListener('keydown', e=>{ if (e.key==='Enter') loadCurves(); });
</script>
</body></html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5055))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    print(f"[web_dashboard] serving on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
