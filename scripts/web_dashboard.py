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
]

# Remote shell snippet — returns one-line JSON-like ASCII parsable by parse_box_status.
# Kept tiny: SSH transports it inline. Avoids needing python on the remote.
REMOTE_PROBE = (
    'gpu_idx="$1"; '
    'proc=$(ps -eo pid,etime,cmd --no-headers 2>/dev/null '
    '| grep -E "run_benchmark" | grep -v grep '
    '| awk \'{cmd=""; for(i=3;i<=NF;i++) cmd=cmd" "$i; printf "%s|%s|%s\\n", $1, $2, cmd}\'); '
    'gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total '
    '--format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null | head -1); '
    'cpu=$(top -bn1 2>/dev/null | grep -E "^%Cpu" | head -1 '
    '| awk \'{printf "%.0f", 100-$8}\'); '
    'printf "GPU=%s\\nCPU=%s\\n%s\\n" "$gpu" "$cpu" "$proc"'
)


def parse_box_status(raw: str):
    """Parse REMOTE_PROBE output into dict."""
    gpu_util = mem_used = mem_total = None
    cpu_util = None
    procs = []
    for line in raw.splitlines():
        line = line.strip()
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
        elif "|" in line:
            try:
                pid, etime, cmd = line.split("|", 2)
                m_seed = re.search(r"--seed\s+(\S+)", cmd)
                m_ns = re.search(r"--mppi_n_samples\s+(\S+)", cmd)
                m_algo = re.search(r"--algos\s+(\S+)", cmd)
                m_tag = []
                if "--knee_penalty_coef" in cmd:
                    m_tag.append("knee")
                if "--use_cluster_obs" in cmd:
                    m_tag.append("cobs")
                if "--glass_num_super_clusters" in cmd:
                    m_tag.append("hier")
                procs.append({
                    "pid": pid.strip(),
                    "etime": etime.strip(),
                    "seed": m_seed.group(1) if m_seed else "?",
                    "ns": m_ns.group(1) if m_ns else "512",
                    "algo": m_algo.group(1) if m_algo else "?",
                    "tag": "+".join(m_tag) if m_tag else "",
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
        if st.st_size < 100:
            continue
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


# ─── Render jobs ──────────────────────────────────────────────────────────

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def find_best_ckpt(phase: str, seed: str):
    """Locate best_mppi.pkl for a phase+seed under exp/tdmpc_glass/."""
    candidates = list(LOCAL_EXP.rglob(f"HopperHop_{phase}/seed_{seed}/checkpoints/best_mppi.pkl"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


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
    for tag, port, host, gpu_idx, label in BOXES:
        info = results.get(tag, {"reachable": False, "error": "no-result", "procs": []})
        boxes.append({"tag": tag, "label": label, "host": host, "port": port,
                      "gpu_idx": gpu_idx, **info})
    return jsonify({"boxes": boxes, "ts": time.time()})


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
    """List (phase, seed) tuples for which a best_mppi.pkl exists locally."""
    out = []
    for pkl in LOCAL_EXP.rglob("HopperHop_*/seed_*/checkpoints/best_mppi.pkl"):
        phase = pkl.parents[2].name.replace("HopperHop_", "")
        seed = pkl.parents[1].name.replace("seed_", "")
        out.append({"phase": phase, "seed": seed,
                    "ckpt": str(pkl), "mtime": pkl.stat().st_mtime,
                    "size_mb": round(pkl.stat().st_size / 1e6, 1)})
    out.sort(key=lambda r: (r["phase"], int(r["seed"]) if r["seed"].isdigit() else 99999))
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
  section h2{font-size:13px;font-weight:600;margin:0 0 10px 0;color:var(--accent);letter-spacing:.04em;text-transform:uppercase}
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

  <section><h2>Box Fleet</h2>
    <table id="boxes"><thead>
      <tr><th>Tag</th><th>Label</th><th>GPU</th><th>Mem</th><th>CPU</th><th>Running</th></tr>
    </thead><tbody></tbody></table>
  </section>

  <section><h2>Learning Curves <span class="small" id="curves-count"></span></h2>
    <div class="small" style="margin-bottom:8px">
      Filter:
      <label><input type="checkbox" id="only-mppi" checked> only MPPI evals</label>
      &nbsp;&nbsp;
      <label>Phase contains: <input id="phase-filter" type="text" style="background:#222a3b;color:var(--fg);border:1px solid var(--line);border-radius:3px;padding:2px 6px;width:160px"></label>
      <button onclick="loadCurves()">apply</button>
    </div>
    <div id="curves"></div>
  </section>

  <section><h2>Render Rollout <span class="small">pick a checkpoint → click render → MP4 plays inline</span></h2>
    <div id="ckpts" class="video-row"></div>
    <div id="jobs" style="margin-top:14px"></div>
  </section>

</div>

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
const $ = sel => document.querySelector(sel);

function loadBoxes(){
  fetch('/api/boxes').then(r=>r.json()).then(j=>{
    $('#ts').textContent = new Date(j.ts*1000).toLocaleTimeString();
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
        return `<div class="mono">PID ${p.pid} · ${p.etime} · ${p.algo} seed=${p.seed} NS=${p.ns} ${tag}</div>`;
      }).join('') || '<span class="small">(idle)</span>';
      tr.innerHTML = `
        <td class="mono ${ok?'':'box-bad'}">${b.tag}</td>
        <td>${b.label}${ok?'':'<span class="small box-bad"> · unreachable</span>'}</td>
        <td>${gpuUtil==null?'—':`<span class="util-bar ${hotG?'hot':''}"><span style="width:${gpuUtil}%"></span></span>${gpuUtil}%`}</td>
        <td>${memPct==null?'—':`<span class="util-bar ${hotM?'hot':''}"><span style="width:${memPct}%"></span></span>${b.mem_used}/${b.mem_total} MiB`}</td>
        <td>${b.cpu_util==null?'—':b.cpu_util+'%'}</td>
        <td>${procHTML}</td>
      `;
      tbody.appendChild(tr);
    });
  });
}

function loadCurves(){
  const phaseFilter = $('#phase-filter').value.trim();
  const url = '/api/curves' + (phaseFilter ? '?phase='+encodeURIComponent(phaseFilter) : '');
  fetch(url).then(r=>r.json()).then(j=>{
    $('#curves-count').textContent = `(${j.curves.length} traces)`;
    const onlyMppi = $('#only-mppi').checked;
    const traces = [];
    j.curves.forEach(c=>{
      let pts = c.points;
      if (onlyMppi) pts = pts.filter(p=>p.eval_type==='mppi');
      if (!pts.length) return;
      const best = Math.max(...pts.map(p=>p.reward));
      const color = best>=500 ? '#7dd87b' : (best>=300 ? '#e0a44c' : '#7e8ba0');
      traces.push({
        x: pts.map(p=>p.step), y: pts.map(p=>p.reward),
        type:'scattergl', mode:'lines',
        name: `${c.phase} s${c.seed} (${best.toFixed(0)})`,
        line:{width:1.2, color},
        hovertemplate: `${c.phase} s${c.seed}<br>step %{x:,d} → %{y:.1f}<extra></extra>`,
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

function loadCheckpoints(){
  fetch('/api/checkpoints').then(r=>r.json()).then(j=>{
    const root = $('#ckpts'); root.innerHTML = '';
    if (!j.checkpoints.length) { root.innerHTML = '<span class="small">no checkpoints found yet</span>'; return; }
    j.checkpoints.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'video-card';
      card.innerHTML = `
        <div><b>${c.phase}</b> · seed ${c.seed} <span class="small">(${c.size_mb} MB)</span></div>
        <div class="small">${new Date(c.mtime*1000).toLocaleString()}</div>
        <button data-phase="${c.phase}" data-seed="${c.seed}">Render rollout</button>
      `;
      root.appendChild(card);
    });
    root.querySelectorAll('button').forEach(btn=>{
      btn.addEventListener('click', () => startRender(btn.dataset.phase, btn.dataset.seed, btn));
    });
  });
}

function startRender(phase, seed, btn){
  btn.disabled = true; btn.textContent = 'queued…';
  fetch('/api/render', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({phase, seed, env_id:'HopperHop', camera:'cam0', n_episodes:2, episode_length:200})
  }).then(r=>r.json()).then(j=>{
    if (j.error) { btn.textContent='Render rollout'; btn.disabled=false; alert(j.error); return; }
    addJobCard(j.job_id, phase, seed);
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
    const stEl = $('#st-'+jobId);
    const pgEl = $('#pg-'+jobId);
    const logEl = $('#log-'+jobId);
    const vidEl = $('#vid-'+jobId);
    pgEl.value = (j.progress||0)*100;
    stEl.textContent = j.status;
    stEl.className = 'pill ' + (j.status==='done' ? 'green' : 'gray');
    logEl.textContent = (j.log||[]).slice(-3).join('\n');
    if (j.status==='done' && j.video){
      vidEl.innerHTML = `<video src="${j.video}" controls style="width:100%;margin-top:6px;border-radius:4px"></video>`;
      if (btn) { btn.disabled=false; btn.textContent='Render rollout'; }
      return;
    }
    if (j.status==='failed'){
      vidEl.innerHTML = `<span class="box-bad small">render failed — see log</span>`;
      if (btn) { btn.disabled=false; btn.textContent='Render rollout'; }
      return;
    }
    setTimeout(()=>pollJob(jobId, btn), 1500);
  });
}

// initial + periodic refresh
loadBoxes(); loadCurves(); loadCheckpoints();
setInterval(loadBoxes, 30000);
setInterval(loadCurves, 60000);
</script>
</body></html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5055))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    print(f"[web_dashboard] serving on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
