#!/usr/bin/env python3
"""gpu-fleet web dashboard.

Generic Flask dashboard for monitoring a multi-box GPU fleet and managing a
task queue. Reads config from config.yaml (or FLEET_CONFIG env var).

Sections
--------
- Box Fleet   : live GPU/CPU stats for every box, SSH-probed every 30 s
- Task Queue  : queue CRUD with priority, retry, force-delete; ETA estimates
- Metrics     : auto-discovers CSVs under exp_dir, plots step vs any column

Usage:
    FLEET_CONFIG=/path/to/config.yaml \\
    nohup python3 tools/gpu-fleet/fleet_dashboard.py \\
        >> tools/gpu-fleet/logs/dashboard.log 2>&1 &
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")
try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required: pip install pandas")
from flask import Flask, jsonify, request

# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
CONFIG_PATH = Path(os.environ.get("FLEET_CONFIG", _DEFAULT_CONFIG))

def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = _load_cfg()
REPO        = Path(CFG["fleet"]["repo"])
SSH_KEY     = os.environ.get("SSH_IDENTITY_FILE",
                  CFG.get("ssh", {}).get("key", "/home/coder/.ssh/id_ed25519"))
SSH_USER    = CFG.get("ssh", {}).get("user", "root")
SSH_TO      = int(CFG.get("ssh", {}).get("connect_timeout", 10))
QUEUE_FILE  = REPO / CFG.get("queue", {}).get("file", "tools/gpu-fleet/queue.json")
PORT        = int(CFG.get("dashboard", {}).get("port", 5055))
EXP_DIR     = REPO / CFG.get("dashboard", {}).get("exp_dir", "exp")
STEP_COL    = CFG.get("dashboard", {}).get("step_col", "step")
LOG_DIR     = REPO / CFG.get("dashboard", {}).get("log_dir", "tools/gpu-fleet/logs")
PROC_PAT    = CFG.get("proc_pattern", "run_benchmark")
XLA_DEFAULT = CFG.get("xla_mem_default", "0.65")
FLEET_NAME  = CFG.get("fleet", {}).get("name", "GPU Fleet")

LOG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

BOXES = CFG.get("boxes", [])  # raw dicts with tag/host/port/gpu_idx/label/...

DEFAULT_TASK_DURATION_S = 14400  # 4-hour ETA fallback

# ── Remote probe script ───────────────────────────────────────────────────────
# Passed verbatim to bash on the remote (or locally). Outputs:
#   GPU=<util>,<mem_used>,<mem_total>
#   CPU=<util>
#   PROC|<pid>|<etime>|<cmd>
#   ...

REMOTE_PROBE = r'''
gpu_idx="$1"
proc_pat="${2:-python}"
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null | head -1)
cpu=$(awk '/^cpu /{u=$2+$4;t=$2+$3+$4+$5+$6+$7+$8;printf "%.0f",u*100/t;exit}' \
      /proc/stat 2>/dev/null || echo 0)
printf "GPU=%s\nCPU=%s\n" "$gpu" "$cpu"
ps -eo pid,etime,cmd --no-headers 2>/dev/null | grep "$proc_pat" | grep -v grep \
  | awk '{e=$2; pid=$1; cmd=""; for(i=3;i<=NF;i++) cmd=cmd" "$i;
          printf "PROC|%s|%s|%s\n", pid, e, cmd}'
'''


def _parse_probe(raw: str) -> dict:
    gpu_util = mem_used = mem_total = cpu_util = None
    procs = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("GPU="):
            parts = [x.strip() for x in line[4:].split(",")]
            try:
                gpu_util  = int(parts[0].split()[0])
                mem_used  = int(parts[1].split()[0])
                mem_total = int(parts[2].split()[0])
            except Exception:
                pass
        elif line.startswith("CPU="):
            try:
                cpu_util = int(float(line[4:].strip()))
            except Exception:
                pass
        elif line.startswith("PROC|"):
            fields = line.split("|", 3)
            if len(fields) == 4:
                procs.append({"pid": fields[1], "etime": fields[2], "cmd": fields[3]})
    return {"gpu_util": gpu_util, "mem_used": mem_used, "mem_total": mem_total,
            "cpu_util": cpu_util, "procs": procs}


def _probe_box(box: dict) -> dict:
    tag, host, port = box["tag"], box.get("host"), box.get("port")
    gpu_idx = box.get("gpu_idx", 0)
    try:
        if tag == "local" or not host:
            r = subprocess.run(
                ["bash", "-c", REMOTE_PROBE, "_", str(gpu_idx), PROC_PAT],
                capture_output=True, text=True, timeout=12,
            )
        else:
            r = subprocess.run(
                ["ssh", "-p", str(port), "-i", SSH_KEY,
                 "-o", "StrictHostKeyChecking=no",
                 "-o", f"ConnectTimeout={SSH_TO}", "-o", "BatchMode=yes",
                 f"{SSH_USER}@{host}", "bash", "-s", "--", str(gpu_idx), PROC_PAT],
                input=REMOTE_PROBE, capture_output=True, text=True, timeout=20,
            )
        status = _parse_probe(r.stdout)
        status["reachable"] = (r.returncode == 0)
        return status
    except subprocess.TimeoutExpired:
        return {"reachable": False, "error": "timeout", "procs": []}
    except Exception as e:
        return {"reachable": False, "error": str(e), "procs": []}


# ── Queue helpers ─────────────────────────────────────────────────────────────

def _load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def _save_queue(tasks: list[dict]):
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tasks, indent=2))
    tmp.replace(QUEUE_FILE)

def _with_lock(fn):
    lock = QUEUE_FILE.with_suffix(".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tasks = _load_queue()
            result = fn(tasks)
            if result is not None:
                _save_queue(result)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _compute_etas(tasks: list[dict]) -> tuple[list[dict], str | None]:
    """Add elapsed/remaining/eta fields. Returns (annotated_tasks, queue_eta_iso)."""
    import heapq
    now = datetime.now(timezone.utc)

    def _p(s):
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _fmt(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Per-launcher average from done tasks.
    durs: dict[str, list[float]] = {}
    for t in tasks:
        if t["status"] == "done" and t.get("started_at") and t.get("ended_at"):
            s, e = _p(t["started_at"]), _p(t["ended_at"])
            if s and e and e > s:
                durs.setdefault(t.get("launcher", ""), []).append((e - s).total_seconds())
    avg: dict[str, float] = {l: sum(v)/len(v) for l, v in durs.items()}

    def est(t):
        return avg.get(t.get("launcher", ""), DEFAULT_TASK_DURATION_S)

    all_tags = [b["tag"] for b in BOXES]
    box_free: dict[str, datetime] = {tag: now for tag in all_tags}
    for t in tasks:
        if t["status"] == "running" and t.get("box") and t.get("started_at"):
            s = _p(t["started_at"])
            if s:
                box_free[t["box"]] = max(s + timedelta(seconds=est(t)), now)

    heap = [(ts, tag) for tag, ts in box_free.items()]
    heapq.heapify(heap)
    pending = sorted(
        [t for t in tasks if t["status"] == "pending"],
        key=lambda t: (t.get("priority", 10), t.get("created_at", ""))
    )
    sched: dict[str, tuple[datetime, datetime]] = {}
    for t in pending:
        if not heap:
            break
        free_ts, tag = heapq.heappop(heap)
        start = max(free_ts, now)
        finish = start + timedelta(seconds=est(t))
        sched[t["id"]] = (start, finish)
        heapq.heappush(heap, (finish, tag))

    result = []
    for t in tasks:
        t = dict(t)
        dur = est(t)
        t["estimated_duration_s"] = int(dur)
        if t["status"] == "running" and t.get("started_at"):
            s = _p(t["started_at"])
            if s:
                eta = s + timedelta(seconds=dur)
                t["elapsed_s"]  = int((now - s).total_seconds())
                t["remaining_s"]= int(max(0, (eta - now).total_seconds()))
                t["eta_iso"]    = _fmt(eta)
        elif t["status"] == "pending" and t["id"] in sched:
            start, finish = sched[t["id"]]
            t["estimated_start_iso"] = _fmt(start)
            t["eta_iso"]             = _fmt(finish)
        result.append(t)

    etas = [t["eta_iso"] for t in result
            if t["status"] in ("running", "pending") and t.get("eta_iso")]
    return result, (max(etas) if etas else None)


# ── CSV discovery ─────────────────────────────────────────────────────────────

def _discover_csvs() -> dict[str, list[dict]]:
    """Return {tag: [{name, path, cols}]} for all CSVs with step_col."""
    grouped: dict[str, list] = {}
    if not EXP_DIR.exists():
        return grouped
    for csv_path in sorted(EXP_DIR.rglob("*.csv")):
        try:
            header = pd.read_csv(csv_path, nrows=0)
            cols = list(header.columns)
            if STEP_COL not in cols:
                continue
            rel   = csv_path.relative_to(EXP_DIR)
            tag   = rel.parts[0] if len(rel.parts) > 1 else "_root"
            entry = {"name": csv_path.stem, "path": str(csv_path), "cols": cols}
            grouped.setdefault(tag, []).append(entry)
        except Exception:
            pass
    return grouped


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Box fleet ─────────────────────────────────────────────────────────────────

@app.route("/api/boxes")
def api_boxes():
    threads, results = [], {}
    def _worker(box):
        results[box["tag"]] = _probe_box(box)
    for box in BOXES:
        t = threading.Thread(target=_worker, args=(box,))
        t.start(); threads.append(t)
    for t in threads:
        t.join(timeout=25)
    out = []
    for box in BOXES:
        info = results.get(box["tag"], {"reachable": False, "procs": []})
        out.append({**box, **info})
    return jsonify({"boxes": out, "ts": time.time()})


# ── Task queue ────────────────────────────────────────────────────────────────

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@app.route("/api/queue")
def api_queue_get():
    tasks = _load_queue()
    tasks = sorted(tasks, key=lambda t: (t.get("priority", 10), t.get("created_at", "")))
    annotated, queue_eta = _compute_etas(tasks)
    return jsonify({"tasks": annotated, "queue_eta": queue_eta})

@app.route("/api/queue", methods=["POST"])
def api_queue_add():
    body = request.get_json(force=True)
    label    = (body.get("label") or "").strip()
    launcher = (body.get("launcher") or "").strip()
    env      = (body.get("env") or "").strip()
    priority = int(body.get("priority", 10))
    if not label or not launcher:
        return jsonify({"error": "label and launcher are required"}), 400
    task = {
        "id": "t" + uuid.uuid4().hex[:7],
        "label": label, "launcher": launcher, "env": env,
        "priority": priority, "status": "pending",
        "box": None, "created_at": _now_iso(),
        "started_at": None, "ended_at": None,
    }
    _with_lock(lambda tasks: tasks + [task])
    return jsonify({"ok": True, "id": task["id"]})

@app.route("/api/queue/<task_id>", methods=["DELETE"])
def api_queue_delete(task_id):
    removed = [False]
    def _del(tasks):
        new = [t for t in tasks if t["id"] != task_id]
        removed[0] = len(new) < len(tasks)
        return new
    _with_lock(_del)
    return jsonify({"ok": True}) if removed[0] else (jsonify({"error": "not found"}), 404)

@app.route("/api/queue/<task_id>/priority", methods=["POST"])
def api_queue_priority(task_id):
    delta = int((request.get_json(force=True) or {}).get("delta", -1))
    def _bump(tasks):
        for t in tasks:
            if t["id"] == task_id and t["status"] == "pending":
                t["priority"] = max(1, t["priority"] + delta)
        return tasks
    _with_lock(_bump)
    return jsonify({"ok": True})

@app.route("/api/queue/<task_id>/retry", methods=["POST"])
def api_queue_retry(task_id):
    def _retry(tasks):
        for t in tasks:
            if t["id"] == task_id and t["status"] in ("running", "failed", "done"):
                t.update({"status": "pending", "box": None,
                           "started_at": None, "ended_at": None})
        return tasks
    _with_lock(_retry)
    return jsonify({"ok": True})


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    grouped = _discover_csvs()
    tags = [{"tag": tag, "n": len(csvs)} for tag, csvs in sorted(grouped.items())]
    all_cols: set[str] = set()
    for csvs in grouped.values():
        for c in csvs:
            all_cols.update(c["cols"])
    all_cols.discard(STEP_COL)
    return jsonify({"tags": tags, "cols": sorted(all_cols), "step_col": STEP_COL})

@app.route("/api/metrics/series")
def api_metrics_series():
    tag    = request.args.get("tag", "")
    metric = request.args.get("metric", "")
    grouped = _discover_csvs()
    csvs = grouped.get(tag, [])
    series = []
    for c in csvs:
        if metric not in c["cols"]:
            continue
        try:
            df = pd.read_csv(c["path"], usecols=[STEP_COL, metric])
            df = df.dropna(subset=[STEP_COL, metric]).sort_values(STEP_COL)
            series.append({
                "name": c["name"],
                "xs":   df[STEP_COL].tolist(),
                "ys":   df[metric].tolist(),
            })
        except Exception:
            pass
    return jsonify({"series": series, "step_col": STEP_COL, "metric": metric})


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font:14px/1.5 'Segoe UI',system-ui,sans-serif;padding:16px}}
  h1{{font-size:18px;margin-bottom:12px;color:#e6edf3}}
  h2{{font-size:14px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}}
  section{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px 16px;margin-bottom:14px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:4px 8px;color:#8b949e;font-weight:500;border-bottom:1px solid #30363d}}
  td{{padding:5px 8px;border-bottom:1px solid #21262d;vertical-align:top}}
  tr:hover td{{background:#1c2128}}
  .mono{{font-family:'Cascadia Code','JetBrains Mono','Consolas',monospace;font-size:12px}}
  .small{{font-size:11px;opacity:.7}}
  .bad{{color:#f85149}} .ok{{color:#3fb950}} .muted{{color:#484f58}}
  .util-bar{{display:inline-block;width:60px;height:6px;background:#21262d;
              border-radius:3px;vertical-align:middle;margin-right:4px;overflow:hidden}}
  .util-bar span{{display:block;height:100%;background:#3fb950;border-radius:3px}}
  .util-bar.hot span{{background:#f85149}}
  button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;
           padding:3px 10px;cursor:pointer;font-size:12px}}
  button:hover{{background:#30363d}}
  .chip{{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;
          background:#1c2950;color:#58a6ff;margin-left:3px}}
  select,input[type=text]{{background:#161b22;color:#c9d1d9;border:1px solid #30363d;
    border-radius:4px;padding:3px 7px;font-size:13px}}
  #metrics-plot{{width:100%;min-height:340px}}
  .progress-bar{{background:#1c2128;border-radius:2px;height:3px;margin-top:3px}}
  .progress-bar-fill{{background:#3fb950;height:3px;border-radius:2px}}
</style>
</head>
<body>
<h1>{name} &nbsp;<span class="small mono" id="ts"></span></h1>

<section>
  <h2>Box Fleet <button onclick="loadBoxes()">↻ refresh</button></h2>
  <table id="box-table">
    <thead><tr>
      <th>Tag</th><th>Label</th><th>GPU util</th><th>Mem</th><th>CPU</th>
      <th>Procs</th><th>Queue task / ETA</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</section>

<section>
  <h2>Task Queue
    <button onclick="loadQueue()">↻ refresh</button>
    <button onclick="toggleAdd()" style="margin-left:4px">+ add</button>
    <span id="queue-eta-hdr" class="small" style="margin-left:10px;opacity:.65"></span>
  </h2>
  <div id="add-form" style="display:none;background:#0d1117;border:1px solid #30363d;
    border-radius:5px;padding:10px;margin-bottom:10px">
    <div style="display:grid;grid-template-columns:60px 1fr 2fr 1fr 80px;gap:8px;align-items:end">
      <label class="small">Priority<br>
        <input id="at-pri" type="number" value="10" min="1" max="99" style="width:100%"></label>
      <label class="small">Label<br>
        <input id="at-label" type="text" placeholder="experiment name" style="width:100%"></label>
      <label class="small">Launcher script<br>
        <input id="at-launcher" type="text" placeholder="scripts/run_foo.sh" style="width:100%"></label>
      <label class="small">Env vars (optional)<br>
        <input id="at-env" type="text" placeholder="SEED=1 LR=3e-4" style="width:100%"></label>
      <button onclick="addTask()" style="padding:6px 0">Add</button>
    </div>
    <div id="at-err" class="small bad" style="display:none;margin-top:6px"></div>
  </div>
  <table id="queue-table">
    <thead><tr>
      <th style="width:50px">Pri</th><th>Label</th>
      <th style="width:80px">Status</th><th style="width:90px">Box</th>
      <th style="width:180px">ETA / Progress</th><th style="width:90px">Actions</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <div id="queue-empty" class="small muted" style="display:none;padding:6px">Queue is empty.</div>
</section>

<section>
  <h2>Metrics</h2>
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap">
    <label class="small">Experiment tag:
      <select id="metric-tag" onchange="loadMetricCols()"><option value="">— pick —</option></select>
    </label>
    <label class="small">Metric column:
      <select id="metric-col" onchange="loadMetricSeries()"><option value="">— pick —</option></select>
    </label>
    <span id="metric-info" class="small muted"></span>
  </div>
  <div id="metrics-plot"></div>
</section>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// ── Box Fleet ─────────────────────────────────────────────────────────────
let QUEUE_TASKS = [];

const STATUS_COLOR = {{
  pending:'#58a6ff', running:'#3fb950', done:'#484f58', failed:'#f85149'
}};

function fmtDur(s){{
  if(s==null) return '—';
  const h=Math.floor(s/3600),m=Math.floor((s%3600)/60);
  return h>0?`${{h}}h ${{m}}m`:`${{m}}m`;
}}
function fmtEta(iso, short){{
  if(!iso) return '';
  const d=new Date(iso), now=new Date(), diff=d-now;
  const t=d.toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}});
  const label=d.toDateString()===now.toDateString()?t:'tmrw '+t;
  if(diff<0) return '<span class="muted">overdue</span>';
  const h=Math.floor(diff/3600000),m=Math.floor((diff%3600000)/60000);
  const rel=h>0?`${{h}}h ${{m}}m`:`${{m}}m`;
  return short?`~${{rel}}`:`~${{rel}} · ${{label}}`;
}}

function loadBoxes(){{
  fetch('/api/boxes').then(r=>r.json()).then(j=>{{
    $('#ts').textContent=new Date(j.ts*1000).toLocaleTimeString();
    const tbody=$('#box-table tbody'); tbody.innerHTML='';
    j.boxes.forEach(b=>{{
      const ok=b.reachable;
      const gPct=b.mem_total?(b.mem_used/b.mem_total*100):null;
      const hotG=b.gpu_util!=null&&b.gpu_util>=90;
      const hotM=gPct!=null&&gPct>=80;
      const qTask=QUEUE_TASKS.find(t=>t.status==='running'&&t.box===b.tag);
      const etaHtml=qTask&&qTask.eta_iso
        ?`<div class="small" style="color:#58a6ff;margin-top:2px">
            ⏱ ${{fmtDur(qTask.elapsed_s)}} elapsed · done ${{fmtEta(qTask.eta_iso,false)}}
            <br><span class="muted">${{qTask.label}}</span></div>`:'';
      const procHtml=(b.procs||[]).map(p=>
        `<div class="mono small">PID ${{p.pid}} · ${{p.etime}}</div>`
      ).join('')||'<span class="small muted">(idle)</span>';
      const tr=document.createElement('tr');
      tr.innerHTML=`
        <td class="mono ${{ok?'':'bad'}}">${{b.tag}}</td>
        <td>${{b.label||''}}${{ok?'':'<span class="small bad"> · unreachable</span>'}}</td>
        <td>${{b.gpu_util==null?'—':`<span class="util-bar ${{hotG?'hot':''}}"><span style="width:${{b.gpu_util}}%"></span></span>${{b.gpu_util}}%`}}</td>
        <td>${{gPct==null?'—':`<span class="util-bar ${{hotM?'hot':''}}"><span style="width:${{gPct.toFixed(0)}}%"></span></span>${{b.mem_used}}/${{b.mem_total}} MiB`}}</td>
        <td>${{b.cpu_util==null?'—':b.cpu_util+'%'}}</td>
        <td>${{procHtml}}</td>
        <td>${{etaHtml}}</td>
      `;
      tbody.appendChild(tr);
    }});
  }});
}}

// ── Task Queue ────────────────────────────────────────────────────────────
function loadQueue(){{
  fetch('/api/queue').then(r=>r.json()).then(j=>{{
    QUEUE_TASKS=j.tasks||[];
    const hdr=$('#queue-eta-hdr');
    const running=QUEUE_TASKS.filter(t=>t.status==='running').length;
    const pending=QUEUE_TASKS.filter(t=>t.status==='pending').length;
    if(hdr) hdr.innerHTML=(j.queue_eta&&(running||pending))
      ?`all done in ${{fmtEta(j.queue_eta,false)}}`:'';
    const tbody=$('#queue-table tbody'); tbody.innerHTML='';
    const empty=$('#queue-empty');
    if(!QUEUE_TASKS.length){{empty.style.display='';return;}}
    empty.style.display='none';
    QUEUE_TASKS.forEach(t=>{{
      const isPending=t.status==='pending';
      const canRetry=['running','failed','done'].includes(t.status);
      const color=STATUS_COLOR[t.status]||'';
      const delBtn=`<button title="delete" onclick="delTask('${{t.id}}')" style="color:#f85149">✕</button>`;
      let actions='';
      if(isPending) actions=`
        <button onclick="movePri('${{t.id}}',-1)">↑</button>
        <button onclick="movePri('${{t.id}}',1)">↓</button>
        ${{delBtn}}`;
      else if(canRetry) actions=`<button onclick="retryTask('${{t.id}}')">↺ retry</button> ${{delBtn}}`;
      // ETA cell
      let etaCell='<span class="muted small">—</span>';
      if(t.status==='running'&&t.eta_iso){{
        const bar=t.estimated_duration_s>0
          ?Math.min(100,Math.round(t.elapsed_s/t.estimated_duration_s*100)):0;
        etaCell=`<div style="font-size:11px;line-height:1.5">
          <div style="color:#3fb950">⏱ ${{fmtDur(t.elapsed_s)}} elapsed</div>
          <div style="color:#58a6ff">→ ${{fmtEta(t.eta_iso,false)}}</div>
          <div class="progress-bar"><div class="progress-bar-fill" style="width:${{bar}}%"></div></div>
        </div>`;
      }} else if(t.status==='pending'&&t.eta_iso){{
        etaCell=`<div style="font-size:11px;line-height:1.5;color:#8b949e">
          <div>starts ${{fmtEta(t.estimated_start_iso,true)}}</div>
          <div>done ${{fmtEta(t.eta_iso,false)}}</div>
        </div>`;
      }} else if(t.status==='done'&&t.started_at&&t.ended_at){{
        const dur=(new Date(t.ended_at)-new Date(t.started_at))/1000;
        etaCell=`<span class="small muted">${{fmtDur(dur)}}</span>`;
      }}
      const boxStr=t.box?`<span class="mono" style="font-size:11px">${{t.box}}</span>`:'<span class="muted">—</span>';
      const tr=document.createElement('tr');
      tr.innerHTML=`
        <td class="mono" style="text-align:center">${{t.priority}}</td>
        <td><span title="${{t.launcher}}&#10;${{t.env||''}}">${{t.label}}</span>
            ${{t.env?`<div class="small mono" style="opacity:.5;margin-top:1px">${{t.env}}</div>`:''}}</td>
        <td><span style="color:${{color}}">${{t.status}}</span></td>
        <td>${{boxStr}}</td>
        <td>${{etaCell}}</td>
        <td style="white-space:nowrap">${{actions}}</td>
      `;
      tbody.appendChild(tr);
    }});
    loadBoxes(); // refresh fleet ETAs whenever queue updates
  }});
}}
function toggleAdd(){{
  const f=$('#add-form'); f.style.display=f.style.display==='none'?'':'none';
}}
function addTask(){{
  const label=$('#at-label').value.trim(), launcher=$('#at-launcher').value.trim();
  const env=$('#at-env').value.trim(), priority=parseInt($('#at-pri').value)||10;
  const err=$('#at-err'); err.style.display='none';
  if(!label||!launcher){{err.textContent='Label and launcher required';err.style.display='';return;}}
  fetch('/api/queue',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{label,launcher,env,priority}})}}).then(r=>r.json()).then(j=>{{
    if(j.error){{err.textContent=j.error;err.style.display='';return;}}
    $('#at-label').value=''; $('#at-env').value='';
    loadQueue();
  }});
}}
function delTask(id){{ fetch('/api/queue/'+id,{{method:'DELETE'}}).then(()=>loadQueue()); }}
function movePri(id,d){{
  fetch('/api/queue/'+id+'/priority',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{delta:d}})}}).then(()=>loadQueue());
}}
function retryTask(id){{ fetch('/api/queue/'+id+'/retry',{{method:'POST'}}).then(()=>loadQueue()); }}

// ── Metrics ───────────────────────────────────────────────────────────────
let _metricData = null;

function loadMetricMeta(){{
  fetch('/api/metrics').then(r=>r.json()).then(j=>{{
    _metricData=j;
    const tagSel=$('#metric-tag'); tagSel.innerHTML='<option value="">— pick tag —</option>';
    j.tags.forEach(t=>{{
      const o=document.createElement('option');
      o.value=t.tag; o.textContent=`${{t.tag}} (${{t.n}} csv)`;
      tagSel.appendChild(o);
    }});
  }});
}}
function loadMetricCols(){{
  const tag=$('#metric-tag').value;
  if(!tag||!_metricData) return;
  fetch('/api/metrics/series?tag='+encodeURIComponent(tag)+'&metric='+encodeURIComponent((_metricData.cols||[])[0]||''))
    .then(r=>r.json()).then(j=>{{
      // discover available cols for this tag from actual CSVs
      fetch('/api/metrics').then(r=>r.json()).then(meta=>{{
        const colSel=$('#metric-col'); colSel.innerHTML='<option value="">— pick metric —</option>';
        (meta.cols||[]).forEach(c=>{{
          const o=document.createElement('option'); o.value=c; o.textContent=c;
          colSel.appendChild(o);
        }});
      }});
    }});
}}
function loadMetricSeries(){{
  const tag=$('#metric-tag').value, metric=$('#metric-col').value;
  if(!tag||!metric) return;
  const info=$('#metric-info'); info.textContent='loading…';
  fetch(`/api/metrics/series?tag=${{encodeURIComponent(tag)}}&metric=${{encodeURIComponent(metric)}}`)
    .then(r=>r.json()).then(j=>{{
      info.textContent=`${{j.series.length}} series`;
      const COLORS=['#58a6ff','#3fb950','#e3b341','#f85149','#bc8cff',
                    '#79c0ff','#7ee787','#ffa657','#ff7b72','#d2a8ff'];
      const traces=j.series.map((s,i)=>{{
        const c=COLORS[i%COLORS.length];
        return {{
          x:s.xs, y:s.ys, mode:'lines', name:s.name,
          line:{{color:c,width:1.5}},
          hovertemplate:`${{s.name}}<br>step:%{{x}}<br>${{metric}}:%{{y:.2f}}<extra></extra>`,
        }};
      }});
      const layout={{
        paper_bgcolor:'#0d1117', plot_bgcolor:'#161b22',
        font:{{color:'#c9d1d9',size:12}},
        xaxis:{{title:j.step_col,gridcolor:'#21262d',zerolinecolor:'#30363d'}},
        yaxis:{{title:j.metric,gridcolor:'#21262d',zerolinecolor:'#30363d'}},
        legend:{{bgcolor:'#161b22',bordercolor:'#30363d',borderwidth:1}},
        margin:{{t:20,b:50,l:60,r:20}},
        hovermode:'x unified',
      }};
      Plotly.newPlot('metrics-plot', traces, layout, {{responsive:true,displayModeBar:false}});
    }});
}}

// ── Boot ──────────────────────────────────────────────────────────────────
loadBoxes(); loadQueue(); loadMetricMeta();
setInterval(loadBoxes, 30000);
setInterval(loadQueue, 10000);
</script>
</body>
</html>
""".format(name=FLEET_NAME)

@app.route("/")
def index():
    return HTML


if __name__ == "__main__":
    print(f"[fleet-dashboard] {FLEET_NAME}  http://localhost:{PORT}", flush=True)
    print(f"[fleet-dashboard] config: {CONFIG_PATH}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
