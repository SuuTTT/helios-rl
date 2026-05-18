# Agent-driven multi-GPU experiment workflow

A playbook for using a single AI agent (e.g. Claude Code) to run a research
sweep across 4+ heterogeneous GPU boxes (local + vast.ai remotes), distilled
from running TD-MPC-Glass HopperHop across local 4070 Ti + ssh3/ssh6/ssh17637.

Audience: someone starting an ML research project where:
- You have one fast local box and several rented (or borrowed) remote GPUs
- The remotes are flaky (vast.ai instances recycle, drivers vary, SSH drops)
- Experiments take hours each, you want >2x throughput from parallelism
- You don't want to babysit; the agent should self-manage

The end state should let you say "use this box too" and walk away.

## 1. Inventory the fleet on first contact

For every new box: probe in one ssh round-trip:

```bash
ssh -p PORT root@HOST \
  "echo --GPU--; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
   echo --DRIVER--; nvidia-smi | head -3 | tail -1
   echo --PYTHON--; python3 --version
   echo --VENV--; ls /root/venv/bin/activate 2>/dev/null && echo yes || echo no
   echo --DISK--; df -h /root | tail -1"
```

What you must know before touching it:
- Driver version (determines CUDA wheel compatibility — kills many runs)
- VRAM per GPU (determines `MEM_FRACTION` + which experiments fit)
- Python version (Python 3.10 forces older JAX which breaks modern stacks)
- Free disk (>30 GB for JAX wheels alone)

Mismatches that have killed runs in this project:
- Driver 535 + JAX 0.6 (needs cuSPARSE 12.6) — 3090 box stayed BLOCKED
- Driver 570 + Blackwell sm_120 — RTX 5070 couldn't load CUDA 13 wheels
- 6 GB shared between two processes — OOM-killed seeds repeatedly
- Old `mujoco_warp` against new JAX — `jax.tree.map_with_path` missing

Hard rule: **if the driver is ≥3 months older than your JAX release, expect
incompatibility**. Don't burn a day adapting; release the box.

## 2. First-time setup checklist (5 min per box)

Parallel push + clone:

```bash
# In two background subshells:
rsync -av -e "ssh -p PORT" --exclude='exp/' --exclude='__pycache__/' --exclude='.git/' \
  /local/repo/ root@HOST:/root/repo/ &
ssh -p PORT root@HOST \
  "apt-get install -y git rsync python3-venv >/dev/null 2>&1
   git clone --depth=1 https://example.com/data-deps.git /root/data &
   python3 -m venv /root/venv" &
wait

# Then install Python deps — pick the right requirements file by driver
ssh -p PORT root@HOST \
  "source /root/venv/bin/activate
   pip install -r /root/repo/requirements-<arch>.txt
   pip install -e /root/repo
   python3 -c 'import jax; print(jax.devices())'"
# Expect: [CudaDevice(id=0)]
```

If that last `jax.devices()` shows CPU, the box is BLOCKED — diagnose vs
release per §1.

## 3. Standardize the run script and output path

The biggest single source of lost data in this project: training script silently
overwrote per-seed CSV on every restart. Iron rule:

- **One CSV per seed** at a predictable, tag-namespaced path.
- **Append, never overwrite** when the file exists at run start.
- Log file uses `tee -a`, not `tee`.

Standard layout (used in our `run_benchmark.py`):

```
exp/<algo>/<env><TAG>/seed_N.csv          # main eval log
exp/<algo>/<env><TAG>/seed_N_diag.csv     # secondary diagnostics
exp/<algo>/logs/<TAG>/HopperHop_seed_N.log  # stdout/stderr tee'd
```

Where `<TAG>` comes from `TDMPC_GLASS_OUTPUT_TAG` env var (default empty). Setting
the tag lets you run two phases of the same algorithm-env pair side by side.

Anti-pattern to remove if you see it:

```python
elif env_id == "X":
    eval_type_csv = SHARED_FILE_PATH   # ← shared, get clobbered by parallel seeds
    with open(eval_type_csv, "w") as cf:  # ← truncates!
        cf.write(header)
```

Fix it _before_ launching multi-seed experiments. We hit this twice (Phase-z,
Phase-q) and lost several hours of data each time.

## 4. Per-seed launcher script, not a queue script

**Why**: a queue script like `SEEDS="1 2 3 4 5"; for s in $SEEDS; do ... done` is
fragile — if a watcher relaunches after seed 3 finished, the queue script
starts again from seed 1, overwriting seed_1/seed_2 CSVs.

Pattern that works:

```bash
# scripts/run_<exp>_<box>.sh — ONE SEED per invocation, env var-driven
#!/usr/bin/env bash
set -u; set +e
SEED=${SEED:-1}
NS=${NS:-2048}
log=$REPO/exp/.../HopperHop_seed_${SEED}.log
echo "[<exp>_<box>] === seed=${SEED} start $(date -u +%FT%TZ) ===" | tee -a "$log"
python3 -u run_benchmark.py --seed "$SEED" --mppi_n_samples "$NS" ...
echo "[<exp>_<box>] === seed=${SEED} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log"
```

Pass seed as env var so the same script runs every seed:
`SEED=3 nohup setsid bash scripts/run_<exp>_<box>.sh ... & disown`

## 5. Stream results back continuously

Set up a single rsync loop that pulls all per-seed CSVs from every box every
~10 min. Keep it minimal:

```bash
# scripts/stream_remotes.sh (one cycle):
for box in box1 box2 box3; do
  rsync -a -e "ssh -p PORT -o ConnectTimeout=10" \
    --include='HopperHop_*/' --include='HopperHop_*/**' \
    --exclude='**/checkpoints/**' --exclude='**/*.pkl' \
    root@$HOST:/root/repo/exp/.../ $LOCAL_MIRROR/$box/ >/dev/null 2>&1 &
done
wait
# Summarize: for each LIVE csv (size>100, recent mtime), grep best metric
```

Two filters to keep the stream signal-rich:
- **Size > 100 bytes** filters out empty header-only CSVs (run just started)
- **mtime within last 2 days** filters out archived/old phase data

Run this as a persistent background Monitor; each cycle emits one line per box
with all currently-running seeds and their best metric so far.

## 6. Per-box queue files + auto-launcher

When a GPU goes idle (run completed or crashed), an agent + human can both
forget to launch the next experiment. Solve this with explicit per-box queues:

```
scripts/queues/<box>.queue:
  <port>|<host>|<launcher_path>|<env_vars>
  <port>|<host>|<launcher_path>|<env_vars>
```

And a single watcher loop that polls each box:

```bash
# scripts/auto_queue.sh:
while true; do
  for box in box1 box2 box3; do
    if is_box_busy "$box"; then continue; fi
    next=$(pop_next_queue_line "scripts/queues/$box.queue") || continue
    ssh -p $port root@$host "cd /root/repo && $envvars nohup setsid bash $launcher ... & disown"
  done
  sleep 300
done
```

Two important details:
- **`is_box_busy` checks GPU memory > threshold** on the specific CUDA index for
  shared-GPU boxes. `ps -eo cmd | grep run_benchmark` is enough for single-GPU.
- **Mark consumed lines with `# DONE <ts>`** in place (sed -i) so retry-of-a-retry
  isn't possible. Comments are skipped.

If a launch crashes immediately (e.g. unknown CLI flag → old code), the next
cycle will detect the box idle again and try the NEXT queue line. So fix code
bugs aggressively before queuing many lines.

## 7. Watcher for OOM/recycle + CSV backup

Boxes get OOM-killed or vast.ai recycles them. Separately from the auto-queue
launcher, a "health watcher" detects DEAD runs and either:
- Relaunches (if the run hadn't finished naturally)
- OR skips (if the run already early-stopped — preserve its data)

**CRITICAL**: backup CSV BEFORE every relaunch:

```bash
ssh root@$host "[[ -f $csv ]] && cp $csv ${csv%.csv}_v${retry}_${ts}.csv"
ssh root@$host "$launcher"
```

Why: when `run_benchmark.py` restarts, it re-opens `seed_N.csv` with `"w"` and
truncates. The 382-MPPI trajectory we'd just built up vanishes. Versioned
backups (`seed_N_v1_<ts>.csv`) let you recover.

We lost ~5 trajectories in iter 5 before adding this. **Don't repeat that
mistake.**

## 8. Code-sync hygiene

Every box has its own copy of the code, and they drift. Rules:

1. **Rsync code to ALL remotes after every meaningful edit**:
   ```bash
   for box in ...; do
     rsync -av scripts/ src/ root@$host:/root/repo/ &
   done
   wait
   ```

2. **Running processes keep their old code in memory.** A code fix doesn't apply
   to in-flight runs. Either restart them (carefully — back up CSV first) or
   accept that they finish on the old behaviour.

3. **Smoke-test new flags before queuing**. Two bugs in this project came from
   adding a kwarg to one code path but forgetting another (`use_cluster_obs`,
   `smoothing_enabled`):
   ```bash
   timeout 60 python3 run_benchmark.py --new_flag ... --total_steps 1000 --seed 99
   # Just need it to reach the JIT-compile boundary, not finish.
   ```

## 9. Hygiene rules / foot-guns to avoid

These cost real hours when violated:

| Anti-pattern | Symptom | Fix |
|---|---|---|
| `tee` instead of `tee -a` in launcher | Log truncated on watcher relaunch | Always `tee -a` |
| `open(csv, "w")` on every run start | CSV cleared on restart | Append mode if exists |
| Queue script looping seeds | Watcher restart re-runs from seed 1 | Per-seed launcher (§4) |
| pgrep `'pattern'` matches the pgrep cmd itself | False "still running" reports | Use `ps -eo cmd \| grep pattern \| grep -v grep \| wc -l` |
| `2>&1 >/dev/null` (wrong order) | stderr still leaks | `>/dev/null 2>&1` |
| Long sleeps in watcher (~5 min) | Slow reaction to box returning | Sleep < SSH cooldown; box should be tested every cycle |
| Mem fraction at 0.85 for shared GPU | OOM-kills the other run | 0.35–0.45 when shared |
| Launch from in-script `mkdir` after redirect | bash redirect happens before script runs | `mkdir -p logs/` BEFORE the `nohup ... > logs/x.log` |

## 10. Documentation discipline (for the agent's future self)

- **One short markdown file per iteration** (`iteration_N_findings.md`). Each
  ends with §"What works / What's falsified". Future-you reads §0 first.
- **A live `dashboard.md`** with current GPU map, current best results, "what to
  run next". Updated whenever something material changes.
- **An `experiment_ledger.md`** (or equivalent in iter doc): every multi-seed
  result with the seed numbers + best MPPI. Iter 6 §0 in our project is this.
- **Falsified ideas stay falsified.** Cross them off, don't quietly retry. We
  almost re-launched Path P twice across iterations because we'd forgotten.

## 11. Recovery patterns

### A box returns after going down

1. `ssh -p PORT -o ConnectTimeout=5 root@HOST "echo ok"` — confirm reachable.
2. `ps -eo cmd | grep run_benchmark` — did anything survive? Usually no.
3. Check `exp/.../seed_N.csv` — if it has data, BACK IT UP before any relaunch.
4. Watcher v2 does step 3 automatically.

### Box keeps OOM-killing the same seed

1. Drop `XLA_PYTHON_CLIENT_MEM_FRACTION` from 0.85 → 0.55 → 0.45 → 0.35.
2. Use only one GPU when 2-GPU box; the slot-isolation isn't enforced strongly.
3. Try a smaller experiment variant (e.g. NS=1024 instead of NS=2048) — the
   eval phase has a memory peak proportional to NS.

### Vanilla driver issue you can't fix

Release the box. We spent 30 min trying to make the 3090 work — its driver was
just too old. The hour saved by giving up was worth more than the GPU.

## 12. The minimum agent loop

A good autonomous agent loop has 4 components running concurrently:

1. **Active runs** — what we're actually training.
2. **Stream** — mirror per-seed CSVs from remotes; surfaces metric every ~10 min.
3. **Health watcher** — detects DEAD runs, backs up CSV, relaunches if appropriate.
4. **Auto-queue** — when a box's slot is idle, pop next experiment from queue file.

These are 3 background tasks (stream, watcher, queue) plus your foreground work.
The agent can dispatch one of these per box and forget about it until events
arrive. The human says "use this new box" and the agent adds it to all 3
managers in one step.

## 13. Quick reference: minimum file set

```
docs/
  iteration_N_findings.md     ← results log per iteration
  dashboard.md                 ← live state, "what's running"
  hardware_req.md              ← per-box compatibility notes
  agent_multi_gpu_workflow.md  ← this file

scripts/
  run_<exp>_<box>.sh           ← per-seed launchers
  queues/<box>.queue           ← per-box experiment queues
  stream_remotes.sh            ← mirror CSVs (background Monitor)
  auto_queue.sh                ← pop queue on idle (background Monitor)
  health_watcher.sh            ← OOM/recycle relaunch (background Monitor)
  iter5_dashboard.sh           ← human-readable snapshot of current state

exp/<algo>/<env>_<TAG>/seed_N.csv              ← canonical run output
exp/<algo>/<env>_<TAG>/seed_N_diag.csv         ← diagnostics sibling
exp/<algo>/<env>_<TAG>/seed_N_v<n>_<ts>.csv    ← versioned backups
exp/<algo>/remote_mirror/<box>/...             ← stream destination
```

Keep this minimal. Every script you don't need is one more thing that drifts
out of sync across 4 boxes.
