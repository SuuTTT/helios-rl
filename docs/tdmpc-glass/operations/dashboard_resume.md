# Resume the live monitoring stack (after a session restart)

Read this first when reopening Claude to continue working on TD-MPC-Glass — it
points at every long-running piece so you can confirm what's still alive and
restart anything that died.

## URLs

- **Web dashboard**: http://localhost:5055
  - Box Fleet: live SSH probe of all 6 boxes (GPU/CPU, running phase·seed·best·last, dup warnings)
  - Learning Curves: Plotly view of every HopperHop_phase*/seed_*.csv (filters: only-MPPI, only-currently-running, phase contains)
  - Render Rollout: click-to-render, length selector (short/medium/long/extra), archived videos surfaced inline
- **Terminal dashboard**: `bash scripts/iter5_dashboard.sh`

## What runs in the background (and how to confirm each is alive)

| Component | What it does | PID file / how to check |
|---|---|---|
| `scripts/web_dashboard.py` | Flask app on :5055 | `pgrep -fa scripts/web_dashboard.py` |
| `scripts/iter5_stream_remotes.sh` | rsync mirror every 300 s | `pgrep -fa iter5_stream_remotes.sh` |
| `scripts/iter6_auto_queue.sh` | poll boxes every 300 s, launch next queue line | `pgrep -fa iter6_auto_queue.sh` |

All three are daemonized with `nohup setsid ... & disown` and end up with PPID=1 (parented to init). They survive Claude session close. Verify with `ps -o pid,ppid,cmd <PID>` — the second column should be `1`.

Live logs:
```
/root/helios-rl/exp/tdmpc_glass/logs/daemons/stream.log
/root/helios-rl/exp/tdmpc_glass/logs/daemons/autoqueue.log
/tmp/web_dashboard.log
```

## Restart playbook (any process died)

```bash
# 1. Stream rsync (mirrors remote CSVs → local exp/tdmpc_glass/remote_mirror/<box>/)
nohup setsid /root/helios-rl/scripts/iter5_stream_remotes.sh \
  > /root/helios-rl/exp/tdmpc_glass/logs/daemons/stream.log 2>&1 < /dev/null & disown

# 2. Auto-queue (claims next idle box, launches queue file's next non-DONE line)
nohup setsid /root/helios-rl/scripts/iter6_auto_queue.sh \
  > /root/helios-rl/exp/tdmpc_glass/logs/daemons/autoqueue.log 2>&1 < /dev/null & disown

# 3. Web dashboard
nohup setsid /root/venv/bin/python3 -u /root/helios-rl/scripts/web_dashboard.py \
  > /tmp/web_dashboard.log 2>&1 < /dev/null & disown
```

To stop one: `pkill -f scripts/iter5_stream_remotes.sh` (or the script's full name). Always grep PIDs first — `pkill -f web_dashboard` could match other things.

## Queue files (where you add experiments)

`scripts/queues/*.queue` — one file per box. Format per line:
```
<port>|<host>|<launcher_script_path>|<env_vars_space_separated>
```
Lines starting with `#` are comments. The autoqueue marks consumed lines as
`# DONE <ts> <original line>`. Re-adding the same line below the DONE marker
re-runs that experiment.

Currently-tracked boxes (must match `BOX_PORT`/`BOX_HOST`/`BOX_GPUMASK` in `iter6_auto_queue.sh`):

| Tag | Queue file |
|---|---|
| `ssh6_4060` | `ssh6_4060.queue` |
| `ssh17637_gpu0` | `ssh17637_gpu0.queue` |
| `ssh17637_gpu1` | `ssh17637_gpu1.queue` |
| `ssh1_2080ti` | `ssh1_2080ti.queue` |
| `ssh6_3080` | `ssh6_3080.queue` |

## What can break (and the fix)

| Symptom | Likely cause | Fix |
|---|---|---|
| Box fleet shows `best —` / `last —` for a running proc | Local mirror outdated (stream died, or remote CSV has no eval rows yet) | Check `pgrep -fa iter5_stream`; restart with the snippet above. If stream is alive, the proc just hasn't done its first eval (~250k env steps at the box's sps). |
| Box marked "unreachable" | SSH timeout / vast.ai box rebooted | Re-test by hand: `ssh -p <port> -o ConnectTimeout=5 root@<host> echo ok`. If the box is gone, comment its row out of `BOXES` in `scripts/web_dashboard.py` AND in `scripts/iter5_dashboard.sh` AND in `iter6_auto_queue.sh`. |
| Render button says `queued…` forever | render_glass_rollout.py crashed mid-run | `curl -s http://localhost:5055/api/render/<job_id>` → look at `log` array for the traceback. Kill the orphan job dict by restarting the dashboard. |
| Auto-queue keeps re-launching the same seed | Launcher exit-code != 0 fast enough that GPU probe still says idle | Tail `/root/helios-rl/exp/tdmpc_glass/logs/daemons/autoqueue.log` + the box's `/tmp/autoqueue_<tag>.log` for the actual error |
| Stream loop "no active csvs" for a freshly-launched seed | CSV is header-only (<30 bytes) | Wait for first eval to be appended (will show up automatically), OR launcher script writing to a wrong tag path |

## Where the live state lives

- **Local mirror of remote CSVs**: `/root/helios-rl/exp/tdmpc_glass/remote_mirror/<box>/HopperHop_<phase>/seed_*.csv`
- **Local-only runs**: `/root/helios-rl/exp/tdmpc_glass/HopperHop_<phase>/seed_*.csv`
- **Diagnostic sidecars**: `seed_N_diag.csv` (per-eval `full_reward_rate / standing_rate / fall_count / time_to_first_full`)
- **Checkpoints**: `HopperHop_<phase>/seed_N/checkpoints/{best_mppi.pkl, last.pkl}`
- **Rendered videos**:
  - new (dashboard-triggered): `exp/tdmpc_glass/rollout_videos/<job_id>.mp4`
  - archived: `exp/tdmpc_glass/videos/<phase>/seed_N_*.mp4`
- **Per-phase run logs**: `exp/tdmpc_glass/logs/<phase>/HopperHop_seed_N.log`

## Quick "what's running now" one-liner

```bash
bash /root/helios-rl/scripts/iter5_dashboard.sh
```
Iterates all 6 boxes, per-box: running run_benchmark procs, GPU/CPU util, all
active HopperHop_phase*/seed_*.csv with best/last MPPI. Bold-green if best ≥ 500.

## Where the project context lives (read these first when resuming)

1. `docs/tdmpc-glass/operations/dashboard.md` — hardware fleet table, current top-line results, goals (G1: 5/5 > 500, G2: break 600)
2. `docs/tdmpc-glass/iterations/iteration_6_plan.md` — current iteration: §3 active runs, §6 ref experiments, §7 env-only roadmap
3. `docs/tdmpc-glass/operations/hardware_req.md` — vast.ai filter criteria when renting new boxes
4. `docs/agent_multi_gpu_workflow.md` — generalized playbook for using Claude+SSH across multiple boxes
5. `AGENT_HANDOFF_CONTEXT.md` — most-recent experiment status (older sessions may not have updated it)
