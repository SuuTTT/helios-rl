# Agent Handoff Context — TD-MPC-Glass / Phase 1c

Date: 2026-05-13. Author: previous Copilot agent on the workstation. Target: Copilot agent running on the remote 4070 Ti (vast.ai).

This file is the single source of truth for picking the work back up on the remote box after the vast.ai instance is stopped + a new one is started (or after `ssh` migration). Read this end-to-end before doing anything else.

---

## 1. Project: helios-rl / TD-MPC-Glass on HopperHop

**Goal (blog v5):** prove that TD-MPC2 + Glass (prototype clustering of latents) escapes the "seed 3/4 stuck" basin in `HopperHop` by annealing the action-collection noise.

- Repo: `helios-rl/` (this folder is the entire project; `cleanrl/`, `tdmpc2/` etc. at workspace root are reference only).
- Algorithm entrypoint: [scripts/run_benchmark.py](scripts/run_benchmark.py), `--algos tdmpc-glass`.
- Glass-specific code: `src/helios_rl/algos/tdmpc_glass/` (encoder, dynamics, π, prototype heads).
- Env: `mujoco_playground.registry.load('HopperHop')` (lives at `/root/mujoco_playground_repo` on remote).

### 1.1 Experiment ladder so far
| Phase | Knobs | Seeds | Outcome |
|-------|-------|-------|---------|
| Phase 1 | Glass defaults, T=1.0 | 1–5 | mean ~280, 2 seeds stuck |
| Phase 1b | T=0.7, init_scale=0.5 | 1–5 | finals 438, 526, 294, 187, 562 → mean 401±158, seeds 3/4 stuck despite identical Glass diagnostics (ent=1.386, active=4, max_mass=0.250) |
| Phase 2 | stopgrad_graph=false | 1–3 | seed 1=242, seed 2=294, worse → **KILLED**, fix hypothesis rejected |
| **Phase 1c** | **same Phase 1b knobs + act-noise anneal 0.30→0.10 over 1M steps** | 1–5 | **RUNNING on remote, the load-bearing experiment** |

### 1.2 Phase 1c verbatim CLI (already running, do NOT change)
```
--algos tdmpc-glass --tasks HopperHop --total_steps 4000000 --seed <S>
--glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true
--act_noise_start 0.30 --act_noise_end 0.10 --act_noise_anneal_steps 1000000
--no_plot
```
Driver script: [scripts/run_phase1c_remote.sh](scripts/run_phase1c_remote.sh) — runs seeds 1..5 sequentially, then dumps a 3-episode cam0 MP4 per seed via [scripts/render_glass_rollout.py](scripts/render_glass_rollout.py).

---

## 2. Current remote run state (snapshot 2026-05-13T17:36Z)

Old box `ssh -p 20305 root@ssh8.vast.ai` (4070 Ti):
- PID 1830 `bash scripts/run_phase1c_remote.sh` (queue)
- PID 1843 `python3 scripts/run_benchmark.py ... --seed 1`
- Seed 1 progress: **es=245,760, sps≈735, loss=0.51**. ETA per seed ≈ 1.5 h at this sps (4M steps / 735sps ≈ 90 min). Total queue ≈ **~8 h** (revised down from 45 h after JIT warmed up to 735 sps).
- Queue log: `/root/helios-rl/exp/tdmpc_glass/logs/phase1c/queue.log` (header: `[phase1c] start 2026-05-13T17:13:08Z seeds=[1 2 3 4 5]`).
- Per-seed log: `/root/helios-rl/exp/tdmpc_glass/logs/phase1c/HopperHop_seed_<S>.log`.

**⚠️ If user stops this instance NOW, seed 1's run is lost** (no checkpoint mid-run, only `best_mppi/latest_eval/final` at end). Restart from scratch on new box.

---

## 3. Files that must move to the new 4070 Ti

You said you're stopping the current vast.ai instance and starting a new one. The new box will be empty. Here is exactly what needs to be on it.

### 3.1 System prerequisites (vast.ai image probably already has these)
- Python 3.10+
- CUDA 12.x driver matching JAX `jax[cuda12]`
- `ffmpeg` (apt-get install ffmpeg if missing)
- A venv at `/root/venv` with packages — see §3.3.

### 3.2 Code (rsync from local `/workspace/helios-rl/`)
The whole repo can be rsynced. Minimum mandatory files:
- `scripts/run_benchmark.py` (contains the act-noise anneal CLI flags; commit 5c473d1)
- `scripts/render_glass_rollout.py` (cam0 default, cluster overlay)
- `scripts/run_phase1c_remote.sh` (queue, activates venv, sets `MUJOCO_GL=egl`)
- `src/helios_rl/...` (all algorithm code)
- `pyproject.toml` / `requirements.txt`

External dep (NOT in repo): `mujoco_playground` checkout at `/root/mujoco_playground_repo`. Recloneable from upstream; the `HopperHop` env is unmodified.

### 3.3 venv at `/root/venv`
Verified package list from old box: `jax[cuda12]`, `flax`, `optax`, `mujoco`, `mujoco_mjx`, `mediapy`, `Pillow`, `imageio`, `imageio-ffmpeg`, `tqdm`, `pyyaml`, `numpy`, `scipy`. (Plus whatever helios-rl's `pyproject.toml` declares.)

### 3.4 SSH key
The vast.ai key you used for old box is presumably uploaded; the new box gets a new port + maybe new host. Update `~/.ssh/config` (or just remember the new `ssh -p <PORT> root@<HOST>`).

### 3.5 What NOT to bother transferring
- Old `exp/tdmpc_glass/HopperHop_phase1c/seed_1.csv` — incomplete; Phase 1c restarts from scratch.
- Old `logs/phase1c/queue.log` — same.
- Old `videos/` — same.

---

## 4. Step-by-step transfer + relaunch checklist (do these in order on the NEW box)

> Run on your **local workstation** unless prefixed `[REMOTE]`.

1. **Snapshot any partial Phase 1c data you want to keep from the OLD box** (optional — recommend skipping since seed 1 isn't done):
   ```bash
   OLD=ssh8.vast.ai; OLD_PORT=20305
   rsync -avz -e "ssh -p $OLD_PORT" root@$OLD:/root/helios-rl/exp/tdmpc_glass/HopperHop_phase1c/ /workspace/helios-rl/exp/tdmpc_glass/HopperHop_phase1c_partial/ || true
   rsync -avz -e "ssh -p $OLD_PORT" root@$OLD:/root/helios-rl/exp/tdmpc_glass/logs/phase1c/ /workspace/helios-rl/exp/tdmpc_glass/logs/phase1c_partial/ || true
   ```
2. **Stop the OLD vast.ai instance** via the vast.ai web UI.
3. **Start the NEW 4070 Ti instance.** Note its `HOST` and `PORT`.
4. **Push local commits** (so the new box can `git pull` instead of relying on rsync):
   ```bash
   cd /workspace/helios-rl && git push origin main
   ```
   Local commits to push: `81cdf6c` (render tool), `5c473d1` (act-noise anneal + phase1c).
5. **Bootstrap the new box** (one-shot):
   ```bash
   NEW=<host>; NEW_PORT=<port>
   ssh -p $NEW_PORT root@$NEW '
     set -e
     apt-get update && apt-get install -y ffmpeg rsync git
     [ -d /root/helios-rl ] || git clone <YOUR_REMOTE_URL> /root/helios-rl
     cd /root/helios-rl && git fetch && git checkout main && git pull
     [ -d /root/mujoco_playground_repo ] || git clone https://github.com/google-deepmind/mujoco_playground /root/mujoco_playground_repo
     python3 -m venv /root/venv
     source /root/venv/bin/activate
     pip install --upgrade pip
     pip install "jax[cuda12]" flax optax mujoco mujoco-mjx mediapy Pillow imageio imageio-ffmpeg tqdm pyyaml numpy scipy
     pip install -e /root/helios-rl || true
     python3 -c "import jax, mediapy, PIL, mujoco; print(\"OK\", jax.devices())"
   '
   ```
6. **If you have NOT pushed to git remote**, rsync code directly instead of step 5's git clone:
   ```bash
   rsync -avz --exclude exp --exclude wandb --exclude videos --exclude __pycache__ \
     -e "ssh -p $NEW_PORT" /workspace/helios-rl/ root@$NEW:/root/helios-rl/
   ```
7. **Launch the Phase 1c queue on the new box:**
   ```bash
   ssh -p $NEW_PORT root@$NEW '
     cd /root/helios-rl && mkdir -p exp/tdmpc_glass/logs/phase1c
     nohup setsid bash scripts/run_phase1c_remote.sh \
       > exp/tdmpc_glass/logs/phase1c/queue.log 2>&1 < /dev/null & disown
     sleep 10
     pgrep -fa "run_phase1c|run_benchmark"
     head -5 exp/tdmpc_glass/logs/phase1c/queue.log
   '
   ```
8. **Verify after ~3 minutes** the JIT compiled and sps stabilized ~700:
   ```bash
   ssh -p $NEW_PORT root@$NEW 'tail -5 /root/helios-rl/exp/tdmpc_glass/logs/phase1c/HopperHop_seed_1.log'
   ```
9. **Tell the new agent: read this file** (`/root/helios-rl/AGENT_HANDOFF_CONTEXT.md`).

---

## 5. What the agent on the remote should do next (priority order)

1. **Verify queue is alive every ~30 min:**
   ```bash
   pgrep -fa "run_phase1c|run_benchmark"
   tail -5 /root/helios-rl/exp/tdmpc_glass/logs/phase1c/queue.log
   for s in 1 2 3 4 5; do
     f=/root/helios-rl/exp/tdmpc_glass/HopperHop_phase1c/seed_$s.csv
     [ -f "$f" ] && echo "=== seed $s ===" && awk -F, '$3=="mppi"' "$f" | tail -5
   done
   ```
2. **When all 5 seeds finish**, compute peak/final returns vs Phase 1b baseline `[438.3, 526.3, 294.4, 186.5, 562.1]`. Key question: **did seeds 3 and 4 escape the stuck basin?**
3. **Render videos**: the queue auto-renders cam0 MP4s after each seed via `scripts/render_glass_rollout.py --camera cam0`. Output: `/root/helios-rl/exp/tdmpc_glass/videos/phase1c/seed_<S>_best_mppi.mp4`.
4. **Mirror results back to workstation:**
   ```bash
   rsync -avz -e "ssh -p $NEW_PORT" root@$NEW:/root/helios-rl/exp/tdmpc_glass/HopperHop_phase1c/ \
     /workspace/helios-rl/exp/tdmpc_glass/HopperHop_phase1c/
   rsync -avz -e "ssh -p $NEW_PORT" root@$NEW:/root/helios-rl/exp/tdmpc_glass/videos/phase1c/ \
     /workspace/helios-rl/exp/tdmpc_glass/videos/phase1c/
   ```
5. **Update blog v5** §5.7 with Phase 1c verdict, regenerate figures via `scripts/plot_blog_figs.py`, embed video frames.

---

## 6. Hard-won pitfalls (don't re-discover these)

- **`'int' is not iterable` in NormMLP**: the `hidden` field of Encoder/Dynamics is `tuple[int,...]`, not `int`. When building models in [scripts/render_glass_rollout.py](scripts/render_glass_rollout.py) use `hidden=(512,512)`.
- **`Pi` returns `(mean, log_std)` tuple**, not a single tensor. Action = `jnp.tanh(mean[0])`.
- **Headless OpenGL**: `MUJOCO_GL=egl` (already set in queue script). Without it the env render crashes with "platform library not loaded".
- **Hopper leaves frame**: default camera is a static free camera. Use `--camera cam0` (mode=trackcom in HopperHop XML). `back` is another trackcom angle.
- **Stopping/restarting vast.ai instances**: SSH may briefly refuse on the new box for ~30 s — retry.
- **Module not found** (jax/mediapy) on remote: queue script MUST `source /root/venv/bin/activate` before invoking `python3`. Already in [run_phase1c_remote.sh](scripts/run_phase1c_remote.sh).
- **stopgrad_graph=false is NOT the fix** (Phase 2 falsified that hypothesis). Don't propose it again.
- **Phase 1c is the load-bearing test** — don't add more knobs until it finishes.

---

## 7. Memory dump (previous agent's `/memories/` snapshots)

### `/memories/repo/workspace-layout.md`
- Top-level layout: `cleanrl/` upstream, `wiki/` content, `benchmark|runs|videos|logs` outputs, `scripts/` helpers.
- Helper scripts reorganized into `scripts/wiki_tools`, `scripts/probes`, `scripts/scratch` on 2026-05-05.

### `/memories/repo/ppo_jax_findings.md` (unrelated to Phase 1c but archived)
- JAX PPO CheetahRun: best v34s3, seed=3, eps=1e-5, crash_threshold=150, **keep optimizer state on recovery** → 904.5@74M (matches Brax).
- Bad: reset optimizer → effective lr×100 → catastrophe.
- eps=1e-8 = faster early, more crashes. eps=1e-5 = stable.

### `/memories/repo/tdmpc2_sps_benchmark.md`
- Official TD-MPC2 (torch, single-env, hopper-hop): 10.5 SPS env, MPPI-bound, ~26 h/1M.
- Our JAX (N_ENVS=256): 57–67 SPS × 256 envs ≈ 14.6–17.2 k env transitions/s, **~1400× official**. But MPPI quality at 1M = 156.9 vs 338 official (53% gap).
- Phase 1c box: sps≈735 (single env steps/s after JIT warmup).

---

## 8. Git state at handoff

- Branch `main`, HEAD `5c473d1` (local), `origin/main` at `9f83e1f` → **2 commits ahead, not pushed**.
- Modified-but-uncommitted: `.gitignore`, `exp/benchmark/tdmpc-glass_HopperHop.csv`, `exp/tdmpc_glass/HopperHop_phase1b_remote/seed_4.csv`, `exp/tdmpc_glass/HopperHop_phase2/seed_1.csv`, `scripts/run_phase1c_remote.sh`.
- Untracked: `exp/tdmpc_glass/HopperHop_phase1b_remote/seed_5.csv`, `exp/tdmpc_glass/HopperHop_phase2/seed_2.csv`, `exp/tdmpc_glass/HopperHop_phase2/seed_3.csv`.
- Recommend: commit the experimental CSVs into an `exp/` branch or just `git add` them locally before pushing; **push commits 81cdf6c + 5c473d1 before relaunching** so the new box can `git pull`.

---

## 9. TL;DR for the human

Before you stop the old instance:
1. `cd /workspace/helios-rl && git push origin main` ← critical, otherwise new box has no act-noise-anneal code.
2. (optional) rsync partial seed_1 results back for forensics.

After starting new 4070 Ti:
3. Note new `HOST:PORT`.
4. Run §4 step 5 bootstrap one-liner.
5. Run §4 step 7 launch one-liner.
6. Tell the new agent: *"read AGENT_HANDOFF_CONTEXT.md and continue from §5".*

That's it. Phase 1c will run ~8 h then dump CSVs + MP4s.
