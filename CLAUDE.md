# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo at a glance

helios-rl is a JAX/Flax research repo for model-free and model-based RL on **MuJoCo Playground** tasks (CartpoleBalance, HopperHop, HopperStand, CheetahRun, FishSwim, …). The current research focus is **TD-MPC-Glass** — TD-MPC2 augmented with prototype clustering of latents — running local 4070 Ti (a vast.ai instance). env for exp is /root/venv. we also have a remote 5070 at ssh -p 37645 root@ssh8.vast.ai -L 8080:localhost:8080; all context about the tdmpc-glass and how to run exp at remote is under /root/helios-rl/docs/tdmpc-glass/blog_phase1.md

Always read `AGENT_HANDOFF_CONTEXT.md` first — it is the live status doc for whatever experiment is running. The handoff doc is more authoritative than `README.md` for what's actually in use; `README.md` describes an older Hydra/`helios.main` skeleton that is **not the day-to-day entry point**.



## Two parallel structures (don't confuse them)

1. **Production training path — `scripts/run_benchmark.py`** (this is what we actually run).
   - Trains `ppo`, `sac`, `tdmpc2`, `tdmpc-glass` on MuJoCo Playground envs via `mujoco_playground.registry.load(...)` + `mujoco_playground.wrapper.wrap_for_brax_training(...)`.
   - Imports the algorithm code from `src/helios/algorithms/*.py` directly (NOT through Hydra).
   - Writes CSVs to `exp/benchmark/<algo>_<task>.csv` (columns: `task,seed,step,reward`).
   - TD-MPC-Glass writes per-seed CSVs/checkpoints to `exp/tdmpc_glass/<Env>[_<TAG>]/seed_<S>/...` (CSV columns: `step,reward,eval_type,seed` with `eval_type ∈ {pi, mppi}`). The optional `<TAG>` suffix comes from the `TDMPC_GLASS_OUTPUT_TAG` env var and is how experimental phases (`phase1b`, `phase1c`, …) keep their outputs separate without overwriting.

2. **Hydra skeleton — `src/helios/main.py` + `configs/`** (older, partly aspirational).
   - `python -m helios.main agent=ppo …` form from `README.md`.
   - `configs/experiment/default.yaml` is minimal (`env_id`, `num_envs`, `num_steps`, `total_timesteps`) and does not match the agent configs in `configs/agent/*.yaml`.
   - Treat this path as not-load-bearing unless a task explicitly asks for it.

## Running things

The benchmark expects `mujoco_playground` as an **out-of-tree checkout**, not a pip-installed package. Set `PYTHONPATH` to include both `src/` and the playground repo. On the workstation it lives at `/workspace/wiki/learn_mujoco_playground/repo`; on remote it is `/root/mujoco_playground_repo`. `run_benchmark.py` itself inserts the workstation path at import time (`sys.path.insert(0, parents[2] / "wiki/learn_mujoco_playground/repo")`), so when running from a different machine layout you usually need to override via `PYTHONPATH=`.

```bash
# Workstation single run
PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
python3 scripts/run_benchmark.py --algos tdmpc-glass --tasks HopperHop \
    --total_steps 250000 --seed 42 --no_plot

# Remote multi-seed Glass queue (sets PYTHONPATH/MUJOCO_GL=egl/venv internally)
nohup setsid bash scripts/run_phase1c_remote.sh \
    > exp/tdmpc_glass/logs/phase1c/queue.log 2>&1 < /dev/null & disown

# Render a Glass policy rollout to MP4 (needs MUJOCO_GL=egl, ffmpeg)
python3 scripts/render_glass_rollout.py --ckpt <best_mppi.pkl> \
    --env_id HopperHop --out out.mp4 --camera cam0
```

Install: pick the requirements file that matches the GPU — `requirements-rtx3090.txt` (CUDA 12 JAX wheels) or `requirements-rtx50series.txt` (CUDA 13 JAX wheels, driver ≥ 580). Then `pip install -e .`. `pyproject.toml`'s declared deps are out of date relative to these files; the requirements files are authoritative.

Memory-fraction guidance (from `docs/tdmpc-glass/env_setup.md`): 0.85 on RTX 3090, 0.55–0.65 on 8 GB RTX 50-series.

There is no `tests/` directory. `pyproject.toml` configures pytest to look there but nothing currently lives there; `scripts/test_*.py` and `scripts/smoke_test_coder.py` are manually-run smoke runners, not pytest tests. Lint/typecheck (`ruff check src/`, `black --check src/`, `mypy src/`) are configured but not part of any required workflow.

## Algorithm code in `src/helios/algorithms/`

Each algo file is a self-contained **versioned milestone**, not an abstract base implementation. The module-level docstring records the milestone reward, architecture choices, and key fixes vs earlier attempts. Treat those docstrings as the spec — when changing an algo, update the docstring to track the new milestone.

- `ppo.py` — v34s3 Brax-exact. PolicyNet 4×Dense(32)+swish, ValueNet 5×Dense(256)+swish, NormalTanh head. Stable on CheetahRun (904.5 @ 74M).
- `sac.py` — v1, custom on top of `brax.training.replay_buffers.UniformSamplingQueue` (GPU-resident replay, no host transfers). Three separately-`jit`-ed update fns (critic/actor/alpha) to avoid retracing.
- `tdmpc2.py` — v24 milestone. SimNorm(V=8) encoder/dynamics, two-hot distributional Q/reward with symlog, MPPI planner with elite selection + pi-trajectory seeding, RunningScale (IQR EMA τ=0.01, clipped [1.0, 4.0]), `lax.scan`-fused K updates per dispatch.
- `tdmpc_glass.py` — TD-MPC2 + prototype-clustering structural-entropy auxiliary loss. The Glass-specific knobs are exposed as `--glass_*` CLI flags on `run_benchmark.py` (proto_temperature, lambda_se, lambda_balance, lambda_temporal, num_prototypes, stopgrad_graph, assign_logits_init_scale, …) and as `glass_overrides` into `train_tdmpc2(..., use_glass=True, ...)`.
- `tdmpc2_patched.py`, `tdmpc.py`, `dreamer.py` — auxiliary / older variants.

Common JAX conventions across these files: pure functions returning `(new_state, metrics)`, params/optimizer state passed explicitly, networks compiled with `jax.jit` and rolled out with `jax.lax.scan` / `jax.vmap`. Single-environment effective `sps≈700` after JIT warm-up on a 4070 Ti for TD-MPC-Glass HopperHop; expect ~140 s JIT compile.

## Conventions that matter

- **Output tagging**: when running a new experimental phase, set `TDMPC_GLASS_OUTPUT_TAG=<name>` so outputs land in `exp/tdmpc_glass/<Env>_<name>/...` and `exp/tdmpc_glass/glass_diag/<Env>_<name>/...` instead of clobbering the untagged baseline.
- **PPO recovery from divergence**: keep optimizer state across the recovery — resetting it produces an effective lr×100 spike. Use `eps=1e-5` for Adam (1e-8 is faster early but unstable). See `AGENT_HANDOFF_CONTEXT.md` §7 memory dump for the falsification record.
- **Glass `stopgrad_graph=false` is not a fix** — Phase 2 falsified it. Don't propose it again.
- **MuJoCo rendering**: `MUJOCO_GL=egl` for headless; otherwise rendering crashes with "platform library not loaded". HopperHop needs `--camera cam0` (or `back`) — the default free camera doesn't track the body.
- **`scripts/run_benchmark.py` is the only training driver that matters.** The dozens of `train_tdmpc_hopper_v3.py`…`v24.py`, `run_sac_*.py`, `run_cartpole*.py` etc. in `scripts/` are historical iteration artifacts. The lessons from them are encoded as the current milestones in `src/helios/algorithms/`.
- **Root-level one-off scripts** `fix_cleanrl.py` and `run_eval.py` hardcode `/workspace/...` paths from an older host layout — they're patches, not part of the workflow.
