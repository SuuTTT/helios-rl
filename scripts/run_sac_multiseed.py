#!/usr/bin/env python3
"""
Multi-seed SAC runner for DMC Suite envs.

Usage:
  # Run 5 seeds on BallInCup (the bimodal PPO env):
  PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
    python3 scripts/run_sac_multiseed.py --env_id BallInCup

  # Multiple envs:
  python3 scripts/run_sac_multiseed.py --env_id BallInCup CartpoleSwingupSparse

  # Override steps:
  python3 scripts/run_sac_multiseed.py --env_id FingerSpin --total_timesteps 10000000

Output CSV: exp/sac/csv/sac_{env}.csv
Log files:  runs/sac_{env}_s{seed}.log
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path("/workspace")
HELIOS    = WORKSPACE / "helios-rl"
SCRIPT    = HELIOS / "scripts" / "run_sac_mjx.py"
LOG_DIR   = WORKSPACE / "runs"
CSV_DIR   = HELIOS / "exp" / "sac" / "csv"
PYTHONPATH = str(WORKSPACE / "wiki/learn_mujoco_playground/repo")

DEFAULT_SEEDS = [1, 2, 3, 4, 5]

# Envs where SAC is most needed (bimodal/hard for PPO)
PRIORITY_ENVS = [
    "BallInCup",
    "CartpoleSwingupSparse",
    "HopperStand",
    "FingerSpin",
]

ALL_ENVS = [
    "BallInCup",
    "CartpoleSwingupSparse",
    "HopperStand",
    "FingerSpin",
    "CheetahRun",
    "CartpoleSwingup",
    "FishSwim",
    "AcrobotSwingup",
]


def env_tag(env_id):
    return env_id.lower().replace(" ", "")


def run_seed(env_id, seed, total_timesteps, extra_args=None):
    csv_path = CSV_DIR / f"sac_{env_tag(env_id)}.csv"
    log_path = LOG_DIR / f"sac_{env_id}_s{seed}.log"

    cmd = [
        sys.executable, str(SCRIPT),
        "--env_id",    env_id,
        "--seed",      str(seed),
        "--csv_log",   str(csv_path),
        "--exp_name",  f"sac_{env_tag(env_id)}_s{seed}",
    ]
    if total_timesteps:
        cmd += ["--total_timesteps", str(total_timesteps)]
    if extra_args:
        cmd += extra_args

    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHONPATH

    print(f"\n{'='*64}")
    print(f"[SAC] {env_id} seed={seed}  log -> {log_path}")
    print(f"{'='*64}")
    t0 = time.time()
    with open(log_path, "w") as fout:
        proc = subprocess.run(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[SAC] {env_id} seed={seed} done in {elapsed:.0f}s (exit={proc.returncode})")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", nargs="+", default=["BallInCup"],
                    help="Environment(s) to run")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--total_timesteps", type=int, default=0,
                    help="Override total_timesteps (0 = reference config)")
    ap.add_argument("--all_envs", action="store_true",
                    help="Run all 8 DMC Suite envs")
    ap.add_argument("--priority", action="store_true",
                    help="Run priority (bimodal) envs only")
    args, extra = ap.parse_known_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    envs = args.env_id
    if args.all_envs:
        envs = ALL_ENVS
    elif args.priority:
        envs = PRIORITY_ENVS

    for env_id in envs:
        for seed in args.seeds:
            rc = run_seed(env_id, seed, args.total_timesteps, extra)
            if rc != 0:
                print(f"WARNING: {env_id} seed={seed} exit={rc}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
