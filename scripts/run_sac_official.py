#!/usr/bin/env python3
"""
Run official Brax SAC on MuJoCo Playground DMC Suite envs.
Uses brax.training.agents.sac.train — fully XLA, zero Python per step.

Output CSV: helios-rl/exp/sac/csv/sac_{env}.csv  (task,seed,step,reward)
"""

import argparse
import csv
import functools
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

# JAX 0.10.0 removed jax.device_put_replicated; brax still calls it.
# Patch it back: replicate by stacking along a new leading axis (1 per device).
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(val, devices):
        n = len(devices)
        return jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * n), val
        )
    jax.device_put_replicated = _device_put_replicated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "wiki/learn_mujoco_playground/repo"))
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params

PRIORITY_ENVS = ["BallInCup", "CartpoleSwingupSparse", "HopperStand", "FingerSpin"]


def run_sac(env_id: str, seed: int, csv_path: Path, total_timesteps: int | None = None):
    print(f"\n{'='*60}")
    print(f"  SAC (official Brax) | env={env_id} | seed={seed}")
    print(f"{'='*60}", flush=True)

    env = registry.load(env_id)
    sac_params = dm_control_suite_params.brax_sac_config(env_id)

    sac_training_params = dict(sac_params)
    if total_timesteps is not None:
        sac_training_params["num_timesteps"] = total_timesteps

    # Remove keys not accepted by brax sac.train
    for unsupported in ["num_resets_per_eval"]:
        sac_training_params.pop(unsupported, None)

    # Extract network_factory if present
    network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            **sac_params.network_factory,
        )

    # CSV writer
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["task", "seed", "step", "reward"])

    times = [datetime.now()]

    def progress(num_steps, metrics):
        reward = float(metrics.get("eval/episode_reward", 0.0))
        reward_std = float(metrics.get("eval/episode_reward_std", 0.0))
        now = datetime.now()
        elapsed = (now - times[0]).total_seconds()
        sps = num_steps / elapsed if elapsed > 0 else 0
        print(
            f"  step={num_steps:>10,}  reward={reward:>8.3f}  ±{reward_std:.3f}"
            f"  sps={sps:>8.0f}  elapsed={elapsed:.0f}s",
            flush=True,
        )
        writer.writerow([env_id, seed, num_steps, reward])
        csv_file.flush()

    t0 = time.time()
    make_inference_fn, params, metrics = sac.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=seed,
        progress_fn=progress,
        network_factory=network_factory,
        **sac_training_params,
    )
    elapsed = time.time() - t0
    final_reward = float(metrics.get("eval/episode_reward", 0.0))
    print(f"\n  Done. reward={final_reward:.3f}  total_time={elapsed:.0f}s", flush=True)

    csv_file.close()
    return final_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="BallInCup", choices=PRIORITY_ENVS + [
        "AcrobotSwingup", "CartpoleBalance", "CartpoleSwingup", "CheetahRun",
        "FingerTurnEasy", "FingerTurnHard", "FishSwim", "HopperHop",
        "HumanoidStand", "HumanoidWalk", "PendulumSwingup", "ReacherEasy",
        "ReacherHard", "WalkerRun", "WalkerStand", "WalkerWalk",
    ])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override num_timesteps (default: use config)")
    parser.add_argument("--priority", action="store_true",
                        help="Run all 4 priority envs sequentially")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1],
                        help="Seeds to run (used with --priority)")
    args = parser.parse_args()

    base_csv_dir = Path(__file__).resolve().parent.parent / "exp" / "sac" / "csv"

    if args.priority:
        for env_id in PRIORITY_ENVS:
            for seed in args.seeds:
                csv_path = base_csv_dir / f"sac_{env_id.lower()}.csv"
                run_sac(env_id, seed, csv_path, args.total_timesteps)
    else:
        csv_path = base_csv_dir / f"sac_{args.env_id.lower()}.csv"
        run_sac(args.env_id, args.seed, csv_path, args.total_timesteps)


if __name__ == "__main__":
    main()
