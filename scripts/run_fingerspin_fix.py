#!/usr/bin/env python3
"""
EXP-001: FingerSpin gamma fix experiment.

Hypothesis: Our baseline used gamma=0.995 for FingerSpin, but the reference
config uses gamma=0.95. This mismatch is responsible for the 200-point gap
(408 vs ~600 in the paper).

This script runs 5 seeds for each gamma value to isolate the effect, then
generates comparison plots.

Usage (from workspace root or helios-rl root):
  PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
    python3 helios-rl/scripts/run_fingerspin_fix.py

  # Quick test (3M steps):
  python3 helios-rl/scripts/run_fingerspin_fix.py --total_timesteps 10000000

  # Only run the fix (skip baseline re-run, use existing CSV):
  python3 helios-rl/scripts/run_fingerspin_fix.py --only fix

Outputs:
  helios-rl/exp/ppo/csv/ours_fingerspin_g095.csv   (EXP-001 gamma=0.95)
  helios-rl/exp/ppo/plots/fingerspin_comparison.png
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE    = Path("/workspace")
HELIOS       = WORKSPACE / "helios-rl"
PPO_SCRIPT   = WORKSPACE / "run_ppo_continuous_mjx.py"
CSV_DIR      = HELIOS / "exp/ppo/csv"
PLOT_DIR     = HELIOS / "exp/ppo/plots"
LOG_DIR      = WORKSPACE / "runs"
PYTHONPATH   = str(WORKSPACE / "wiki/learn_mujoco_playground/repo")

SEEDS = [1, 2, 3, 4, 5]
ENV   = "FingerSpin"

COMMON_ARGS = [
    "--env-id",          ENV,
    "--num-envs",        "2048",
    "--num-steps",       "30",
    "--learning-rate",   "1e-3",
    "--update-epochs",   "16",
    "--num-minibatches", "32",
    "--ent-coef",        "0.01",
    "--clip-coef",       "0.3",
    "--max-grad-norm",   "1.0",
    "--no-anneal-lr",
    "--normalize-obs",
    "--reward-scaling",  "10.0",
    "--eval-freq",       "1",
]

VARIANTS = {
    "baseline": {
        "gamma":   "0.995",
        "csv":     CSV_DIR / "ours_fingerspin.csv",
        "label":   "gamma=0.995 (baseline, wrong)",
        "color":   "#e08030",
        "already_run": True,   # We already have this data — skip unless forced
    },
    "fix": {
        "gamma":   "0.95",
        "csv":     CSV_DIR / "ours_fingerspin_g095.csv",
        "label":   "gamma=0.95 (fix, reference value)",
        "color":   "#1f6dbf",
        "already_run": False,
    },
}


# ── Training ──────────────────────────────────────────────────────────────────

def run_seed(variant_name, seed, total_timesteps, csv_log):
    cfg = VARIANTS[variant_name]
    exp_name = f"fingerspin_{variant_name}_s{seed}"
    log_path = LOG_DIR / f"{exp_name}.log"
    cmd = [
        sys.executable, str(PPO_SCRIPT),
        *COMMON_ARGS,
        "--gamma",           cfg["gamma"],
        "--seed",            str(seed),
        "--total-timesteps", str(total_timesteps),
        "--csv-log",         str(csv_log),
        "--exp-name",        exp_name,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHONPATH

    print(f"\n{'='*64}")
    print(f"[{variant_name.upper()}] FingerSpin seed={seed} gamma={cfg['gamma']} "
          f"steps={total_timesteps:,}")
    print(f"  log -> {log_path}")
    print(f"{'='*64}")
    t0 = time.time()
    with open(log_path, "w") as fout:
        proc = subprocess.run(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[{variant_name.upper()}] seed={seed} done in {elapsed:.0f}s  "
          f"(exit={proc.returncode})")
    return proc.returncode


# ── Plotting ──────────────────────────────────────────────────────────────────

def smooth_ewa(arr, alpha=0.4):
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * arr[i]
    return out


def build_curves(csv_path, smooth=0.4):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    seeds = sorted(df["seed"].unique())
    step_max = df["step"].max()
    step_min = df["step"].min()
    grid = np.linspace(step_min, step_max, 300)
    curves = []
    for s in seeds:
        sub = df[df["seed"] == s].sort_values("step")
        interp = np.interp(grid, sub["step"].values, sub["reward"].values)
        curves.append(smooth_ewa(interp, smooth))
    return grid, np.array(curves)


def ci95(curves):
    n = curves.shape[0]
    mean = np.mean(curves, axis=0)
    se   = stats.sem(curves, axis=0)
    t    = stats.t.ppf(0.975, df=max(n - 1, 1))
    return mean, mean - t * se, mean + t * se


def make_comparison_plot(out_path):
    """Two-panel: left = CI style, right = optic style."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = ["95% CI", "Individual Seeds + Mean"]

    for ax, title in zip(axes, titles):
        for v_name, cfg in VARIANTS.items():
            if not cfg["csv"].exists():
                continue
            grid, curves = build_curves(cfg["csv"])
            x = grid / 1e6
            color = cfg["color"]
            label = cfg["label"]

            if "CI" in title:
                mean, lo, hi = ci95(curves)
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
                ax.plot(x, mean, color=color, lw=2.0, label=label)
            else:
                for c in curves:
                    ax.plot(x, c, color=color, lw=0.85, alpha=0.45)
                mean = np.mean(curves, axis=0)
                ax.plot(x, mean, color=color, lw=2.2, label=label)

        # Paper reference line
        ax.axhline(600, color="gray", lw=1.2, ls="--", alpha=0.6, label="Paper ref (~600)")
        ax.set_title(f"FingerSpin — {title}", fontsize=11)
        ax.set_xlabel("Steps (M)", fontsize=9)
        ax.set_ylabel("Episode Reward", fontsize=9)
        ax.set_ylim(0, 700)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison plot: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["baseline", "fix", "all", "plot"],
                    default="all",
                    help="Which variants to run (default: all). "
                         "'plot' skips training, regenerates plot only.")
    ap.add_argument("--total_timesteps", type=int, default=75_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.only != "plot":
        for v_name, cfg in VARIANTS.items():
            if args.only not in ("all", v_name):
                continue
            if cfg["already_run"] and v_name == "baseline":
                print(f"[{v_name}] Skipping — baseline CSV already exists at {cfg['csv']}")
                continue
            for seed in args.seeds:
                rc = run_seed(v_name, seed, args.total_timesteps, cfg["csv"])
                if rc != 0:
                    print(f"WARNING: seed={seed} exited with code {rc}")

    # Generate comparison plot
    plot_out = PLOT_DIR / "fingerspin_comparison.png"
    make_comparison_plot(plot_out)

    # Print summary
    print("\n── Summary ──────────────────────────────────────────────────────")
    for v_name, cfg in VARIANTS.items():
        if cfg["csv"].exists():
            df = pd.read_csv(cfg["csv"])
            df.columns = [c.strip() for c in df.columns]
            seeds = sorted(df["seed"].unique())
            bests = [df[df["seed"] == s]["reward"].max() for s in seeds]
            print(f"  {cfg['label']}")
            print(f"    seeds={seeds}  mean_best={np.mean(bests):.1f}  "
                  f"range=[{min(bests):.1f}, {max(bests):.1f}]")
        else:
            print(f"  {cfg['label']}  — not yet run")


if __name__ == "__main__":
    main()
