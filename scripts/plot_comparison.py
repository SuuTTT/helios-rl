#!/usr/bin/env python3
"""Multi-algorithm comparison plot for helios benchmark results.

Reads  helios-rl/exp/benchmark/<algo>_<task>.csv files (task,seed,step,reward),
then plots all algorithms on the same axes per task.

Usage:
    python3 helios-rl/scripts/plot_comparison.py \\
        --exp_dir  helios-rl/exp/benchmark/ \\
        --out_dir  helios-rl/exp/benchmark/plots/

The script auto-discovers all <algo>_<task>.csv files in exp_dir.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Colour palette (algo → colour)
ALGO_COLORS = {
    "ppo":    "#1f6dbf",   # steel blue
    "sac":    "#e07b39",   # burnt orange
    "tdmpc2": "#2ca02c",   # forest green
}
ALGO_LABELS = {
    "ppo":    "PPO (v34s3)",
    "sac":    "SAC (custom v1)",
    "tdmpc2": "TD-MPC2 (v24)",
}

N_GRID = 200   # interpolation points per curve


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth_ewa(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


def build_curve(df: pd.DataFrame, n_grid: int = N_GRID, smooth: float = 1.0):
    """Mean ± 95% CI over seeds, interpolated to uniform step grid."""
    seeds    = sorted(df["seed"].unique())
    step_max = df["step"].max()
    step_min = df["step"].min()
    grid     = np.linspace(step_min, step_max, n_grid)
    curves   = []
    for s in seeds:
        sub = df[df["seed"] == s].sort_values("step")
        interp = np.interp(grid, sub["step"].values, sub["reward"].values)
        if smooth < 1.0:
            interp = smooth_ewa(interp, smooth)
        curves.append(interp)
    curves = np.array(curves)
    n = curves.shape[0]
    mean = np.mean(curves, axis=0)
    if n > 1:
        se = stats.sem(curves, axis=0)
        t  = stats.t.ppf(0.975, df=max(n - 1, 1))
        lo, hi = mean - t * se, mean + t * se
    else:
        lo, hi = mean, mean
    return grid, mean, lo, hi


def _decorate_ax(ax, title):
    ax.set_title(title, fontsize=12, pad=5)
    ax.set_xlabel("Steps (M)", fontsize=10)
    ax.set_ylabel("Episode Reward", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover(exp_dir: Path):
    """
    Returns dict:  task_name → { algo_name → DataFrame }
    Discovers files matching  <algo>_<task>.csv
    """
    data = {}
    pattern = re.compile(r"^([a-z0-9]+)_(.+)\.csv$", re.IGNORECASE)
    for f in sorted(exp_dir.glob("*.csv")):
        m = pattern.match(f.name)
        if not m:
            continue
        algo, task = m.group(1).lower(), m.group(2)
        df = pd.read_csv(f)
        if df.empty or not {"task", "seed", "step", "reward"}.issubset(df.columns):
            print(f"  Skipping {f.name}: missing columns or empty")
            continue
        data.setdefault(task, {})[algo] = df
    return data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-algo comparison plot")
    ap.add_argument("--exp_dir",  required=True,
                    help="Directory containing <algo>_<task>.csv files")
    ap.add_argument("--out_dir",  default=None,
                    help="Output directory (default: exp_dir/plots)")
    ap.add_argument("--smooth",   type=float, default=0.6,
                    help="EWA smoothing factor (1=off, 0.3=heavy, default: 0.6)")
    ap.add_argument("--ncols",    type=int,   default=3,
                    help="Columns in grid plot (default: 3)")
    ap.add_argument("--dpi",      type=int,   default=150)
    ap.add_argument("--n_grid",   type=int,   default=N_GRID)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = discover(exp_dir)
    if not data:
        print(f"No <algo>_<task>.csv files found in {exp_dir}")
        return

    tasks = sorted(data.keys())
    print(f"Found {len(tasks)} task(s): {tasks}")

    # ── Per-task plots ──────────────────────────────────────────────────────
    for task in tasks:
        algo_data = data[task]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=args.dpi)

        for algo in sorted(algo_data.keys()):
            df     = algo_data[algo]
            color  = ALGO_COLORS.get(algo, "#888888")
            label  = ALGO_LABELS.get(algo, algo.upper())
            grid, mean, lo, hi = build_curve(df, n_grid=args.n_grid, smooth=args.smooth)
            x = grid / 1e6

            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, lw=2.0, label=label)
            # Mark best achieved
            best = mean.max()
            ax.annotate(f"{best:.0f}", xy=(x[mean.argmax()], best),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=7, color=color, ha="center")

        _decorate_ax(ax, task)
        ax.legend(fontsize=8.5, loc="upper left",
                  framealpha=0.85, edgecolor="#cccccc")

        per_env_path = out_dir / f"{task}_comparison.png"
        fig.tight_layout()
        fig.savefig(per_env_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {per_env_path}")

    # ── Grid plot (all tasks in one figure) ──────────────────────────────────
    ncols = min(args.ncols, len(tasks))
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 4),
                             dpi=args.dpi)
    axes = np.array(axes).flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        algo_data = data[task]

        for algo in sorted(algo_data.keys()):
            df     = algo_data[algo]
            color  = ALGO_COLORS.get(algo, "#888888")
            label  = ALGO_LABELS.get(algo, algo.upper())
            grid, mean, lo, hi = build_curve(df, n_grid=args.n_grid, smooth=args.smooth)
            x = grid / 1e6
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, lw=2.0, label=label)

        _decorate_ax(ax, task)
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper left",
                      framealpha=0.85, edgecolor="#cccccc")

    # Hide empty subplots
    for j in range(len(tasks), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Helios Benchmark: PPO vs SAC vs TD-MPC2",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    grid_path = out_dir / "all_comparison.png"
    fig.savefig(grid_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {grid_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Task':<22} {'Algo':<14} {'MaxReward':>10}  {'@Step':>10}")
    print("-" * 60)
    for task in tasks:
        for algo in sorted(data[task].keys()):
            df = data[task][algo]
            best_row = df.loc[df["reward"].idxmax()]
            print(f"  {task:<20} {algo:<14} {best_row['reward']:>10.1f}"
                  f"  {int(best_row['step']):>10,}")


if __name__ == "__main__":
    main()
