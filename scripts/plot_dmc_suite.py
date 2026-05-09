#!/usr/bin/env python3
"""
Plot PPO DMC suite learning curves in two styles.

Style 1 — ci:     mean line + shaded 95% Student-t confidence interval
Style 2 — optic:  5 individual thin seed lines (low alpha) + bold mean line

Outputs
-------
Per-env files  : <out_dir>/<env>_ci.png   and  <env>_optic.png
All-in-one grid: <out_dir>/all_ci.png     and  all_optic.png

Input CSV format (task,seed,step,reward)
----------------------------------------
task,seed,step,reward
FingerSpin,1,983040,12.5
FingerSpin,1,1966080,34.2
...

Usage
-----
# Auto-discover all ours_*.csv in a directory:
python3 scripts/plot_dmc_suite.py --csv_dir exp/ppo/csv --out_dir exp/ppo/plots

# Explicit files with custom labels:
python3 scripts/plot_dmc_suite.py \
    --csv_files exp/ppo/csv/ours_fingerspin.csv:FingerSpin \
                exp/ppo/csv/ours_cheetahrun.csv:CheetahRun \
    --out_dir exp/ppo/plots

# Smooth noisy curves (EWA window) before plotting:
python3 scripts/plot_dmc_suite.py --csv_dir exp/ppo/csv --smooth 0.3
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Defaults ──────────────────────────────────────────────────────────────────
BLUE_DARK  = "#1f6dbf"
BLUE_LIGHT = "#90b8e0"
N_GRID     = 300

# Canonical env order for grid layout (tag -> display label)
ENV_ORDER = [
    ("cheetahrun",           "CheetahRun"),
    ("ballincup",            "BallInCup"),
    ("cartpoleswingup",      "CartpoleSwingup"),
    ("cartpoleswingupsparse","CartpoleSwingupSparse"),
    ("fingerspin",           "FingerSpin"),
    ("fishswim",             "FishSwim"),
    ("acrobotswingup",       "AcrobotSwingup"),
    ("hopperstand",          "HopperStand"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth_ewa(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential weighted average smoothing. alpha in (0,1]: 1 = no smoothing."""
    if alpha >= 1.0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


def build_curves(df: pd.DataFrame, n_grid: int = N_GRID, smooth: float = 1.0):
    """Interpolate each seed onto a common step grid. Returns (grid, curves)."""
    seeds = sorted(df["seed"].unique())
    step_max = df["step"].max()
    step_min = df["step"].min()
    grid = np.linspace(step_min, step_max, n_grid)
    curves = []
    for s in seeds:
        sub = df[df["seed"] == s].sort_values("step")
        interp = np.interp(grid, sub["step"].values, sub["reward"].values)
        if smooth < 1.0:
            interp = smooth_ewa(interp, smooth)
        curves.append(interp)
    return grid, np.array(curves)  # (n_seeds, n_grid)


def ci95(curves: np.ndarray):
    n = curves.shape[0]
    mean = np.mean(curves, axis=0)
    se   = stats.sem(curves, axis=0)
    t    = stats.t.ppf(0.975, df=max(n - 1, 1))
    return mean, mean - t * se, mean + t * se


def _decorate_ax(ax, title, xlabel="Steps (M)", ylabel="Episode Reward"):
    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_ci(ax, grid, curves, title):
    mean, lo, hi = ci95(curves)
    x = grid / 1e6
    ax.fill_between(x, lo, hi, color=BLUE_DARK, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=BLUE_DARK, lw=2.0,
            label=f"mean ± 95% CI  (n={curves.shape[0]})")
    _decorate_ax(ax, title)
    ax.legend(fontsize=7.5, loc="lower right")


def plot_optic(ax, grid, curves, title):
    x = grid / 1e6
    for c in curves:
        ax.plot(x, c, color=BLUE_LIGHT, lw=0.85, alpha=0.5)
    mean = np.mean(curves, axis=0)
    ax.plot(x, mean, color=BLUE_DARK, lw=2.2,
            label=f"mean  (n={curves.shape[0]})")
    _decorate_ax(ax, title)
    ax.legend(fontsize=7.5, loc="lower right")


# ── Discover / load CSVs ──────────────────────────────────────────────────────

def _tag_from_filename(path: str) -> str:
    """ours_fingerspin.csv  →  fingerspin"""
    stem = Path(path).stem
    stem = re.sub(r"^(ours|brax)_", "", stem)
    return stem.lower()


def _label_from_tag(tag: str) -> str:
    for t, label in ENV_ORDER:
        if t == tag:
            return label
    return tag.title()


def discover_csvs(csv_dir: str):
    """Return list of (tag, label, path) sorted by ENV_ORDER."""
    found = {}
    for f in Path(csv_dir).glob("ours_*.csv"):
        tag = _tag_from_filename(str(f))
        found[tag] = str(f)
    # sort by ENV_ORDER, append unknowns at end
    ordered = []
    for tag, label in ENV_ORDER:
        if tag in found:
            ordered.append((tag, label, found[tag]))
    for tag, path in found.items():
        if tag not in [t for t, _, _ in ordered]:
            ordered.append((tag, _label_from_tag(tag), path))
    return ordered


def parse_explicit_files(specs):
    """Parse list of 'path:label' or 'path' strings."""
    result = []
    for spec in specs:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path = spec
            tag  = _tag_from_filename(spec)
            label = _label_from_tag(tag)
        tag = _tag_from_filename(path)
        result.append((tag, label, path))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Plot PPO DMC suite learning curves")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv_dir",   help="Directory with ours_*.csv files")
    src.add_argument("--csv_files", nargs="+", metavar="PATH[:LABEL]",
                     help="Explicit CSV files, optionally with :Label suffix")
    ap.add_argument("--out_dir",  default="runs/plots/suite",
                    help="Output directory (default: runs/plots/suite)")
    ap.add_argument("--smooth",   type=float, default=1.0,
                    help="EWA smoothing factor 0–1 (1=off, 0.3=heavy). "
                         "Applied before plotting.")
    ap.add_argument("--ncols",    type=int, default=4,
                    help="Columns in the all-in-one grid (default: 4)")
    ap.add_argument("--n_grid",   type=int, default=N_GRID,
                    help="Number of interpolation points per seed (default: 300)")
    ap.add_argument("--dpi",      type=int, default=150)
    ap.add_argument("--no_per_env", action="store_true",
                    help="Skip individual per-env plots, only write grid")
    args = ap.parse_args()

    if args.csv_dir:
        envs = discover_csvs(args.csv_dir)
    else:
        envs = parse_explicit_files(args.csv_files)

    if not envs:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all data
    all_data = {}
    for tag, label, path in envs:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        grid, curves = build_curves(df, n_grid=args.n_grid, smooth=args.smooth)
        all_data[tag] = (grid, curves, label)
        n_seeds = curves.shape[0]
        bests   = curves.max(axis=1)
        print(f"  {label:<26} seeds={n_seeds}  "
              f"best mean={bests.mean():.1f}  range=[{bests.min():.1f}, {bests.max():.1f}]")

    # ── Per-env plots ─────────────────────────────────────────────────────────
    if not args.no_per_env:
        for tag, (grid, curves, label) in all_data.items():
            for style, fn in [("ci", plot_ci), ("optic", plot_optic)]:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                fn(ax, grid, curves, label)
                plt.tight_layout()
                p = out / f"{tag}_{style}.png"
                plt.savefig(p, dpi=args.dpi, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {p}")

    # ── All-in-one grid ───────────────────────────────────────────────────────
    ncols = min(args.ncols, len(all_data))
    nrows = (len(all_data) + ncols - 1) // ncols
    for style, fn in [("ci", plot_ci), ("optic", plot_optic)]:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 5.4, nrows * 4.0))
        axes = np.array(axes).flatten()
        for i, (tag, (grid, curves, label)) in enumerate(all_data.items()):
            fn(axes[i], grid, curves, label)
        for j in range(len(all_data), len(axes)):
            axes[j].set_visible(False)
        smooth_note = f" (EWA α={args.smooth})" if args.smooth < 1.0 else ""
        fig.suptitle(
            f"PPO · DMC Suite — Our Implementation{smooth_note}",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        p = out / f"all_{style}.png"
        plt.savefig(p, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Env':<26} {'Seeds':>5} {'Mean best':>10} {'Min':>8} {'Max':>8}")
    print("-" * 63)
    for tag, (grid, curves, label) in all_data.items():
        bests = curves.max(axis=1)
        print(f"{label:<26} {curves.shape[0]:>5} {bests.mean():>10.1f} "
              f"{bests.min():>8.1f} {bests.max():>8.1f}")


if __name__ == "__main__":
    main()
