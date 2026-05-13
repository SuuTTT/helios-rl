#!/usr/bin/env python3
"""Plot TD-MPC-Glass seed CI against official TD-MPC2 results.

This is intentionally narrow for the TD-MPC-Glass milestone workflow:

    python3 scripts/plot_tdmpc_glass_ci.py \
      --task HopperHop \
      --glass_dir exp/tdmpc_glass/HopperHop \
      --official_csv /workspace/tdmpc2/results/tdmpc2/hopper-hop.csv

TD-MPC-Glass seed files use:
    step,reward,eval_type,seed

Official TD-MPC2 files use:
    step,reward,seed
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


COLORS = {
    "tdmpc_glass_pi": "#1f77b4",
    "tdmpc_glass_mppi": "#d62728",
    "official_tdmpc2": "#2ca02c",
}

LABELS = {
    "tdmpc_glass_pi": "TD-MPC-Glass pi",
    "tdmpc_glass_mppi": "TD-MPC-Glass MPPI",
    "official_tdmpc2": "Official TD-MPC2",
}


def mean_ci(df: pd.DataFrame, n_grid: int, step_min: float, step_max: float):
    """Interpolate per seed to a shared grid and return mean and 95% CI."""
    grid = np.linspace(step_min, step_max, n_grid)
    curves = []
    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed].sort_values("step")
        if sub.empty:
            continue
        curves.append(np.interp(grid, sub["step"].to_numpy(), sub["reward"].to_numpy()))

    curves = np.asarray(curves, dtype=np.float64)
    mean = curves.mean(axis=0)
    if curves.shape[0] > 1:
        sem = stats.sem(curves, axis=0)
        tval = stats.t.ppf(0.975, df=curves.shape[0] - 1)
        lo = mean - tval * sem
        hi = mean + tval * sem
    else:
        lo = mean.copy()
        hi = mean.copy()
    return grid, mean, lo, hi, curves


def load_glass(glass_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(glass_dir.glob("seed_*.csv")):
        df = pd.read_csv(path)
        required = {"step", "reward", "eval_type", "seed"}
        if df.empty or not required.issubset(df.columns):
            print(f"Skipping {path}: missing required columns")
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No valid seed_*.csv files found in {glass_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize(label: str, df: pd.DataFrame) -> dict:
    finals = []
    bests = []
    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed].sort_values("step")
        finals.append(float(sub.iloc[-1]["reward"]))
        bests.append(float(sub["reward"].max()))
    finals = np.asarray(finals, dtype=np.float64)
    bests = np.asarray(bests, dtype=np.float64)
    return {
        "series": label,
        "n_seeds": int(len(finals)),
        "final_mean": float(finals.mean()),
        "final_ci95_halfwidth": ci_halfwidth(finals),
        "best_per_seed_mean": float(bests.mean()),
        "best_per_seed_ci95_halfwidth": ci_halfwidth(bests),
        "final_values": " ".join(f"{x:.1f}" for x in finals),
        "best_values": " ".join(f"{x:.1f}" for x in bests),
    }


def ci_halfwidth(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(stats.t.ppf(0.975, df=values.size - 1) * stats.sem(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="HopperHop")
    ap.add_argument("--glass_dir", type=Path, required=True)
    ap.add_argument("--official_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("exp/tdmpc_glass/plots"))
    ap.add_argument("--n_grid", type=int, default=300)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    glass = load_glass(args.glass_dir)
    official = pd.read_csv(args.official_csv)
    if not {"step", "reward", "seed"}.issubset(official.columns):
        raise ValueError(f"Official CSV missing required columns: {args.official_csv}")

    glass_pi = glass[glass["eval_type"].str.lower() == "pi"].copy()
    glass_mppi = glass[glass["eval_type"].str.lower() == "mppi"].copy()
    series = {
        "tdmpc_glass_pi": glass_pi,
        "tdmpc_glass_mppi": glass_mppi,
        "official_tdmpc2": official,
    }

    step_min = min(float(df["step"].min()) for df in series.values() if not df.empty)
    step_max = min(float(df["step"].max()) for df in series.values() if not df.empty)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=args.dpi)
    curve_rows = []
    summary_rows = []

    for key, df in series.items():
        grid, mean, lo, hi, _ = mean_ci(df, args.n_grid, step_min, step_max)
        x = grid / 1e6
        ax.fill_between(x, lo, hi, color=COLORS[key], alpha=0.14, linewidth=0)
        ax.plot(x, mean, color=COLORS[key], lw=2.0, label=LABELS[key])
        best_idx = int(np.argmax(mean))
        ax.annotate(
            f"{mean[best_idx]:.0f}",
            xy=(x[best_idx], mean[best_idx]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[key],
            ha="center",
        )
        for s, m, l, h in zip(grid, mean, lo, hi):
            curve_rows.append(
                {"series": key, "step": int(round(s)), "mean": m, "ci95_lo": l, "ci95_hi": h}
            )
        summary_rows.append(summarize(key, df))

    ax.set_title(f"{args.task}: TD-MPC-Glass vs Official TD-MPC2", fontsize=12, pad=8)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Episode Reward")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88, edgecolor="#cccccc")
    fig.tight_layout()

    slug = args.task.lower().replace("_", "-")
    png_path = args.out_dir / f"{slug}_tdmpc_glass_vs_official_95ci.png"
    curve_path = args.out_dir / f"{slug}_tdmpc_glass_vs_official_95ci_curve.csv"
    summary_path = args.out_dir / f"{slug}_tdmpc_glass_vs_official_95ci_summary.csv"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    print(f"Saved plot: {png_path}")
    print(f"Saved curve: {curve_path}")
    print(f"Saved summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

