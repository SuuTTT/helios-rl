#!/usr/bin/env python3
"""Plot Phase-1 TD-MPC-Glass HopperHop progress.

Generates two figures:

1. ``hopperhop_phase1_progress_95ci.png`` — per-seed MPPI return curves for
   Phase-1 alongside the per-seed official TD-MPC2 curves and a mean +/- 95% CI
   band over the completed seeds.
2. ``hopperhop_phase1_glass_diag.png`` — Glass diagnostics (cluster-mass
   entropy, max cluster mass, active clusters, transition cut mass) parsed
   from the training logs.

Inputs are the per-seed CSVs under ``exp/tdmpc_glass/HopperHop/`` and the
official baseline ``/workspace/tdmpc2/results/tdmpc2/hopper-hop.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/workspace/helios-rl")
GLASS_DIR = ROOT / "exp/tdmpc_glass/HopperHop"
LOG_DIR = ROOT / "exp/tdmpc_glass/logs/phase1"
OFFICIAL = Path("/workspace/tdmpc2/results/tdmpc2/hopper-hop.csv")
OUT_DIR = ROOT / "exp/tdmpc_glass/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_glass_mppi() -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for csv in sorted(GLASS_DIR.glob("seed_*.csv")):
        seed = int(csv.stem.split("_")[1])
        df = pd.read_csv(csv)
        if df.empty or "eval_type" not in df.columns:
            continue
        out[seed] = df[df["eval_type"] == "mppi"].sort_values("step").reset_index(drop=True)
    return out


def load_official() -> dict[int, pd.DataFrame]:
    df = pd.read_csv(OFFICIAL)
    return {
        int(s): sub.sort_values("step").reset_index(drop=True)
        for s, sub in df.groupby("seed")
    }


def mean_ci(curves: list[np.ndarray], grid: np.ndarray):
    arr = np.asarray(curves, dtype=np.float64)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = stats.sem(arr, axis=0)
        tval = stats.t.ppf(0.975, df=arr.shape[0] - 1)
        return mean, mean - tval * sem, mean + tval * sem
    return mean, mean, mean


def plot_progress():
    glass = load_glass_mppi()
    official = load_official()

    # Common grid: 0 .. 4M, 100 ticks
    grid = np.linspace(0, 4_000_000, 201)
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=170)

    # Per-seed Phase 1
    phase1_curves = []
    for seed, df in glass.items():
        x = df["step"].to_numpy()
        y = df["reward"].to_numpy()
        if x.size == 0:
            continue
        last_x = x[-1]
        completed = last_x >= 3_750_000
        ax.plot(
            x, y,
            color=plt.cm.viridis(0.15 + 0.18 * (seed - 1)),
            lw=1.4,
            alpha=0.7 if completed else 0.35,
            label=f"Phase1 seed {seed}{'' if completed else f' (in-flight {last_x/1e6:.1f}M)'}",
        )
        if completed:
            phase1_curves.append(np.interp(grid, x, y, left=np.nan, right=np.nan))

    # Per-seed official
    official_curves = []
    for seed, df in official.items():
        x = df["step"].to_numpy()
        y = df["reward"].to_numpy()
        ax.plot(x, y, color="tab:gray", lw=1.0, alpha=0.45,
                label=f"Official TD-MPC2 seed {seed}")
        official_curves.append(np.interp(grid, x, y, left=np.nan, right=np.nan))

    # Phase 1 mean +/- 95% CI (completed seeds only)
    if len(phase1_curves) >= 2:
        valid = ~np.any(np.isnan(np.stack(phase1_curves)), axis=0)
        if valid.any():
            m, lo, hi = mean_ci([c[valid] for c in phase1_curves], grid[valid])
            ax.plot(grid[valid], m, color="tab:red", lw=2.4,
                    label=f"Phase1 mean ({len(phase1_curves)} seeds)")
            ax.fill_between(grid[valid], lo, hi, color="tab:red", alpha=0.18,
                            label="Phase1 95% CI")

    # Official mean +/- 95% CI
    if len(official_curves) >= 2:
        valid = ~np.any(np.isnan(np.stack(official_curves)), axis=0)
        if valid.any():
            m, lo, hi = mean_ci([c[valid] for c in official_curves], grid[valid])
            ax.plot(grid[valid], m, color="tab:green", lw=2.2,
                    label=f"Official mean ({len(official_curves)} seeds)")
            ax.fill_between(grid[valid], lo, hi, color="tab:green", alpha=0.15,
                            label="Official 95% CI")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("MPPI return (HopperHop)")
    ax.set_title("TD-MPC-Glass Phase 1 vs Official TD-MPC2 — HopperHop, 4M steps")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 4_000_000)
    ax.set_ylim(bottom=-10)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()

    out = OUT_DIR / "hopperhop_phase1_progress_95ci.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return out


def parse_glass_diag(log_path: Path) -> pd.DataFrame:
    """Pull `step` and Glass diagnostics out of a training log."""
    rows = []
    step = None
    step_re = re.compile(r"\s*step=\s*([\d,]+)\s+pi_reward=")
    diag_re = re.compile(
        r"glass se=([-0-9.]+) ent=([-0-9.]+) active=([\d.]+) "
        r"max_mass=([-0-9.]+) cut=([-0-9.]+)"
    )
    if not log_path.exists():
        return pd.DataFrame()
    with log_path.open() as fh:
        for ln in fh:
            m = step_re.match(ln)
            if m:
                step = int(m.group(1).replace(",", ""))
                continue
            m = diag_re.search(ln)
            if m and step is not None:
                rows.append({
                    "step": step,
                    "se": float(m.group(1)),
                    "ent": float(m.group(2)),
                    "active": float(m.group(3)),
                    "max_mass": float(m.group(4)),
                    "cut": float(m.group(5)),
                })
                step = None
    return pd.DataFrame(rows)


def plot_glass_diag():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), dpi=170, sharex=True)
    metrics = [
        ("ent", "cluster-mass entropy   (max = log(K) = 2.079)"),
        ("max_mass", "max cluster mass   (uniform = 1/K = 0.125)"),
        ("active", "active clusters   (out of K=8)"),
        ("cut", "transition cut mass"),
    ]
    # Phase-1 logs
    phase1_logs = {
        1: LOG_DIR / "HopperHop_seed_1.log",
        2: LOG_DIR / "HopperHop_seed_2.log",
        3: LOG_DIR / "HopperHop_seed_3_d.log",
        4: LOG_DIR / "HopperHop_seed_4.log",
    }
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for seed, log in phase1_logs.items():
            df = parse_glass_diag(log)
            if df.empty:
                continue
            ax.plot(df["step"], df[col],
                    color=plt.cm.viridis(0.15 + 0.18 * (seed - 1)),
                    lw=1.4, label=f"Phase1 seed {seed}")
        # Pre-phase1 reference: known constant uniform values from analysis.
        if col == "ent":
            ax.axhline(2.0794, ls="--", color="tab:gray", lw=1.0,
                       label="Pre-Phase1 (inert, ent=log K)")
        elif col == "max_mass":
            ax.axhline(0.1287, ls="--", color="tab:gray", lw=1.0,
                       label="Pre-Phase1 (inert, ~1/K)")
        elif col == "active":
            ax.axhline(8.0, ls="--", color="tab:gray", lw=1.0,
                       label="Pre-Phase1 (inert, all K)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("environment steps")
    fig.suptitle("Glass diagnostics — Phase 1 vs Pre-Phase 1 (inert reference)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "hopperhop_phase1_glass_diag.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return out


def plot_transition_matrix():
    """Visualise the prototype transition matrix P at end of seed 3."""
    diag_dir = Path("/workspace/helios-rl/exp/benchmark/glass_diag/HopperHop/seed_3")
    npzs = sorted(diag_dir.glob("step_*.npz"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not npzs:
        print("no glass_diag dumps for seed 3 yet, skipping matrix plot")
        return None
    last = np.load(npzs[-1])
    P, A, S = last["P"], last["A"], last["S"]
    labels = np.argmax(S, axis=-1)
    order = np.argsort(labels)
    P_sorted = P[order][:, order]
    A_sorted = A[order][:, order]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=170)
    im0 = axes[0].imshow(P_sorted, cmap="magma", vmin=0)
    axes[0].set_title(f"P (reordered by cluster) — step {npzs[-1].stem.split('_')[1]}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(A_sorted, cmap="magma", vmin=0)
    axes[1].set_title("A = (P+P^T)/2 (reordered)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(S[order], cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("S (prototype → cluster)")
    axes[2].set_xlabel("cluster")
    axes[2].set_ylabel("prototype (sorted)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    out = OUT_DIR / "hopperhop_phase1_glass_matrix_seed3.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    plot_progress()
    plot_glass_diag()
    plot_transition_matrix()
