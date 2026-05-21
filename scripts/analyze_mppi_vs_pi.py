#!/usr/bin/env python3
"""Analyze paired HopperHop pi vs MPPI eval rows under exp/tdmpc_glass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


SEED_RE = re.compile(r"seed_(\d+)\.csv$")


@dataclass(frozen=True)
class EvalPair:
    path: Path
    source: str
    phase: str
    seed: int
    step: int
    pi: float
    mppi: float

    @property
    def delta(self) -> float:
        return self.mppi - self.pi


def source_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] == "remote_mirror":
        return f"remote:{rel.parts[1]}" if len(rel.parts) > 1 else "remote"
    return "local"


def iter_csvs(root: Path, include_snapshots: bool) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("seed_*.csv"):
        rel_parts = path.relative_to(root).parts
        if not include_snapshots and "_final_snapshot" in rel_parts:
            continue
        if not SEED_RE.match(path.name):
            continue
        if "HopperHop" not in path.parent.name:
            continue
        out.append(path)
    return sorted(out)


def dedupe_csvs(paths: list[Path]) -> list[Path]:
    """Drop exact duplicate CSV contents, preferring shorter/local paths."""
    best_by_hash: dict[str, Path] = {}
    for path in paths:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        current = best_by_hash.get(digest)
        if current is None:
            best_by_hash[digest] = path
            continue
        current_key = ("remote_mirror" in current.parts, len(str(current)), str(current))
        path_key = ("remote_mirror" in path.parts, len(str(path)), str(path))
        if path_key < current_key:
            best_by_hash[digest] = path
    return sorted(best_by_hash.values())


def read_pairs(path: Path, root: Path) -> list[EvalPair]:
    match = SEED_RE.match(path.name)
    if not match:
        return []
    seed = int(match.group(1))
    rows: dict[int, dict[str, float]] = defaultdict(dict)
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if not {"step", "reward", "eval_type"}.issubset(reader.fieldnames or []):
                return []
            for row in reader:
                eval_type = (row.get("eval_type") or "").strip().lower()
                if eval_type not in {"pi", "mppi"}:
                    continue
                try:
                    step = int(float(row["step"]))
                    reward = float(row["reward"])
                except (TypeError, ValueError):
                    continue
                rows[step][eval_type] = reward
    except OSError:
        return []

    phase = path.parent.name.removeprefix("HopperHop_")
    if phase == "HopperHop":
        phase = "default"
    source = source_for(path, root)
    pairs = []
    for step, vals in sorted(rows.items()):
        if "pi" in vals and "mppi" in vals:
            pairs.append(
                EvalPair(
                    path=path,
                    source=source,
                    phase=phase,
                    seed=seed,
                    step=step,
                    pi=vals["pi"],
                    mppi=vals["mppi"],
                )
            )
    return pairs


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def summarize_pairs(pairs: list[EvalPair]) -> dict[str, float]:
    if not pairs:
        return {}
    deltas = [p.delta for p in pairs]
    return {
        "pairs": len(pairs),
        "mppi_lt_pi": sum(p.delta < 0 for p in pairs),
        "mppi_eq_pi": sum(p.delta == 0 for p in pairs),
        "mppi_gt_pi": sum(p.delta > 0 for p in pairs),
        "mppi_lt_pi_pct": pct(sum(p.delta < 0 for p in pairs), len(pairs)),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "mean_pi": mean(p.pi for p in pairs),
        "mean_mppi": mean(p.mppi for p in pairs),
    }


def fmt(x: float) -> str:
    if isinstance(x, int):
        return str(x)
    if math.isfinite(x):
        return f"{x:.1f}"
    return str(x)


def run_key(pair: EvalPair) -> tuple[str, str, int, str]:
    return (pair.source, pair.phase, pair.seed, str(pair.path))


def markdown_report(root: Path, discovered_csvs: list[Path], csvs: list[Path], pairs: list[EvalPair]) -> str:
    summary = summarize_pairs(pairs)
    by_phase: dict[str, list[EvalPair]] = defaultdict(list)
    by_source: dict[str, list[EvalPair]] = defaultdict(list)
    by_run: dict[tuple[str, str, int, str], list[EvalPair]] = defaultdict(list)
    for pair in pairs:
        by_phase[pair.phase].append(pair)
        by_source[pair.source].append(pair)
        by_run[run_key(pair)].append(pair)

    run_summaries = []
    for key, run_pairs in by_run.items():
        best_pi = max(p.pi for p in run_pairs)
        best_mppi = max(p.mppi for p in run_pairs)
        best_pi_pair = max(run_pairs, key=lambda p: p.pi)
        best_mppi_pair = max(run_pairs, key=lambda p: p.mppi)
        final_pair = max(run_pairs, key=lambda p: p.step)
        run_summaries.append(
            {
                "source": key[0],
                "phase": key[1],
                "seed": key[2],
                "path": key[3],
                "pairs": len(run_pairs),
                "under_pct": pct(sum(p.delta < 0 for p in run_pairs), len(run_pairs)),
                "mean_delta": mean(p.delta for p in run_pairs),
                "best_pi": best_pi,
                "best_pi_step": best_pi_pair.step,
                "best_mppi": best_mppi,
                "best_mppi_step": best_mppi_pair.step,
                "best_delta": best_mppi - best_pi,
                "final_pi": final_pair.pi,
                "final_mppi": final_pair.mppi,
                "final_delta": final_pair.delta,
            }
        )

    lines = [
        "# MPPI vs pi evaluation analysis",
        "",
        "Generated by `scripts/analyze_mppi_vs_pi.py`.",
        "",
        "## Dataset",
        "",
        f"- Root: `{root}`",
        f"- Discovered canonical HopperHop result CSVs: {len(discovered_csvs)}",
        f"- Unique CSV contents analyzed: {len(csvs)}",
        f"- Paired eval points with both `pi` and `mppi`: {len(pairs)}",
        "- Excluded: `_diag.csv`, backup/versioned seed CSVs, `_final_snapshot` archives, and exact duplicate CSV contents from mirrors.",
        "",
        "## Headline statistics",
        "",
        f"- MPPI < pi at the same eval step: {summary.get('mppi_lt_pi', 0):.0f} / {summary.get('pairs', 0):.0f} ({summary.get('mppi_lt_pi_pct', 0):.1f}%)",
        f"- MPPI > pi at the same eval step: {summary.get('mppi_gt_pi', 0):.0f} / {summary.get('pairs', 0):.0f}",
        f"- MPPI = pi at the same eval step: {summary.get('mppi_eq_pi', 0):.0f} / {summary.get('pairs', 0):.0f}",
        f"- Mean delta `mppi - pi`: {summary.get('mean_delta', 0):.1f}",
        f"- Median delta `mppi - pi`: {summary.get('median_delta', 0):.1f}",
        f"- Worst step delta: {summary.get('min_delta', 0):.1f}",
        f"- Best step delta: {summary.get('max_delta', 0):.1f}",
        "",
        "## By pi reward band",
        "",
        "| pi band | pairs | MPPI < pi | under % | mean delta | median delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    bands = [
        ("pi < 50", lambda p: p.pi < 50),
        ("50 <= pi < 200", lambda p: 50 <= p.pi < 200),
        ("200 <= pi < 400", lambda p: 200 <= p.pi < 400),
        ("400 <= pi < 500", lambda p: 400 <= p.pi < 500),
        ("pi >= 500", lambda p: p.pi >= 500),
    ]
    for label, pred in bands:
        band_pairs = [p for p in pairs if pred(p)]
        if not band_pairs:
            continue
        deltas = [p.delta for p in band_pairs]
        under = sum(d < 0 for d in deltas)
        lines.append(
            f"| {label} | {len(band_pairs)} | {under} | {pct(under, len(band_pairs)):.1f} | {mean(deltas):.1f} | {median(deltas):.1f} |"
        )

    lines += [
        "",
        "## By source",
        "",
        "| source | pairs | MPPI < pi | under % | mean delta | best MPPI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, source_pairs in sorted(by_source.items()):
        s = summarize_pairs(source_pairs)
        lines.append(
            f"| {source} | {s['pairs']:.0f} | {s['mppi_lt_pi']:.0f} | {s['mppi_lt_pi_pct']:.1f} | {s['mean_delta']:.1f} | {max(p.mppi for p in source_pairs):.1f} |"
        )

    lines += [
        "",
        "## Phase summary",
        "",
        "| phase | seeds | pairs | MPPI < pi % | mean delta | best pi | best MPPI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    phase_rows = []
    for phase, phase_pairs in by_phase.items():
        s = summarize_pairs(phase_pairs)
        phase_rows.append(
            (
                s["mppi_lt_pi_pct"],
                phase,
                len({(p.source, p.seed, p.path) for p in phase_pairs}),
                s["pairs"],
                s["mean_delta"],
                max(p.pi for p in phase_pairs),
                max(p.mppi for p in phase_pairs),
            )
        )
    for _, phase, seeds, n, md, best_pi, best_mppi in sorted(phase_rows, reverse=True)[:30]:
        lines.append(
            f"| {phase} | {seeds} | {n:.0f} | {_:.1f} | {md:.1f} | {best_pi:.1f} | {best_mppi:.1f} |"
        )

    lines += [
        "",
        "## Worst same-step MPPI underperformance cases",
        "",
        "| source | phase | seed | step | pi | MPPI | delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(pairs, key=lambda p: p.delta)[:20]:
        lines.append(
            f"| {p.source} | {p.phase} | {p.seed} | {p.step} | {p.pi:.1f} | {p.mppi:.1f} | {p.delta:.1f} |"
        )

    lines += [
        "",
        "## Runs where best pi beats best MPPI",
        "",
        "| source | phase | seed | best pi | best pi step | best MPPI | best MPPI step | best MPPI - best pi | under-step % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best_pi_beats = [r for r in run_summaries if r["best_mppi"] < r["best_pi"]]
    best_pi_beats.sort(key=lambda r: r["best_delta"])
    for r in best_pi_beats[:30]:
        lines.append(
            f"| {r['source']} | {r['phase']} | {r['seed']} | {r['best_pi']:.1f} | {r['best_pi_step']} | "
            f"{r['best_mppi']:.1f} | {r['best_mppi_step']} | {r['best_delta']:.1f} | {r['under_pct']:.1f} |"
        )

    high_pi_pairs = [p for p in pairs if p.pi >= 400]
    high_under = [p for p in high_pi_pairs if p.delta < 0]
    run_best_under = len(best_pi_beats)
    lines += [
        "",
        "## Interpretation",
        "",
        f"- MPPI underperforms pi at {summary.get('mppi_lt_pi_pct', 0):.1f}% of paired eval steps. This is not rare enough to treat MPPI as a uniformly better evaluator.",
        f"- The effect is more important at high policy quality: among evals with `pi >= 400`, MPPI is lower in {len(high_under)} / {len(high_pi_pairs)} cases ({pct(len(high_under), len(high_pi_pairs)):.1f}%).",
        f"- At the run level, best pi beats best MPPI in {run_best_under} / {len(run_summaries)} runs ({pct(run_best_under, len(run_summaries)):.1f}%). This means selecting checkpoints only by `best_mppi.pkl` can miss policies whose deterministic actor is already better than the planner.",
        "- Likely mechanism: MPPI plans through the learned latent dynamics/reward model. When the actor has learned a coherent gait but the short-horizon model or reward head is locally miscalibrated, MPPI can search into model-favored but real-environment-bad actions. This is especially plausible on HopperHop because small contact-timing errors flip foot-hop into knee-walk/fall outcomes.",
        "- MPPI also uses finite samples and horizon `H=3`; with noisy value estimates, the planner can be worse than the actor's direct action even though it evaluates more candidates.",
        "",
        "## Recommendations",
        "",
        "- Track `best_pi.pkl` alongside `best_mppi.pkl` and render both when they disagree by more than 50 reward.",
        "- For dashboard success criteria, show both `best_pi` and `best_mppi`; consider a run solved if either evaluator is >= 500, then verify by video/original reward rollout.",
        "- Add a `pi_minus_mppi` warning to the dashboard for evals where `pi - mppi >= 100`; those are model/planner mismatch cases worth inspecting.",
        "- For future fair sweeps, report three metrics: best pi, best MPPI, and best of either. MPPI-only reporting is too pessimistic for some seeds and can bias algorithm decisions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("exp/tdmpc_glass"))
    parser.add_argument("--output", type=Path, default=Path("docs/tdmpc-glass/mppi_vs_pi_analysis.md"))
    parser.add_argument("--include-snapshots", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    discovered_csvs = iter_csvs(root, include_snapshots=args.include_snapshots)
    csvs = dedupe_csvs(discovered_csvs)
    pairs: list[EvalPair] = []
    for path in csvs:
        pairs.extend(read_pairs(path, root))
    report = markdown_report(root, discovered_csvs, csvs, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    summary = summarize_pairs(pairs)
    print(f"csvs={len(csvs)} discovered_csvs={len(discovered_csvs)} pairs={len(pairs)}")
    print(
        "mppi_lt_pi="
        f"{summary.get('mppi_lt_pi', 0):.0f}/{summary.get('pairs', 0):.0f} "
        f"({summary.get('mppi_lt_pi_pct', 0):.1f}%) "
        f"mean_delta={summary.get('mean_delta', 0):.1f}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
