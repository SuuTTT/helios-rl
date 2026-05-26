#!/usr/bin/env python3
"""Find low-cost Vast.ai GPU offers for TD-MPC/Glass probes.

The script intentionally does not store or accept API keys directly. Configure
the Vast CLI once with `vastai set api-key ...`, then run this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

DEFAULT_EXCLUDED_OFFER_IDS = {
    # 2026-05-26: good paper specs but SSH route repeatedly timed out during
    # env setup / rsync, making it unsuitable for queue workers.
    "34624617",  # created as contract 37907664, ssh4.vast.ai:27665
}


def money(v: Any) -> str:
    return "-" if v is None else f"{float(v):.4f}"


def one(v: Any) -> str:
    return "-" if v is None else f"{float(v):.1f}"


def build_query(args: argparse.Namespace) -> str:
    parts = [
        "verified=true",
        "rentable=true",
        "rented=false",
        f"dph < {args.max_dph}",
        f"dlperf_usd > {args.min_dlperf_usd}",
        f"gpu_ram >= {args.min_gpu_ram}",
        f"reliability > {args.min_reliability}",
        f"direct_port_count >= {args.min_ports}",
    ]
    if args.min_cuda:
        parts.append(f"cuda_vers >= {args.min_cuda}")
    if args.num_gpus:
        parts.append(f"num_gpus == {args.num_gpus}")
    if args.gpu:
        gpu_names = ",".join(f'"{name.strip()}"' for name in args.gpu.split(","))
        parts.append(f"gpu_name in [{gpu_names}]")
    if args.exclude_geolocation:
        locs = ",".join(loc.strip() for loc in args.exclude_geolocation.split(",") if loc.strip())
        if locs:
            parts.append(f"geolocation notin [{locs}]")
    return " ".join(parts)


def search(args: argparse.Namespace) -> list[dict[str, Any]]:
    cmd = [
        "vastai",
        "search",
        "offers",
        build_query(args),
        "--raw",
        "--storage",
        str(args.storage),
        "--limit",
        str(args.limit),
        "-o",
        args.order,
    ]
    out = subprocess.check_output(cmd, text=True)
    rows = json.loads(out)
    # Vast query fields can differ from total-cost fields after storage is
    # included, so enforce the bar again client-side.
    filtered = []
    for row in rows:
        if str(row.get("id")) in args.exclude_offer_ids:
            continue
        dph = row.get("dph_total", row.get("dph"))
        dlusd = row.get("dlperf_per_dphtotal", row.get("dlperf_usd"))
        if dph is None or dlusd is None:
            continue
        if float(dph) < args.max_dph and float(dlusd) > args.min_dlperf_usd:
            filtered.append(row)
    return filtered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-dph", type=float, default=0.10, help="maximum total $/hour")
    ap.add_argument("--min-dlperf-usd", type=float, default=200.0, help="minimum DLPerf/$")
    ap.add_argument("--min-gpu-ram", type=float, default=8.0, help="minimum GPU RAM in GB")
    ap.add_argument("--min-reliability", type=float, default=0.95)
    ap.add_argument("--min-ports", type=int, default=2)
    ap.add_argument("--min-cuda", type=float, default=13.0)
    ap.add_argument("--storage", type=float, default=50.0, help="storage GB used for pricing")
    ap.add_argument("--num-gpus", type=int, default=None)
    ap.add_argument("--gpu", default="", help='comma list, e.g. "RTX 3070,RTX 3080"')
    ap.add_argument(
        "--exclude-offer-id",
        action="append",
        default=[],
        help="offer ID to exclude; can be repeated",
    )
    ap.add_argument(
        "--exclude-geolocation",
        default="",
        help="comma list of Vast geolocation tokens to exclude in the query, e.g. CN,VN",
    )
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--order", default="dlperf_usd-,dph")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()
    args.exclude_offer_ids = set(args.exclude_offer_id) | DEFAULT_EXCLUDED_OFFER_IDS

    rows = search(args)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"Query: `{build_query(args)}`")
    print()
    print("| Offer | GPU | GPUs | $/h | DLPerf | DLPerf/$ | RAM GB | Rel | Location | Net down/up |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        dph = r.get("dph_total", r.get("dph"))
        dlusd = r.get("dlperf_per_dphtotal", r.get("dlperf_usd"))
        loc = r.get("geolocation") or r.get("country_code") or "-"
        net = f"{one(r.get('inet_down'))}/{one(r.get('inet_up'))}"
        print(
            f"| {r.get('id')} | {r.get('gpu_name')} | {r.get('num_gpus')} | "
            f"{money(dph)} | {one(r.get('dlperf'))} | {one(dlusd)} | "
            f"{one(r.get('gpu_ram'))} | {one(r.get('reliability'))} | {loc} | {net} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
