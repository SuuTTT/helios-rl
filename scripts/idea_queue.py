#!/usr/bin/env python3
"""File-backed research idea queue for TD-MPC-Glass.

This is intentionally small and conservative. It does not run an LLM by itself;
it stores ideas, agent claims, generated probe specs, and can enqueue approved
probes into the existing central experiment queue.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/helios-rl")
QUEUE_DIR = REPO / "scripts" / "queues"
IDEA_QUEUE = QUEUE_DIR / "idea_queue.json"
CENTRAL_QUEUE = QUEUE_DIR / "central_queue.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:7]}"


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def with_lock(path: Path, fn):
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = read_json(path, [])
            result = fn(data)
            if result is not None:
                write_json(path, result)
                return result
            return data
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def find_item(items: list[dict], idea_id: str) -> dict:
    for item in items:
        if item.get("id") == idea_id:
            return item
    raise SystemExit(f"idea not found: {idea_id}")


def append_event(item: dict, event: str, **fields):
    item.setdefault("events", []).append({"at": now_iso(), "event": event, **fields})
    item["updated_at"] = now_iso()


def validate_env(env: str):
    try:
        shlex.split(env or "")
    except ValueError as e:
        raise SystemExit(f"invalid env string: {e}") from e


def cmd_add(args):
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]

    def mutate(items):
        item = {
            "id": short_id("i"),
            "status": "new",
            "priority": args.priority,
            "title": args.title,
            "goal": args.goal,
            "hypothesis": args.hypothesis or "",
            "metric": args.metric or "HopperHop G1: 5/5 seeds best_any >= 500",
            "notes": args.notes or "",
            "tags": tags,
            "owner": args.owner or "",
            "claimed_by": "",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "probe_specs": [],
            "evidence": [],
            "events": [],
        }
        append_event(item, "created", by=args.owner or "")
        items.append(item)
        print(item["id"])
        return items

    with_lock(IDEA_QUEUE, mutate)


def cmd_list(args):
    items = read_json(IDEA_QUEUE, [])
    statuses = set(args.status or [])
    rows = []
    for item in items:
        if statuses and item.get("status") not in statuses:
            continue
        rows.append(item)
    rows.sort(key=lambda x: (x.get("priority", 10), x.get("created_at", "")))
    for item in rows:
        probes = item.get("probe_specs", [])
        queued = sum(1 for p in probes if p.get("central_task_id"))
        print(
            f"{item['id']} p{item.get('priority')} {item.get('status'):<12} "
            f"probes={len(probes)} queued={queued} "
            f"claim={item.get('claimed_by') or '-'} :: {item.get('title')}"
        )


def cmd_show(args):
    item = find_item(read_json(IDEA_QUEUE, []), args.idea_id)
    print(json.dumps(item, indent=2))


def cmd_claim(args):
    def mutate(items):
        item = find_item(items, args.idea_id)
        if item.get("claimed_by") and item.get("claimed_by") != args.agent and not args.force:
            raise SystemExit(f"already claimed by {item.get('claimed_by')}; use --force to steal")
        item["claimed_by"] = args.agent
        if item.get("status") == "new":
            item["status"] = "triage"
        append_event(item, "claimed", by=args.agent)
        return items

    with_lock(IDEA_QUEUE, mutate)


def cmd_status(args):
    def mutate(items):
        item = find_item(items, args.idea_id)
        old = item.get("status")
        item["status"] = args.status
        append_event(item, "status", old=old, new=args.status, by=args.by or "")
        return items

    with_lock(IDEA_QUEUE, mutate)


def cmd_add_probe(args):
    validate_env(args.env)

    def mutate(items):
        item = find_item(items, args.idea_id)
        spec = {
            "id": short_id("p"),
            "status": "designed",
            "label": args.label,
            "launcher": args.launcher,
            "env": args.env,
            "priority": args.priority,
            "pass_rule": args.pass_rule or "",
            "kill_rule": args.kill_rule or "",
            "notes": args.notes or "",
            "created_at": now_iso(),
            "central_task_id": "",
        }
        item.setdefault("probe_specs", []).append(spec)
        if item.get("status") in ("new", "triage"):
            item["status"] = "probe_designed"
        append_event(item, "probe_designed", probe_id=spec["id"], label=args.label)
        print(spec["id"])
        return items

    with_lock(IDEA_QUEUE, mutate)


def cmd_enqueue(args):
    def mutate_ideas(items):
        item = find_item(items, args.idea_id)
        probes = item.get("probe_specs", [])
        if args.probe_id:
            probes = [p for p in probes if p.get("id") == args.probe_id]
            if not probes:
                raise SystemExit(f"probe not found: {args.probe_id}")
        to_enqueue = [p for p in probes if not p.get("central_task_id")]
        if not to_enqueue:
            print("nothing to enqueue")
            return items

        def mutate_central(tasks):
            for probe in to_enqueue:
                task_id = short_id("t")
                tasks.append({
                    "id": task_id,
                    "status": "pending",
                    "priority": probe.get("priority", 10),
                    "label": f"idea {item['id']} {probe['label']}",
                    "launcher": probe["launcher"],
                    "env": probe["env"],
                    "created_at": now_iso(),
                    "idea_id": item["id"],
                    "idea_probe_id": probe["id"],
                    "auto_promoted": [],
                })
                probe["central_task_id"] = task_id
                probe["status"] = "queued"
                print(f"{probe['id']} -> {task_id}")
            return tasks

        with_lock(CENTRAL_QUEUE, mutate_central)
        item["status"] = "queued"
        append_event(item, "enqueued", count=len(to_enqueue))
        return items

    with_lock(IDEA_QUEUE, mutate_ideas)


def cmd_evidence(args):
    def mutate(items):
        item = find_item(items, args.idea_id)
        item.setdefault("evidence", []).append({
            "at": now_iso(),
            "by": args.by or "",
            "summary": args.summary,
            "path": args.path or "",
            "decision": args.decision or "",
        })
        append_event(item, "evidence", by=args.by or "", decision=args.decision or "")
        return items

    with_lock(IDEA_QUEUE, mutate)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add")
    s.add_argument("--title", required=True)
    s.add_argument("--goal", required=True)
    s.add_argument("--hypothesis")
    s.add_argument("--metric")
    s.add_argument("--notes")
    s.add_argument("--tags")
    s.add_argument("--owner")
    s.add_argument("--priority", type=int, default=10)
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("list")
    s.add_argument("--status", action="append")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("idea_id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("claim")
    s.add_argument("idea_id")
    s.add_argument("--agent", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_claim)

    s = sub.add_parser("status")
    s.add_argument("idea_id")
    s.add_argument("status")
    s.add_argument("--by")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("add-probe")
    s.add_argument("idea_id")
    s.add_argument("--label", required=True)
    s.add_argument("--launcher", required=True)
    s.add_argument("--env", required=True)
    s.add_argument("--priority", type=int, default=10)
    s.add_argument("--pass-rule")
    s.add_argument("--kill-rule")
    s.add_argument("--notes")
    s.set_defaults(func=cmd_add_probe)

    s = sub.add_parser("enqueue")
    s.add_argument("idea_id")
    s.add_argument("--probe-id")
    s.set_defaults(func=cmd_enqueue)

    s = sub.add_parser("evidence")
    s.add_argument("idea_id")
    s.add_argument("--summary", required=True)
    s.add_argument("--path")
    s.add_argument("--decision")
    s.add_argument("--by")
    s.set_defaults(func=cmd_evidence)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
