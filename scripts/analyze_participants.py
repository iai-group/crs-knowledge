#!/usr/bin/env python3
"""Analyze participant expertise from a JSONL export.

Reads a file like `exports/screen_results.jsonl` where each line is a JSON
object with at least `answers` and `prolific.id`. Computes:
 - per-domain counts (Novice/Intermediate/Expert)
 - per-participant dominant expertise (majority across domains, tie-breaker)

Usage: python scripts/analyze_participants.py [path/to/screen_results.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

PRIORITY = {"Expert": 3, "Intermediate": 2, "Novice": 1}


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip/continue on bad line
                continue


def analyze(path: Path):
    # per-domain counters
    domains = set()
    per_domain: Dict[str, Counter] = defaultdict(Counter)

    # group responses by participant id; keep latest entry per participant
    by_participant: Dict[str, Dict] = {}

    for obj in read_jsonl(path):
        prolific = obj.get("prolific") or {}
        pid = prolific.get("id")
        if not pid:
            continue
        # keep latest by timestamp (file order assumed chronological) -> overwrite
        by_participant[pid] = obj

    for pid, obj in by_participant.items():
        answers = obj.get("answers") or {}
        # count per-domain
        for domain, value in answers.items():
            domains.add(domain)
            if value in ("Novice", "Intermediate", "Expert"):
                per_domain[domain][value] += 1

    # compute per-participant dominant expertise
    dominant = Counter()

    for pid, obj in by_participant.items():
        answers = obj.get("answers") or {}
        vals = [v for v in answers.values() if v in PRIORITY]
        if not vals:
            continue
        # majority by count; if tie, choose by PRIORITY
        counts = Counter(vals)
        top_count = max(counts.values())
        candidates = [k for k, c in counts.items() if c == top_count]
        if len(candidates) == 1:
            dominant[candidates[0]] += 1
        else:
            # tie-breaker by PRIORITY
            winner = max(candidates, key=lambda k: PRIORITY.get(k, 0))
            dominant[winner] += 1

    # print results
    print(f"File: {path}")
    print(f"Total unique participants: {len(by_participant)}")
    print()
    print("Per-domain distribution:")
    for domain in sorted(domains):
        counts = per_domain[domain]
        total = sum(counts.values())
        print(f" - {domain}: total={total}")
        for k in ["Novice", "Intermediate", "Expert"]:
            print(f"    {k}: {counts.get(k,0)}")

    print()
    print("Per-participant dominant expertise (majority across domains):")
    total_dom = sum(dominant.values())
    print(f" total_participants_with_answers={total_dom}")
    for k in ["Novice", "Intermediate", "Expert"]:
        print(f"  {k}: {dominant.get(k,0)}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv
    path = (
        Path(argv[1]) if len(argv) > 1 else Path("exports/screen_results.jsonl")
    )
    if not path.exists():
        print(f"File not found: {path}")
        return 2
    analyze(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
