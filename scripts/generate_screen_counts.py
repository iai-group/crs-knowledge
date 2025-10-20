"""Generate domain-expertise counts from exported conversations.

Heuristic: a conversation is considered 'completed' if it contains a
system message whose content starts with 'Recommend' (case-sensitive
match as in existing data). For each such conversation, use
`current_domain` and `screen_answers[<current_domain>]` to increment
the appropriate domain-expertise count (expertise normalized to
lowercase).

Writes output to `exports/screen_counts.json` as a JSON object mapping
"domain-expertise" -> count.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from glob import glob
from typing import Dict, Tuple


def conversation_is_recommended(messages: list) -> bool:
    """Return True if any system message appears to indicate a recommendation."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            content = m.get("content", "")
            # Heuristic: starts with "Recommend" (as in existing exports)
            if content.strip().startswith("Recommend "):
                return True
    return False


def build_counts_and_targets(
    conversations_dir: str,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    counts = Counter()
    # domain -> target_label -> count
    target_dist = defaultdict(Counter)
    files = sorted(glob(os.path.join(conversations_dir, "*.json")))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # ignore malformed files
            continue

        messages = data.get("messages", [])
        if not conversation_is_recommended(messages):
            continue

        current_domain = data.get("current_domain")
        screen_answers = data.get("screen_answers", {}) or {}
        if not current_domain:
            continue

        # screen_answers may use domain display names or internal names;
        # assume internal names (bicycle, digital_camera, etc.) are used.
        expertise = screen_answers.get(current_domain)
        if not expertise:
            # try fallback: maybe the keys are display names (e.g., 'bicycles')
            # try a simple mapping by checking for a key that contains the domain
            for k in screen_answers:
                if (
                    current_domain.replace("_", " ")
                    in k.replace("_", " ").lower()
                ):
                    expertise = screen_answers.get(k)
                    break

        if not expertise:
            continue

        key = f"{current_domain}-{expertise.lower()}"
        counts[key] += 1

        # target distribution per domain
        id = data.get("target", {}).get("id", "First")

        target_dist[current_domain][id] += 1

    # normalize defaultdicts to plain dicts of dicts
    return dict(counts), {d: dict(c) for d, c in target_dist.items()}


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    conv_dir = os.path.join(repo_root, "exports", "conversations")
    out_dir = os.path.join(repo_root, "exports")
    os.makedirs(out_dir, exist_ok=True)

    counts, target_dist = build_counts_and_targets(conv_dir)

    out_path = os.path.join(out_dir, "screen_counts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    # Print totals per domain (sum across expertise levels)
    domain_totals: Dict[str, int] = {}
    for k, v in counts.items():
        # keys are like 'laptop-intermediate' -> split on last '-'
        if "-" in k:
            domain = k.rsplit("-", 1)[0]
        else:
            domain = k
        domain_totals[domain] = domain_totals.get(domain, 0) + v

    # print one line per domain
    for domain, tot in sorted(domain_totals.items()):
        print(f"{domain}: {tot}")
        # print target distribution for this domain
        td = target_dist.get(domain, {})
        if td:
            # sort by count desc, then label asc for stability
            for label, cnt in sorted(td.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {label}: {cnt}")

    # Print overall total number of valid conversations (sum of all counts)
    total = sum(counts.values())
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()
