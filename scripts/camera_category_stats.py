"""Compute category statistics for filtered camera items.

Reads a JSONL file (default: data/items/cameras_filtered.jsonl) and computes:
- total items
- frequency of each category (counted once per item)
- top N categories

Writes a JSON summary to `exports/cameras_category_stats.json` by default.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def flatten_categories(cat) -> List[str]:
    """Flatten categories to a list of strings. Handles nested lists/tuples and comma-separated strings."""
    out: List[str] = []
    if cat is None:
        return out
    if isinstance(cat, str):
        # sometimes categories might be a single string with separators
        # split by '>' or '|' or ',' if present
        if ">" in cat:
            parts = [p.strip() for p in cat.split(">") if p.strip()]
            return parts
        if "|" in cat:
            parts = [p.strip() for p in cat.split("|") if p.strip()]
            return parts
        if "," in cat:
            parts = [p.strip() for p in cat.split(",") if p.strip()]
            return parts
        return [cat.strip()]
    if isinstance(cat, (list, tuple)):
        for c in cat:
            out.extend(flatten_categories(c))
        return out
    # fallback
    return [str(cat)]


def compute_stats(items: Iterable[dict]):
    total = 0
    category_counter = Counter()

    for item in items:
        total += 1
        cats = item.get("categories") or item.get("category") or []
        flat = [c for c in (flatten_categories(c) for c in [cats])]
        # flatten_categories when given list returns list; the above results in nested list
        # so flatten one more time:
        flat2: List[str] = []
        for part in flat:
            if isinstance(part, list):
                flat2.extend(part)
            else:
                flat2.append(part)

        # normalize and unique per item
        cleaned = {c.strip() for c in flat2 if c and isinstance(c, str)}
        for c in cleaned:
            category_counter[c] += 1

    return {
        "total_items": total,
        "unique_categories": len(category_counter),
        "category_counts": category_counter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default="data/items/cameras_filtered.jsonl",
        help="Input JSONL of filtered camera items",
    )
    parser.add_argument(
        "--out",
        default="exports/cameras_category_stats.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show top N categories",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(inp)
    stats = compute_stats(items)

    total = stats["total_items"]
    counter: Counter = stats["category_counts"]
    unique = stats["unique_categories"]

    top = counter.most_common(args.top)

    summary = {
        "input": str(inp),
        "output": str(out),
        "total_items": total,
        "unique_categories": unique,
        "top_categories": [{"category": k, "count": v} for k, v in top],
    }

    # write full counts to JSON (convert Counter to dict)
    full = {
        "total_items": total,
        "unique_categories": unique,
        "category_counts": dict(counter),
    }

    out.write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Input: {inp}")
    print(f"Total items: {total}")
    print(f"Unique categories: {unique}")
    print("Top categories:")
    for cat, cnt in top:
        print(f"  {cat}: {cnt}")

    print(f"Wrote full counts to {out}")


if __name__ == "__main__":
    main()
