"""Parse a .jsonl file of items and filter camera-related items.

Filters for items whose `categories` list contains the following hierarchy:
- "Electronics"
- "Camera & Photo"
- "Digital Cameras"

Writes filtered items to an output JSONL file and prints a summary.

Usage:
    python scripts/parse_cameras.py data/items/cameras_meta.jsonl --out exports/cameras_filtered.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

REQUIRED_CATEGORIES = ["Electronics", "Camera & Photo", "Digital Cameras"]


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip invalid lines
                continue


def has_category_hierarchy(categories) -> bool:
    """Return True if `categories` contains the required hierarchy in order.

    The function checks that each required category appears somewhere in the categories list with
    the required order (not necessarily contiguous), e.g. Electronics -> Camera & Photo -> Digital Cameras.
    If `categories` is nested (list of lists), flattens it first.
    """
    if not categories:
        return False

    # Flatten one level of nested lists/tuples
    flat = []
    for c in categories:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        else:
            flat.append(c)

    # Normalize to strings
    flat = [str(x) for x in flat]

    # Find each required category in order
    idx = 0
    for req in REQUIRED_CATEGORIES:
        try:
            i = flat.index(req, idx)
        except ValueError:
            return False
        idx = i + 1
    return True


def filter_items(items: Iterable[dict]) -> Iterable[dict]:
    for item in items:
        categories = item.get("categories") or item.get("category") or []
        if has_category_hierarchy(categories):
            yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input .jsonl file")
    parser.add_argument(
        "--out",
        default="exports/cameras_filtered.jsonl",
        help="Output .jsonl path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of items written (0 = all)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(inp)
    matched = list(filter_items(items))

    total = len(matched)
    write_count = total if args.limit <= 0 else min(total, args.limit)

    with out.open("w", encoding="utf-8") as f:
        for i, item in enumerate(matched):
            if args.limit and i >= args.limit:
                break
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")

    print(f"Input: {inp}")
    print(f"Output: {out} (wrote {write_count} items)")
    print(f"Total matched: {total}")


if __name__ == "__main__":
    main()
