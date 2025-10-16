"""Explore running shoes categories across multiple Amazon metadata files.

This script searches for running shoes in Clothing, Shoes and Jewelry and
Sports & Outdoors to identify the correct categories and filtering logic.

Run this before creating the running shoes preprocessing script.
"""

import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Metadata files to search
DATA_FILES = {
    "Clothing_Shoes_and_Jewelry": "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl",
    "Sports_and_Outdoors": "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl",
}

# Keywords to identify running shoes
RUNNING_SHOE_KEYWORDS = [
    "running shoe",
    "running shoes",
    "runner shoe",
    "runner shoes",
    "jogging shoe",
    "jogging shoes",
    "road running",
    "trail running",
]

# Keywords to exclude (accessories, not actual shoes)
EXCLUDE_KEYWORDS = [
    "lace",
    "insole",
    "sock",
    "cleaner",
    "spray",
    "bag",
    "holder",
    "rack",
    "organizer",
    "deodorizer",
    "polish",
    "cream",
    "brush",
    "horn",
    "tree",
    "stretcher",
]


def open_jsonl_source(source: str):
    """Yield parsed JSON objects from source."""
    parsed = urlparse(source)

    if parsed.scheme == "hf":
        hf_path = source[len("hf://") :]
        parts = hf_path.split("/")
        if parts[0] == "datasets":
            if len(parts) < 4:
                raise ValueError(f"Invalid hf:// dataset path: {source}")
            owner = parts[1]
            repo = parts[2]
            filename = "/".join(parts[3:])
        else:
            if len(parts) < 3:
                raise ValueError(f"Invalid hf:// path: {source}")
            owner = parts[0]
            repo = parts[1]
            filename = "/".join(parts[2:])

        repo_id = f"{owner}/{repo}"
        url = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        )
        print(f"Streaming from Hugging Face URL: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except Exception as e:
                    print(f"Failed to parse JSON line: {e}", file=sys.stderr)
        return

    raise FileNotFoundError(f"Could not open source: {source}")


def is_running_shoe(title: str) -> bool:
    """Check if title indicates a running shoe (not accessory)."""
    title_lower = title.lower()

    # Check for exclude keywords first
    for exclude in EXCLUDE_KEYWORDS:
        if exclude in title_lower:
            return False

    # Check for running shoe keywords
    for keyword in RUNNING_SHOE_KEYWORDS:
        if keyword in title_lower:
            return True

    return False


def explore_metadata_file(
    file_key: str, file_path: str, max_items: int = 100000
):
    """Explore one metadata file for running shoes."""
    print(f"\n{'='*80}")
    print(f"EXPLORING: {file_key}")
    print(f"{'='*80}\n")

    main_categories = Counter()
    shoe_samples = []
    category_paths = Counter()
    total_running_shoes = 0
    max_samples = 30

    print(f"Processing up to {max_items:,} items from {file_key}...")

    with tqdm(total=max_items, unit="items", desc=file_key) as pbar:
        for i, item in enumerate(open_jsonl_source(file_path)):
            if i >= max_items:
                break

            pbar.update(1)

            title = item.get("title", "")
            main_cat = item.get("main_category", "")
            categories = item.get("categories", [])

            # Track main categories
            if main_cat:
                main_categories[main_cat] += 1

            # Check if this is a running shoe
            if is_running_shoe(title):
                total_running_shoes += 1

                # Collect sample
                if len(shoe_samples) < max_samples:
                    shoe_samples.append(
                        {
                            "title": title,
                            "main_category": main_cat,
                            "categories": categories,
                            "parent_asin": item.get("parent_asin", ""),
                            "average_rating": item.get("average_rating", ""),
                            "rating_number": item.get("rating_number", 0),
                        }
                    )

                # Track category paths
                if categories:
                    cat_path = " > ".join(categories)
                    category_paths[cat_path] += 1

    # Print results for this file
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {file_key}:")
    print(f"{'='*80}")
    print(f"Total running shoes found: {total_running_shoes:,}")

    if total_running_shoes > 0:
        print(f"\nMain Categories (Top 10):")
        for cat, count in main_categories.most_common(10):
            print(f"  {cat}: {count:,}")

        print(
            f"\nSample Running Shoe Items (showing {min(10, len(shoe_samples))}):"
        )
        for i, sample in enumerate(shoe_samples[:10], 1):
            print(f"\n{i}. {sample['title'][:80]}...")
            print(f"   ASIN: {sample['parent_asin']}")
            print(f"   Main Category: {sample['main_category']}")
            print(
                f"   Rating: {sample['average_rating']} ({sample['rating_number']} reviews)"
            )
            print(f"   Categories: {sample['categories']}")

        print(f"\nCategory Paths (Top 20):")
        for cat_path, count in category_paths.most_common(20):
            print(f"  [{count:3d}] {cat_path}")

    return {
        "file": file_key,
        "total_running_shoes": total_running_shoes,
        "main_categories": main_categories,
        "category_paths": category_paths,
        "samples": shoe_samples,
    }


def main():
    print("=" * 80)
    print("RUNNING SHOES CATEGORY EXPLORATION")
    print("=" * 80)
    print("\nSearching for running shoes across multiple metadata files...")
    print(f"Files to explore: {', '.join(DATA_FILES.keys())}")

    # Explore each file
    results = {}
    for file_key, file_path in DATA_FILES.items():
        try:
            result = explore_metadata_file(
                file_key, file_path, max_items=100000
            )
            results[file_key] = result
        except Exception as e:
            print(f"\nError processing {file_key}: {e}")
            continue

    # Summary across all files
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL FILES:")
    print("=" * 80)

    total_across_all = sum(r["total_running_shoes"] for r in results.values())
    print(f"\nTotal running shoes found: {total_across_all:,}")

    print("\nBreakdown by source file:")
    for file_key in DATA_FILES.keys():
        if file_key in results:
            count = results[file_key]["total_running_shoes"]
            pct = (
                (count / total_across_all * 100) if total_across_all > 0 else 0
            )
            print(f"  {file_key}: {count:,} ({pct:.1f}%)")

    # Aggregate main categories
    print("\nMain Categories Across All Files:")
    all_main_cats = Counter()
    for result in results.values():
        all_main_cats.update(result["main_categories"])

    for cat, count in all_main_cats.most_common(15):
        print(f"  {cat}: {count:,}")

    # Aggregate category paths
    print("\nMost Common Category Paths Across All Files (Top 30):")
    all_cat_paths = Counter()
    for result in results.values():
        all_cat_paths.update(result["category_paths"])

    for cat_path, count in all_cat_paths.most_common(30):
        print(f"  [{count:3d}] {cat_path}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print(
        """
Based on this exploration, consider the following for running shoes preprocessing:

1. Which source file(s) have the most running shoes?
2. What are the common category patterns to filter on?
3. Are there false positives (accessories) that need better filtering?
4. Which main_category values should be included?
5. Should you combine multiple source files or focus on one?

Next steps:
- Review the category paths above
- Identify the most reliable filtering criteria
- Create running_shoes_preprocess.py with appropriate filters
    """
    )


if __name__ == "__main__":
    main()
