"""Explore laptop-related categories in Amazon Electronics metadata.

This script helps identify the correct categories to use for laptop filtering
by sampling items that might be laptops and showing their category structure.

Run this before finalizing the laptop_preprocess.py filtering logic.
"""

import json
import sys
from collections import Counter
from typing import Any, Dict
from urllib.parse import urlparse

import requests
from tqdm import tqdm

DATA_FILES = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Electronics.jsonl"


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


def main():
    print("Exploring laptop-related items in Electronics metadata...\n")

    # Track statistics
    main_categories = Counter()
    laptop_titles_sample = []
    laptop_categories_sample = []
    computer_main_cat_count = 0
    laptop_keyword_count = 0
    max_samples = 50
    max_items = 100000  # Limit total items to process for faster exploration

    print(f"Processing up to {max_items:,} items...")

    with tqdm(total=max_items, unit="items") as pbar:
        for i, item in enumerate(open_jsonl_source(DATA_FILES)):
            if i >= max_items:
                break

            pbar.update(1)

            main_cat = item.get("main_category", "")
            title = item.get("title", "").lower()
            categories = item.get("categories", [])

            # Track all main categories
            if main_cat:
                main_categories[main_cat] += 1

            # Look for laptop-related items
            has_laptop_keyword = "laptop" in title or "notebook" in title

            if has_laptop_keyword:
                laptop_keyword_count += 1

                if len(laptop_titles_sample) < max_samples:
                    laptop_titles_sample.append(
                        {
                            "title": item.get("title", ""),
                            "main_category": main_cat,
                            "categories": categories,
                        }
                    )

            if main_cat == "Computers & Accessories":
                computer_main_cat_count += 1

                if (
                    has_laptop_keyword
                    and len(laptop_categories_sample) < max_samples
                ):
                    laptop_categories_sample.append(
                        {
                            "title": item.get("title", ""),
                            "categories": categories,
                        }
                    )

    # Print results
    print("\n" + "=" * 80)
    print("MAIN CATEGORIES DISTRIBUTION (Top 15):")
    print("=" * 80)
    for cat, count in main_categories.most_common(15):
        print(f"  {cat}: {count:,}")

    print("\n" + "=" * 80)
    print("STATISTICS:")
    print("=" * 80)
    print(
        f"  Items with 'Computers & Accessories' main_category: {computer_main_cat_count:,}"
    )
    print(
        f"  Items with 'laptop' or 'notebook' in title: {laptop_keyword_count:,}"
    )

    print("\n" + "=" * 80)
    print(f"SAMPLE LAPTOP ITEMS (showing {len(laptop_titles_sample)}):")
    print("=" * 80)
    for i, sample in enumerate(laptop_titles_sample[:10], 1):
        print(f"\n{i}. Title: {sample['title'][:100]}...")
        print(f"   Main Category: {sample['main_category']}")
        print(f"   Categories: {sample['categories']}")

    print("\n" + "=" * 80)
    print("CATEGORY PATTERNS IN LAPTOP ITEMS:")
    print("=" * 80)

    # Analyze category patterns
    all_category_strings = []
    for sample in laptop_categories_sample:
        if sample["categories"]:
            cat_str = " > ".join(sample["categories"])
            all_category_strings.append(cat_str)

    category_counter = Counter(all_category_strings)
    print(f"\nMost common category paths in laptop items (top 20):")
    for cat_path, count in category_counter.most_common(20):
        print(f"  [{count:3d}] {cat_path}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print(
        """
Based on this exploration, review the filtering logic in laptop_preprocess.py:

1. Check if 'Computers & Accessories' is the right main_category
2. Verify the category paths that actually contain laptops
3. Consider if title-based filtering is sufficient or too broad
4. Look for false positives (laptop bags, accessories, etc.)
    """
    )


if __name__ == "__main__":
    main()
