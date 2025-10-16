"""Running shoes metadata preprocessing.

This script processes Clothing, Shoes and Jewelry and Sports & Outdoors metadata
files to extract running shoes and automatically adds a 'content' field for
embedding generation.

Based on exploration results:
- Clothing, Shoes & Jewelry: Uses paths with 'Women > Shoes > Athletic > Running'
  and 'Men > Shoes > Athletic > Running'
- Sports & Outdoors: Uses 'Sports & Outdoors > Running Shoes'

The script supports:
- local file paths
- http(s) URLs (streaming)
- hf:// paths which are downloaded via `huggingface_hub`

Dependencies:
- requests
- tqdm

Example:
    Run the script to produce `data/items/running_shoes_meta.jsonl` with embedded content fields.
"""

import json
import os
import sys
import time
from typing import Any, Dict, Iterator
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Metadata files to process
DATA_FILES = {
    "Clothing_Shoes_and_Jewelry": "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl",
    "Sports_and_Outdoors": "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl",
}

# Category filters for each source
# For Clothing, Shoes & Jewelry - must contain all of these in sequence
CLOTHING_REQUIRED_SEQUENCES = [
    ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Athletic", "Running"],
    ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic", "Running"],
]

# For Sports & Outdoors - exact match
SPORTS_EXACT_PATH = ["Sports & Outdoors", "Running Shoes"]

# Exclude kids' shoes categories
EXCLUDE_CATEGORIES = ["Boys", "Girls", "Baby"]


def is_running_shoe(item: Dict[str, Any], source: str) -> bool:
    """Check if an item is a running shoe based on its category path.

    Args:
        item: The item dictionary with metadata
        source: The source file key ("Clothing_Shoes_and_Jewelry" or "Sports_and_Outdoors")

    Returns:
        True if the item matches running shoe category criteria (excluding kids' shoes)
    """
    categories = item.get("categories", [])

    if not categories:
        return False

    # Exclude kids' shoes (Boys, Girls, Baby)
    if any(exclude_cat in categories for exclude_cat in EXCLUDE_CATEGORIES):
        return False

    if source == "Clothing_Shoes_and_Jewelry":
        # Check if categories contain any of the required sequences
        for required_seq in CLOTHING_REQUIRED_SEQUENCES:
            # Check if all elements of required_seq appear in categories in order
            if all(cat in categories for cat in required_seq):
                return True
        return False

    elif source == "Sports_and_Outdoors":
        # Exact match for Sports & Outdoors
        return categories == SPORTS_EXACT_PATH

    return False


def open_jsonl_source(source: str):
    """Yield parsed JSON objects from source.

    source can be:
    - local path: /path/to/file.jsonl
    - http(s) URL
    - hf://<repo>/<path> (will download the file using huggingface_hub)
    """
    parsed = urlparse(source)

    # hf:// scheme -> translate to huggingface.co resolve URL and stream
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


def load_running_shoe_metadata() -> Iterator[Dict[str, Any]]:
    """Yield running shoe items from metadata sources (streaming).

    This generator yields raw items that are running shoes based on
    category filtering. It does not accumulate all items in memory.
    """
    for source_key, source_path in DATA_FILES.items():
        print(f"\nProcessing {source_key}...")
        for item in open_jsonl_source(source_path):
            try:
                if is_running_shoe(item, source_key):
                    yield item
            except Exception as e:
                print(f"Error processing item: {e}")


def clean_item(obj: Any) -> Any:
    """Recursively remove null/empty fields from JSON-like structures.

    Removes: None, empty strings, empty lists, empty dicts.
    Returns the cleaned object (may be None).
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return s if s != "" else None
    if isinstance(obj, list):
        cleaned = [clean_item(x) for x in obj]
        cleaned = [x for x in cleaned if x is not None]
        return cleaned if cleaned else None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cv = clean_item(v)
            if cv is not None:
                out[k] = cv
        return out if out else None
    return obj


def format_features(features):
    """Format features list into text."""
    if not features:
        return ""
    return " features: " + " ".join(features)


def format_description(description):
    """Format description list into text."""
    if not description:
        return ""
    if isinstance(description, list):
        return " description: " + " ".join(description)
    return " description: " + str(description)


def format_details(details):
    """Format details dict into text."""
    if not details:
        return ""

    parts = []
    for key, value in details.items():
        if value:
            parts.append(f"{key}: {value}")

    if parts:
        return " details: " + " ".join(parts)
    return ""


def create_content_field(item):
    """
    Create a content field from item metadata.

    Combines: title, features, description, details (including specifications)
    """
    parts = []

    # Add title
    if "title" in item and item["title"]:
        parts.append(f"title: {item['title']}")

    # Add features
    if "features" in item:
        features_text = format_features(item["features"])
        if features_text:
            parts.append(features_text)

    # Add description
    if "description" in item:
        desc_text = format_description(item["description"])
        if desc_text:
            parts.append(desc_text)

    # Add details (includes specifications and other metadata)
    if "details" in item:
        details_text = format_details(item["details"])
        if details_text:
            parts.append(details_text)

    return " ".join(parts)


# Main processing
OUTPUT_PATH = "data/items/running_shoes_meta.jsonl"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

count = 0
start_ts = time.time()

print("=" * 80)
print("RUNNING SHOES METADATA PREPROCESSING")
print("=" * 80)
print(f"\nSources:")
for key in DATA_FILES.keys():
    print(f"  - {key}")
print(f"\nCategory filters:")
print(f"  - Clothing, Shoes & Jewelry: Women/Men > Shoes > Athletic > Running")
print(f"  - Sports & Outdoors: Running Shoes")
print(f"\nOutput: {OUTPUT_PATH}")
print()

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    # Use tqdm to show a live progress bar: we don't know total, so leave total=None
    with tqdm(unit="items", desc="Processing", mininterval=1) as pbar:
        for shoe in load_running_shoe_metadata():
            cleaned = clean_item(shoe)
            pbar.update(1)
            if not cleaned:
                continue

            # Add content field automatically
            cleaned["content"] = create_content_field(cleaned)

            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            count += 1
            pbar.set_postfix({"written": count})

elapsed = time.time() - start_ts
print(f"\n{'='*80}")
print(f"COMPLETED")
print(f"{'='*80}")
print(f"Wrote {count} cleaned running shoe items to {OUTPUT_PATH}")
print(f"Processing completed in {elapsed:.2f} seconds")
