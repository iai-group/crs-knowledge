"""
Preprocess bike items from Amazon Reviews 2023 dataset.

This script downloads bike metadata from the Sports and Outdoors category,
filters for bikes, cleans the data, and outputs structured JSON for use
in the retrieval system.
"""

import json
import os
from typing import Any, Dict, List

from datasets import load_dataset

DATA_FILES = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl"
OUTPUT_PATH = "data/items/bikes_meta.jsonl"


def is_bike(item: Dict[str, Any]) -> bool:
    """Check if an item is a bike based on its details."""
    return item.get("details", {}).get("Bike Type") is not None


def load_bike_metadata() -> List[Dict[str, Any]]:
    """Load and filter bike metadata from the Amazon dataset."""
    print("Loading Sports and Outdoors metadata...")

    # Load metadata (not reviews) from Sports and Outdoors
    metadata = load_dataset(
        "json",
        data_files=DATA_FILES,
        split="train",
        streaming=True,
    )

    print("Filtering for bikes...")
    bikes = []
    for item in metadata:
        if is_bike(item):
            bikes.append(item)

    print(f"Found {len(bikes)} bike items")
    return bikes


def clean_bike_data(bikes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean bike data by removing null values and unnecessary fields."""
    print("Cleaning bike data...")

    bikes_cleaned = []
    for bike in bikes:
        # Remove null values from top level
        bike_cleaned = {}
        for k, v in bike.items():
            if v is not None:
                bike_cleaned[k] = v
        bikes_cleaned.append(bike_cleaned)

    # Remove unnecessary fields and null values from details
    for bike in bikes_cleaned:
        # Remove unwanted top-level fields
        to_delete = ["videos", "rating_number", "main_category"]
        for k, v in bike.items():
            if not v:
                to_delete.append(k)
        for k in to_delete:
            if k in bike:
                del bike[k]

        # Clean details section
        if "details" in bike:
            to_delete = ["Best Sellers Rank"]
            for k, v in bike["details"].items():
                if not v:
                    to_delete.append(k)
            for k in to_delete:
                if k in bike["details"]:
                    del bike["details"][k]

    print(f"Cleaned {len(bikes_cleaned)} bike items")
    return bikes_cleaned


def save_raw_data(bikes: List[Dict[str, Any]], filename: str) -> None:
    """Save raw cleaned bike data to JSONL file."""
    os.makedirs("data/raw", exist_ok=True)

    print(f"Saving raw data to {filename}...")
    with open(filename, "w") as f:
        for bike in bikes:
            f.write(json.dumps(bike) + "\n")


def load_raw_data(filename: str) -> List[Dict[str, Any]]:
    """Load raw bike data from JSONL file."""
    print(f"Loading raw data from {filename}...")
    bikes = []
    with open(filename, "r") as f:
        for line in f:
            bikes.append(json.loads(line.strip()))

    print(f"Loaded {len(bikes)} bike items")
    return bikes


def add_content_field(bikes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add content field to each bike for embedding/retrieval."""
    print("Adding content field to bike data...")

    for bike in bikes:
        # Extract the essential fields as clean text
        title = bike.get("title", "").strip()
        features = " ".join(bike.get("features", [])).strip()
        description = " ".join(bike.get("description", [])).strip()
        details = " ".join(
            [f"{k}: {v}" for k, v in bike.get("details", {}).items() if v]
        ).strip()

        # Create content field combining title, features, description, and details
        content_parts = []
        if title:
            content_parts.append(f"title: {title}")
        if features:
            content_parts.append(f"features: {features}")
        if description:
            content_parts.append(f"description: {description}")
        if details:
            content_parts.append(f"details: {details}")

        bike["content"] = " ".join(content_parts)

    print(f"Added content field to {len(bikes)} bike items")
    return bikes


def save_enhanced_data(bikes: List[Dict[str, Any]], filename: str) -> None:
    """Save enhanced bike data with content field to JSONL file."""
    os.makedirs("data/raw", exist_ok=True)

    print(f"Saving enhanced data to {filename}...")
    with open(filename, "w") as f:
        for bike in bikes:
            f.write(json.dumps(bike) + "\n")


def merge_existing_files() -> None:
    """Merge existing raw and items files to add content field to raw data."""
    raw_file = "data/raw/bikes.jsonl"
    items_file = "data/items/bikes.jsonl"

    if not os.path.exists(raw_file):
        print(f"❌ Raw file {raw_file} not found")
        return

    if not os.path.exists(items_file):
        print(f"❌ Items file {items_file} not found")
        return

    print("Loading existing files...")

    # Load raw data
    raw_bikes = {}
    with open(raw_file, "r") as f:
        for line in f:
            bike = json.loads(line.strip())
            raw_bikes[bike["parent_asin"]] = bike

    # Load items data and create content mapping
    items_content = {}
    with open(items_file, "r") as f:
        for line in f:
            item_data = json.loads(line.strip())
            # The items file format is {bike_id: content_string}
            for bike_id, content in item_data.items():
                items_content[bike_id] = content

    print(
        f"Loaded {len(raw_bikes)} raw items and {len(items_content)} processed items"
    )

    # Merge content into raw data
    merged_count = 0
    for bike_id, bike in raw_bikes.items():
        if bike_id in items_content:
            bike["content"] = items_content[bike_id]
            merged_count += 1

    print(f"Merged content for {merged_count} bikes")

    # Save enhanced data to new file
    os.makedirs("data/items", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for bike in raw_bikes.values():
            f.write(json.dumps(bike) + "\n")

    print(f"✅ Created merged file: {OUTPUT_PATH} with content field")


def main():
    """Main preprocessing pipeline."""
    print("Starting bike item preprocessing...")

    # Check if both files exist for merging
    raw_file = "data/raw/bikes.jsonl"
    items_file = "data/items/bikes.jsonl"

    if os.path.exists(raw_file) and os.path.exists(items_file):
        print("Found existing raw and items files. Merging them...")

        # Check if content field already exists in raw data
        with open(raw_file, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                first_bike = json.loads(first_line)
                if "content" in first_bike:
                    print("✅ Content field already exists in raw data!")
                    bikes = [first_bike]
                    for line in f:
                        bikes.append(json.loads(line.strip()))
                else:
                    merge_existing_files()
                    # Reload the enhanced data from merged file
                    if os.path.exists(OUTPUT_PATH):
                        bikes = load_raw_data(OUTPUT_PATH)
                    else:
                        print(f"❌ Failed to create merged file {OUTPUT_PATH}")
                        return
            else:
                print("❌ Raw file is empty")
                return
    else:
        print("Raw or items files not found. Running full preprocessing...")
        # Check if raw data already exists
        if os.path.exists(raw_file):
            print(
                f"Raw data file {raw_file} already exists. Loading from file..."
            )
            bikes = load_raw_data(raw_file)

            # Check if content field already exists
            if bikes and "content" not in bikes[0]:
                print("Content field missing, adding it...")
                bikes = add_content_field(bikes)
                save_enhanced_data(bikes, raw_file)
            else:
                print("Content field already exists in raw data.")
        else:
            # Download and process from scratch
            bikes = load_bike_metadata()
            bikes_cleaned = clean_bike_data(bikes)
            bikes_with_content = add_content_field(bikes_cleaned)
            save_enhanced_data(bikes_with_content, raw_file)
            bikes = bikes_with_content

    # Print some stats
    if bikes:
        print(f"\nPreprocessing complete!")
        print(f"Total items: {len(bikes)}")
        print(f"Enhanced file: {raw_file}")

        # Show example of first item
        print(f"\nExample item structure:")
        example_bike = bikes[0]
        print(f"Title: {example_bike.get('title', '')[:100]}...")
        print(f"Content: {example_bike.get('content', '')[:150]}...")

        # Show available fields
        print(f"\nAvailable fields: {list(example_bike.keys())}")


def update_content_field() -> bool:
    """Update the content field of a specific bike by its ID."""
    with open("data/items/bikes_meta_clean.jsonl", "r") as f:
        bikes = [json.loads(line.strip()) for line in f]

    for bike in bikes:
        details = bike.get("details", {})
        clean_details = {}
        for k, v in details.items():
            if k and v:
                if isinstance(v, dict):
                    clean_details[k] = {}
                    for sub_k, sub_v in v.items():
                        if sub_k and sub_v:
                            if isinstance(sub_v, str):
                                clean_details[k][sub_k] = sub_v.strip()
                            else:
                                clean_details[k][sub_k] = sub_v
                elif isinstance(v, str):
                    clean_details[k] = v.strip()
                else:
                    clean_details[k] = v
        bike["details"] = clean_details
    bikes = add_content_field(bikes)
    with open("data/items/bikes_meta_clean_again.jsonl", "w") as f:
        for bike in bikes:
            f.write(json.dumps(bike) + "\n")


if __name__ == "__main__":
    update_content_field()
