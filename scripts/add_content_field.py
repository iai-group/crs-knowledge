#!/usr/bin/env python3
"""
Script to add a 'content' field to item metadata files for embedding generation.

The content field combines: title, features, description, details, and skill level
into a single text representation suitable for semantic search.

Usage:
    python scripts/add_content_field.py data/items/digital_camera_meta.jsonl
    python scripts/add_content_field.py data/items/bicycle_meta.jsonl
"""

import argparse
import json
from pathlib import Path


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

    Combines: title, features, description, details (including skill level)
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

    # Add details (includes skill level and other metadata)
    if "details" in item:
        details_text = format_details(item["details"])
        if details_text:
            parts.append(details_text)

    return " ".join(parts)


def process_file(input_path, output_path=None, overwrite=False):
    """
    Process a JSONL metadata file and add content fields.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (if None, overwrites input)
        overwrite: If True, overwrite existing content fields
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Read all items
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} items from {input_path}")

    # Add content field to each item
    updated_count = 0
    for item in items:
        if "content" not in item or overwrite:
            item["content"] = create_content_field(item)
            updated_count += 1

    print(f"Updated {updated_count} items with content field")

    # Write back to file
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote updated items to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add content field to item metadata for embedding generation"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file (e.g., data/items/digital_camera_meta.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output JSONL file (default: overwrite input file)",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing content fields",
        default=False,
    )

    args = parser.parse_args()

    process_file(args.input_file, args.output, args.overwrite)


if __name__ == "__main__":
    main()
