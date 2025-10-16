#!/usr/bin/env python3
"""
Prefilter large item collections before embedding generation.

Reduces item space by:
- Removing items with low ratings or few reviews
- Deduplicating by normalized title
- Optionally sampling by brand/category diversity
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def normalize_title(title):
    """Normalize title for duplicate detection."""
    if not title:
        return ""
    # Lowercase, remove extra spaces
    normalized = " ".join(title.lower().split())
    return normalized


def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def prefilter_items(items, min_rating=3.5, min_reviews=5, max_items=None):
    """
    Filter items by quality metrics and deduplicate.
    Uses stratified sampling by brand to ensure diversity if max_items is set.

    Args:
        items: List of item dictionaries
        min_rating: Minimum average rating (default 3.5)
        min_reviews: Minimum number of reviews (default 5)
        max_items: Maximum items to keep (default None - no limit)

    Returns:
        List of filtered items
    """
    import math

    print(f"Initial items: {len(items)}")

    # Step 1: Filter by rating and review count
    filtered = []
    for item in items:
        rating = item.get("average_rating") or item.get("rating")
        review_count = (
            item.get("rating_number") or item.get("review_count") or 0
        )

        # Convert rating to float if string
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except (ValueError, TypeError):
                rating = 0.0
        elif rating is None:
            rating = 0.0

        # Keep items with good ratings and sufficient reviews
        if rating >= min_rating and review_count >= min_reviews:
            filtered.append(item)

    print(
        f"After rating/review filter (>={min_rating} stars, >={min_reviews} reviews): {len(filtered)}"
    )

    # Step 2: Deduplicate by normalized title (keep highest rated)
    title_groups = defaultdict(list)
    for item in filtered:
        title = item.get("title", "")
        norm_title = normalize_title(title)
        if norm_title:
            title_groups[norm_title].append(item)

    # Keep best item from each title group (highest rating, then most reviews)
    deduped = []
    for norm_title, group in title_groups.items():
        # Sort by rating (desc) then review count (desc)
        best = max(
            group,
            key=lambda x: (
                x.get("average_rating") or x.get("rating") or 0,
                x.get("rating_number") or x.get("review_count") or 0,
            ),
        )
        deduped.append(best)

    print(f"After deduplication by title: {len(deduped)}")

    # Step 3: If max_items is set and we have too many, use stratified sampling by brand for diversity
    if max_items is not None and len(deduped) > max_items:
        # Add quality scores to all items
        for item in deduped:
            rating = item.get("average_rating") or item.get("rating") or 0
            reviews = item.get("rating_number") or item.get("review_count") or 0
            item["_prefilter_score"] = rating * math.log(reviews + 1)

        # Group by brand (extract from title or use dedicated field if available)
        brand_groups = defaultdict(list)
        for item in deduped:
            # Try to extract brand from title (first word often is brand)
            title = item.get("title", "")
            brand = item.get("brand", "")
            if not brand and title:
                # Extract first word as potential brand
                brand = title.split()[0] if title.split() else "Unknown"
            else:
                brand = brand or "Unknown"
            brand_groups[brand].append(item)

        print(f"Found {len(brand_groups)} unique brands/groups")

        # Sort items within each brand by quality score
        for brand, group in brand_groups.items():
            group.sort(key=lambda x: x.get("_prefilter_score", 0), reverse=True)

        # Stratified sampling: take items proportionally from each brand
        # Calculate items per brand based on brand size, but ensure each brand gets at least 1
        total_brands = len(brand_groups)
        min_per_brand = 1
        remaining_slots = max_items - (min_per_brand * total_brands)

        if remaining_slots < 0:
            # Too many brands, just take top brands by their best item
            brand_scores = {
                brand: max(group, key=lambda x: x.get("_prefilter_score", 0))[
                    "_prefilter_score"
                ]
                for brand, group in brand_groups.items()
            }
            top_brands = sorted(
                brand_scores.items(), key=lambda x: x[1], reverse=True
            )[:max_items]
            selected = []
            for brand, _ in top_brands:
                selected.append(brand_groups[brand][0])
        else:
            # Allocate remaining slots proportionally to brand size
            total_items = len(deduped)
            selected = []

            # First pass: give each brand its minimum
            brand_allocations = {brand: min_per_brand for brand in brand_groups}

            # Second pass: distribute remaining slots proportionally
            for brand, group in brand_groups.items():
                proportion = len(group) / total_items
                additional = int(remaining_slots * proportion)
                brand_allocations[brand] += additional

            # Third pass: if we still have slots due to rounding, give to largest brands
            allocated_total = sum(brand_allocations.values())
            if allocated_total < max_items:
                sorted_brands = sorted(
                    brand_groups.items(), key=lambda x: len(x[1]), reverse=True
                )
                for brand, _ in sorted_brands:
                    if allocated_total >= max_items:
                        break
                    brand_allocations[brand] += 1
                    allocated_total += 1

            # Select top items from each brand according to allocation
            for brand, allocation in brand_allocations.items():
                group = brand_groups[brand]
                n_to_take = min(allocation, len(group))
                selected.extend(group[:n_to_take])

        # Remove temporary score field
        for item in selected:
            item.pop("_prefilter_score", None)

        print(
            f"After stratified sampling to {len(selected)} items (from {len(brand_groups)} brands)"
        )
        return selected

    return deduped


def write_jsonl(items, output_path):
    """Write items to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {output_path}")


def write_report(
    input_path,
    output_path,
    report_path,
    original_count,
    filtered_count,
    min_rating,
    min_reviews,
    max_items,
    filtered_items,
):
    """Write filtering report with diversity metrics."""
    # Calculate brand diversity
    brand_counts = defaultdict(int)
    for item in filtered_items:
        title = item.get("title", "")
        brand = item.get("brand", "")
        if not brand and title:
            brand = title.split()[0] if title.split() else "Unknown"
        else:
            brand = brand or "Unknown"
        brand_counts[brand] += 1

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Prefilter Report\n")
        f.write(f"================\n\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n\n")
        f.write(f"Filters Applied:\n")
        f.write(f"  - Minimum rating: {min_rating}\n")
        f.write(f"  - Minimum reviews: {min_reviews}\n")
        f.write(
            f"  - Maximum items: {max_items if max_items is not None else 'No limit'}\n\n"
        )
        f.write(f"Results:\n")
        f.write(f"  - Original items: {original_count}\n")
        f.write(f"  - Filtered items: {filtered_count}\n")
        f.write(
            f"  - Reduction: {original_count - filtered_count} ({100*(original_count-filtered_count)/original_count:.1f}%)\n\n"
        )
        f.write(f"Diversity Metrics:\n")
        f.write(f"  - Unique brands/groups: {len(brand_counts)}\n")
        f.write(
            f"  - Average items per brand: {filtered_count / len(brand_counts):.1f}\n\n"
        )
        f.write(f"Top 10 Brands by Item Count:\n")
        sorted_brands = sorted(
            brand_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for brand, count in sorted_brands:
            f.write(f"  - {brand}: {count} items\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prefilter large item collections before embedding generation"
    )
    parser.add_argument(
        "--meta", required=True, help="Path to input metadata JSONL file"
    )
    parser.add_argument(
        "--output",
        help="Path to output filtered JSONL file (default: input_prefiltered.jsonl)",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=3.5,
        help="Minimum average rating (default: 3.5)",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=5,
        help="Minimum number of reviews (default: 5)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum items to keep using stratified sampling (default: None - no limit)",
    )

    args = parser.parse_args()

    # Set output path
    input_path = Path(args.meta)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_prefiltered.jsonl"

    report_path = output_path.parent / f"{output_path.stem}.report.txt"

    # Load items
    print(f"Loading items from {input_path}...")
    items = load_jsonl(input_path)
    original_count = len(items)

    # Filter items
    print(f"\nApplying filters...")
    filtered_items = prefilter_items(
        items,
        min_rating=args.min_rating,
        min_reviews=args.min_reviews,
        max_items=args.max_items,
    )

    # Write output
    print(f"\nWriting output...")
    write_jsonl(filtered_items, output_path)
    write_report(
        input_path,
        output_path,
        report_path,
        original_count,
        len(filtered_items),
        args.min_rating,
        args.min_reviews,
        args.max_items,
        filtered_items,
    )

    print(f"\nDone! Report written to {report_path}")


if __name__ == "__main__":
    main()
