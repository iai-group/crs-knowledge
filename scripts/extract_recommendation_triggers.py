#!/usr/bin/env python3
"""Extract user utterances that triggered recommendations.

This script processes conversation JSON files and identifies user messages that
immediately preceded a system message containing "Recommend". It extracts the
user utterance along with the percentile and expertise level, saving them to CSV.

Usage:
  python3 scripts/extract_recommendation_triggers.py [conversations_dir] [output_csv]

Defaults:
  conversations_dir: post_process/selected_conversations
  output_csv: post_process/recommendation_triggers.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


def get_expertise_level(data: Dict) -> str:
    """Get expertise level for the conversation's domain."""
    domain = data.get("current_domain")
    screen = data.get("screen_answers") or {}

    if domain and isinstance(screen, dict) and domain in screen:
        expertise = screen.get(domain)
        if expertise and isinstance(expertise, str):
            return expertise.strip()

    return "unknown"


def extract_recommendation_triggers(file_path: Path) -> List[Dict[str, str]]:
    """
    Extract user utterances that triggered recommendations.

    Args:
        file_path: Path to conversation JSON file

    Returns:
        List of dicts with 'percentile', 'utterance', 'expertise' keys
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Failed to read {file_path.name}: {e}")
        return []

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return []

    # Extract percentile from JSON data
    percentile = data.get("knowledge_percentile")
    if percentile is None:
        print(f"Warning: No knowledge_percentile found in {file_path.name}")
        return []

    # Get expertise level
    expertise = get_expertise_level(data)

    # Find user utterances that precede recommendation system messages
    triggers = []
    for i in range(len(messages)):
        msg = messages[i]

        # Check if this is a system message with "Recommend"
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content") or ""

        if role == "system" and "Recommend" in content:
            # Look backwards to find the last user message
            for j in range(i - 1, -1, -1):
                prev_msg = messages[j]
                prev_role = (prev_msg.get("role") or "").strip().lower()

                if prev_role in ["human", "user"]:
                    utterance = (prev_msg.get("content") or "").strip()
                    if utterance:
                        triggers.append(
                            {
                                "percentile": str(percentile),
                                "utterance": utterance,
                                "expertise": expertise,
                            }
                        )
                    break

    return triggers


def process_directory(conversations_dir: Path, output_csv: Path) -> None:
    """
    Process all conversation files and save recommendation triggers to CSV.

    Args:
        conversations_dir: Directory containing conversation JSON files
        output_csv: Path to output CSV file
    """
    files = sorted(conversations_dir.glob("*.json"))

    if not files:
        print(f"No JSON files found in {conversations_dir}")
        return

    all_triggers = []
    processed = 0

    print(f"Processing {len(files)} conversation files...")
    for file_path in files:
        triggers = extract_recommendation_triggers(file_path)
        all_triggers.extend(triggers)
        if triggers:
            processed += 1
            print(
                f"  {file_path.name}: Found {len(triggers)} recommendation trigger(s)"
            )

    if not all_triggers:
        print("No recommendation triggers found.")
        return

    # Write to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["percentile", "utterance", "expertise"]
        )
        writer.writeheader()
        writer.writerows(all_triggers)

    print(
        f"\nExtracted {len(all_triggers)} recommendation triggers from {processed} conversations"
    )
    print(f"Results saved to: {output_csv}")

    # Print summary statistics
    by_expertise = {}
    for trigger in all_triggers:
        exp = trigger["expertise"]
        by_expertise[exp] = by_expertise.get(exp, 0) + 1

    print("\nTriggers by expertise level:")
    for exp, count in sorted(by_expertise.items()):
        print(f"  {exp}: {count}")


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv

    conversations_dir = (
        Path(argv[1])
        if len(argv) > 1
        else Path("post_process/selected_conversations")
    )
    output_csv = (
        Path(argv[2])
        if len(argv) > 2
        else Path("post_process/recommendation_triggers.csv")
    )

    if not conversations_dir.exists():
        print(f"Error: Conversations directory not found: {conversations_dir}")
        return 1

    process_directory(conversations_dir, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
