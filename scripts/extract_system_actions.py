#!/usr/bin/env python3
"""Extract system actions from conversations.

This script processes conversation JSON files and extracts all system actions
(Recommend, Elicit, Answer, Question_About_Recommendation, Redirect) along with
the turn number, percentile, and expertise level.

Usage:
  python3 scripts/extract_system_actions.py [conversations_dir] [output_csv]

Defaults:
  conversations_dir: post_process/conversations
  output_csv: post_process/system_actions.csv
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


def extract_system_actions(file_path: Path) -> List[Dict[str, str]]:
    """
    Extract system actions from a conversation.

    Args:
        file_path: Path to conversation JSON file

    Returns:
        List of dicts with 'prolific_id', 'domain', 'percentile', 'expertise', 'turn', 'action' keys
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

    # Get prolific ID
    prolific = data.get("prolific") or {}
    prolific_id = prolific.get("id", "unknown")

    # Get domain
    domain = data.get("current_domain", "unknown")

    # Get expertise level
    expertise = get_expertise_level(data)

    # Valid action types
    valid_actions = {
        "Recommend",
        "Elicit",
        "Answer",
        "Question_About_Recommendation",  # Also accept this variant
        "Answer_About_Recommendation",
        "Redirect",
    }

    # Extract actions from system messages
    actions = []
    human_turn_count = 0

    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content") or ""

        # Count human turns
        if role in ["human", "user"]:
            human_turn_count += 1
            continue

        # Check if this is a system message with an action
        if role == "system":
            # Check if content starts with one of the valid actions
            first_line = content.split("\n")[0].strip()

            # Check each valid action
            action_found = None
            for action in valid_actions:
                if first_line.startswith(action):
                    action_found = action
                    # Normalize Question_About_Recommendation to Answer_About_Recommendation
                    if action_found == "Question_About_Recommendation":
                        action_found = "Answer_About_Recommendation"
                    break

            if action_found:
                # Turn number is the count of human utterances before this action + 1
                turn = human_turn_count + 1

                actions.append(
                    {
                        "prolific_id": prolific_id,
                        "domain": domain,
                        "percentile": str(percentile),
                        "expertise": expertise,
                        "turn": str(turn),
                        "action": action_found,
                    }
                )

    return actions


def process_directory(conversations_dir: Path, output_csv: Path) -> None:
    """
    Process all conversation files and save system actions to CSV.

    Args:
        conversations_dir: Directory containing conversation JSON files
        output_csv: Path to output CSV file
    """
    files = sorted(conversations_dir.glob("*.json"))

    if not files:
        print(f"No JSON files found in {conversations_dir}")
        return

    all_actions = []
    processed = 0

    print(f"Processing {len(files)} conversation files...")
    for file_path in files:
        actions = extract_system_actions(file_path)
        all_actions.extend(actions)
        if actions:
            processed += 1
            print(f"  {file_path.name}: Found {len(actions)} action(s)")

    if not all_actions:
        print("No system actions found.")
        return

    # Write to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prolific_id",
                "domain",
                "percentile",
                "expertise",
                "turn",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(all_actions)

    print(
        f"\nExtracted {len(all_actions)} system actions from {processed} conversations"
    )
    print(f"Results saved to: {output_csv}")

    # Print summary statistics
    by_action = {}
    by_expertise = {}
    by_domain = {}

    for action_entry in all_actions:
        action = action_entry["action"]
        exp = action_entry["expertise"]
        domain = action_entry["domain"]

        by_action[action] = by_action.get(action, 0) + 1
        by_expertise[exp] = by_expertise.get(exp, 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1

    print("\nActions by type:")
    for action, count in sorted(by_action.items()):
        print(f"  {action}: {count}")

    print("\nActions by expertise level:")
    for exp, count in sorted(by_expertise.items()):
        print(f"  {exp}: {count}")

    print("\nActions by domain:")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv

    conversations_dir = (
        Path(argv[1]) if len(argv) > 1 else Path("post_process/conversations")
    )
    output_csv = (
        Path(argv[2])
        if len(argv) > 2
        else Path("post_process/system_actions.csv")
    )

    if not conversations_dir.exists():
        print(f"Error: Conversations directory not found: {conversations_dir}")
        return 1

    process_directory(conversations_dir, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
