#!/usr/bin/env python3
"""Filter and copy conversations that meet specific criteria.

This script processes conversation JSON files and copies those that meet the following criteria:
1. At least 3 human utterances
2. At least one agent message containing "I recommend"

The copied conversations will have consecutive "Great! You've selected" messages
deduplicated (keeping only the last one in each sequence).

Usage:
  python3 scripts/filter_conversations.py [input_dir] [output_dir]

If not specified:
  input_dir defaults to 'exports/conversations'
  output_dir defaults to 'post_process/conversations'
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List


def deduplicate_selected_messages(messages: List[dict]) -> List[dict]:
    """Deduplicate consecutive 'Great! You've selected' messages.

    Keeps only the last one in each consecutive sequence.
    """
    deduplicated = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content") or ""

        # Check if this is an agent message starting with "Great! You've selected"
        if role in (
            "assistant",
            "ai",
            "agent",
        ) and content.strip().startswith("Great! You've selected"):
            # Look ahead to find consecutive similar messages
            j = i
            while j < len(messages):
                next_msg = messages[j]
                next_role = (next_msg.get("role") or "").strip().lower()
                next_content = next_msg.get("content") or ""

                # Stop if we hit a user message, system message, or a different type of agent message
                if next_role in ("human", "user", "participant", "system"):
                    break
                if next_role in (
                    "assistant",
                    "ai",
                    "agent",
                ) and not next_content.strip().startswith(
                    "Great! You've selected"
                ):
                    break
                j += 1

            # Keep only the last "Great! You've selected" message before the user/different message
            if j > i + 1:
                # Multiple consecutive "Great! You've selected" messages found
                # Keep only the one at position j-1
                deduplicated.append(messages[j - 1])
                i = j
            else:
                # Just one message, keep it
                deduplicated.append(msg)
                i += 1
        else:
            deduplicated.append(msg)
            i += 1

    return deduplicated


def meets_criteria(data: dict) -> bool:
    """Check if conversation meets the filtering criteria.

    Returns True if:
    - Has a prolific id
    - At least 3 human utterances
    - At least one agent message containing "I recommend"
    """
    # Check for prolific id
    prolific = data.get("prolific")
    if not isinstance(prolific, dict):
        return False
    prolific_id = prolific.get("id")
    if (
        not prolific_id
        or not isinstance(prolific_id, str)
        or not prolific_id.strip()
    ):
        return False

    msgs = data.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    # Count human utterances
    human_count = 0
    has_recommendation = False

    for msg in msgs:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()

        if role in ("human", "user", "participant"):
            human_count += 1
        elif role in ("assistant", "ai", "agent"):
            if "I recommend" in content:
                has_recommendation = True

    return human_count >= 3 and has_recommendation


def filter_conversations(src_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Filter conversations and copy to output directory.

    Returns:
        tuple: (total_files_processed, files_copied)
    """
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Error: Source directory does not exist: {src_dir}")
        return 0, 0

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.json"))
    copied = 0

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to read {p.name}: {e}")
            continue

        # Check if conversation meets criteria
        if not meets_criteria(data):
            continue

        # Deduplicate "Great! You've selected" messages
        if "messages" in data and isinstance(data["messages"], list):
            data["messages"] = deduplicate_selected_messages(data["messages"])

        # Write to output directory
        out_path = out_dir / p.name
        try:
            out_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            copied += 1
        except Exception as e:
            print(f"Warning: Failed to write {out_path.name}: {e}")
            continue

    return len(files), copied


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv

    # Parse arguments
    src = Path(argv[1]) if len(argv) > 1 else Path("exports/conversations")
    out = Path(argv[2]) if len(argv) > 2 else Path("post_process/conversations")

    # Check source directory
    if not src.exists() or not src.is_dir():
        print(f"Error: Source directory not found: {src}")
        return 1

    print(f"Filtering conversations from: {src}")
    print(f"Output directory: {out}")
    print()

    total, copied = filter_conversations(src, out)

    print(f"Total files processed: {total}")
    print(f"Files copied: {copied}")
    print(f"Files filtered out: {total - copied}")

    if copied > 0:
        print(f"\nFiltered conversations saved to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
