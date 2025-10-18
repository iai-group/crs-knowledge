#!/usr/bin/env python3
"""Parse conversation JSON files into human-readable text files.

For each JSON file in a directory (default: `exports/Prolific-0919/conversations` then
`exports/conversations`), write a `.md` file to `exports/parsed_conversations` containing
only non-system messages in the order they appeared. Each line is prefixed with the role
(`human` or `ai/assistant`) and the message content.

This script also deduplicates consecutive agent messages that start with "Great! You've
selected", keeping only the last one before a user turn or the end of the conversation.

Usage:
  python3 scripts/parse_conversations.py [conversations_dir] [output_dir]

If `output_dir` is omitted, it defaults to `exports/parsed_conversations`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List


def parse_dir(src: Path, out_dir: Path) -> int:
    files = sorted(src.glob("*.json"))
    written = 0
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        msgs = data.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        # filter out system
        non_system = [m for m in msgs if (m.get("role") or "") != "system"]
        if not non_system:
            continue

        # Count human turns and skip if 2 or fewer
        human_turns = sum(
            1 for m in non_system if (m.get("role") or "").strip() == "human"
        )
        if human_turns <= 2:
            continue

        # Determine expertise level for this conversation
        domain = data.get("current_domain")
        screen = data.get("screen_answers") or {}
        expertise = None
        if domain and isinstance(screen, dict) and domain in screen:
            expertise = screen.get(domain)

        # Create output path directly in output directory (no subdirectories)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / (p.stem + ".md")

        lines: List[str] = []
        # header: use prolific id if available, otherwise filename
        prof = data.get("prolific") or {}
        pid = prof.get("id") or ""
        if pid:
            lines.append(f"# Conversation — prolific_id: {pid}")
        else:
            lines.append(f"# Conversation — {p.stem}")

        # Add link to original JSON file
        lines.append("")
        # Get relative path from workspace root with leading slash
        try:
            relative_path = p.relative_to(Path.cwd())
        except ValueError:
            relative_path = p
        lines.append(f"**Source:** [{p.name}](/{relative_path})")

        # Add domain and expertise level for that domain
        if domain:
            lines.append("")
            lines.append(f"**Domain:** {domain}")
            # expertise was already extracted above
            if expertise:
                lines.append(f"**Expertise Level:** {expertise}")

        # include declared expertise from screen_answers if available (all domains)
        if isinstance(screen, dict) and screen:
            bikes = screen.get("bicycles")
            movies = screen.get("movies")
            laptops = screen.get("laptops")
            parts = []
            if bikes:
                parts.append(f"Bicycles: {bikes}")
            if movies:
                parts.append(f"Movies: {movies}")
            if laptops:
                parts.append(f"Laptops: {laptops}")
            if parts:
                lines.append("")
                lines.append("**All Expertise Levels:** " + "; ".join(parts))
        lines.append("")

        # Deduplicate consecutive "Great! You've selected..." messages
        # Keep only the last one before a user turn or end of conversation
        deduplicated = []
        i = 0
        while i < len(non_system):
            msg = non_system[i]
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
                while j < len(non_system):
                    next_msg = non_system[j]
                    next_role = (next_msg.get("role") or "").strip().lower()
                    next_content = next_msg.get("content") or ""

                    # Stop if we hit a user message or a different type of agent message
                    if next_role in ("human", "user", "participant"):
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
                    deduplicated.append(non_system[j - 1])
                    i = j
                else:
                    # Just one message, keep it
                    deduplicated.append(msg)
                    i += 1
            else:
                deduplicated.append(msg)
                i += 1

        for m in deduplicated:
            role = (m.get("role") or "").strip()
            content = m.get("content") or ""
            # normalize role names to uppercase labels
            r = role.lower()
            if r in ("assistant", "ai", "agent"):
                label = "Agent"
            elif r in ("human", "user", "participant"):
                label = "User"
            # format: bold the User utterance content, labels in bold uppercase
            if label == "User":
                # Split by newlines, wrap each line with **, then combine back
                content_lines = content.strip().split("\n")
                wrapped_lines = [
                    f"**{line}**" for line in content_lines if line.strip()
                ]
                content_str = "\n\t".join(wrapped_lines)
            else:
                content_str = content
            lines.append(f"**{label}**: {content_str}")
        # join with double newlines to make paragraphs in Markdown
        out_p.write_text("\n\n".join(lines), encoding="utf-8")
        written += 1
    return written


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv
    src = Path(argv[1]) if len(argv) > 1 else Path("exports/conversations")
    if not src.exists() or not src.is_dir():
        alt = Path("exports/conversations")
        if alt.exists() and alt.is_dir():
            src = alt
        else:
            print(f"No conversations directory found at {src} or {alt}")
            return 2
    out = (
        Path(argv[2]) if len(argv) > 2 else src.parent / "parsed_conversations"
    )
    n = parse_dir(src, out)
    print(f"Parsed {n} conversation(s) to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
