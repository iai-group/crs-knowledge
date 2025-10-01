#!/usr/bin/env python3
"""Analyze conversation files in exports/conversations.

Loads all JSON files in a directory (default: `exports/conversations`), keeps
those that contain a top-level `messages` list, removes `system` messages,
and computes statistics on message lengths (characters and words) and the
average `q1` score from `post_task_answers` when available.

Usage: python scripts/analyze_conversations.py [path/to/conversations_dir]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional


def safe_num(v: Any) -> Optional[float]:
    """Try to convert v to float; return None on failure."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def summarize_numbers(xs: Iterable[float]) -> Dict[str, float]:
    xs = list(xs)
    if not xs:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stdev": 0.0,
        }
    cnt = len(xs)
    try:
        st = pstdev(xs) if cnt > 1 else 0.0
    except Exception:
        st = 0.0
    return {
        "count": cnt,
        "mean": mean(xs),
        "median": median(xs),
        "min": min(xs),
        "max": max(xs),
        "stdev": st,
    }


def analyze_dir(dirpath: Path):
    files = sorted(dirpath.glob("*.json"))
    convo_files = 0
    total_messages = 0
    total_non_system = 0
    messages_per_convo: List[int] = []
    non_system_per_convo: List[int] = []

    char_lengths: List[int] = []
    word_lengths: List[int] = []
    # per-role lengths
    human_char_lengths: List[int] = []
    human_word_lengths: List[int] = []
    ai_char_lengths: List[int] = []
    ai_word_lengths: List[int] = []

    q1_scores: List[float] = []
    bike_counts: Counter[str] = Counter()

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # skip malformed
            continue
        msgs = data.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        convo_files += 1
        total_messages += len(msgs)
        messages_per_convo.append(len(msgs))

        # filter out system messages
        non_system = [m for m in msgs if (m.get("role") or "") != "system"]
        total_non_system += len(non_system)
        non_system_per_convo.append(len(non_system))

        for m in non_system:
            content = m.get("content")
            if content is None:
                content = ""
            content = str(content)
            char_lengths.append(len(content))
            # simple word split
            wcount = len(content.split())
            word_lengths.append(wcount)

            # per-role splits (human / ai)
            role = (m.get("role") or "").strip()
            if role == "human":
                human_char_lengths.append(len(content))
                human_word_lengths.append(wcount)
            elif role in ("ai", "assistant"):
                ai_char_lengths.append(len(content))
                ai_word_lengths.append(wcount)

        # post task answers: try to extract q1 numeric score
        post = data.get("post_task_answers") or {}
        if isinstance(post, dict):
            v = safe_num(post.get("q1"))
            if v is not None:
                q1_scores.append(v)

        # screen answers: collect bicycle expertise if present
        screen = data.get("screen_answers") or {}
        if isinstance(screen, dict):
            b = screen.get("bicycles")
            if isinstance(b, str) and b.strip():
                bike_counts[b.strip()] += 1

    print(f"Conversations dir: {dirpath}")
    print(f"Files scanned: {len(files)}")
    print(f"Conversation files with messages: {convo_files}")
    print(f"Total messages (including system): {total_messages}")
    print(f"Total non-system messages: {total_non_system}")

    def fmt(s: Dict[str, float]) -> str:
        return (
            f"count={s['count']}, mean={s['mean']:.2f}, median={s['median']:.2f}, "
            f"min={s['min']:.0f}, max={s['max']:.0f}, stdev={s['stdev']:.2f}"
        )

    print()
    print("Message length (characters):")
    print("  ", fmt(summarize_numbers(char_lengths)))
    print("Message length (words):")
    print("  ", fmt(summarize_numbers(word_lengths)))

    # per-role word-lengths
    print()
    print("Message length (words) by role:")
    print("  human:", fmt(summarize_numbers(human_word_lengths)))
    print("  ai/assistant:", fmt(summarize_numbers(ai_word_lengths)))

    print()
    print("Messages per conversation (including system):")
    print("  ", fmt(summarize_numbers(messages_per_convo)))
    print("Non-system messages per conversation:")
    print("  ", fmt(summarize_numbers(non_system_per_convo)))

    print()
    if q1_scores:
        s = summarize_numbers(q1_scores)
        print("Post-task 'q1' score summary:")
        print("  ", fmt(s))
        print(f"  Conversations with q1 score: {s['count']}")
        print(f"  Average q1 score: {s['mean']:.2f}")
    else:
        print("No numeric 'q1' post-task scores found.")

    print()
    print(
        "Bicycle domain expertise counts (from conversation files' screen_answers):"
    )
    if bike_counts:
        for k in ["Novice", "Intermediate", "Expert"]:
            print(f"  {k}: {bike_counts.get(k, 0)}")
    else:
        print("  No bicycle expertise data found in conversation files.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv
    dirpath = Path(argv[1]) if len(argv) > 1 else Path("exports/conversations")
    # fallback to alternate folder if default missing
    if not dirpath.exists() or not dirpath.is_dir():
        alt = Path("exports/Prolific-0919/conversations")
        if alt.exists() and alt.is_dir():
            dirpath = alt
        else:
            print(
                f"Directory not found: {dirpath} (and fallback {alt} missing)"
            )
            return 2
    analyze_dir(dirpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
