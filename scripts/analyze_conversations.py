#!/usr/bin/env python3
"""Analyze conversation files in exports/conversations.

Loads all JSON files in a directory (default: `exports/conversations`), keeps
those that contain a top-level `messages` list, removes `system` messages,
and computes statistics on message lengths (characters and words) and the
average `q1` score from `post_task_answers` when available.

Also extracts the number of preferences elicited from system messages.

Usage: python scripts/analyze_conversations.py [path/to/conversations_dir]
"""
from __future__ import annotations

import json
import re
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


def extract_preference_count(messages: List[dict]) -> Optional[int]:
    """Extract the final preference count from system messages.

    Iterates from the end of messages to find the last system message
    with format "Updated to X preferences ..." and extracts X.

    Returns:
        The preference count as an integer, or None if not found.
    """
    for msg in reversed(messages):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            # Look for pattern "Updated to X preferences"
            match = re.search(r"Updated to (\d+) preferences?", content)
            if match:
                return int(match.group(1))
    return None


def analyze_dir(dirpath: Path):
    files = sorted(dirpath.glob("*.json"))
    convo_files = 0
    completed_convo_files = 0
    total_messages = 0
    total_non_system = 0
    messages_per_convo: List[int] = []
    non_system_per_convo: List[int] = []
    human_messages_per_convo: List[int] = []

    # char_lengths: List[int] = []
    word_lengths: List[int] = []
    # per-role lengths
    # human_char_lengths: List[int] = []
    human_word_lengths: List[int] = []
    # ai_char_lengths: List[int] = []aa
    ai_word_lengths: List[int] = []

    q1_scores: List[float] = []
    bike_counts: Counter[str] = Counter()

    # Track distributions per domain and expertise
    domain_counts: Counter[str] = Counter()
    expertise_by_domain: Dict[str, Counter[str]] = {}

    # Track preference counts
    preference_counts: List[int] = []
    preference_counts_by_domain: Dict[str, List[int]] = {}
    preference_counts_by_expertise: Dict[str, List[int]] = {}

    # Track feedback comments
    feedback_comments: List[Dict[str, str]] = []

    # Track human turns per conversation
    human_turns_per_convo: List[int] = []
    human_turns_by_domain: Dict[str, List[int]] = {}
    human_turns_by_expertise: Dict[str, List[int]] = {}

    # Track q1 scores by domain and expertise
    q1_scores_by_domain: Dict[str, List[float]] = {}
    q1_scores_by_expertise: Dict[str, List[float]] = {}

    # Track selected items at the end of conversations
    selected_items: Counter[str] = Counter()
    selected_items_by_domain: Dict[str, Counter[str]] = {}

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

        # Track completed conversations
        if data.get("completed"):
            completed_convo_files += 1

        total_messages += len(msgs)
        messages_per_convo.append(len(msgs))

        # Extract preference count from system messages
        pref_count = extract_preference_count(msgs)
        if pref_count is not None:
            preference_counts.append(pref_count)

        # filter out system messages
        non_system = [m for m in msgs if (m.get("role") or "") != "system"]

        total_non_system += len(non_system)
        non_system_per_convo.append(len(non_system))

        # Count human turns in this conversation
        human_turns = sum(
            1 for m in non_system if (m.get("role") or "").strip() == "human"
        )
        human_turns_per_convo.append(human_turns)
        human_messages_per_convo.append(human_turns)

        for m in non_system:
            content = m.get("content")
            if content is None:
                content = ""
            content = str(content)
            # char_lengths.append(len(content))
            # simple word split
            wcount = len(content.split())
            word_lengths.append(wcount)

            # per-role splits (human / ai)
            role = (m.get("role") or "").strip()
            if role == "human":
                # human_char_lengths.append(len(content))
                human_word_lengths.append(wcount)
            elif role in ("ai", "assistant"):
                # ai_char_lengths.append(len(content))
                ai_word_lengths.append(wcount)

        # post task answers: try to extract q1 numeric score and q2 feedback
        post = data.get("post_task_answers") or {}
        domain = data.get("current_domain")
        screen = data.get("screen_answers") or {}

        if isinstance(post, dict):
            v = safe_num(post.get("q1"))
            if v is not None:
                q1_scores.append(v)

                # Track q1 scores by domain
                if domain and isinstance(domain, str) and domain.strip():
                    domain_str = domain.strip()
                    if domain_str not in q1_scores_by_domain:
                        q1_scores_by_domain[domain_str] = []
                    q1_scores_by_domain[domain_str].append(v)

                    # Track q1 scores by expertise
                    if isinstance(screen, dict):
                        expertise = screen.get(domain_str)
                        if (
                            expertise
                            and isinstance(expertise, str)
                            and expertise.strip()
                        ):
                            expertise_str = expertise.strip()
                            if expertise_str not in q1_scores_by_expertise:
                                q1_scores_by_expertise[expertise_str] = []
                            q1_scores_by_expertise[expertise_str].append(v)

            # Collect feedback comments (q2) with associated q1 score
            q2_feedback = post.get("q2")
            if (
                q2_feedback
                and isinstance(q2_feedback, str)
                and q2_feedback.strip()
            ):
                domain = data.get("current_domain", "unknown")
                # Get relative path from current working directory
                try:
                    file_path = str(p.relative_to(Path.cwd()))
                except ValueError:
                    # If p is already relative or can't compute relative path
                    file_path = (
                        str(p)
                        .replace(
                            "/conversations/",
                            f"/parsed_conversations/{expertise}/",
                        )
                        .replace(".json", ".md")
                    )
                feedback_comments.append(
                    {
                        "domain": domain,
                        "feedback": q2_feedback.strip(),
                        "file": file_path,
                        "q1_score": v,  # Include q1 score with feedback
                    }
                )

        # Track domain and expertise distributions
        # (domain and screen already extracted above)
        if domain and isinstance(domain, str) and domain.strip():
            domain = domain.strip()
            domain_counts[domain] += 1

            # Track expertise level for this domain
            if domain not in expertise_by_domain:
                expertise_by_domain[domain] = Counter()

            expertise = None
            if isinstance(screen, dict):
                # Get expertise for the current domain
                expertise = screen.get(domain)
                if (
                    expertise
                    and isinstance(expertise, str)
                    and expertise.strip()
                ):
                    expertise = expertise.strip()
                    expertise_by_domain[domain][expertise] += 1

            # Track human turns by domain
            if domain not in human_turns_by_domain:
                human_turns_by_domain[domain] = []
            human_turns_by_domain[domain].append(human_turns)

            # Track human turns by expertise
            if expertise:
                if expertise not in human_turns_by_expertise:
                    human_turns_by_expertise[expertise] = []
                human_turns_by_expertise[expertise].append(human_turns)

            # Track preference counts by domain
            if pref_count is not None:
                if domain not in preference_counts_by_domain:
                    preference_counts_by_domain[domain] = []
                preference_counts_by_domain[domain].append(pref_count)

                # Track preference counts by expertise
                if expertise:
                    if expertise not in preference_counts_by_expertise:
                        preference_counts_by_expertise[expertise] = []
                    preference_counts_by_expertise[expertise].append(pref_count)

        # Extract selected item from the end of the conversation
        # Look for pattern: AI asks "Are you sure you want to go with this recommendation?"
        # followed by human saying "Yes, I want this one." or similar
        selected_item = None
        for i in range(len(msgs) - 1, -1, -1):
            msg = msgs[i]
            if msg.get("role") == "ai":
                content = msg.get("content", "")
                # Check if this is the confirmation question
                if (
                    "Are you sure you want to go with this recommendation?"
                    in content
                ):
                    # Extract item name from the message (usually in bold between **)
                    import re

                    match = re.search(r"\*\*(.*?)\*\*", content)
                    if match:
                        selected_item = match.group(1).strip()
                        # Remove any "Great! You've selected" prefix if present
                        if selected_item.startswith("Great! You've selected "):
                            selected_item = selected_item[
                                len("Great! You've selected ") :
                            ]
                        break

        if selected_item:
            selected_items[selected_item] += 1
            if domain:
                if domain not in selected_items_by_domain:
                    selected_items_by_domain[domain] = Counter()
                selected_items_by_domain[domain][selected_item] += 1

    print(f"Conversations dir: {dirpath}")
    print(f"Files scanned: {len(files)}")
    print(f"Conversation files with messages: {convo_files}")
    print(f"  Completed: {completed_convo_files}")
    # print(f"Total messages (including system): {total_messages}")
    print(f"Total non-system messages: {total_non_system}")

    def fmt(s: Dict[str, float]) -> str:
        return (
            f"count={s['count']}, mean={s['mean']:.2f}, median={s['median']:.2f}, "
            f"min={s['min']:.0f}, max={s['max']:.0f}, stdev={s['stdev']:.2f}"
        )

    print()
    # print("Message length (characters):")
    # print("  ", fmt(summarize_numbers(char_lengths)))
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
    print("Human messages per conversation:")
    print("  ", fmt(summarize_numbers(human_messages_per_convo)))

    print()
    if q1_scores:
        s = summarize_numbers(q1_scores)
        print("Post-task 'q1' score summary:")
        print("  ", fmt(s))
        print(f"  Conversations with q1 score: {s['count']}")
        print(f"  Average q1 score: {s['mean']:.2f}")
    else:
        print("No numeric 'q1' post-task scores found.")

    # Print q1 scores by domain
    print()
    print("Post-task 'q1' scores by domain:")
    if q1_scores_by_domain:
        for domain in sorted(q1_scores_by_domain.keys()):
            s = summarize_numbers(q1_scores_by_domain[domain])
            print(f"  {domain}:", fmt(s))
    else:
        print("  No domain-specific q1 scores available.")

    # Print q1 scores by expertise
    print()
    print("Post-task 'q1' scores by expertise level:")
    if q1_scores_by_expertise:
        for level in ["Novice", "Intermediate", "Expert"]:
            if level in q1_scores_by_expertise:
                s = summarize_numbers(q1_scores_by_expertise[level])
                print(f"  {level}:", fmt(s))
    else:
        print("  No expertise-specific q1 scores available.")

    # Print domain distribution
    print()
    print("Domain distribution:")
    if domain_counts:
        for domain, count in sorted(
            domain_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {domain}: {count}")
    else:
        print("  No domain information found.")

    # Print expertise distribution per domain
    print()
    print("Expertise distribution per domain:")
    if expertise_by_domain:
        for domain in sorted(expertise_by_domain.keys()):
            print(f"  {domain}:")
            expertise_counter = expertise_by_domain[domain]
            for level in ["Novice", "Intermediate", "Expert"]:
                count = expertise_counter.get(level, 0)
                print(f"    {level}: {count}")
    else:
        print("  No expertise information found.")

    # Print human turns statistics
    print()
    print("Human turns per conversation:")
    print("  ", fmt(summarize_numbers(human_turns_per_convo)))

    # Print human turns by domain
    print()
    print("Human turns per conversation by domain:")
    if human_turns_by_domain:
        for domain in sorted(human_turns_by_domain.keys()):
            s = summarize_numbers(human_turns_by_domain[domain])
            total_turns = sum(human_turns_by_domain[domain])
            print(f"  {domain}:", fmt(s), f"(total: {total_turns})")
    else:
        print("  No domain data available.")

    # Print human turns by expertise
    print()
    print("Human turns per conversation by expertise level:")
    if human_turns_by_expertise:
        for level in ["Novice", "Intermediate", "Expert"]:
            if level in human_turns_by_expertise:
                s = summarize_numbers(human_turns_by_expertise[level])
                total_turns = sum(human_turns_by_expertise[level])
                print(f"  {level}:", fmt(s), f"(total: {total_turns})")
    else:
        print("  No expertise data available.")

    # Print preference counts
    print()
    print("Preferences elicited per conversation:")
    if preference_counts:
        print("  ", fmt(summarize_numbers(preference_counts)))
    else:
        print("  No preference count data available.")

    # Print preference counts by domain
    print()
    print("Preferences elicited per conversation by domain:")
    if preference_counts_by_domain:
        for domain in sorted(preference_counts_by_domain.keys()):
            s = summarize_numbers(preference_counts_by_domain[domain])
            total_prefs = sum(preference_counts_by_domain[domain])
            print(f"  {domain}:", fmt(s), f"(total: {total_prefs})")
    else:
        print("  No domain data available.")

    # Print preference counts by expertise
    print()
    print("Preferences elicited per conversation by expertise level:")
    if preference_counts_by_expertise:
        for level in ["Novice", "Intermediate", "Expert"]:
            if level in preference_counts_by_expertise:
                s = summarize_numbers(preference_counts_by_expertise[level])
                total_prefs = sum(preference_counts_by_expertise[level])
                print(f"  {level}:", fmt(s), f"(total: {total_prefs})")
    else:
        print("  No expertise data available.")

    # Print selected items distribution
    print()
    print("=" * 80)
    print(
        f"Selected items at end of conversation: {sum(selected_items.values())} total"
    )
    print("=" * 80)
    if selected_items:
        print("\nTop selected items (overall):")
        for item, count in selected_items.most_common(10):
            # Truncate long item names for display
            display_name = item if len(item) <= 70 else item[:67] + "..."
            print(f"  {count:2d}x {display_name}")

        print("\nSelected items by domain:")
        for domain in sorted(selected_items_by_domain.keys()):
            print(f"  {domain}:")
            items = selected_items_by_domain[domain]
            for item, count in items.most_common(5):
                display_name = item if len(item) <= 65 else item[:62] + "..."
                print(f"    {count:2d}x {display_name}")
    else:
        print("  No selected items found.")

    # # Print feedback comments summary
    # print()
    # print("=" * 80)
    # print(
    #     f"Feedback comments from post-task questionnaire (q2): {len(feedback_comments)} total"
    # )
    # print("=" * 80)
    # if feedback_comments:
    #     # Show a summary by domain
    #     feedback_by_domain = Counter(
    #         comment["domain"] for comment in feedback_comments
    #     )
    #     print("\nFeedback count by domain:")
    #     for domain, count in sorted(
    #         feedback_by_domain.items(), key=lambda x: x[1], reverse=True
    #     ):
    #         print(f"  {domain}: {count}")

    #     print("\n" + "=" * 80)
    #     print(
    #         "Individual feedback comments (sorted by q1 score, lowest first):"
    #     )
    #     print("=" * 80)

    #     # Sort by q1 score (lowest first), with None values at the end
    #     sorted_feedback = sorted(
    #         feedback_comments,
    #         key=lambda x: (
    #             x.get("q1_score") is None,
    #             x.get("q1_score") or float("inf"),
    #         ),
    #     )

    #     for i, comment in enumerate(sorted_feedback, 1):
    #         q1_score = comment.get("q1_score")
    #         score_str = f"q1={q1_score}" if q1_score is not None else "q1=N/A"
    #         print(
    #             f"\n[{i}] {score_str} | Domain: {comment['domain']} | File: {comment['file']}"
    #         )
    #         # Wrap long feedback text for better readability
    #         feedback_text = comment["feedback"]
    #         # Print with slight indentation
    #         for line in feedback_text.split("\n"):
    #             print(f"    {line}")
    # else:
    #     print("  No feedback comments found.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv
    dirpath = Path(argv[1]) if len(argv) > 1 else Path("exports/conversations")
    # fallback to alternate folder if default missing
    if not dirpath.exists() or not dirpath.is_dir():
        print(f"Directory not found: {dirpath}")
        return 2
    analyze_dir(dirpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
