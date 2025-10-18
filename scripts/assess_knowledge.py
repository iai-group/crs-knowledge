#!/usr/bin/env python3
"""Assess participant knowledge levels by comparing answers to ground truth.

This script loads ground truth answers from data/questionnaires/answers/ and
compares them to participants' pre-task answers in conversation files.

Scoring system:
- Correct "Definitely True/False": +2 points
- Correct "Probably True/False": +1 point
- Incorrect "Definitely True/False": -2 points
- Incorrect "Probably True/False": -1 point
- "I don't know": 0 points

Usage:
  python3 scripts/assess_knowledge.py [conversations_dir] [ground_truth_dir]

Defaults:
  conversations_dir: post_process/conversations
  ground_truth_dir: data/questionnaires/answers
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_ground_truth(gt_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load ground truth answers from text files.

    Returns:
        Dict mapping domain to dict mapping question_id to answer (Yes/No)
    """
    ground_truth = {}

    # Map file names to domain names
    domain_mapping = {
        "bicycle_answers.txt": "bicycle",
        "digital_camera_answers.txt": "digital_camera",
        "laptop_answers.txt": "laptop",
        "running_shoes_answers.txt": "running_shoes",
        "smartwatch_answers.txt": "smartwatch",
    }

    for filename, domain in domain_mapping.items():
        file_path = gt_dir / filename
        if not file_path.exists():
            print(f"Warning: Ground truth file not found: {file_path}")
            continue

        answers = {}
        lines = file_path.read_text(encoding="utf-8").strip().split("\n")

        current_category = None
        question_idx = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a category header (no Yes/No at start)
            if not (line.startswith("Yes") or line.startswith("No")):
                # This is a category header
                current_category = line
                question_idx = 0
                continue

            # Extract answer (Yes or No)
            answer = "Yes" if line.startswith("Yes") else "No"

            # Create question ID: q_Category_Index
            if current_category:
                # Clean category name: only replace spaces with underscores
                # Keep commas and hyphens to match the user data format exactly
                clean_category = current_category.replace(" ", "_")
                question_id = f"q_{clean_category}_{question_idx}"
                answers[question_id] = answer
                question_idx += 1

        ground_truth[domain] = answers

    return ground_truth


def score_answer(user_answer: str, correct_answer: str) -> int:
    """Score a single answer based on the ground truth.

    Args:
        user_answer: User's answer (e.g., "Definitely True", "Probably False", "I don't know")
        correct_answer: Ground truth answer ("Yes" or "No")

    Returns:
        Score: +2, +1, 0, -1, or -2
    """
    user_answer = user_answer.strip()

    # Handle "I don't know"
    if (
        "don't know" in user_answer.lower()
        or "dont know" in user_answer.lower()
    ):
        return 0

    # Determine if user said True or False
    is_true = "True" in user_answer
    is_false = "False" in user_answer

    if not (is_true or is_false):
        # Unknown format, treat as "I don't know"
        return 0

    # Determine if answer is correct
    user_says_yes = is_true
    correct_is_yes = correct_answer == "Yes"
    is_correct = user_says_yes == correct_is_yes

    # Determine confidence level
    is_definitely = "Definitely" in user_answer
    is_probably = "Probably" in user_answer

    # Calculate score
    if is_correct:
        if is_definitely:
            return 2
        elif is_probably:
            return 1
        else:
            return 1  # Default to probably if unclear
    else:
        if is_definitely:
            return -2
        elif is_probably:
            return -1
        else:
            return -1  # Default to probably if unclear


def assess_participant(
    conversation_data: dict, ground_truth: Dict[str, Dict[str, str]]
) -> Optional[Dict]:
    """Assess a single participant's knowledge.

    Returns:
        Dict with assessment results or None if no pre-task answers
    """
    domain = conversation_data.get("current_domain")
    if not domain:
        return None

    pre_task = conversation_data.get("pre_task_answers")
    if not isinstance(pre_task, dict) or not pre_task:
        return None

    # Get ground truth for this domain
    gt_answers = ground_truth.get(domain)
    if not gt_answers:
        return None

    prolific = conversation_data.get("prolific", {})
    prolific_id = prolific.get("id", "unknown")

    # Calculate scores (both with and without confidence)
    scores_with_conf = []  # +2, +1, 0, -1, -2
    scores_no_conf = []  # +1, 0, -1 (ignoring confidence)
    correct_count = 0
    incorrect_count = 0
    idk_count = 0
    answered_count = 0

    for question_id, user_answer in pre_task.items():
        if question_id not in gt_answers:
            continue

        correct_answer = gt_answers[question_id]

        # Score with confidence
        score_conf = score_answer(user_answer, correct_answer)
        scores_with_conf.append(score_conf)

        # Score without confidence (just correct/incorrect/idk)
        if score_conf > 0:
            score_no_conf = 1  # Correct regardless of confidence
            correct_count += 1
        elif score_conf < 0:
            score_no_conf = -1  # Incorrect regardless of confidence
            incorrect_count += 1
        else:
            score_no_conf = 0  # I don't know
            idk_count += 1
        scores_no_conf.append(score_no_conf)

        if score_conf != 0:
            answered_count += 1

    if not scores_with_conf:
        return None

    # Calculate totals for both scoring methods
    total_score_conf = sum(scores_with_conf)
    total_score_no_conf = sum(scores_no_conf)

    num_questions = len(scores_with_conf)

    # Normalize to fractions (0-1 range)
    # With confidence: range is [-2n, +2n], so max range is 4n
    score_fraction_conf = (total_score_conf + 2 * num_questions) / (
        4 * num_questions
    )

    # Without confidence: range is [-n, +n], so max range is 2n
    score_fraction_no_conf = (total_score_no_conf + num_questions) / (
        2 * num_questions
    )

    # Also keep the 0-100 normalized score for backwards compatibility
    normalized_score = score_fraction_conf * 100

    # Get declared expertise
    screen_answers = conversation_data.get("screen_answers", {})
    declared_expertise = screen_answers.get(domain, "unknown")

    return {
        "prolific_id": prolific_id,
        "domain": domain,
        "declared_expertise": declared_expertise,
        "total_score": total_score_conf,
        "total_score_no_conf": total_score_no_conf,
        "score_fraction_conf": score_fraction_conf,
        "score_fraction_no_conf": score_fraction_no_conf,
        "normalized_score": normalized_score,
        "max_possible": num_questions * 2,
        "questions_answered": num_questions,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "idk": idk_count,
        "answered": answered_count,
        "scores": scores_with_conf,
    }


def assess_directory(
    conversations_dir: Path,
    ground_truth: Dict[str, Dict[str, str]],
    update_files: bool = False,
) -> List[Dict]:
    """Assess all participants in a directory.

    Args:
        conversations_dir: Path to directory containing conversation JSON files
        ground_truth: Ground truth answers loaded from files
        update_files: If True, update the original JSON files with knowledge scores and percentiles

    Returns:
        List of assessment results
    """
    files = sorted(conversations_dir.glob("*.json"))
    results = []
    file_to_data = {}  # Store file_path -> (data, assessment) mapping

    # First pass: assess all participants
    for file_path in files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name}: {e}")
            continue

        assessment = assess_participant(data, ground_truth)
        if assessment:
            assessment["file"] = file_path.name
            results.append(assessment)
            file_to_data[file_path] = (data, assessment)

    # Calculate percentiles based on score_fraction_conf (with confidence scores)
    if results and update_files:
        # Sort all scores to calculate percentiles
        all_scores = sorted([r["score_fraction_conf"] for r in results])
        n = len(all_scores)

        # Second pass: add scores and percentiles to JSON files
        for file_path, (data, assessment) in file_to_data.items():
            score = assessment["score_fraction_conf"]

            # Calculate percentile: percentage of scores less than or equal to this score
            # Using the "exclusive" method (percentage of scores strictly below)
            rank = sum(1 for s in all_scores if s < score)
            percentile = (rank / n) * 100 if n > 1 else 50.0

            # Add scores and percentile to data
            data["knowledge_score_with_confidence"] = assessment[
                "score_fraction_conf"
            ]
            data["knowledge_score_no_confidence"] = assessment[
                "score_fraction_no_conf"
            ]
            data["knowledge_percentile"] = round(percentile, 1)

            try:
                file_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"Warning: Failed to update {file_path.name}: {e}")

    return results


def print_statistics(results: List[Dict]):
    """Print detailed statistics about knowledge assessments."""

    if not results:
        print("No assessment results to display.")
        return

    print(f"Knowledge Assessment Results")
    print("=" * 80)
    print(f"Total participants assessed: {len(results)}")
    print()

    # Overall statistics
    all_scores = [r["normalized_score"] for r in results]
    all_raw_scores = [r["total_score"] for r in results]

    print("Overall Knowledge Scores (% normalized scale):")
    print(f"  Mean: {mean(all_scores):.2f}")
    print(f"  Median: {median(all_scores):.2f}")
    print(
        f"  Std Dev: {pstdev(all_scores):.2f}"
        if len(all_scores) > 1
        else "  Std Dev: 0.00"
    )
    print(f"  Min: {min(all_scores):.2f}")
    print(f"  Max: {max(all_scores):.2f}")
    print()

    print("Raw Scores:")
    print(f"  Mean: {mean(all_raw_scores):.2f}")
    print(f"  Median: {median(all_raw_scores):.2f}")
    print(f"  Min: {min(all_raw_scores):.0f}")
    print(f"  Max: {max(all_raw_scores):.0f}")
    print()

    # By domain
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r["normalized_score"])

    print("Knowledge Scores by Domain:")
    for domain in sorted(by_domain.keys()):
        scores = by_domain[domain]
        print(f"  {domain}:")
        print(f"    Count: {len(scores)}")
        print(f"    Mean: {mean(scores):.2f}")
        print(f"    Median: {median(scores):.2f}")
        print(
            f"    Std Dev: {pstdev(scores):.2f}"
            if len(scores) > 1
            else "    Std Dev: 0.00"
        )
    print()

    # By declared expertise
    by_expertise = defaultdict(list)
    for r in results:
        by_expertise[r["declared_expertise"]].append(r["normalized_score"])

    print("Knowledge Scores by Declared Expertise:")
    for level in ["Novice", "Intermediate", "Expert"]:
        if level in by_expertise:
            scores = by_expertise[level]
            print(f"  {level}:")
            print(f"    Count: {len(scores)}")
            print(f"    Mean: {mean(scores):.2f}")
            print(f"    Median: {median(scores):.2f}")
            print(
                f"    Std Dev: {pstdev(scores):.2f}"
                if len(scores) > 1
                else "    Std Dev: 0.00"
            )

    # Show unknown if any
    if "unknown" in by_expertise:
        scores = by_expertise["unknown"]
        print(f"  unknown: Count={len(scores)}, Mean={mean(scores):.2f}")
    print()

    # Accuracy statistics
    total_questions = sum(r["questions_answered"] for r in results)
    total_correct = sum(r["correct"] for r in results)
    total_incorrect = sum(r["incorrect"] for r in results)
    total_idk = sum(r["idk"] for r in results)

    print("Answer Distribution:")
    print(f"  Total questions answered: {total_questions}")
    print(
        f"  Correct: {total_correct} ({100 * total_correct / total_questions:.1f}%)"
    )
    print(
        f"  Incorrect: {total_incorrect} ({100 * total_incorrect / total_questions:.1f}%)"
    )
    print(
        f"  I don't know: {total_idk} ({100 * total_idk / total_questions:.1f}%)"
    )
    print()

    # Top and bottom performers
    sorted_results = sorted(
        results, key=lambda x: x["normalized_score"], reverse=True
    )

    print("Top 5 Performers:")
    for i, r in enumerate(sorted_results[:5], 1):
        print(
            f"  {i}. Score: {r['normalized_score']:.1f}, "
            f"Domain: {r['domain']}, "
            f"Declared: {r['declared_expertise']}, "
            f"Correct: {r['correct']}/{r['questions_answered']}"
        )
    print()

    print("Bottom 5 Performers:")
    for i, r in enumerate(sorted_results[-5:][::-1], 1):
        print(
            f"  {i}. Score: {r['normalized_score']:.1f}, "
            f"Domain: {r['domain']}, "
            f"Declared: {r['declared_expertise']}, "
            f"Correct: {r['correct']}/{r['questions_answered']}"
        )
    print()

    # Expertise vs actual knowledge correlation
    print("Declared Expertise vs Actual Knowledge:")
    expertise_groups = {}
    for level in ["Novice", "Intermediate", "Expert"]:
        level_results = [r for r in results if r["declared_expertise"] == level]
        if level_results:
            expertise_groups[level] = {
                "count": len(level_results),
                "avg_score": mean(
                    [r["normalized_score"] for r in level_results]
                ),
                "avg_correct_pct": mean(
                    [
                        100 * r["correct"] / r["questions_answered"]
                        for r in level_results
                    ]
                ),
            }

    for level in ["Novice", "Intermediate", "Expert"]:
        if level in expertise_groups:
            g = expertise_groups[level]
            print(
                f"  {level}: {g['count']} participants, "
                f"Avg score: {g['avg_score']:.1f}, "
                f"Avg correct: {g['avg_correct_pct']:.1f}%"
            )


def plot_declared_vs_assessed(
    results: List[Dict], output_file: str = "expertise_comparison.png"
):
    """Plot declared expertise vs assessed knowledge scores.

    Creates a visualization showing:
    1. Box plot of scores by declared expertise level
    2. Scatter plot with jitter showing individual participants
    3. Mean lines for each expertise level

    Args:
        results: List of assessment result dictionaries
        output_file: Path to save the plot (default: expertise_comparison.png)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    if not results:
        print("No results to plot.")
        return

    # Prepare data
    expertise_order = ["Novice", "Intermediate", "Expert"]
    data_by_expertise = {level: [] for level in expertise_order}

    for r in results:
        expertise = r["declared_expertise"]
        if expertise in expertise_order:
            data_by_expertise[expertise].append(r["normalized_score"])

    # Filter out empty categories
    filtered_expertise = [e for e in expertise_order if data_by_expertise[e]]
    filtered_data = [data_by_expertise[e] for e in filtered_expertise]

    if not filtered_data:
        print("No valid expertise data to plot.")
        return

    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create labels with counts included
    labels_with_counts = [
        f"{expertise}\n(n={len(data_by_expertise[expertise])})"
        for expertise in filtered_expertise
    ]

    # Box plot
    bp = ax.boxplot(filtered_data, labels=labels_with_counts, patch_artist=True)

    # Customize box plot colors
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    for patch, color in zip(bp["boxes"], colors[: len(filtered_expertise)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(
        "Assessed Knowledge Score (%)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Knowledge Assessment by Declared Expertise",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # Add mean markers
    means = [mean(data) for data in filtered_data]
    ax.plot(
        range(1, len(means) + 1),
        means,
        "D",
        color="red",
        markersize=8,
        label="Mean",
        zorder=3,
    )
    ax.legend()

    ax.set_xlabel("Declared Expertise Level", fontsize=12, fontweight="bold")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # Also create a summary statistics plot
    create_expertise_summary_plot(
        results, output_file.replace(".png", "_summary.png")
    )


def create_expertise_summary_plot(results: List[Dict], output_file: str):
    """Create a summary plot comparing declared vs assessed expertise."""
    if not MATPLOTLIB_AVAILABLE:
        return

    expertise_order = ["Novice", "Intermediate", "Expert"]

    # Calculate statistics by expertise
    stats = {}
    for level in expertise_order:
        level_results = [r for r in results if r["declared_expertise"] == level]
        if level_results:
            scores = [r["normalized_score"] for r in level_results]
            correct_pcts = [
                100 * r["correct"] / r["questions_answered"]
                for r in level_results
            ]
            stats[level] = {
                "count": len(level_results),
                "mean_score": mean(scores),
                "mean_correct": mean(correct_pcts),
                "std_score": pstdev(scores) if len(scores) > 1 else 0,
            }

    if not stats:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(stats))
    expertise_labels = list(stats.keys())
    means = [stats[e]["mean_score"] for e in expertise_labels]
    stds = [stats[e]["std_score"] for e in expertise_labels]
    counts = [stats[e]["count"] for e in expertise_labels]

    # Create bar plot with error bars
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=10,
        alpha=0.7,
        color=colors[: len(expertise_labels)],
        edgecolor="black",
        linewidth=1.5,
    )

    # Create labels with counts included
    labels_with_counts = [
        f"{label}\n(n={count})"
        for label, count in zip(expertise_labels, counts)
    ]

    ax.set_ylabel(
        "Mean Assessed Knowledge Score (%)", fontsize=13, fontweight="bold"
    )
    ax.set_title(
        "Declared vs Assessed Expertise",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_with_counts, fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    ax.set_xlabel("Declared Expertise Level", fontsize=13, fontweight="bold")

    # Add horizontal reference lines
    overall_mean = mean([r["normalized_score"] for r in results])
    ax.axhline(
        y=overall_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean: {overall_mean:.1f}",
        alpha=0.7,
    )
    ax.legend(fontsize=11)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Summary plot saved to: {output_file}")


def plot_overall_histogram(results: List[Dict], output_file: str):
    """Plot overall histogram of assessed knowledge scores.

    Args:
        results: List of assessment result dictionaries
        output_file: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print(
            "Warning: matplotlib not available. Skipping histogram generation."
        )
        return

    if not results:
        print("No results to plot.")
        return

    scores = [r["normalized_score"] for r in results]

    from scipy import stats

    mean_score = mean(scores)
    median_score = median(scores)

    # Figure 1: Histogram
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.hist(scores, bins=20, edgecolor="black", alpha=0.7, color="#66b3ff")

    ax1.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.1f}",
    )
    ax1.axvline(
        median_score,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_score:.1f}",
    )

    ax1.set_xlabel(
        "Assessed Knowledge Score (%)", fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Distribution of Assessed Knowledge Scores\n(Histogram, n={len(results)})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=11)

    plt.tight_layout()
    histogram_file = output_file.replace(".png", "_histogram.png")
    plt.savefig(histogram_file, dpi=300, bbox_inches="tight")
    print(f"Overall histogram saved to: {histogram_file}")
    plt.close(fig1)

    # Figure 2: Density
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    density = stats.gaussian_kde(scores)
    x_range = np.linspace(0, 100, 300)
    ax2.plot(
        x_range,
        density(x_range),
        linewidth=2.5,
        color="#0055aa",
    )
    ax2.fill_between(x_range, density(x_range), alpha=0.3, color="#66b3ff")

    ax2.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.1f}",
    )
    ax2.axvline(
        median_score,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_score:.1f}",
    )

    ax2.set_xlabel(
        "Assessed Knowledge Score (%)", fontsize=12, fontweight="bold"
    )
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Distribution of Assessed Knowledge Scores\n(Kernel Density Estimate, n={len(results)})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=11)

    plt.tight_layout()
    density_file = output_file.replace(".png", "_density.png")
    plt.savefig(density_file, dpi=300, bbox_inches="tight")
    print(f"Overall density saved to: {density_file}")
    plt.close(fig2)


def plot_histograms_by_expertise(results: List[Dict], output_file: str):
    """Plot overlapping density distributions by declared expertise level.

    Creates a single figure with overlapping density curves for all expertise levels.

    Args:
        results: List of assessment result dictionaries
        output_file: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print(
            "Warning: matplotlib not available. Skipping histogram generation."
        )
        return

    if not results:
        print("No results to plot.")
        return

    from scipy import stats

    expertise_order = ["Novice", "Intermediate", "Expert"]
    colors = ["#ff6666", "#6666ff", "#66ff66"]
    darker_colors = ["#cc0000", "#0000cc", "#00cc00"]

    # Prepare data by expertise
    data_by_expertise = {}
    for level in expertise_order:
        level_scores = [
            r["normalized_score"]
            for r in results
            if r["declared_expertise"] == level
        ]
        if level_scores:
            data_by_expertise[level] = level_scores

    if not data_by_expertise:
        print("No valid expertise data to plot.")
        return

    x_range = np.linspace(0, 100, 300)

    # Figure 1: Overlapping histograms
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for level, color, dark_color in zip(
        data_by_expertise.keys(), colors, darker_colors
    ):
        scores = data_by_expertise[level]

        ax1.hist(
            scores,
            bins=20,
            alpha=0.4,
            color=color,
            edgecolor=dark_color,
            linewidth=1.5,
            label=f"{level} (n={len(scores)})",
        )

        # Add mean line
        mean_score = mean(scores)
        ax1.axvline(
            mean_score,
            color=dark_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

    ax1.set_xlabel(
        "Assessed Knowledge Score (%)", fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Distribution of Assessed Knowledge by Declared Expertise\n(Histograms)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    histogram_file = output_file.replace(".png", "_histograms.png")
    plt.savefig(histogram_file, dpi=300, bbox_inches="tight")
    print(f"Histograms by expertise saved to: {histogram_file}")
    plt.close(fig1)

    # Figure 2: Overlapping density curves
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for level, color, dark_color in zip(
        data_by_expertise.keys(), colors, darker_colors
    ):
        scores = data_by_expertise[level]

        # Add kernel density estimate
        density = stats.gaussian_kde(scores)
        ax2.plot(
            x_range,
            density(x_range),
            linewidth=2.5,
            color=dark_color,
            linestyle="-",
            label=f"{level} (n={len(scores)})",
        )
        ax2.fill_between(
            x_range,
            density(x_range),
            alpha=0.2,
            color=color,
        )

        # Add mean line
        mean_score = mean(scores)
        ax2.axvline(
            mean_score,
            color=dark_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

    ax2.set_xlabel(
        "Assessed Knowledge Score (%)", fontsize=12, fontweight="bold"
    )
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Distribution of Assessed Knowledge by Declared Expertise",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    density_file = output_file.replace(".png", "_density.png")
    plt.savefig(density_file, dpi=300, bbox_inches="tight")
    print(f"Density curves by expertise saved to: {density_file}")
    plt.close(fig2)


def plot_scoring_methods_comparison(
    results: List[Dict[str, Any]], output_file: str
) -> None:
    """
    Create overlapping histograms comparing the two scoring methods.

    Args:
        results: List of participant results
        output_file: Path to save the plot
    """
    scores_with_conf = [r["score_fraction_conf"] * 100 for r in results]
    scores_no_conf = [r["score_fraction_no_conf"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create overlapping histograms with 50 bins
    ax.hist(
        scores_with_conf,
        bins=50,
        alpha=0.6,
        label=f"With Confidence (mean={np.mean(scores_with_conf):.1f}%)",
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        scores_no_conf,
        bins=50,
        alpha=0.6,
        label=f"Without Confidence (mean={np.mean(scores_no_conf):.1f}%)",
        color="#A23B72",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Knowledge Score (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        "Comparison of Scoring Methods: With vs Without Confidence",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Scoring methods comparison saved to: {output_file}")
    plt.close(fig)


def copy_top_bottom_conversations(
    conversations_dir: Path,
    output_dir: Path,
    percentile_threshold: float = 10.0,
) -> None:
    """
    Copy top and bottom percentile conversations to a single directory with percentile prefixes.

    Args:
        conversations_dir: Directory containing original conversation files
        output_dir: Directory to copy selected conversations to
        percentile_threshold: Threshold for top/bottom selection (default: 10%)
    """
    # Load all conversation files with percentiles
    all_data = []
    for file_path in sorted(conversations_dir.glob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if "knowledge_percentile" in data:
                all_data.append(
                    {
                        "file_path": file_path,
                        "data": data,
                        "percentile": data["knowledge_percentile"],
                        "score_conf": data.get(
                            "knowledge_score_with_confidence", 0
                        ),
                        "score_no_conf": data.get(
                            "knowledge_score_no_confidence", 0
                        ),
                    }
                )
        except Exception as e:
            print(f"Warning: Failed to read {file_path.name}: {e}")
            continue

    if not all_data:
        print("No conversations with percentiles found.")
        return

    # Sort by percentile
    all_data.sort(key=lambda x: x["percentile"])

    # Calculate how many files for top and bottom
    total_files = len(all_data)
    n_select = max(1, int(total_files * percentile_threshold / 100))

    # Select bottom and top performers
    bottom_performers = all_data[:n_select]
    top_performers = all_data[-n_select:]

    # Combine both groups
    selected_conversations = bottom_performers + top_performers

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write all selected conversations with percentile prefix
    print(
        f"\nCopying top and bottom {percentile_threshold}% ({len(selected_conversations)} files total):"
    )
    for i, item in enumerate(
        sorted(selected_conversations, key=lambda x: x["percentile"]), 1
    ):
        src = item["file_path"]
        # Format percentile as integer (e.g., "00", "05", "99")
        percentile_str = f"{int(item['percentile']):02d}"
        new_filename = f"{percentile_str}_{src.name}"
        dst = output_dir / new_filename

        try:
            dst.write_text(
                json.dumps(item["data"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"  {i}. {new_filename} - Percentile: {item['percentile']:.1f}, "
                f"Score: {item['score_conf']*100:.1f}%"
            )
        except Exception as e:
            print(f"Warning: Failed to write {new_filename}: {e}")

    print(f"\nSelected conversations copied to: {output_dir}")
    print(f"  Total files: {len(selected_conversations)}")
    print(f"  Bottom {percentile_threshold}%: {len(bottom_performers)} files")
    print(f"  Top {percentile_threshold}%: {len(top_performers)} files")


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv

    conversations_dir = (
        Path(argv[1]) if len(argv) > 1 else Path("post_process/conversations")
    )
    gt_dir = (
        Path(argv[2]) if len(argv) > 2 else Path("data/questionnaires/answers")
    )

    if not conversations_dir.exists():
        print(f"Error: Conversations directory not found: {conversations_dir}")
        return 1

    if not gt_dir.exists():
        print(f"Error: Ground truth directory not found: {gt_dir}")
        return 1

    print(f"Loading ground truth from: {gt_dir}")
    ground_truth = load_ground_truth(gt_dir)

    domains_loaded = len(ground_truth)
    total_questions = sum(len(answers) for answers in ground_truth.values())
    print(
        f"Loaded {domains_loaded} domains with {total_questions} total questions"
    )
    print()

    print(f"Assessing participants from: {conversations_dir}")
    results = assess_directory(
        conversations_dir, ground_truth, update_files=True
    )
    print(f"Assessed {len(results)} participants")
    print(f"Updated JSON files with knowledge scores")
    print()

    print_statistics(results)

    # Generate plots
    output_dir = Path("post_process/plots")
    output_dir.mkdir(exist_ok=True)

    # Box plot and summary
    plot_file = output_dir / "expertise_comparison.png"
    plot_declared_vs_assessed(results, str(plot_file))

    # Histograms
    histogram_file = output_dir / "knowledge_histogram.png"
    plot_overall_histogram(results, str(histogram_file))

    histogram_by_expertise_file = (
        output_dir / "knowledge_histogram_by_expertise.png"
    )
    plot_histograms_by_expertise(results, str(histogram_by_expertise_file))

    # Comparison of scoring methods
    scoring_comparison_file = output_dir / "scoring_methods_comparison.png"
    plot_scoring_methods_comparison(results, str(scoring_comparison_file))

    # Copy top and bottom 10% to selected_conversations
    selected_dir = Path("post_process/selected_conversations")
    copy_top_bottom_conversations(
        conversations_dir, selected_dir, percentile_threshold=20.0
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
