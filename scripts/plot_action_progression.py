#!/usr/bin/env python3
"""Plot progression of system actions through conversation turns.

This script reads the system_actions.csv file and creates a line plot showing
the frequency of each action type across conversation turns (capped at 12).

Usage:
  python3 scripts/plot_action_progression.py [actions_csv] [output_plot] [--normalize] [--min-percentile N] [--max-percentile N]

Defaults:
  actions_csv: post_process/system_actions.csv
  output_plot: post_process/plots/action_progression.png (or action_progression_normalized.png with --normalize)

Options:
  --normalize: Normalize values by dividing by total actions at each turn
  --min-percentile N: Only include actions from conversations with percentile >= N
  --max-percentile N: Only include actions from conversations with percentile <= N

Examples:
  # Bottom 20% performers
  python3 scripts/plot_action_progression.py --max-percentile 20

  # Top 40% performers
  python3 scripts/plot_action_progression.py --min-percentile 60

  # Middle range
  python3 scripts/plot_action_progression.py --min-percentile 40 --max-percentile 60
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_action_progression(
    actions_csv: Path,
    output_plot: Path,
    max_turn: int = 12,
    normalize: bool = False,
    min_percentile: Optional[float] = None,
    max_percentile: Optional[float] = None,
) -> None:
    """
    Plot the progression of actions through conversation turns.

    Args:
        actions_csv: Path to CSV file with system actions
        output_plot: Path to save the plot
        max_turn: Maximum turn number to include (default: 12)
        normalize: If True, normalize values by total actions at each turn
        min_percentile: If set, only include actions with percentile >= this value
        max_percentile: If set, only include actions with percentile <= this value
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        return

    # Read the CSV file
    with actions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        actions = list(reader)

    print(f"Loaded {len(actions)} actions from {actions_csv}")

    # Filter by percentile if specified
    if min_percentile is not None or max_percentile is not None:
        original_count = len(actions)
        filtered_actions = []
        for entry in actions:
            percentile = float(entry["percentile"])
            if min_percentile is not None and percentile < min_percentile:
                continue
            if max_percentile is not None and percentile > max_percentile:
                continue
            filtered_actions.append(entry)
        actions = filtered_actions

        filter_desc = []
        if min_percentile is not None:
            filter_desc.append(f"≥{min_percentile}")
        if max_percentile is not None:
            filter_desc.append(f"≤{max_percentile}")
        print(
            f"Filtered to {len(actions)} actions (percentile {' and '.join(filter_desc)})"
        )

    # Count actions by turn for each action type
    # Structure: {action_type: {turn: count}}
    action_by_turn = defaultdict(lambda: defaultdict(int))

    for entry in actions:
        action = entry["action"]
        turn = int(entry["turn"])

        # Cap at max_turn
        if turn <= max_turn:
            action_by_turn[action][turn] += 1

    # Define action types we want to plot (excluding Redirect due to low frequency)
    action_types = [
        "Recommend",
        "Elicit",
        "Answer",
        "Answer_About_Recommendation",
    ]

    # Calculate total actions per turn for normalization
    total_by_turn = defaultdict(int)
    for action in action_types:
        for turn in range(1, max_turn + 1):
            total_by_turn[turn] += action_by_turn[action][turn]

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Color scheme
    colors = {
        "Recommend": "#2E86AB",
        "Elicit": "#A23B72",
        "Answer": "#F18F01",
        "Answer_About_Recommendation": "#06A77D",
    }

    # Line styles for better distinction
    line_styles = {
        "Recommend": "-",
        "Elicit": "--",
        "Answer": "-.",
        "Answer_About_Recommendation": ":",
    }

    # Markers for data points
    markers = {
        "Recommend": "o",
        "Elicit": "s",
        "Answer": "^",
        "Answer_About_Recommendation": "D",
    }

    # Plot each action type
    turns = list(range(1, max_turn + 1))

    for action in action_types:
        if normalize:
            # Normalize by dividing by total actions at each turn
            values = []
            for turn in turns:
                if total_by_turn[turn] > 0:
                    values.append(
                        action_by_turn[action][turn] / total_by_turn[turn]
                    )
                else:
                    values.append(0.0)
        else:
            # Use raw counts
            values = [action_by_turn[action][turn] for turn in turns]

        ax.plot(
            turns,
            values,
            label=action.replace("_", " ").replace("Question", "Answer"),
            color=colors[action],
            linestyle=line_styles[action],
            marker=markers[action],
            markersize=6,
            linewidth=2.5,
            markevery=2,  # Show marker every 2 points to reduce clutter
        )

    # Customize the plot
    ax.set_xlabel("Turn", fontsize=14, fontweight="bold")

    if normalize:
        ax.set_ylabel("Proportion", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")

    # Set x-axis to show all turns
    ax.set_xticks(turns)
    ax.set_xlim(0.5, max_turn + 0.5)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Add legend
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9, edgecolor="black")

    # Tight layout
    plt.tight_layout()

    # Save the plot
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_plot}")

    # Print summary statistics (normalized)
    print("\nNormalized action proportions by turn (first 10 turns):")
    print(f"{'Turn':<6}", end="")
    for action in action_types:
        print(f"{action.replace('_', ' '):<20}", end="")
    print()

    for turn in range(1, min(11, max_turn + 1)):
        print(f"{turn:<6}", end="")
        for action in action_types:
            if total_by_turn[turn] > 0:
                proportion = action_by_turn[action][turn] / total_by_turn[turn]
                print(f"{proportion:<20.3f}", end="")
            else:
                print(f"{'0.000':<20}", end="")
        print()

    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else argv

    # Parse flags and options
    normalize = "--normalize" in argv
    min_percentile = None
    max_percentile = None

    # Extract percentile filters
    filtered_argv = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--normalize":
            pass  # Already handled
        elif arg == "--min-percentile":
            if i + 1 < len(argv):
                min_percentile = float(argv[i + 1])
                i += 1  # Skip the next argument
            else:
                print("Error: --min-percentile requires a value")
                return 1
        elif arg == "--max-percentile":
            if i + 1 < len(argv):
                max_percentile = float(argv[i + 1])
                i += 1  # Skip the next argument
            else:
                print("Error: --max-percentile requires a value")
                return 1
        else:
            filtered_argv.append(arg)
        i += 1

    argv = filtered_argv

    # Parse arguments
    if len(argv) > 1:
        actions_csv = Path(argv[1])
    else:
        actions_csv = Path("post_process/system_actions.csv")

    if len(argv) > 2:
        output_plot = Path(argv[2])
    else:
        # Auto-adjust output path based on normalization and percentile filters
        parts = ["action_progression"]

        if min_percentile is not None:
            parts.append(f"min{int(min_percentile)}")
        if max_percentile is not None:
            parts.append(f"max{int(max_percentile)}")
        if normalize:
            parts.append("normalized")

        filename = "_".join(parts) + ".png"
        output_plot = Path("post_process/plots") / filename

    if not actions_csv.exists():
        print(f"Error: Actions CSV not found: {actions_csv}")
        return 1

    plot_action_progression(
        actions_csv,
        output_plot,
        normalize=normalize,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
