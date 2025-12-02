"""Prompt loading utilities."""

import os
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=128)
def _cached_file_read(file_path: str) -> str:
    """Load a file with caching."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")


class PromptLoader:
    """Handles loading and caching of external prompt files."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))
            )
            base_path = os.path.join(project_root, "data/prompts")
        self.base_path = base_path

    def load_prompt(self, file_name: str) -> str:
        """Load a prompt from an external file with caching."""
        file_path = os.path.join(self.base_path, file_name)
        return _cached_file_read(file_path)

    def get_all_prompts(self) -> Dict[str, str]:
        """Load all available prompts."""
        return {
            "decision_stage": self.load_prompt("decision_stage_prompt.txt"),
            "history_summarization": self.load_prompt(
                "history_summarization_prompt.txt"
            ),
            "recommendation": self.load_prompt(
                "recommendation_stage_prompt.txt"
            ),
            "response_stage": self.load_prompt("response_stage_prompt.txt"),
        }


_prompt_loader = PromptLoader()


def load_prompt(file_name: str) -> str:
    """Load a prompt from an external file."""
    return _prompt_loader.load_prompt(file_name)


def decision_stage_prompt() -> str:
    return _prompt_loader.load_prompt("decision_stage_prompt.txt")


def history_summarization_prompt() -> str:
    return _prompt_loader.load_prompt("history_summarization_prompt.txt")


def recommendation_prompt() -> str:
    return _prompt_loader.load_prompt("recommendation_stage_prompt.txt")


def response_stage_prompt() -> str:
    return _prompt_loader.load_prompt("response_stage_prompt.txt")
