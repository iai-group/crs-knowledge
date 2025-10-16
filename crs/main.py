"""Example of a main file."""

import argparse
import json
import logging
import os
import random
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from crs.components.pages import render_current_page
from crs.config_loader import get_config_loader

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_new_save_filename() -> str:
    """Generates a new filename for saving the conversation."""
    unique_id = str(uuid.uuid4())
    date_part = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{date_part}_{unique_id}.json"


def init(args, model_name: str = None):
    """Initialize the Streamlit application with configuration.

    Args:
        model_name: Optional model name override. If None, uses config default.
    """
    # Load configuration
    config_loader = get_config_loader()

    if model_name is None:
        model_name = config_loader.get_default_model()

    # Set page config using UI configuration
    ui_config = config_loader.get_ui_config()
    st.set_page_config(
        page_title=ui_config.get("page_title", "Recommendation Game"),
        page_icon="🤖",
        layout="wide",
    )

    if "prolific" not in st.session_state:
        st.session_state.prolific = {}
        prolific_id = st.query_params.get("prolific_pid", "")
        if prolific_id:
            st.session_state.prolific["id"] = prolific_id
        study_id = st.query_params.get("study_id", "")
        if study_id:
            st.session_state.prolific["study_id"] = study_id
        session_id = st.query_params.get("session_id", "")
        if session_id:
            st.session_state.prolific["session_id"] = session_id

    if "auto_save_filename" not in st.session_state:
        st.session_state.auto_save_filename = get_new_save_filename()

    if "current_page" not in st.session_state:
        if args.page:
            st.session_state.current_page = args.page
        else:
            st.session_state.current_page = "screen"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    if "model_name" not in st.session_state:
        st.session_state.model_name = model_name

    if "domains" not in st.session_state:
        # Store domains in lowercase internally with underscores for file paths
        st.session_state.domains = [
            "bicycle",
            "digital_camera",
            "smartwatch",
            "running_shoes",
            "laptop",
        ]

    if args.domain:
        # Convert argument to lowercase for consistency
        domain_lower = args.domain.lower()
        if domain_lower in st.session_state.domains:
            st.session_state.current_domain = domain_lower
        else:
            raise ValueError(
                f"Invalid domain '{args.domain}'. Must be one of {st.session_state.domains}"
            )

    if "task_version" not in st.session_state:
        # Randomly select short or long version for tasks
        st.session_state.task_version = random.choice(["short", "long"])


def auto_save_conversation() -> None:
    """Automatically saves the current conversation to a JSON file."""
    os.makedirs("exports/conversations", exist_ok=True)
    filepath = os.path.join(
        "exports", "conversations", st.session_state.auto_save_filename
    )
    conversation_data = {
        "last_saved": datetime.now().isoformat(),
        "prolific": st.session_state.get("prolific", {}),
        "screen_answers": st.session_state.get("screen_answers", {}),
        "pre_task_answers": st.session_state.get("pre_answers", {}),
        "post_task_answers": st.session_state.get("post_answers", {}),
        "current_domain": st.session_state.get("current_domain", ""),
        "task_version": st.session_state.get("task_version", ""),
        "messages": [],
    }
    # Extract messages from chat_history.
    for message in st.session_state.chat_log:
        role = (
            "human"
            if isinstance(message, HumanMessage)
            else "ai" if isinstance(message, AIMessage) else "system"
        )
        conversation_data["messages"].append(
            {"role": role, "content": message.content}
        )
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Conversation autosaved to {filepath}")


def main(args: argparse.Namespace) -> None:
    """Main function to set up the Streamlit UI.

    Args:
        args: Command-line arguments.
    """
    logger.info("Starting main function.")
    logger.debug(f"Arguments: {args}")

    # Provide helper functions to the pages module via st.session_state or a
    # small helpers namespace. Pages expect `helpers` to have the same helpers
    # that were present in this module previously.
    st.session_state.auto_save_conversation = auto_save_conversation
    st.session_state.debug = args.debug

    render_current_page()


def parse_args() -> argparse.Namespace:
    """Parses arguments from command-line call.

    Returns:
        Arguments from command-line call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        dest="debug",
        help="Debugging mode",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--page",
        type=str,
        choices=[
            "front",
            "knowledge",
            "start",
            "chat",
            "post",
            "end",
            "prolific_redirect",
            "screen",
        ],
        help="Page to start on",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "bicycle",
            "digital_camera",
            "smartwatch",
            "running_shoes",
            "laptop",
        ],
        help="Domain to use (lowercase with underscores)",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    init(args)
    main(args)
