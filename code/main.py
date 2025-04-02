"""Example of a main file."""

import argparse
import json
import logging
import os
import random
import uuid
from code.ui import (
    build_chatbot,
    build_introduction,
    build_questionnaire,
    build_task,
)
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

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


def init():
    st.set_page_config(
        page_title="Recommendation Game",
        page_icon="🤖",
        layout="wide",
    )

    if "auto_save_filename" not in st.session_state:
        st.session_state.auto_save_filename = get_new_save_filename()

    if "current_page" not in st.session_state:
        st.session_state.current_page = "start"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "domains" not in st.session_state:
        st.session_state.domains = ["Bicycle", "Movies"]

    if "domain_pair" not in st.session_state:
        if len(st.session_state.domains) >= 2:
            st.session_state.domain_pair = random.sample(
                st.session_state.domains, 2
            )
        else:
            st.session_state.domain_pair = st.session_state.domains


def reset():
    st.session_state.auto_save_filename = get_new_save_filename()
    del st.session_state.domain_pair
    del st.session_state.current_domain
    st.session_state.chat_history = []
    st.session_state.post_answers = {}

    if len(st.session_state.domains) > 1:
        st.session_state.current_page = "pre"
        st.session_state.pre_answers = {}
    else:
        st.session_state.current_page = "chat"
    st.rerun()


def auto_save_conversation() -> None:
    """Automatically saves the current conversation to a JSON file."""
    os.makedirs("exports/conversations", exist_ok=True)
    filepath = os.path.join(
        "exports", "conversations", st.session_state.auto_save_filename
    )
    conversation_data = {
        "last_saved": datetime.now().isoformat(),
        "pre_task_answers": st.session_state.get("pre_answers", {}),
        "post_task_answers": st.session_state.get("post_answers", {}),
        "current_domain": st.session_state.get("current_domain", ""),
        "messages": [],
    }
    # Extract messages from chat_history.
    for message in st.session_state.chat_history:
        role = (
            "human"
            if isinstance(message, HumanMessage)
            else "ai" if isinstance(message, AIMessage) else "unknown"
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

    st.session_state.auto_save_conversation = auto_save_conversation
    col1, col2 = st.columns(2)

    if st.session_state.current_page == "start":
        with col1:
            build_introduction()
        if st.button("Next"):
            st.session_state.current_page = "pre"
            st.rerun()

    if st.session_state.current_page == "pre":
        with col1:
            build_questionnaire("pre", next_page="chat")

    elif st.session_state.current_page == "chat":
        if "current_domain" not in st.session_state:
            # st.session_state.current_domain = random.choice(
            #     st.session_state.domain_pair
            # )
            chosen = st.session_state.pre_answers.get("chosen")
            if chosen in st.session_state.domain_pair:
                st.session_state.current_domain = chosen
            else:
                st.session_state.current_domain = random.choice(
                    st.session_state.domain_pair
                )
            st.session_state.domains.remove(st.session_state.current_domain)
        with col1:
            build_chatbot()
        with col2:
            build_task()

            if st.button("Done"):
                st.session_state.current_page = "post"
                st.rerun()

    elif st.session_state.current_page == "post":
        with col1:
            build_questionnaire("post", next_page="end")

    elif st.session_state.current_page == "end":
        with col1:
            st.header("Thank you for participating!")
            st.markdown("Your responses have been recorded.")
            if len(st.session_state.domains) > 0:
                st.markdown(
                    "If you wish to try another domain, please click the button below."
                )
                if st.button("Restart"):
                    reset()


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    init()
    main(args)
