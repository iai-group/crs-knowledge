"""Page rendering helpers extracted from crs/main.py to simplify routing.

This module exposes a single function `render_current_page(st, helpers)` which
renders the current page based on `st.session_state.current_page`.

`helpers` is a small namespace object (or module) containing functions the
pages need from the main module, e.g. `save_response`, `count_bicycle_category`,
`reset`, and `auto_save_conversation`. This avoids circular imports and keeps
`crs/main.py` as the entry point.
"""

import streamlit as st

from crs.components import (
    build_chatbot,
    build_introduction,
    build_questionnaire,
    build_task,
    build_timer,
)
from crs.components.task import build_recommended_items_tracker


def render_current_page():
    """Render the page currently set in st.session_state.current_page.

    helpers: an object or module with attributes used by the pages:
      - save_response
      - count_bicycle_category
      - reset
      - auto_save_conversation

    """
    col1, col2 = st.columns(2)

    current = st.session_state.get("current_page", "screen")
    print(f"🎬 render_current_page called, current_page = '{current}'")

    if current == "screen":
        build_questionnaire("screen", next_page="pre")
        return

    if current == "pre":
        build_questionnaire("pre", next_page="start")
        return

    if current == "start":
        with col1:
            build_introduction()
        if st.button("Next"):
            st.session_state.current_page = "chat"
            st.rerun()
        return

    if current == "chat":
        # Initialize timer when first entering chat page
        if "chat_start_time" not in st.session_state:
            import time

            st.session_state.chat_start_time = time.time()
            st.session_state.chat_timer_expired = False

        # Build task first to ensure it's initialized before chatbot
        with col2:
            build_task()

        # Now build chatbot (task should be initialized)
        with col1:
            st.header("Assistant")
            build_timer()
            build_chatbot()

        with col2:
            build_recommended_items_tracker()

        return

    if current == "post":
        with col1:
            build_questionnaire("post", next_page="end")
        return

    if current == "end":
        # Mark study as completed and save
        if not st.session_state.get("study_completed", False):
            st.session_state.study_completed = True
            if "auto_save_conversation" in st.session_state:
                st.session_state.auto_save_conversation()

        # Show explicit completion instructions and Prolific code
        st.title("Study complete")
        completion_url = (
            "https://app.prolific.com/submissions/complete?cc=COF2N3M0"
        )
        completion_code = "COF2N3M0"
        st.write("Thanks. Click the button below to finish on Prolific:")
        st.code(completion_url)
        if st.button("Open Prolific completion link in new tab"):
            html = (
                f"<script>window.open('{completion_url}', '_blank');</script>"
            )
            st.components.v1.html(html, height=0)
        st.write("If it doesn’t open, copy this code into Prolific:")
        st.code(completion_code)
        return

    if current == "prolific_redirect":
        st.title("Study completion")
        prolific_url = (
            "https://app.prolific.com/submissions/complete?cc=CWOANW8B"
        )
        completion_code = "CWOANW8B"
        st.markdown("### Condition full")
        st.write(
            "Unfortunately, we have enough participants in this condition. "
            "please use the link below to mark your study as complete on Prolific."
        )
        st.code(prolific_url)
        if st.button("Open completion link in new tab"):
            html = f"<script>window.open('{prolific_url}', '_blank');</script>"
            st.components.v1.html(html, height=0)
        st.write("If it doesn’t open, copy this code into Prolific:")
        st.code(completion_code)
        return
