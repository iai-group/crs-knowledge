"""Page rendering helpers extracted from crs/main.py to simplify routing.

This module exposes a single function `render_current_page(st, helpers)` which
renders the current page based on `st.session_state.current_page`.

`helpers` is a small namespace object (or module) containing functions the
pages need from the main module, e.g. `save_response`, `count_bicycle_category`,
`reset`, and `auto_save_conversation`. This avoids circular imports and keeps
`crs/main.py` as the entry point.
"""

import os
import random

import streamlit as st

from crs.components import (
    build_chatbot,
    build_introduction,
    build_questionnaire,
    build_task,
)


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
        if "current_domain" not in st.session_state:
            if st.session_state.get("domains"):
                st.session_state.current_domain = random.choice(
                    st.session_state.domains
                )
                try:
                    st.session_state.domains.remove(
                        st.session_state.current_domain
                    )
                except ValueError:
                    pass
            else:
                st.session_state.current_domain = "Bicycle"
        with col2:
            build_task()

            # Determine whether the conversation already found the target.
            conversation_state = getattr(
                st.session_state, "conversation_state", None
            )
            target_found = (
                conversation_state and conversation_state.is_target_found()
                if conversation_state
                else False
            )

            # Only render the Give up button when the target has NOT been found.
            if not target_found:
                # When clicked, open a confirmation modal instead of navigating immediately.
                if st.button("Give up", type="secondary"):
                    # Use a session state flag to show the confirmation modal.
                    st.session_state.show_give_up_confirm = True

            # Render confirmation modal if requested in session state.
            if st.session_state.get("show_give_up_confirm"):
                # Use an expander-style modal built from st.modal when available,
                # fall back to an info box with buttons if not. Streamlit >=1.18
                # provides st.modal; to keep compatibility we try to use it if present.
                try:
                    with st.modal("Confirm Give up"):
                        st.warning(
                            "You have not completed the task yet. Are you sure you want to give up?"
                        )
                        cola, colb = st.columns(2)
                        with cola:
                            if st.button("Yes, give up"):
                                st.session_state.current_page = "post"
                                # clear the modal flag so it doesn't re-open
                                st.session_state.show_give_up_confirm = False
                                st.rerun()
                        with colb:
                            if st.button("Cancel"):
                                st.session_state.show_give_up_confirm = False
                                st.rerun()
                except Exception:
                    # Fallback UI if st.modal is not available in this Streamlit version.
                    st.warning(
                        "You have not completed the task yet. Are you sure you want to give up?"
                    )
                    cola, colb = st.columns(2)
                    with cola:
                        if st.button("Yes, give up"):
                            st.session_state.current_page = "post"
                            st.session_state.show_give_up_confirm = False
                            st.rerun()
                    with colb:
                        if st.button("Cancel"):
                            st.session_state.show_give_up_confirm = False
                            st.rerun()

        with col1:
            build_chatbot()
        return

    if current == "post":
        with col1:
            build_questionnaire("post", next_page="end")
        return

    if current == "end":
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
