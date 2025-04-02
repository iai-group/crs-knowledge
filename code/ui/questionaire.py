import json
import logging
import os
import random

import streamlit as st

logger = logging.getLogger(__name__)

QUESTIONNAIRE_DIR = "data/questionnaires/"


def load_questionnaire(directory: str, name: str) -> list:
    """
    If needed, load additional questions for 'post' or other questionnaires
    from JSON. For 'pre', we do a custom pairwise question, so we might not
    even need to load a file.
    """
    questions = []
    filepath = os.path.join(directory, f"{name}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions.extend(data)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    return questions


def build_questionnaire(page: str, next_page: str = None) -> None:
    """Builds the pre/post questionnaire pages."""
    st.header(f"{page.capitalize()} Questionnaire")

    # ----------- Pre-Questionnaire (Pairwise) -----------
    if page == "pre":

        # Build a simple form
        with st.form("pre_questionnaire_form"):
            answers = {}
            st.write("Which domain would you say you know more about?")
            chosen = st.radio(
                "", st.session_state.domain_pair, key="pre_question"
            )
            submitted = st.form_submit_button("Submit Questionnaire")
            if submitted:
                # Store the user's chosen domain
                answers["options"] = st.session_state.domain_pair
                answers["chosen"] = chosen

                st.session_state["pre_answers"] = answers

                # Auto-save conversation if desired
                if "auto_save_conversation" in st.session_state:
                    st.session_state.auto_save_conversation()

                # Move on to the next page
                if next_page:
                    st.session_state.current_page = next_page
                    st.rerun()

    # ----------- Post-Questionnaire (Simple Example) -----------
    elif page == "post":
        # If you still want to load from JSON for your post questions, do so:
        questions = load_questionnaire(QUESTIONNAIRE_DIR, "post")

        # If you do have a JSON structure, handle it similarly in a loop
        with st.form("post_questionnaire_form"):
            answers = {}
            for i, q in enumerate(questions):
                question_text = q.get("question", f"Question {i+1}")
                question_type = q.get("type", "text")
                key = f"post_q{i+1}"

                if question_type == "scale":
                    scale_min = q.get("scale_min", 1)
                    scale_max = q.get("scale_max", 5)
                    answers[f"q{i+1}"] = st.slider(
                        question_text,
                        min_value=scale_min,
                        max_value=scale_max,
                        key=key,
                    )
                elif question_type == "radio":
                    options = q.get("options", ["Yes", "No"])
                    answers[f"q{i+1}"] = st.radio(
                        question_text, options, key=key
                    )
                else:
                    answers[f"q{i+1}"] = st.text_input(question_text, key=key)

            submitted = st.form_submit_button("Submit Post-Questionnaire")
            if submitted:
                st.session_state["post_answers"] = answers
                if "auto_save_conversation" in st.session_state:
                    st.session_state.auto_save_conversation()
                st.success("Thank you for completing the post-questionnaire!")
                if next_page:
                    st.session_state.current_page = next_page
                    st.rerun()

    # Fallback if page is neither 'pre' nor 'post'
    else:
        st.info("No questionnaire defined for this page.")
