import json
import logging
import random

import streamlit as st

logger = logging.getLogger(__name__)

TASKS_PATH = "data/tasks/"


def load_task(task: str) -> dict:
    """
    Loads task data and returns a randomly selected task.
    Each task is expected to be a dict with "story" and "img path" keys.
    """
    task_file = f"{TASKS_PATH}/{task}.json"
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if tasks:
            selected_task = random.choice(tasks)
            return selected_task
        else:
            logger.error("No tasks found in %s", task_file)
            return None
    except Exception as e:
        logger.error("Error loading tasks from %s: %s", task_file, e)
        return None


def build_task():
    domain = st.session_state.get("current_domain", "")
    task = load_task(domain.lower())

    st.subheader("Your Task")
    st.write(task.get("story", "No story provided."))
    img_path = task.get("img_path")
    if img_path:
        st.image(img_path, width=300, caption="Task Image")

    st.session_state.task = task
