import json
import logging
import random

import streamlit as st
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

TASKS_PATH = "data/tasks/"

# Common tip for all tasks
TASK_TIP = "**Tip:** Don’t simply copy the text above. Work with the assistant to identify which features matter most and how they fit the story. When you find the best match, select it from the list below and explain why it fits better than the others, referring to specific features or technical details. You can also ask the assistant to clarify any features you’re unsure about."


def load_task(domain: str, version: str = None) -> dict:
    """
    Loads task data and returns a randomly selected task.
    Each task is expected to be a dict with "story" and "img path" keys.

    Args:
        domain: The domain name (e.g., "Bicycle", "Digital_Camera")
        version: Optional version ("short" or "long"). If None, randomly selected.

    Returns:
        The selected task dict, or None if loading fails
    """
    # Determine version if not provided
    if version is None:
        version = random.choice(["short", "long"])

    # Construct the task file path
    domain_lower = domain.lower().replace(" ", "_")
    task_file = f"{TASKS_PATH}/{domain_lower}_{version}.json"

    try:
        with open(task_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if tasks:
            selected_task = random.choice(tasks)
            return selected_task, version
        else:
            logger.error("No tasks found in %s", task_file)
            return None, None
    except Exception as e:
        logger.error("Error loading tasks from %s: %s", task_file, e)
        return None, None


def build_task():
    domain = st.session_state.get("current_domain", "")

    # Get the task version from session state, or it will be randomly selected
    version = st.session_state.get("task_version", None)

    task, selected_version = load_task(domain, version)

    # Store the version in session state if it was just selected
    if selected_version and "task_version" not in st.session_state:
        st.session_state.task_version = selected_version

    if task:
        st.subheader("Your Task")
        st.write(task.get("story", "No story provided."))

        # Display the tip in an info box for better visibility
        st.info(TASK_TIP)

        # img_path = task.get("img_path")
        # if img_path:
        #     st.image(img_path, width=300, caption="Task Image")

        st.session_state.task = task
    else:
        # Capitalize domain for display to user
        domain_display = domain.replace("_", " ").title()
        st.error(f"Could not load task for domain: {domain_display}")


def get_item_image_url(item: dict) -> str:
    """Extract the best available image URL from item metadata."""
    images = item.get("images", [])

    if not images:
        return ""

    # Get the first image (usually the main product image)
    main_image = images[0]

    # Prefer large, then hi_res, then thumb
    if "large" in main_image and main_image["large"]:
        return main_image["large"]
    elif "hi_res" in main_image and main_image["hi_res"]:
        return main_image["hi_res"]
    elif "thumb" in main_image and main_image["thumb"]:
        return main_image["thumb"]

    return ""


def build_recommended_items_tracker():
    """Display and track recommended items with checkboxes."""
    # Initialize displayed items list if not exists
    if "displayed_items" not in st.session_state:
        st.session_state.displayed_items = []
    if "selected_item" not in st.session_state:
        st.session_state.selected_item = None

    # Extract recommended items from conversation state if available
    conversation_state = getattr(st.session_state, "conversation_state", None)
    if hasattr(conversation_state, "recommended_items"):
        # Update our local list with new recommendations
        for recommended_item in conversation_state.recommended_items:
            # Extract item ID for comparison
            item_id = (
                recommended_item.get("id")
                or recommended_item.get("parent_asin")
                or recommended_item.get("metadata", {}).get("parent_asin")
            )

            if item_id and item_id not in [
                item["id"] for item in st.session_state.displayed_items
            ]:
                # Extract item name/title for display
                item_name = None
                if "title" in recommended_item:
                    item_name = recommended_item["title"]
                elif "content" in recommended_item:
                    content = recommended_item["content"]
                    if isinstance(content, dict):
                        item_name = content.get("title", content.get("name"))
                    elif isinstance(content, str) and content.startswith(
                        "title:"
                    ):
                        # Extract title from content string
                        title_part = (
                            content.split("features:")[0]
                            .replace("title: ", "")
                            .strip()
                        )
                        item_name = title_part

                # Fallback to a generic name if no title found
                if not item_name:
                    item_name = f"Item {item_id}"

                st.session_state.displayed_items.append(
                    {
                        "id": item_id,
                        "name": item_name,
                        "details": recommended_item,
                    }
                )

    # Display recommended items section
    if st.session_state.displayed_items:
        st.subheader("Recommended Items")

        for item in st.session_state.displayed_items:
            item_id = item["id"]
            item_name = item["name"]
            item_details = item["details"]

            image_url = get_item_image_url(item_details)

            with st.container():
                c1, c2 = st.columns([1, 0.2], vertical_alignment="center")

                with c1:
                    st.checkbox(
                        item_name,
                        key=f"item_checkbox_{item_id}",
                        value=item_id == st.session_state.selected_item,
                        on_change=process_item_selection,
                        args=(item_id, item_name),
                    )

                with c2:
                    if image_url:
                        # Use HTML/CSS for hover preview
                        hover_html = f"""
                        <style>
                        .preview-hover-container {{
                            position: relative;
                            cursor: pointer;
                        }}
                        .preview-hover-label {{
                            text-decoration: underline dotted;
                            color: #0072C6;
                        }}
                        .preview-hover-image {{
                            visibility: hidden;
                            opacity: 0;
                            transition: opacity 0.2s;
                            position: absolute;
                            z-index: 10;
                            right: 110%;
                            top: 0;
                            background: #fff;
                            border: 1px solid #ccc;
                            padding: 4px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                        }}
                        .preview-hover-container:hover .preview-hover-image {{
                            visibility: visible;
                            opacity: 1;
                        }}
                        </style>
                        <span class="preview-hover-container">
                            <span class="preview-hover-label">Show image</span>
                            <span class="preview-hover-image">
                                <img src="{image_url}" alt="Preview" style="max-width:250px;max-height:250px;display:block;margin:0 auto;" />
                            </span>
                        </span>
                        """
                        st.markdown(hover_html, unsafe_allow_html=True)

    else:
        st.subheader("Recommended Items")
        st.write("*No items recommended yet.*")


def deselect_other_checkboxes(selected_id):
    """Deselect all other checkboxes except the one with selected_id."""
    for item in st.session_state.displayed_items:
        item_id = item["id"]
        if item_id != selected_id:
            checkbox_key = f"item_checkbox_{item_id}"
            if checkbox_key in st.session_state:
                st.session_state[checkbox_key] = False


def process_item_selection(selected_id, item_name):
    """Handle item selection and trigger confirmation flow."""
    st.session_state.selected_item = selected_id
    if st.session_state.get(f"item_checkbox_{selected_id}"):
        deselect_other_checkboxes(selected_id)
        add_confirmation_message(item_name)
    else:
        st.session_state.conversation_state.turn_state[
            "awaiting_confirmation"
        ] = False


def add_confirmation_message(item_name: str):
    """Add an AI message asking the user to confirm their selection.

    the confirmation flow in the orchestrator.
    """
    # Check how many items are displayed
    num_displayed_items = len(st.session_state.get("displayed_items", []))

    if num_displayed_items < 2:
        # Not enough recommendations to choose from
        content = (
            f"I see you're interested in **{item_name}**. However, to help you make "
            f"the best decision, I'd like to provide you with at least one more "
            f"recommendation to compare. Could you tell me a bit more about your "
            f"preferences or what other features are important to you?"
        )
        ai_msg = AIMessage(content)
        st.session_state.chat_history.append(ai_msg)
        st.session_state.chat_log.append(ai_msg)

        # Don't set awaiting_confirmation flag - allow normal conversation flow
        if hasattr(st.session_state, "conversation_state"):
            if not hasattr(st.session_state.conversation_state, "turn_state"):
                st.session_state.conversation_state.turn_state = {}
            st.session_state.conversation_state.turn_state[
                "awaiting_confirmation"
            ] = False
    else:
        # Enough recommendations - proceed with confirmation
        content = f"Great! You've selected **{item_name}**. Are you sure you want to go with this recommendation?"

        ai_msg = AIMessage(content)
        st.session_state.chat_history.append(ai_msg)
        st.session_state.chat_log.append(ai_msg)

        if hasattr(st.session_state, "conversation_state"):
            if not hasattr(st.session_state.conversation_state, "turn_state"):
                st.session_state.conversation_state.turn_state = {}
            st.session_state.conversation_state.turn_state[
                "awaiting_confirmation"
            ] = True
            st.session_state.conversation_state.turn_state[
                "selected_item_name"
            ] = item_name
            st.session_state.conversation_state.turn_state[
                "selected_item_id"
            ] = st.session_state.selected_item

    st.session_state.auto_save_conversation()
