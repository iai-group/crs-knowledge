import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


class ImageMessage(AIMessage):
    """AI message with a separate image URL field for clean rendering."""

    def __init__(self, content: str = "", image_url: str = ""):
        super().__init__(content)
        self.image_url = image_url


from crs.agents.chains import get_response_stream


@st.fragment(run_every="1s")
def build_timer() -> bool:
    """
    Builds a real-time countdown timer display.

    Returns:
        bool: True if timer is still active, False if expired
    """
    # Stop timer updates if we're not on the chat page anymore
    # This prevents fragment errors when navigating to other pages
    if st.session_state.get("current_page") != "chat":
        return False

    if getattr(st.session_state, "chat_timer_expired", False):
        st.error("⏰ Time's up! Please choose one recommended item.")
        return False

    TIMER_DURATION = 900  # 15 minutes in seconds
    current_time = time.time()
    chat_start_time = st.session_state.get("chat_start_time", current_time)
    elapsed_time = current_time - chat_start_time
    remaining_time = max(0, TIMER_DURATION - elapsed_time)

    # Update timer expired status
    timer_expired = remaining_time <= 0
    st.session_state.chat_timer_expired = timer_expired

    # Display timer with real-time updates
    if remaining_time > 0:
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        st.info(f"⏱️ Time remaining: {minutes:02d}:{seconds:02d}")
    else:
        # If there are no recommended items at all, just proceed to the next
        # page (post questionnaire). This avoids trapping the participant on
        # the chat page when no recommendations were produced.
        displayed = st.session_state.get("displayed_items")
        # displayed_items may be None, empty list, or a populated list of dicts
        if not displayed:
            # Advance to post questionnaire page and force a rerun so the
            # UI updates immediately.
            st.session_state.current_page = "post"
            # Ensure timer expired flag is set (maintain invariant)
            st.session_state.chat_timer_expired = True
            st.rerun()
        else:
            st.error("⏰ Time's up! Please choose one recommended item.")

    return not timer_expired


def build_chatbot() -> None:
    """Builds the chatbot interface in the left column."""

    if not st.session_state.chat_history:
        welcome = AIMessage(
            "I am your personal recommender assistant. What are you looking for today?"
        )
        st.session_state.chat_history.append(welcome)

    messages_container = st.container(height=500)

    with messages_container:
        # Display the conversation history
        for entry in st.session_state.chat_history:
            if isinstance(entry, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(entry.content)
            elif isinstance(entry, ImageMessage):
                # ImageMessage: render the image at fixed width and the text content
                with st.chat_message("ai"):
                    st.image(entry.image_url, width=300)
                    st.markdown(entry.content)
            elif isinstance(entry, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(entry.content)

    # Check if we're in confirmation mode
    conversation_state = getattr(st.session_state, "conversation_state", None)
    awaiting_confirmation = False
    if conversation_state and hasattr(conversation_state, "turn_state"):
        awaiting_confirmation = conversation_state.turn_state.get(
            "awaiting_confirmation", False
        )

    user_message = None

    # Show confirmation buttons if awaiting confirmation
    if awaiting_confirmation:
        with messages_container:
            # Center the buttons using columns with padding on sides
            col_left, col1, col2, col_right = st.columns([1, 2, 2, 1])
            with col1:
                if st.button("✅ Yes", use_container_width=True):
                    user_message = "Yes, I want this one."
                    # Set confirmation flag immediately when user clicks Yes
                    if conversation_state and hasattr(
                        conversation_state, "turn_state"
                    ):
                        conversation_state.turn_state[
                            "user_confirmed_selection"
                        ] = True
            with col2:
                if st.button("🤔 Let me think", use_container_width=True):
                    user_message = "Let me think about it."
        # Still allow text input in confirmation mode
        text_input = st.chat_input("Or type your response...")
        if text_input and text_input.strip():
            user_message = text_input
    else:
        # Normal chat input
        user_message = st.chat_input("What are you looking for?")

    # # Only show chat input if timer hasn't expired
    # if not st.session_state.get(
    #     "chat_timer_expired", False
    # ) or st.session_state.get("debug", False):
    #     user_message = st.chat_input("What are you looking for?")
    # else:
    #     user_message = None
    #     st.chat_input(
    #         "Time expired -- please select one of the recommended items",
    #         disabled=True,
    #     )

    if user_message is not None and user_message.strip() != "":
        with messages_container, st.chat_message("user"):
            st.markdown(user_message)

        st.session_state.chat_history.append(HumanMessage(user_message))
        st.session_state.chat_log.append(HumanMessage(user_message))

        # Check if we're in special confirmation/explanation flows
        conversation_state = getattr(
            st.session_state, "conversation_state", None
        )

        # Check if user provided explanation after confirmation
        if conversation_state and hasattr(conversation_state, "turn_state"):
            if conversation_state.turn_state.get("awaiting_explanation"):
                print(
                    "📝 User provided explanation, transitioning to post questionnaire..."
                )
                print(
                    f"Current page before transition: {st.session_state.get('current_page')}"
                )

                # Clear the explanation flag
                conversation_state.turn_state["awaiting_explanation"] = False

                # Add user's explanation to the conversation log
                st.session_state.auto_save_conversation()

                # Transition to the post questionnaire page
                st.session_state.current_page = "post"
                print(
                    f"Current page after setting: {st.session_state.current_page}"
                )
                st.rerun()

            # Check if user confirmed their selection (either via button or detected by LLM)
            elif conversation_state.turn_state.get("user_confirmed_selection"):
                print("✅ User confirmed selection, asking for explanation...")
                # Clear the confirmation flag
                conversation_state.turn_state["user_confirmed_selection"] = (
                    False
                )
                # Clear awaiting_confirmation to hide buttons
                conversation_state.turn_state["awaiting_confirmation"] = False

                # Ask user to explain why this choice is best
                selected_item_name = conversation_state.turn_state.get(
                    "selected_item_name", "this item"
                )
                explanation_prompt = (
                    "Great! Before we finalize, could you briefly explain in a "
                    "sentence or two why you think this option is the best "
                    "choice for you?"
                )

                explanation_msg = AIMessage(explanation_prompt)
                st.session_state.chat_history.append(explanation_msg)
                st.session_state.chat_log.append(explanation_msg)
                st.session_state.auto_save_conversation()

                # Set flag to indicate we're awaiting explanation
                conversation_state.turn_state["awaiting_explanation"] = True

                # Rerun to show the new message and hide buttons
                st.rerun()

        # Normal conversation flow - call LLM orchestrator
        response_stream = get_response_stream(
            st.session_state.task,
            st.session_state.chat_history,
            st.session_state.get("model_name", "gpt-4.1-mini"),
        )
        # Show the response stream to the user (and the image while streaming)
        image_url = response_stream.get("image_url", "")
        with messages_container, st.chat_message("ai"):
            if image_url:
                st.image(image_url, width=300)
            response = st.write_stream(response_stream["stream"])

        # Persist the AI message. If there is an image, use ImageMessage so
        # the UI can render the image consistently later without HTML hacks.
        if image_url:
            ai_msg = ImageMessage(response or "", image_url=image_url)
        else:
            ai_msg = AIMessage(response or "")

        st.session_state.chat_history.append(ai_msg)
        st.session_state.chat_log.append(ai_msg)
        st.session_state.auto_save_conversation()
