import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from crs.agents.chains import get_response_stream


def build_chatbot() -> None:
    """Builds the chatbot interface in the left column."""
    st.header("Assistant")

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
            elif isinstance(entry, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(entry.content)

    # Check if target has been found (success achieved)
    conversation_state = getattr(st.session_state, "conversation_state", None)
    target_found = (
        conversation_state and conversation_state.is_target_found()
        if conversation_state
        else False
    )

    # Input area for user's message (disabled if target found)
    if target_found:
        user_message = st.chat_input(
            "🎉 Target found! Conversation complete.", disabled=True
        )
        st.success(
            "🎯 **Congratulations!** You have successfully discovered the target item!"
        )
        if st.button("Complete", type="primary", use_container_width=True):
            st.session_state.current_page = "post"
            st.rerun()
    else:
        user_message = st.chat_input("Ask for recommendation")

    # Process the user's message when the "Send" button is clicked (only if target not found)
    if user_message is not None and user_message.strip() != "":
        with messages_container, st.chat_message("user"):
            st.markdown(user_message)

        st.session_state.chat_history.append(HumanMessage(user_message))
        st.session_state.chat_log.append(HumanMessage(user_message))

        response_stream = get_response_stream(
            st.session_state.task,
            st.session_state.chat_history,
            st.session_state.get("model_name", "gpt-4.1-nano"),
        )
        with messages_container, st.chat_message("ai"):
            print("Response stream:", response_stream)
            if response_stream.get("image_url"):
                st.image(response_stream["image_url"], width=300)
            response = st.write_stream(response_stream["stream"])

        # st.session_state.chat_history.append(AIMessage(response_stream))
        st.session_state.chat_history.append(AIMessage(response))
        st.session_state.chat_log.append(AIMessage(response))
        st.session_state.auto_save_conversation()

        target_found = (
            conversation_state and conversation_state.is_target_found()
            if conversation_state
            else False
        )
        if target_found:
            st.rerun()
