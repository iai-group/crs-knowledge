from code.agents.chains import get_response_stream

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


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

    # Input area for user's message
    user_message = st.chat_input("Ask for recommendation")

    # Process the user's message when the "Send" button is clicked
    if user_message is not None and user_message.strip() != "":
        with messages_container, st.chat_message("user"):
            st.markdown(user_message)

        response_stream = get_response_stream(
            st.session_state.task, st.session_state.chat_history, user_message
        )
        with messages_container, st.chat_message("ai"):
            response = st.write_stream(response_stream)

        st.session_state.chat_history.append(HumanMessage(user_message))
        st.session_state.chat_history.append(AIMessage(response))
        st.session_state.auto_save_conversation()
