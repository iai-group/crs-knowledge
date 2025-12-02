"""LangChain integration for conversational responses."""

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from crs.agents.orchestrator import create_orchestrator


def get_response_stream(
    task: Dict[str, Any],
    chat_history: List[BaseMessage],
    model_name: str = "gpt-4.1-mini",
) -> str:
    """Get the response from the model based on the chat history and user query.

    Args:
        task: The task configuration containing domain and target information.
        chat_history: The chat history.
        model_name: The name of the model to use.

    Returns:
        The response stream from the model.
    """
    print("Creating orchestrator with model:", model_name)
    orchestrator = create_orchestrator(model_name, use_streamlit=True)
    return orchestrator.process_conversation(task, chat_history)
