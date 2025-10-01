from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from crs.agents.orchestrator import create_orchestrator


def get_response_stream(
    task: Dict[str, Any],
    chat_history: List[BaseMessage],
    model_name: str = "gpt-4.1-nano",
) -> str:
    """
    Get the response from the model based on the chat history and user query.

    This function now uses the new modular orchestrator for better separation of concerns.

    Args:
        task (Dict[str, Any]): The task configuration containing domain and target information.
        chat_history (List[BaseMessage]): The chat history.
        model_name (str): The name of the model to use.

    Returns:
        str: The response stream from the model.
    """
    # Use the new orchestrator for cleaner, more modular processing
    print("Creating orchestrator with model:", model_name)
    orchestrator = create_orchestrator(model_name, use_streamlit=True)
    return orchestrator.process_conversation(task, chat_history)
