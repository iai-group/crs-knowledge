from code.agents.connectors import available_models, get_model_connector
from code.agents.prompts.task_prompt_v0 import base_prompt

from langchain_core.output_parsers import StrOutputParser


def get_response_stream(
    task: str,
    chat_history: str,
    user_query: str,
    model_name: str = "llama3.2:latest",
) -> str:
    """
    Get the response from the model based on the chat history and user query.

    Args:
        chat_history (str): The chat history.
        user_query (str): The user query.
        model_name (str): The name of the model to use.

    Returns:
        str: The response from the model.
    """
    # Get the model connector
    model = get_model_connector(model_name)

    chain = base_prompt | model | StrOutputParser()

    return chain.stream(
        {
            # "task_description": task.get("story"),
            "domain": task.get("current_domain"),
            "target_item": task.get("target_item"),
            "chat_history": chat_history,
            "user_query": user_query,
        }
    )
