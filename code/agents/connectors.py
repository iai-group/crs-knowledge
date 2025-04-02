import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

available_models = {
    "llama3.2:latest": "ollama",
    # "deepseek": "ollama",
    # "gemini-1.5-pro": "google",
    # "chatgpt-3.5-turbo": "openai",
}


def get_model_connector(model_name: str):
    """
    Get the model connector based on the model name.
    """
    if model_name in available_models:
        if available_models[model_name] == "ollama":
            return get_ollama_model(model_name)
        elif available_models[model_name] == "google":
            return get_google_model(model=model_name)
        elif available_models[model_name] == "openai":
            return get_openai_model(model=model_name)
    else:
        raise ValueError(f"Model {model_name} not found in available models.")


def get_ollama_model(model: str):
    ollama_endpoint = "https://ollama.ux.uis.no"

    model = OllamaLLM(model=model, base_url=ollama_endpoint)
    return model


def get_google_model(model: str):
    """
    Get the Google model connector.
    """
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        max_output_tokens=512,
        google_api_key=google_api_key,
    )
    return google_model


def get_openai_model(model: str):
    """
    Get the OpenAI model connector.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=512,
        openai_api_key=openai_api_key,
    )
    return openai_model
