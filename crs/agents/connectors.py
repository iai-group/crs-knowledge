import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

load_dotenv()

CACHE_VERSION = "1"

available_models = {
    "gemma3:latest": "ollama",
    "llama3.2:latest": "ollama",
    "deepseek": "ollama",
    "gemini-1.5-pro": "google",
    "gpt-4.1-nano": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-5-nano": "openai",
    "gpt-5": "openai",
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
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = ChatGoogleGenerativeAI(
        model=model,
        max_output_tokens=2048,
        google_api_key=google_api_key,
    )
    return google_model


@st.cache_resource(show_spinner=False)
def get_openai_model(model: str):
    """
    Get the OpenAI model connector.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = ChatOpenAI(
        model=model,
        max_tokens=4096,
        openai_api_key=openai_api_key,
        # temperature=0.7,
        max_retries=2,
        timeout=60,
    )
    return openai_model
