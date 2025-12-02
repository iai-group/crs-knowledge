"""
CRS Agents module for conversational recommendation systems.

This module provides a modular architecture for building conversational
recommendation systems with the following components:

- prompt_loader: Handles loading and caching of external prompt files
- chain_factory: Factory for creating LangChain chains
- state_manager: Manages conversation state across different backends
- stages: Individual conversation stages (summarization, decision, recommendation, response)
- orchestrator: Main orchestrator for the multi-step conversation flow
- connectors: Model connectors for different LLM providers
- chains: Legacy chains module (maintained for backward compatibility)

Usage:
    from crs.agents.orchestrator import create_orchestrator

    orchestrator = create_orchestrator("gpt-4.1-mini")
    response = orchestrator.process_conversation(task, chat_history)
"""

# flake8: noqa: F401

from .chains import get_response_stream
from .connectors import available_models, get_model_connector

try:
    from .chain_factory import ChainFactory, create_simple_chain
    from .orchestrator import ConversationOrchestrator, create_orchestrator
    from .prompt_loader import PromptLoader, load_prompt
    from .stages import (
        ConversationStage,
        DecisionStage,
        DecisionType,
        ItemRetrievalStage,
        PreferenceStatus,
        PreferenceSummarizationStage,
        RecommendationStage,
        ResponseStage,
    )
    from .state_manager import (
        ConversationState,
        StateManager,
        StreamlitStateManager,
    )

    __all__ = [
        "available_models",
        "get_model_connector",
        "get_response_stream",
        "PromptLoader",
        "load_prompt",
        "ChainFactory",
        "create_simple_chain",
        "ConversationState",
        "StateManager",
        "StreamlitStateManager",
        "ConversationOrchestrator",
        "create_orchestrator",
        "ConversationStage",
        "PreferenceSummarizationStage",
        "PreferenceStatus",
        "ItemRetrievalStage",
        "HistorySummarizationStage",
        "DecisionStage",
        "DecisionType",
        "RecommendationStage",
        "ResponseStage",
        "SuccessDetectionStage",
    ]

except ImportError:
    __all__ = [
        "available_models",
        "get_model_connector",
        "get_response_stream",
    ]
