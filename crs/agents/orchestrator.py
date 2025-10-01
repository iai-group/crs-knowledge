"""Main conversation orchestrator."""

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from crs.config_loader import get_config_loader

from .connectors import get_model_connector
from .prompt_loader import PromptLoader
from .stages import (  # SuccessDetectionStage,; UserIntentAnalysisStage,
    DecisionStage,
    ItemRetrievalStage,
    PreferenceSummarizationStage,
    RecommendationAnalyzerStage,
    RecommendationStage,
    ResponseStage,
)
from .state_manager import StateManager, StreamlitStateManager


class ConversationOrchestrator:
    """Orchestrates the multi-step conversation flow."""

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        state_manager: StateManager = None,
    ):
        self.model = get_model_connector(model_name)
        self.prompt_loader = PromptLoader()
        self.state_manager = state_manager or StreamlitStateManager()

        self.preference_stage = PreferenceSummarizationStage(
            self.model, self.prompt_loader
        )
        self.retrieval_stage = ItemRetrievalStage(
            self.model, self.prompt_loader
        )
        self.decision_stage = DecisionStage(self.model, self.prompt_loader)

        self.recommendation_analyzer_stage = RecommendationAnalyzerStage(
            self.model, self.prompt_loader
        )
        self.recommendation_stage = RecommendationStage(
            self.model, self.prompt_loader
        )
        self.response_stage = ResponseStage(self.model, self.prompt_loader)

    def process_conversation(
        self,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
        session_id: str = "default",
    ) -> Any:
        """Process a complete conversation turn."""
        state = self.state_manager.get_state(session_id)

        # Update turn count
        turn_count = len(chat_history) // 2
        state.turn_count = turn_count

        print(f"\nCurrent turn count: {turn_count}\n")

        number_of_preferences = self.preference_stage.execute(
            state, task, chat_history
        )

        if number_of_preferences > 1:
            self.retrieval_stage.execute(state, task, chat_history)

        decision = self.decision_stage.execute(state, task, chat_history)

        if decision.lower() == "recommend":
            explanation_aspectis = self.recommendation_analyzer_stage.execute(
                state, task, chat_history
            )
            return self.recommendation_stage.execute(
                state, task, chat_history, explanation_aspectis
            )

        response_stream = self.response_stage.execute(
            state, task, chat_history, decision=decision
        )
        return response_stream


# Factory function for easy creation
def create_orchestrator(
    model_name: str = "gpt-4.1-nano", use_streamlit: bool = True
) -> ConversationOrchestrator:
    """Create a conversation orchestrator with appropriate state manager."""
    if use_streamlit:
        state_manager = StreamlitStateManager()
    else:
        state_manager = StateManager()

    return ConversationOrchestrator(model_name, state_manager)
