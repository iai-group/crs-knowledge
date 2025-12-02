"""Main conversation orchestrator."""

import logging
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from .connectors import get_model_connector
from .prompt_loader import PromptLoader
from .stages import (
    DecisionStage,
    DecisionType,
    ItemRetrievalStage,
    ItemSelectionStage,
    PreferenceStatus,
    PreferenceSummarizationStage,
    QuestionAnswerStage,
    RecommendationAnalyzerStage,
    RecommendationStage,
    ResponseStage,
)
from .state_manager import StateManager, StreamlitStateManager

logger = logging.getLogger(__name__)


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
        self.item_selection_stage = ItemSelectionStage(
            self.model, self.prompt_loader
        )
        self.question_answer_stage = QuestionAnswerStage(
            self.model, self.prompt_loader
        )
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
        """Process a complete conversation turn through the recommendation pipeline.

        This method orchestrates a multi-stage conversation flow:
        1. PreferenceSummarizationStage - extracts and tracks user preferences
        2. ItemRetrievalStage (if new preferences) - retrieves matching items from catalog
        3. DecisionStage - determines next action (ALWAYS called)
        4. Special routing for questions about recommended items
        5. ItemSelectionStage (if recommending) - selects best item from candidates
        6. RecommendationAnalyzerStage (if recommending) - analyzes selected item
        7. RecommendationStage OR ResponseStage - generates final output

        Args:
            task: Task information including domain and target item description
            chat_history: Complete conversation history as list of messages
            session_id: Session identifier for state management (default: "default")

        Returns:
            Response dict containing stream generator and optional image_url for recommendation
        """
        state = self.state_manager.get_state(session_id)
        turn_count = len(chat_history) // 2

        existing_turn_state = getattr(state, "turn_state", {}) or {}

        turn_state = {
            "turn_count": turn_count,
            "preferences": None,
            "num_preferences": None,
            "retrieved_items": None,
            "num_retrieved_items": None,
            "decision": None,
            "selected_item_name": existing_turn_state.get("selected_item_name"),
            "selected_item_id": existing_turn_state.get("selected_item_id"),
        }

        state.turn_state = turn_state
        state.turn_count = turn_count

        logger.debug(f"\nCurrent turn count: {turn_count}\n")

        pref_status = self.preference_stage.execute(state, task, chat_history)

        if pref_status == PreferenceStatus.TOO_MANY:
            logger.debug(
                "Too many preferences provided, asking user to prioritize."
            )
            return self.response_stage.execute(
                state, task, chat_history, decision=DecisionType.ASK_PRIORITIZE
            )

        if pref_status == PreferenceStatus.NEW:
            self.retrieval_stage.execute(state, task, chat_history)

        decision = self.decision_stage.execute(state, task, chat_history)

        if decision == DecisionType.ANSWER_ABOUT_RECOMMENDATION:
            if not state.recommended_items:
                logger.debug(
                    "Warning: ANSWER_ABOUT_RECOMMENDATION but no recommended_items. "
                    "Falling back to general ANSWER."
                )
                response_stage = ResponseStage(self.model, self.prompt_loader)
                return response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ANSWER
                )
            return self.question_answer_stage.execute(state, task, chat_history)

        if decision == DecisionType.RECOMMEND:
            if not state.retrieved_items:
                logger.debug(
                    "Warning: RECOMMEND decision but no items retrieved. Falling back to ELICIT."
                )
                return self.response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ELICIT
                )

            selected_item = self.item_selection_stage.execute(
                state, task, chat_history, target=task.get("target")
            )

            if selected_item is None:
                logger.debug(
                    "Item selection stage returned None, falling back to ELICIT."
                )
                return self.response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ELICIT
                )

            explanation_aspects = self.recommendation_analyzer_stage.execute(
                state, task, chat_history, selected_item
            )

            recommendation_result = self.recommendation_stage.execute(
                state, task, chat_history, explanation_aspects, selected_item
            )

            if recommendation_result is None:
                logger.debug(
                    "Recommendation stage returned None, falling back to ELICIT."
                )
                return self.response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ELICIT
                )

            return recommendation_result

        response_stream = self.response_stage.execute(
            state, task, chat_history, decision=decision
        )
        return response_stream


def create_orchestrator(
    model_name: str = "gpt-4.1-mini", use_streamlit: bool = True
) -> ConversationOrchestrator:
    """Create a conversation orchestrator with appropriate state manager."""
    if use_streamlit:
        state_manager = StreamlitStateManager()
    else:
        state_manager = StateManager()

    return ConversationOrchestrator(model_name, state_manager)
