"""Main conversation orchestrator."""

import logging
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from .connectors import get_model_connector
from .prompt_loader import PromptLoader
from .stages import (  # SuccessDetectionStage,; UserIntentAnalysisStage,
    DecisionStage,
    DecisionType,
    ItemRetrievalStage,
    ItemSelectionStage,
    PreferenceStatus,
    PreferenceSummarizationStage,
    QuestionAnswerStage,
    RecommendationAnalyzerStage,
    RecommendationStage,
    RecommendedItemCheckStage,
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
        self.recommended_item_check_stage = RecommendedItemCheckStage(
            self.model, self.prompt_loader
        )
        self.retrieval_stage = ItemRetrievalStage(
            self.model, self.prompt_loader
        )
        self.decision_stage = DecisionStage(self.model, self.prompt_loader)
        self.item_selection_stage = ItemSelectionStage(
            self.model, self.prompt_loader
        )

        # Stage for nuanced answers about a previously recommended item
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
        """Process a complete conversation turn.

        This method implements a consistent flow through all stages:
        1. PreferenceSummarizationStage - extracts user preferences
        1.5. RecommendedItemCheckStage (if new preferences) - checks if previously recommended items satisfy new preferences
        2. ItemRetrievalStage (if new preferences) - retrieves matching items
        3. DecisionStage - decides next action (ALWAYS called)
        4. RecommendationAnalyzerStage (if recommending) - analyzes items
        5. RecommendationStage OR ResponseStage - generates output

        Args:
            task: Task information including domain and target
            chat_history: Complete conversation history
            session_id: Session identifier for state management

        Returns:
            Response dict with stream and optional image_url
        """
        state = self.state_manager.get_state(session_id)
        # Update turn count
        turn_count = len(chat_history) // 2

        # or create a new one
        existing_turn_state = getattr(state, "turn_state", {}) or {}

        # Create a mutable turn_state dict and attach to the global state so
        # stages can read/update turn-specific information during processing.
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

        # Attach turn_state to the overall state object for easy access by
        # downstream components. Keep explicit turn_count field for
        # compatibility with existing code that expects `state.turn_count`.
        state.turn_state = turn_state
        state.turn_count = turn_count

        logger.debug(f"\nCurrent turn count: {turn_count}\n")

        # Stage 1: Summarize preferences
        pref_status = self.preference_stage.execute(state, task, chat_history)

        if pref_status == PreferenceStatus.NEW:
            # Stage 1.5: If new preferences, check if any previously recommended items satisfy them
            # before retrieving new items from the database
            matching_item = self.recommended_item_check_stage.execute(
                state, task, chat_history
            )

            # If a matching item is found, add it back to retrieved_items so it can be recommended again
            if matching_item:
                logger.debug(
                    f"Previously recommended item satisfies new preferences, adding to retrieved items"
                )
                # Add the matching item to the front of retrieved_items
                if matching_item not in state.retrieved_items:
                    state.retrieved_items.insert(0, matching_item)
            else:
                # No match found, proceed with normal retrieval
                logger.debug(
                    "No previously recommended items match new preferences, retrieving new items"
                )
                # Stage 2
                self.retrieval_stage.execute(state, task, chat_history)

        # If user provided too many preferences at once, override decision
        # to ask them to prioritize
        if pref_status == PreferenceStatus.TOO_MANY:
            logger.debug(
                "Too many preferences provided, asking user to prioritize."
            )
            return self.response_stage.execute(
                state, task, chat_history, decision=DecisionType.ASK_PRIORITIZE
            )

        # Stage 3: Make decision (ALWAYS called for consistent flow)
        decision = self.decision_stage.execute(state, task, chat_history)

        # If the user asks a question about a previously recommended item,
        # route to the specialized question-answering stage which produces
        # a nuanced, comparative answer.
        if decision == DecisionType.QUESTION_ABOUT_RECOMMENDATION:
            # Validate we have a recommended item to discuss
            if not state.last_recommended_item:
                logger.debug(
                    "Warning: QUESTION_ABOUT_RECOMMENDATION but no last_recommended_item. "
                    "Falling back to general ANSWER."
                )
                # Fall back to general answer response
                response_stage = ResponseStage(self.model, self.prompt_loader)
                return response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ANSWER
                )
            return self.question_answer_stage.execute(state, task, chat_history)

        # Stage 4, 5, 6 & 7: Execute recommendation or response based on decision
        if decision == DecisionType.RECOMMEND:
            # Validate we have items before recommending
            if not state.retrieved_items:
                logger.debug(
                    "Warning: RECOMMEND decision but no items retrieved. Falling back to ELICIT."
                )
                return self.response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ELICIT
                )

            # Stage 4: Select the best item from retrieved items
            selected_item = self.item_selection_stage.execute(
                state, task, chat_history
            )

            # If item selection fails, fall back to elicit
            if selected_item is None:
                logger.debug(
                    "Item selection stage returned None, falling back to ELICIT."
                )
                return self.response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ELICIT
                )

            # Stage 5: Analyze the selected item
            explanation_aspects = self.recommendation_analyzer_stage.execute(
                state, task, chat_history, selected_item
            )

            # Stage 6: Generate recommendation for the selected item
            recommendation_result = self.recommendation_stage.execute(
                state, task, chat_history, explanation_aspects, selected_item
            )

            # If recommendation stage returns None (no items), fall back to elicit
            if recommendation_result is None:
                logger.debug(
                    "Recommendation stage returned None, falling back to ELICIT."
                )
                return self.response_stage.execute(
                    state, task, chat_history, decision=DecisionType.ELICIT
                )

            return recommendation_result

        # For all other decisions, use the response stage
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
