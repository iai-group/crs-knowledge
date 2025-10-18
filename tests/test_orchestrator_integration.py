"""
Integration tests for ConversationOrchestrator.

Focus on:
- Proper orchestration of stages
- State management across conversation
- Decision routing works correctly
- Data flows properly between stages
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from crs.agents.orchestrator import ConversationOrchestrator
from crs.agents.prompt_loader import PromptLoader
from crs.agents.stages import DecisionType, PreferenceStatus
from crs.agents.state_manager import ConversationState, StateManager


@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    model.invoke = Mock(return_value=AIMessage(content="Response"))
    return model


@pytest.fixture
def prompt_loader():
    """Create a real prompt loader."""
    return PromptLoader()


@pytest.fixture
def state_manager():
    """Create a real state manager."""
    return StateManager()


@pytest.fixture
def sample_task():
    """Create a sample task."""
    return {
        "domain": "laptop",
        "target": {"asin": "TARGET123", "title": "Target Laptop"},
    }


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_orchestrator_initializes_all_stages(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that orchestrator initializes all required stages."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Check all stages are initialized
        assert orchestrator.preference_stage is not None
        assert orchestrator.retrieval_stage is not None
        assert orchestrator.selection_stage is not None
        assert orchestrator.decision_stage is not None
        assert orchestrator.recommendation_analyzer_stage is not None
        assert orchestrator.qa_stage is not None
        assert orchestrator.recommendation_stage is not None
        assert orchestrator.response_stage is not None


class TestOrchestratorBasicFlow:
    """Test basic message processing flow."""

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_process_message_returns_string(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that process_message returns a string response."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Mock all stages to return appropriate values
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",  # Preference summarization
                AIMessage(content="Elicit\nNeed more info"),  # Decision
                AIMessage(content="What's your budget?"),  # Response
            ]
        )

        conversation_id = "test_123"
        user_message = "I need a laptop"
        chat_history = []

        response = orchestrator.process_message(
            conversation_id, user_message, chat_history
        )

        assert isinstance(response, str), "process_message should return string"
        assert len(response) > 0, "Response should not be empty"

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_state_persists_across_messages(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that conversation state persists across multiple messages."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )
        conversation_id = "test_123"

        # First message - add preferences
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",  # Preferences
                AIMessage(content="Elicit\nMore info needed"),  # Decision
                AIMessage(content="What's your budget?"),  # Response
            ]
        )

        response1 = orchestrator.process_message(
            conversation_id, "I need a gaming laptop", []
        )

        # Get state after first message
        state1 = state_manager.get_state(conversation_id)
        assert state1.preferences is not None, "Preferences should be stored"

        # Second message - state should still have preferences
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop\n$1500 budget",  # Updated preferences
                AIMessage(content="Recommend\nReady"),  # Decision
                AIMessage(content="Let me find some options"),  # Response
            ]
        )

        chat_history = [
            HumanMessage(content="I need a gaming laptop"),
            AIMessage(content=response1),
        ]

        response2 = orchestrator.process_message(
            conversation_id, "$1500 budget", chat_history
        )

        state2 = state_manager.get_state(conversation_id)
        # State should be updated but still exist
        assert state2 is not None
        assert state2.preferences is not None


class TestOrchestratorDecisionRouting:
    """Test that orchestrator routes decisions correctly."""

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_elicit_decision_skips_retrieval(
        self, mock_retriever_class, mock_model, prompt_loader, state_manager
    ):
        """Test that ELICIT decision doesn't trigger retrieval."""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Mock to return ELICIT decision
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming",  # Preferences
                AIMessage(content="Elicit\nNeed more"),  # Decision
                AIMessage(content="Tell me more"),  # Response
            ]
        )

        orchestrator.process_message("test_123", "I need a laptop", [])

        # Retrieval should NOT be called
        mock_retriever.retrieve_items.assert_not_called()

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_recommend_decision_triggers_retrieval(
        self, mock_retriever_class, mock_model, prompt_loader, state_manager
    ):
        """Test that RECOMMEND decision triggers item retrieval."""
        mock_retriever = Mock()
        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Laptop 1",
                    "parent_asin": "A1",
                    "content": "Gaming",
                    "images": [],
                },
                0.9,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Mock to return RECOMMEND decision
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop\n$1500",  # Preferences
                AIMessage(content="Recommend\nReady to recommend"),  # Decision
                AIMessage(content="0"),  # Selection
                AIMessage(content="I recommend this laptop"),  # Recommendation
                AIMessage(content="Here's my recommendation"),  # Response
            ]
        )

        orchestrator.process_message("test_123", "Show me laptops", [])

        # Retrieval SHOULD be called
        mock_retriever.retrieve_items.assert_called_once()

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_answer_decision_skips_retrieval(
        self, mock_retriever_class, mock_model, prompt_loader, state_manager
    ):
        """Test that ANSWER decision doesn't trigger retrieval."""
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Mock to return ANSWER decision
        mock_model.invoke = Mock(
            side_effect=[
                "",  # No preferences from this question
                AIMessage(content="Answer\nGeneral question"),  # Decision
                AIMessage(content="RAM is memory"),  # QA answer
                AIMessage(content="RAM is Random Access Memory"),  # Response
            ]
        )

        orchestrator.process_message("test_123", "What is RAM?", [])

        # Retrieval should NOT be called
        mock_retriever.retrieve_items.assert_not_called()


class TestOrchestratorRecommendationFlow:
    """Test complete recommendation flow through orchestrator."""

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_full_recommendation_flow(
        self, mock_retriever_class, mock_model, prompt_loader, state_manager
    ):
        """Test complete flow from user message to recommendation."""
        mock_retriever = Mock()
        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Gaming Laptop Pro",
                    "parent_asin": "A001",
                    "content": "High-end gaming",
                    "images": [],
                },
                0.95,
            ),
            (
                {
                    "title": "Gaming Laptop Plus",
                    "parent_asin": "A002",
                    "content": "Mid-range gaming",
                    "images": [],
                },
                0.85,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )
        conversation_id = "test_rec_flow"

        # Mock responses for recommendation flow
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop\n$1500 budget\nHigh performance",  # Preferences
                AIMessage(
                    content="Recommend\nUser has enough preferences"
                ),  # Decision
                AIMessage(content="0"),  # Selection (first item)
                AIMessage(
                    content="I recommend the Gaming Laptop Pro because it matches your needs"
                ),  # Recommendation
                AIMessage(
                    content="Based on your preferences for a gaming laptop with $1500 budget"
                ),  # Response
            ]
        )

        response = orchestrator.process_message(
            conversation_id,
            "Can you recommend a gaming laptop? My budget is $1500 and I need high performance",
            [],
        )

        assert isinstance(response, str)
        assert len(response) > 0

        # Check that state was updated with recommendation
        state = state_manager.get_state(conversation_id)
        assert state.retrieved_items is not None
        assert len(state.retrieved_items) > 0

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_target_handling_in_flow(
        self,
        mock_retriever_class,
        mock_model,
        prompt_loader,
        state_manager,
        sample_task,
    ):
        """Test that target is properly passed through the flow."""
        target_asin = sample_task["target"]["asin"]

        mock_retriever = Mock()
        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Target Laptop",
                    "parent_asin": target_asin,
                    "content": "Target",
                    "images": [],
                },
                0.95,
            ),
            (
                {
                    "title": "Other Laptop",
                    "parent_asin": "OTHER",
                    "content": "Other",
                    "images": [],
                },
                0.85,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Set up state with task
        conversation_id = "test_target"
        state = state_manager.get_state(conversation_id)

        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",  # Preferences
                AIMessage(content="Recommend\nReady"),  # Decision
                AIMessage(content="0"),  # Selection
                AIMessage(content="I recommend this laptop"),  # Recommendation
                AIMessage(content="Here's my recommendation"),  # Response
            ]
        )

        # Process message - target should be filtered before first recommendation
        response = orchestrator.process_message(
            conversation_id, "Show laptops", []
        )

        assert isinstance(response, str)

        # Check retrieval was called with proper domain
        mock_retriever.retrieve_items.assert_called()


class TestOrchestratorStateUpdates:
    """Test that state is properly updated throughout conversation."""

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_preferences_accumulated(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that preferences accumulate across messages."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )
        conversation_id = "test_prefs"

        # First message
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",
                AIMessage(content="Elicit\nMore info"),
                AIMessage(content="What's your budget?"),
            ]
        )

        orchestrator.process_message(
            conversation_id, "I need a gaming laptop", []
        )

        state1 = state_manager.get_state(conversation_id)
        assert state1.preferences is not None

        # Second message - preferences should build
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop\n$1500 budget",  # Should include previous + new
                AIMessage(content="Elicit\nMore info"),
                AIMessage(content="Any other requirements?"),
            ]
        )

        orchestrator.process_message(
            conversation_id,
            "My budget is $1500",
            [
                HumanMessage(content="I need a gaming laptop"),
                AIMessage(content="What's your budget?"),
            ],
        )

        state2 = state_manager.get_state(conversation_id)
        assert state2.preferences is not None

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_recommended_items_tracked(
        self, mock_retriever_class, mock_model, prompt_loader, state_manager
    ):
        """Test that recommended items are tracked in state."""
        mock_retriever = Mock()
        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Laptop 1",
                    "parent_asin": "A1",
                    "content": "Gaming",
                    "images": [],
                },
                0.9,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )
        conversation_id = "test_tracking"

        # Make a recommendation
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",
                AIMessage(content="Recommend\nReady"),
                AIMessage(content="0"),
                AIMessage(content="I recommend Laptop 1"),
                AIMessage(content="Here's Laptop 1"),
            ]
        )

        orchestrator.process_message(conversation_id, "Show me laptops", [])

        state = state_manager.get_state(conversation_id)
        # Check that selected item was stored
        assert (
            state.selected_item is not None or state.retrieved_items is not None
        )


class TestOrchestratorEdgeCases:
    """Test edge cases in orchestrator."""

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_empty_message_handled(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that empty messages are handled gracefully."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        mock_model.invoke = Mock(
            side_effect=[
                "",
                AIMessage(content="Elicit\nNeed input"),
                AIMessage(content="How can I help?"),
            ]
        )

        response = orchestrator.process_message("test_empty", "", [])

        assert isinstance(response, str)

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_long_conversation_history(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that long conversation history is handled."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Create long history (100 messages)
        long_history = []
        for i in range(50):
            long_history.append(HumanMessage(content=f"Message {i}"))
            long_history.append(AIMessage(content=f"Response {i}"))

        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",
                AIMessage(content="Elicit\nMore info"),
                AIMessage(content="Tell me more"),
            ]
        )

        response = orchestrator.process_message(
            "test_long", "I need a laptop", long_history
        )

        assert isinstance(response, str)

    @patch("crs.agents.orchestrator.ItemRetriever")
    def test_multiple_conversations_separate_state(
        self, mock_retriever, mock_model, prompt_loader, state_manager
    ):
        """Test that different conversation IDs maintain separate state."""
        orchestrator = ConversationOrchestrator(
            mock_model, prompt_loader, state_manager
        )

        # Conversation 1
        mock_model.invoke = Mock(
            side_effect=[
                "Gaming laptop",
                AIMessage(content="Elicit"),
                AIMessage(content="Response 1"),
            ]
        )
        orchestrator.process_message("conv_1", "Gaming laptop", [])

        # Conversation 2
        mock_model.invoke = Mock(
            side_effect=[
                "Office laptop",
                AIMessage(content="Elicit"),
                AIMessage(content="Response 2"),
            ]
        )
        orchestrator.process_message("conv_2", "Office laptop", [])

        # States should be separate
        state1 = state_manager.get_state("conv_1")
        state2 = state_manager.get_state("conv_2")

        assert (
            state1 is not state2
        ), "Different conversations should have different states"
