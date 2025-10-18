"""
Comprehensive tests for ConversationOrchestrator.

Tests orchestration logic:
- Stage initialization
- Decision routing
- State management
- Complete conversation flows
- Error handling
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from crs.agents.orchestrator import ConversationOrchestrator
from crs.agents.stages import DecisionType, PreferenceStatus
from crs.agents.state_manager import ConversationState


class TestOrchestratorInitialization:
    """Test cases for orchestrator initialization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()

    @pytest.fixture
    def mock_prompt_loader(self):
        """Create a mock prompt loader."""
        loader = Mock()
        loader.load_prompt = Mock(return_value="Test prompt: {input}")
        return loader

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        manager.get_state = Mock(return_value=ConversationState())
        return manager

    def test_initialization(
        self, mock_model, mock_prompt_loader, mock_state_manager
    ):
        """Test that orchestrator initializes all stages."""
        with patch("crs.agents.orchestrator.ItemRetriever"):
            orchestrator = ConversationOrchestrator(
                mock_model, mock_prompt_loader, mock_state_manager
            )

            # Check that stages are initialized
            assert orchestrator.preference_stage is not None
            assert orchestrator.retrieval_stage is not None
            assert orchestrator.selection_stage is not None
            assert orchestrator.decision_stage is not None
            assert orchestrator.recommendation_analyzer_stage is not None
            assert orchestrator.qa_stage is not None
            assert orchestrator.recommendation_stage is not None
            assert orchestrator.response_stage is not None

    def test_initialization_with_none_values(self):
        """Test that orchestrator handles None values gracefully."""
        with pytest.raises(Exception):
            ConversationOrchestrator(None, None, None)


class TestOrchestratorProcessMessage:
    """Test cases for message processing."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.invoke = Mock(return_value=AIMessage(content="Test response"))
        return model

    @pytest.fixture
    def mock_prompt_loader(self):
        """Create a mock prompt loader."""
        loader = Mock()
        loader.load_prompt = Mock(return_value="Test prompt")
        return loader

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        state = ConversationState()
        state.domain = "laptop"
        manager.get_state = Mock(return_value=state)
        return manager

    @pytest.fixture
    def orchestrator(self, mock_model, mock_prompt_loader, mock_state_manager):
        """Create an orchestrator instance."""
        with patch("crs.agents.orchestrator.ItemRetriever"):
            return ConversationOrchestrator(
                mock_model, mock_prompt_loader, mock_state_manager
            )

    def test_process_message_basic(self, orchestrator, mock_state_manager):
        """Test basic message processing."""
        conversation_id = "test_conv_123"
        user_message = "I need a laptop for gaming"
        chat_history = []

        # Mock stage executions
        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.PARTIAL,
                "summary": "Gaming laptop",
                "preference_count": 1,
            }
            mock_decision.return_value = {"decision": DecisionType.ELICIT}
            mock_response.return_value = {"response": "What's your budget?"}

            response = orchestrator.process_message(
                conversation_id, user_message, chat_history
            )

            assert isinstance(response, str)
            assert len(response) > 0

    def test_process_message_with_recommendation_flow(
        self, orchestrator, mock_state_manager
    ):
        """Test message processing that leads to recommendations."""
        conversation_id = "test_conv_123"
        user_message = "Can you recommend something?"
        chat_history = [
            HumanMessage(content="I need a gaming laptop"),
            AIMessage(content="What's your budget?"),
            HumanMessage(content="Around $1500"),
        ]

        # Mock the entire recommendation pipeline
        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.retrieval_stage, "execute"
            ) as mock_retrieval,
            patch.object(
                orchestrator.selection_stage, "execute"
            ) as mock_selection,
            patch.object(
                orchestrator.recommendation_stage, "execute"
            ) as mock_recommend,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.SUFFICIENT,
                "summary": "Gaming laptop, $1500 budget",
                "preference_count": 2,
            }
            mock_decision.return_value = {"decision": DecisionType.RECOMMEND}
            mock_retrieval.return_value = {
                "items": [{"title": "Gaming Pro"}],
                "item_count": 1,
            }
            mock_selection.return_value = {
                "selected_items": [{"title": "Gaming Pro"}]
            }
            mock_recommend.return_value = {
                "recommendation_text": "I recommend Gaming Pro"
            }
            mock_response.return_value = {"response": "I recommend Gaming Pro"}

            response = orchestrator.process_message(
                conversation_id, user_message, chat_history
            )

            assert isinstance(response, str)
            # Verify recommendation pipeline was called
            mock_retrieval.assert_called_once()
            mock_selection.assert_called_once()
            mock_recommend.assert_called_once()

    def test_process_message_with_answer_flow(
        self, orchestrator, mock_state_manager
    ):
        """Test message processing for answering questions."""
        conversation_id = "test_conv_123"
        user_message = "What is RAM?"
        chat_history = []

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(orchestrator.qa_stage, "execute") as mock_qa,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.NONE,
                "summary": "",
                "preference_count": 0,
            }
            mock_decision.return_value = {"decision": DecisionType.ANSWER}
            mock_qa.return_value = {"answer": "RAM is Random Access Memory"}
            mock_response.return_value = {
                "response": "RAM is Random Access Memory"
            }

            response = orchestrator.process_message(
                conversation_id, user_message, chat_history
            )

            assert isinstance(response, str)
            mock_qa.assert_called_once()


class TestOrchestratorStateManagement:
    """Test cases for state management in orchestrator."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()

    @pytest.fixture
    def mock_prompt_loader(self):
        """Create a mock prompt loader."""
        loader = Mock()
        loader.load_prompt = Mock(return_value="Test prompt")
        return loader

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        state = ConversationState()
        state.domain = "laptop"
        manager.get_state = Mock(return_value=state)
        manager.update_state = Mock()
        return manager

    @pytest.fixture
    def orchestrator(self, mock_model, mock_prompt_loader, mock_state_manager):
        """Create an orchestrator instance."""
        with patch("crs.agents.orchestrator.ItemRetriever"):
            return ConversationOrchestrator(
                mock_model, mock_prompt_loader, mock_state_manager
            )

    def test_state_updates_after_preference_stage(
        self, orchestrator, mock_state_manager
    ):
        """Test that state is updated after preference summarization."""
        conversation_id = "test_conv_123"
        user_message = "I need a gaming laptop"

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.PARTIAL,
                "summary": "Gaming laptop",
                "preference_count": 1,
            }
            mock_decision.return_value = {"decision": DecisionType.ELICIT}
            mock_response.return_value = {"response": "What's your budget?"}

            orchestrator.process_message(conversation_id, user_message, [])

            # Verify state manager was called to update state
            mock_state_manager.update_state.assert_called()

    def test_state_persists_across_messages(
        self, orchestrator, mock_state_manager
    ):
        """Test that state persists across multiple messages."""
        conversation_id = "test_conv_123"

        # First message
        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.PARTIAL,
                "summary": "Gaming laptop",
                "preference_count": 1,
            }
            mock_decision.return_value = {"decision": DecisionType.ELICIT}
            mock_response.return_value = {"response": "What's your budget?"}

            orchestrator.process_message(
                conversation_id, "I need a gaming laptop", []
            )

            # Verify state manager was called
            first_call_count = mock_state_manager.update_state.call_count

        # Second message - state should be retrieved and updated again
        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.SUFFICIENT,
                "summary": "Gaming laptop, $1500",
                "preference_count": 2,
            }
            mock_decision.return_value = {"decision": DecisionType.RECOMMEND}
            mock_response.return_value = {"response": "I recommend..."}

            orchestrator.process_message(
                conversation_id,
                "Around $1500",
                [
                    HumanMessage(content="I need a gaming laptop"),
                    AIMessage(content="What's your budget?"),
                ],
            )

            # Verify state manager was called again
            assert mock_state_manager.update_state.call_count > first_call_count


class TestOrchestratorDecisionRouting:
    """Test cases for decision routing logic."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()

    @pytest.fixture
    def mock_prompt_loader(self):
        """Create a mock prompt loader."""
        loader = Mock()
        loader.load_prompt = Mock(return_value="Test prompt")
        return loader

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        state = ConversationState()
        state.domain = "laptop"
        manager.get_state = Mock(return_value=state)
        return manager

    @pytest.fixture
    def orchestrator(self, mock_model, mock_prompt_loader, mock_state_manager):
        """Create an orchestrator instance."""
        with patch("crs.agents.orchestrator.ItemRetriever"):
            return ConversationOrchestrator(
                mock_model, mock_prompt_loader, mock_state_manager
            )

    def test_elicit_decision_route(self, orchestrator):
        """Test routing for ELICIT decision."""
        conversation_id = "test_conv_123"

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
            patch.object(
                orchestrator.retrieval_stage, "execute"
            ) as mock_retrieval,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.PARTIAL,
                "summary": "Gaming",
                "preference_count": 1,
            }
            mock_decision.return_value = {"decision": DecisionType.ELICIT}
            mock_response.return_value = {"response": "Tell me more"}

            orchestrator.process_message(conversation_id, "I need a laptop", [])

            # Retrieval should NOT be called for ELICIT
            mock_retrieval.assert_not_called()

    def test_recommend_decision_route(self, orchestrator):
        """Test routing for RECOMMEND decision."""
        conversation_id = "test_conv_123"

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.retrieval_stage, "execute"
            ) as mock_retrieval,
            patch.object(
                orchestrator.selection_stage, "execute"
            ) as mock_selection,
            patch.object(
                orchestrator.recommendation_stage, "execute"
            ) as mock_recommend,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.SUFFICIENT,
                "summary": "Gaming laptop, $1500",
                "preference_count": 2,
            }
            mock_decision.return_value = {"decision": DecisionType.RECOMMEND}
            mock_retrieval.return_value = {
                "items": [{"title": "Laptop 1"}],
                "item_count": 1,
            }
            mock_selection.return_value = {
                "selected_items": [{"title": "Laptop 1"}]
            }
            mock_recommend.return_value = {
                "recommendation_text": "I recommend Laptop 1"
            }
            mock_response.return_value = {"response": "I recommend Laptop 1"}

            orchestrator.process_message(
                conversation_id, "Can you recommend something?", []
            )

            # All recommendation stages should be called
            mock_retrieval.assert_called_once()
            mock_selection.assert_called_once()
            mock_recommend.assert_called_once()

    def test_answer_decision_route(self, orchestrator):
        """Test routing for ANSWER decision."""
        conversation_id = "test_conv_123"

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(orchestrator.qa_stage, "execute") as mock_qa,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
            patch.object(
                orchestrator.retrieval_stage, "execute"
            ) as mock_retrieval,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.NONE,
                "summary": "",
                "preference_count": 0,
            }
            mock_decision.return_value = {"decision": DecisionType.ANSWER}
            mock_qa.return_value = {"answer": "RAM is memory"}
            mock_response.return_value = {"response": "RAM is memory"}

            orchestrator.process_message(conversation_id, "What is RAM?", [])

            # QA stage should be called, retrieval should NOT
            mock_qa.assert_called_once()
            mock_retrieval.assert_not_called()


class TestOrchestratorEdgeCases:
    """Test cases for edge cases and error handling."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()

    @pytest.fixture
    def mock_prompt_loader(self):
        """Create a mock prompt loader."""
        loader = Mock()
        loader.load_prompt = Mock(return_value="Test prompt")
        return loader

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        state = ConversationState()
        state.domain = "laptop"
        manager.get_state = Mock(return_value=state)
        return manager

    @pytest.fixture
    def orchestrator(self, mock_model, mock_prompt_loader, mock_state_manager):
        """Create an orchestrator instance."""
        with patch("crs.agents.orchestrator.ItemRetriever"):
            return ConversationOrchestrator(
                mock_model, mock_prompt_loader, mock_state_manager
            )

    def test_empty_user_message(self, orchestrator):
        """Test handling of empty user message."""
        conversation_id = "test_conv_123"

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.NONE,
                "summary": "",
                "preference_count": 0,
            }
            mock_decision.return_value = {"decision": DecisionType.ELICIT}
            mock_response.return_value = {"response": "How can I help?"}

            response = orchestrator.process_message(conversation_id, "", [])

            assert isinstance(response, str)

    def test_very_long_chat_history(self, orchestrator):
        """Test handling of very long chat history."""
        conversation_id = "test_conv_123"
        long_history = []

        # Create 100 message pairs
        for i in range(100):
            long_history.append(HumanMessage(content=f"Message {i}"))
            long_history.append(AIMessage(content=f"Response {i}"))

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
            patch.object(
                orchestrator.response_stage, "execute"
            ) as mock_response,
        ):

            mock_pref.return_value = {
                "status": PreferenceStatus.SUFFICIENT,
                "summary": "Gaming laptop",
                "preference_count": 5,
            }
            mock_decision.return_value = {"decision": DecisionType.RECOMMEND}
            mock_response.return_value = {"response": "I recommend..."}

            response = orchestrator.process_message(
                conversation_id, "Recommend something", long_history
            )

            assert isinstance(response, str)

    def test_stage_execution_failure_handling(self, orchestrator):
        """Test handling when a stage execution fails."""
        conversation_id = "test_conv_123"

        with (
            patch.object(orchestrator.preference_stage, "execute") as mock_pref,
            patch.object(
                orchestrator.decision_stage, "execute"
            ) as mock_decision,
        ):

            # Simulate preference stage failure
            mock_pref.side_effect = Exception("Stage execution failed")

            # Should handle error gracefully
            with pytest.raises(Exception) as exc_info:
                orchestrator.process_message(
                    conversation_id, "I need a laptop", []
                )

            assert "Stage execution failed" in str(exc_info.value)


class TestOrchestratorIntegration:
    """Integration tests for complete orchestrator workflows."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with realistic responses."""
        model = Mock()
        return model

    @pytest.fixture
    def mock_prompt_loader(self):
        """Create a real prompt loader."""
        from crs.agents.prompt_loader import PromptLoader

        return PromptLoader()

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        from crs.agents.state_manager import StateManager

        return StateManager()

    def test_complete_conversation_flow(
        self, mock_model, mock_prompt_loader, mock_state_manager
    ):
        """Test a complete conversation from start to recommendation."""
        with patch("crs.agents.orchestrator.ItemRetriever"):
            orchestrator = ConversationOrchestrator(
                mock_model, mock_prompt_loader, mock_state_manager
            )

        conversation_id = "integration_test_123"

        # Message 1: Initial preference
        with patch.object(mock_model, "invoke") as mock_invoke:
            mock_invoke.return_value = AIMessage(
                content="Elicit\nNeed more info"
            )

            response1 = orchestrator.process_message(
                conversation_id, "I need a laptop", []
            )
            assert isinstance(response1, str)

        # Message 2: Add more preferences
        history1 = [
            HumanMessage(content="I need a laptop"),
            AIMessage(content=response1),
        ]

        with patch.object(mock_model, "invoke") as mock_invoke:
            mock_invoke.return_value = AIMessage(
                content="Recommend\nUser has enough preferences"
            )

            # Mock retrieval
            with (
                patch.object(
                    orchestrator.retrieval_stage, "execute"
                ) as mock_retrieval,
                patch.object(
                    orchestrator.selection_stage, "execute"
                ) as mock_selection,
            ):

                mock_retrieval.return_value = {
                    "items": [{"title": "Gaming Pro"}],
                    "item_count": 1,
                }
                mock_selection.return_value = {
                    "selected_items": [{"title": "Gaming Pro"}]
                }

                response2 = orchestrator.process_message(
                    conversation_id, "For gaming, budget $1500", history1
                )
                assert isinstance(response2, str)
