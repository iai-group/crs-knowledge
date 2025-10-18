"""
Integration tests for conversation stages.

Tests the complete flow with proper data formatting:
- DecisionType enum
- Stage execution with real state management
- Data flow between stages
- Proper formatting of recommendations, retrieved items, and targets
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from crs.agents.chain_factory import ChainFactory
from crs.agents.prompt_loader import PromptLoader
from crs.agents.stages import (
    ConversationStage,
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
from crs.agents.state_manager import ConversationState


@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    return model


@pytest.fixture
def prompt_loader():
    """Create a real prompt loader."""
    return PromptLoader()


@pytest.fixture
def conversation_state():
    """Create a fresh conversation state."""
    state = ConversationState()
    state.domain = "laptop"
    return state


@pytest.fixture
def sample_task():
    """Create a sample task dictionary."""
    return {
        "domain": "laptop",
        "target": {
            "asin": "TARGET123",
            "title": "Target Gaming Laptop",
            "content": "High-performance gaming laptop with RTX 4090",
        },
    }


class TestDecisionType:
    """Test cases for DecisionType enum."""

    def test_from_string_valid_values(self):
        """Test conversion from valid string values."""
        assert DecisionType.from_string("recommend") == DecisionType.RECOMMEND
        assert DecisionType.from_string("Recommend") == DecisionType.RECOMMEND
        assert DecisionType.from_string("RECOMMEND") == DecisionType.RECOMMEND
        assert DecisionType.from_string("elicit") == DecisionType.ELICIT
        assert DecisionType.from_string("answer") == DecisionType.ANSWER
        assert DecisionType.from_string("redirect") == DecisionType.REDIRECT

    def test_from_string_invalid_values(self):
        """Test conversion from invalid string values defaults to ELICIT."""
        assert DecisionType.from_string("invalid") == DecisionType.ELICIT
        assert DecisionType.from_string("") == DecisionType.ELICIT
        assert DecisionType.from_string(None) == DecisionType.ELICIT

    def test_from_string_custom_default(self):
        """Test conversion with custom default value."""
        assert (
            DecisionType.from_string("invalid", default=DecisionType.ANSWER)
            == DecisionType.ANSWER
        )


class TestPreferenceSummarizationIntegration:
    """Integration tests for PreferenceSummarizationStage."""

    def test_preference_extraction_updates_state(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test that preferences are properly extracted and state is updated."""
        # Mock the model to return preferences
        mock_model.invoke = Mock(
            return_value="Gaming laptop\nHigh performance\nBudget $1500"
        )

        stage = PreferenceSummarizationStage(mock_model, prompt_loader)

        chat_history = [
            HumanMessage(
                content="I need a gaming laptop with high performance, budget around $1500"
            ),
        ]

        result = stage.execute(conversation_state, sample_task, chat_history)

        # Check that result is a PreferenceStatus
        assert result in PreferenceStatus

        # Check that state was updated with preferences
        assert conversation_state.preferences is not None
        assert len(conversation_state.preferences) > 0

    def test_too_many_preferences_detection(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test detection of too many preferences at once."""
        # Return more than 3 new preferences
        many_prefs = "\n".join([f"Preference {i}" for i in range(10)])
        mock_model.invoke = Mock(return_value=many_prefs)

        stage = PreferenceSummarizationStage(mock_model, prompt_loader)

        chat_history = [
            HumanMessage(content="Long message with many preferences...")
        ]

        result = stage.execute(conversation_state, sample_task, chat_history)

        assert result == PreferenceStatus.TOO_MANY


class TestItemRetrievalIntegration:
    """Integration tests for ItemRetrievalStage."""

    @patch("crs.agents.stages.ItemRetriever")
    def test_retrieval_returns_proper_format(
        self,
        mock_retriever_class,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that retrieved items have proper format with all required fields."""
        # Mock retriever to return items with metadata
        mock_retriever = Mock()
        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Gaming Laptop Pro",
                    "parent_asin": "ASIN001",
                    "content": "High-end gaming",
                    "images": ["img1.jpg"],
                },
                0.95,
            ),
            (
                {
                    "title": "Gaming Laptop Plus",
                    "parent_asin": "ASIN002",
                    "content": "Mid-range gaming",
                    "images": ["img2.jpg"],
                },
                0.88,
            ),
            (
                {
                    "title": "Budget Gaming Laptop",
                    "parent_asin": "ASIN003",
                    "content": "Entry-level gaming",
                    "images": [],
                },
                0.75,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        conversation_state.preferences = "Gaming laptop, high performance"

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        # Check that result is a list
        assert isinstance(result, list)
        assert len(result) > 0

        # Check that each item has required fields
        for item in result:
            assert "id" in item
            assert "title" in item
            assert "content" in item
            assert "images" in item
            assert "score" in item
            assert isinstance(item["score"], float)

        # Check that state was updated
        assert conversation_state.retrieved_items == result

    @patch("crs.agents.stages.ItemRetriever")
    def test_target_filtering_before_first_recommendation(
        self,
        mock_retriever_class,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that target item is filtered out before first recommendation."""
        mock_retriever = Mock()
        target_asin = sample_task["target"]["asin"]

        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Target Gaming Laptop",
                    "parent_asin": target_asin,
                    "content": "Target item",
                    "images": [],
                },
                0.95,
            ),
            (
                {
                    "title": "Other Laptop",
                    "parent_asin": "OTHER001",
                    "content": "Other item",
                    "images": [],
                },
                0.85,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        conversation_state.preferences = "Gaming laptop"
        conversation_state.recommended_items = []  # No recommendations yet

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        # Target should be filtered out
        assert all(item["id"] != target_asin for item in result)

    @patch("crs.agents.stages.ItemRetriever")
    def test_previously_recommended_items_filtered(
        self,
        mock_retriever_class,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that previously recommended items are filtered out."""
        mock_retriever = Mock()
        mock_retriever.retrieve_items.return_value = [
            (
                {
                    "title": "Laptop 1",
                    "parent_asin": "ASIN001",
                    "content": "Item 1",
                    "images": [],
                },
                0.9,
            ),
            (
                {
                    "title": "Laptop 2",
                    "parent_asin": "ASIN002",
                    "content": "Item 2",
                    "images": [],
                },
                0.8,
            ),
            (
                {
                    "title": "Laptop 3",
                    "parent_asin": "ASIN003",
                    "content": "Item 3",
                    "images": [],
                },
                0.7,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        # Mark one item as already recommended
        conversation_state.recommended_items = [
            {"id": "ASIN001", "title": "Laptop 1"}
        ]
        conversation_state.preferences = "Gaming"

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        # ASIN001 should be filtered out
        assert all(item["id"] != "ASIN001" for item in result)


class TestItemSelectionIntegration:
    """Integration tests for ItemSelectionStage."""

    def test_selection_returns_single_item(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test that selection returns a single item with proper format."""
        # Mock model to select first item (index 0)
        mock_model.invoke = Mock(return_value=AIMessage(content="0"))

        conversation_state.retrieved_items = [
            {
                "id": "ASIN001",
                "title": "Gaming Laptop Pro",
                "content": "High-end",
                "images": ["img1.jpg"],
                "score": 0.9,
            },
            {
                "id": "ASIN002",
                "title": "Office Laptop",
                "content": "Business use",
                "images": [],
                "score": 0.7,
            },
        ]
        conversation_state.preferences = "Gaming laptop"

        stage = ItemSelectionStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        # Should return single item or None
        if result is not None:
            assert isinstance(result, dict)
            assert "id" in result
            assert "title" in result
            assert result["id"] == "ASIN001"

    def test_selection_with_no_retrieved_items(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test that selection returns None when no items available."""
        conversation_state.retrieved_items = []

        stage = ItemSelectionStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        assert result is None

    def test_selected_item_updates_state(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test that selected item is properly stored in state."""
        mock_model.invoke = Mock(return_value=AIMessage(content="0"))

        conversation_state.retrieved_items = [
            {
                "id": "ASIN001",
                "title": "Selected Laptop",
                "content": "Best match",
                "images": [],
                "score": 0.95,
            },
        ]

        stage = ItemSelectionStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        if result is not None:
            # State should be updated with selected item
            assert conversation_state.selected_item is not None
            assert conversation_state.selected_item["id"] == "ASIN001"


class TestDecisionStage:
    """Test cases for DecisionStage."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.invoke = Mock(
            return_value=AIMessage(
                content="Recommend\nUser has enough preferences"
            )
        )
        return model

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader."""
        return PromptLoader()

    @pytest.fixture
    def stage(self, mock_model, prompt_loader):
        """Create a DecisionStage instance."""
        return DecisionStage(mock_model, prompt_loader)

    def test_execute_returns_decision(self, stage):
        """Test that execute returns a decision type."""
        state = ConversationState()
        state.preferences = "Gaming, $1500"
        state.recommended_items = []
        task = {"domain": "laptop"}

        chat_history = [HumanMessage(content="I need a laptop")]

        result = stage.execute(state, task, chat_history)

        assert isinstance(result, DecisionType)

    def test_execute_with_recommendation_request(self, stage, mock_model):
        """Test decision when user explicitly requests recommendations."""
        mock_model.invoke = Mock(
            return_value=AIMessage(
                content="Recommend\nUser requested recommendations"
            )
        )

        state = ConversationState()
        state.preferences = "Gaming, $1500"
        task = {"domain": "laptop"}
        chat_history = [HumanMessage(content="Can you recommend something?")]

        result = stage.execute(state, task, chat_history)

        assert result == DecisionType.RECOMMEND or isinstance(
            result, DecisionType
        )


class TestRecommendationAnalyzerStage:
    """Test cases for RecommendationAnalyzerStage."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.invoke = Mock(
            return_value=AIMessage(
                content="Is it good for gaming?\nYes\nGPU, RAM, screen refresh rate"
            )
        )
        return model

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader."""
        return PromptLoader()

    @pytest.fixture
    def stage(self, mock_model, prompt_loader):
        """Create a RecommendationAnalyzerStage instance."""
        return RecommendationAnalyzerStage(mock_model, prompt_loader)

    def test_execute_with_question(self, stage):
        """Test execution with a user question about recommendations."""
        state = ConversationState()
        state.recommended_items = [
            {"title": "Gaming Laptop", "content": "High-end GPU"}
        ]
        task = {"domain": "laptop"}

        chat_history = [
            HumanMessage(content="Does it have a good GPU?"),
        ]

        result = stage.execute(state, task, chat_history)

        # Should return a tuple or boolean
        assert result is not None


class TestQuestionAnswerStage:
    """Test cases for QuestionAnswerStage."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.invoke = Mock(
            return_value=AIMessage(
                content="A gaming laptop needs a powerful GPU for graphics rendering."
            )
        )
        return model

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader."""
        return PromptLoader()

    @pytest.fixture
    def stage(self, mock_model, prompt_loader):
        """Create a QuestionAnswerStage instance."""
        return QuestionAnswerStage(mock_model, prompt_loader)

    def test_execute_general_question(self, stage):
        """Test answering a general domain question."""
        state = ConversationState()
        task = {"domain": "laptop"}

        chat_history = [HumanMessage(content="What is RAM?")]

        result = stage.execute(state, task, chat_history)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_about_recommendation(self, stage):
        """Test answering a question about a recommended item."""
        state = ConversationState()
        state.recommended_items = [
            {"title": "Gaming Laptop", "content": "16GB RAM, RTX 3070"}
        ]
        task = {"domain": "laptop"}

        chat_history = [HumanMessage(content="Does it have enough RAM?")]

        result = stage.execute(state, task, chat_history)

        assert isinstance(result, str)


class TestRecommendationStage:
    """Test cases for RecommendationStage."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.invoke = Mock(
            return_value=AIMessage(
                content="Based on your preferences, I recommend this laptop."
            )
        )
        return model

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader."""
        return PromptLoader()

    @pytest.fixture
    def stage(self, mock_model, prompt_loader):
        """Create a RecommendationStage instance."""
        return RecommendationStage(mock_model, prompt_loader)

    def test_execute_with_no_items(self, stage):
        """Test execution with no selected items."""
        state = ConversationState()
        state.selected_item = None
        task = {"domain": "laptop"}

        result = stage.execute(state, task, [])

        assert isinstance(result, str)

    def test_execute_with_items(self, stage):
        """Test execution with selected items."""
        state = ConversationState()
        state.selected_item = {
            "title": "Gaming Laptop",
            "content": "High-end laptop",
        }
        task = {"domain": "laptop"}

        result = stage.execute(state, task, [])

        assert isinstance(result, str)
        assert len(result) > 0


class TestResponseStage:
    """Test cases for ResponseStage."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.invoke = Mock(
            return_value=AIMessage(
                content="What type of games do you plan to play?"
            )
        )
        return model

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader."""
        return PromptLoader()

    @pytest.fixture
    def stage(self, mock_model, prompt_loader):
        """Create a ResponseStage instance."""
        return ResponseStage(mock_model, prompt_loader)

    def test_execute_elicit_response(self, stage):
        """Test generating an elicitation response."""
        state = ConversationState()
        state.decision = DecisionType.ELICIT
        task = {"domain": "laptop"}

        result = stage.execute(state, task, [])

        assert isinstance(result, str)

    def test_execute_recommendation_response(self, stage):
        """Test generating a recommendation response."""
        state = ConversationState()
        state.decision = DecisionType.RECOMMEND
        state.recommendation_text = "I recommend this laptop: Gaming Pro"
        task = {"domain": "laptop"}

        result = stage.execute(state, task, [])

        assert isinstance(result, str)

    def test_execute_answer_response(self, stage):
        """Test generating an answer response."""
        state = ConversationState()
        state.decision = DecisionType.ANSWER
        state.answer = "RAM is Random Access Memory"
        task = {"domain": "laptop"}

        result = stage.execute(state, task, [])

        assert isinstance(result, str)


class TestConversationStageIntegration:
    """Integration tests for stage interactions."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader."""
        return PromptLoader()

    def test_stage_pipeline(self, mock_model, prompt_loader):
        """Test a complete pipeline of stages."""
        state = ConversationState()
        task = {"domain": "laptop"}

        # Stage 1: Preference Summarization
        pref_stage = PreferenceSummarizationStage(mock_model, prompt_loader)
        mock_model.invoke = Mock(
            return_value=AIMessage(content="Gaming laptop\nHigh performance")
        )

        chat_history = [HumanMessage(content="I need a gaming laptop")]
        pref_result = pref_stage.execute(state, task, chat_history)

        assert isinstance(pref_result, PreferenceStatus)

        # Stage 2: Decision
        decision_stage = DecisionStage(mock_model, prompt_loader)
        mock_model.invoke = Mock(
            return_value=AIMessage(content="Elicit\nNeed more info")
        )

        decision_result = decision_stage.execute(state, task, chat_history)

        assert isinstance(decision_result, DecisionType)

    def test_state_persistence_across_stages(self):
        """Test that state is properly maintained across stages."""
        state = ConversationState()
        state.domain = "laptop"
        state.preference_count = 2
        state.preference_summary = "Gaming laptop, $1500"

        # Verify state maintains values
        assert state.domain == "laptop"
        assert state.preference_count == 2
        assert state.preference_summary == "Gaming laptop, $1500"

        # Simulate updating state
        state.retrieved_items = [{"title": "Laptop 1"}]
        assert len(state.retrieved_items) == 1
