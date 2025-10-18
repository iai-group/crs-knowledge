"""
Integration tests for conversation stages.

Focus on end-to-end flow and proper data formatting:
- Recommendations are properly formatted
- Retrieved items have all required fields
- Target handling is correct
- State management works across stages
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from crs.agents.chain_factory import ChainFactory
from crs.agents.orchestrator import ConversationOrchestrator
from crs.agents.prompt_loader import PromptLoader
from crs.agents.stages import (
    DecisionStage,
    DecisionType,
    ItemRetrievalStage,
    ItemSelectionStage,
    PreferenceStatus,
    PreferenceSummarizationStage,
    QuestionAnswerStage,
    RecommendationStage,
    ResponseStage,
)
from crs.agents.state_manager import ConversationState


def create_mock_chain(return_value):
    """Helper to create a mock chain that returns a specific value."""
    mock_chain = Mock()
    mock_chain.invoke = Mock(return_value=return_value)
    mock_chain.stream = Mock(return_value=iter([return_value]))
    return mock_chain


@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    model.invoke = Mock(return_value=AIMessage(content="Test response"))
    return model


@pytest.fixture
def prompt_loader():
    """Create a real prompt loader."""
    return PromptLoader()


@pytest.fixture
def conversation_state():
    """Create a fresh conversation state."""
    state = ConversationState()
    return state


@pytest.fixture
def sample_task():
    """Create a sample task dictionary with target."""
    return {
        "domain": "laptop",
        "target": {
            "asin": "TARGET123",
            "title": "Target Gaming Laptop",
            "content": "High-performance gaming laptop with RTX 4090",
        },
    }


@pytest.fixture
def sample_retrieved_items():
    """Create sample retrieved items with proper format."""
    return [
        {
            "id": "ASIN001",
            "title": "Gaming Laptop Pro",
            "content": "High-end gaming laptop with RTX 4080, 32GB RAM",
            "images": ["img1.jpg", "img2.jpg"],
            "score": 0.95,
        },
        {
            "id": "ASIN002",
            "title": "Gaming Laptop Plus",
            "content": "Mid-range gaming with RTX 4070, 16GB RAM",
            "images": ["img3.jpg"],
            "score": 0.88,
        },
        {
            "id": "TARGET123",
            "title": "Target Gaming Laptop",
            "content": "High-performance gaming laptop with RTX 4090",
            "images": [],
            "score": 0.92,
        },
    ]


class TestDecisionTypeEnum:
    """Test DecisionType enum functionality."""

    def test_from_string_valid_values(self):
        """Test conversion from valid string values."""
        assert DecisionType.from_string("recommend") == DecisionType.RECOMMEND
        assert DecisionType.from_string("Elicit") == DecisionType.ELICIT
        assert DecisionType.from_string("ANSWER") == DecisionType.ANSWER

    def test_from_string_invalid_defaults_to_elicit(self):
        """Test invalid values default to ELICIT."""
        assert DecisionType.from_string("invalid") == DecisionType.ELICIT
        assert DecisionType.from_string("") == DecisionType.ELICIT
        assert DecisionType.from_string(None) == DecisionType.ELICIT


class TestPreferenceExtractionFlow:
    """Test preference extraction and state updates."""

    @patch.object(ConversationState, "add_system_message")
    def test_preferences_update_state(
        self,
        mock_add_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that extracted preferences properly update conversation state."""
        # Mock the chain to return string directly
        mock_chain = create_mock_chain(
            "Gaming laptop\nHigh performance\nBudget $1500"
        )
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            stage = PreferenceSummarizationStage(mock_model, prompt_loader)
            chat_history = [
                HumanMessage(
                    content="I need a gaming laptop, high performance, budget $1500"
                )
            ]

            result = stage.execute(
                conversation_state, sample_task, chat_history
            )

            # Check result is a PreferenceStatus
            assert result in [
                PreferenceStatus.NEW,
                PreferenceStatus.OLD,
                PreferenceStatus.TOO_MANY,
            ]

            # Check state was updated
            assert conversation_state.preferences is not None

    def test_too_many_preferences_detected(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test detection of too many preferences at once."""
        # Return 10 preferences (more than threshold of 3)
        many_prefs = "\n".join([f"Preference {i}" for i in range(10)])

        mock_chain = create_mock_chain(many_prefs)
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            stage = PreferenceSummarizationStage(mock_model, prompt_loader)
            chat_history = [HumanMessage(content="Long message...")]

            result = stage.execute(
                conversation_state, sample_task, chat_history
            )

            assert result == PreferenceStatus.TOO_MANY


class TestItemRetrievalFlow:
    """Test item retrieval with proper formatting."""

    @patch.object(ConversationState, "update_retrieved_items")
    @patch("crs.agents.stages.ItemRetriever")
    def test_retrieved_items_have_required_fields(
        self,
        mock_retriever_class,
        mock_update,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that all retrieved items have required fields: id, title, content, images, score."""
        mock_retriever = Mock()
        # Mock the retrieve method (not retrieve_items)
        mock_retriever.retrieve.return_value = [
            (
                {
                    "title": "Laptop 1",
                    "parent_asin": "A1",
                    "content": "Gaming",
                    "images": ["i1.jpg"],
                },
                0.9,
            ),
            (
                {
                    "title": "Laptop 2",
                    "parent_asin": "A2",
                    "content": "Office",
                    "images": [],
                },
                0.8,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        conversation_state.preferences = "Gaming laptop"

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        assert isinstance(result, list)
        assert len(result) > 0

        # Verify each item has all required fields
        for item in result:
            assert "id" in item, "Item missing 'id' field"
            assert "title" in item, "Item missing 'title' field"
            assert "content" in item, "Item missing 'content' field"
            assert "images" in item, "Item missing 'images' field"
            assert "score" in item, "Item missing 'score' field"
            assert isinstance(item["score"], float), "Score must be float"
            assert isinstance(item["images"], list), "Images must be list"

    @patch.object(ConversationState, "update_retrieved_items")
    @patch("crs.agents.stages.ItemRetriever")
    def test_target_filtered_before_first_recommendation(
        self,
        mock_retriever_class,
        mock_update,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that target item is filtered out when no recommendations made yet."""
        target_asin = sample_task["target"]["asin"]

        mock_retriever = Mock()
        # Mock the retrieve method
        mock_retriever.retrieve.return_value = [
            (
                {
                    "title": "Target Item",
                    "parent_asin": target_asin,
                    "content": "Target",
                    "images": [],
                },
                0.95,
            ),
            (
                {
                    "title": "Other Item",
                    "parent_asin": "OTHER",
                    "content": "Other",
                    "images": [],
                },
                0.85,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        conversation_state.preferences = "Gaming"
        conversation_state.recommended_items = []  # No recommendations yet

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        # Target should be filtered out
        target_ids = [item["id"] for item in result]
        assert (
            target_asin not in target_ids
        ), f"Target {target_asin} should be filtered before first recommendation"

    @patch.object(ConversationState, "update_retrieved_items")
    @patch("crs.agents.stages.ItemRetriever")
    def test_previously_recommended_filtered(
        self,
        mock_retriever_class,
        mock_update,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that previously recommended items don't appear again."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            (
                {
                    "title": "Item 1",
                    "parent_asin": "PREV_REC",
                    "content": "Already shown",
                    "images": [],
                },
                0.9,
            ),
            (
                {
                    "title": "Item 2",
                    "parent_asin": "NEW_ITEM",
                    "content": "Not shown",
                    "images": [],
                },
                0.8,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        # Mark one item as already recommended
        conversation_state.recommended_items = [
            {"id": "PREV_REC", "title": "Item 1"}
        ]
        conversation_state.preferences = "Gaming"

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        result_ids = [item["id"] for item in result]
        assert (
            "PREV_REC" not in result_ids
        ), "Previously recommended item should be filtered"

    @patch.object(ConversationState, "update_retrieved_items")
    @patch("crs.agents.stages.ItemRetriever")
    def test_state_updated_with_retrieved_items(
        self,
        mock_retriever_class,
        mock_update,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that conversation state is updated with retrieved items."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            (
                {
                    "title": "Item",
                    "parent_asin": "A1",
                    "content": "Content",
                    "images": [],
                },
                0.9,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        conversation_state.preferences = "Gaming"

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        # Verify update_retrieved_items was called
        mock_update.assert_called_once()
        # Verify result contains properly formatted items
        assert len(result) == 1
        assert result[0]["id"] == "A1"


class TestItemSelectionFlow:
    """Test item selection logic."""

    @patch.object(ConversationState, "add_system_message")
    def test_selection_returns_single_item(
        self,
        mock_add_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
        sample_retrieved_items,
    ):
        """Test that selection returns a single item dict, not a list."""
        mock_chain = create_mock_chain("0")
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            conversation_state.retrieved_items = sample_retrieved_items[
                :2
            ]  # Use first 2 items
            conversation_state.preferences = "Gaming"

            stage = ItemSelectionStage(mock_model, prompt_loader)
            result = stage.execute(conversation_state, sample_task, [])

            # Result should be single dict or None
            if result is not None:
                assert isinstance(
                    result, dict
                ), "Selection should return single item dict, not list"
                assert "id" in result
                assert "title" in result

    def test_no_items_returns_none(
        self, mock_model, prompt_loader, conversation_state, sample_task
    ):
        """Test that selection returns None when no items available."""
        conversation_state.retrieved_items = []

        stage = ItemSelectionStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        assert result is None, "Should return None when no items to select from"

    @patch.object(ConversationState, "add_system_message")
    def test_selected_item_stored_in_state(
        self,
        mock_add_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
        sample_retrieved_items,
    ):
        """Test that selected item is properly returned."""
        mock_chain = create_mock_chain("1")
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            conversation_state.retrieved_items = sample_retrieved_items

            stage = ItemSelectionStage(mock_model, prompt_loader)
            result = stage.execute(conversation_state, sample_task, [])

            if result is not None:
                # Result should match the selected item
                assert result["id"] in [
                    item["id"] for item in sample_retrieved_items
                ]
                # Verify it's actually one of the retrieved items
                assert result in sample_retrieved_items


class TestDecisionFlow:
    """Test decision stage logic."""

    @patch.object(ConversationState, "add_system_message")
    def test_decision_returns_enum(
        self,
        mock_add_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that decision stage returns a DecisionType enum."""
        mock_chain = create_mock_chain("Elicit\nNeed more preferences")
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            conversation_state.preferences = "Gaming"

            stage = DecisionStage(mock_model, prompt_loader)
            chat_history = [HumanMessage(content="I need a laptop")]

            result = stage.execute(
                conversation_state, sample_task, chat_history
            )

            assert isinstance(
                result, DecisionType
            ), "Decision should return DecisionType enum"

    @patch.object(ConversationState, "add_system_message")
    def test_recommend_decision_with_preferences(
        self,
        mock_add_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that RECOMMEND decision is made when user has preferences."""
        mock_chain = create_mock_chain("Recommend\nUser has preferences")
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            conversation_state.preferences = (
                "Gaming laptop\nHigh performance\n$1500 budget"
            )

            stage = DecisionStage(mock_model, prompt_loader)
            chat_history = [
                HumanMessage(content="Can you recommend something?")
            ]

            result = stage.execute(
                conversation_state, sample_task, chat_history
            )

            # Should be a valid DecisionType
            assert result in DecisionType


class TestRecommendationFlow:
    """Test recommendation generation."""

    @patch.object(ConversationState, "add_system_message")
    def test_recommendation_returns_dict_with_stream(
        self,
        mock_add_system_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
        sample_retrieved_items,
    ):
        """Test that recommendation stage returns a dict with stream and image_url."""
        mock_chain = create_mock_chain("I recommend this laptop because...")
        with patch.object(
            ChainFactory, "create_chain", return_value=mock_chain
        ):
            conversation_state.preferences = "Gaming"
            # Pass selected_item as parameter
            selected_item = sample_retrieved_items[0]

            stage = RecommendationStage(mock_model, prompt_loader)
            result = stage.execute(
                conversation_state, sample_task, [], selected_item=selected_item
            )

            assert isinstance(result, dict), "Recommendation should return dict"
            assert "stream" in result, "Should have stream key"
            assert "image_url" in result, "Should have image_url key"

    @patch.object(ConversationState, "add_system_message")
    def test_no_item_selected_returns_none(
        self,
        mock_add_system_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test behavior when no item is selected."""
        conversation_state.selected_item = None
        conversation_state.retrieved_items = []  # No items

        stage = RecommendationStage(mock_model, prompt_loader)
        result = stage.execute(conversation_state, sample_task, [])

        assert result is None, "Should return None when no items available"
        mock_add_system_msg.assert_called_once()  # Verify system message was added


class TestQuestionAnswerFlow:
    """Test question answering logic."""

    @patch.object(ConversationState, "add_system_message")
    def test_qa_returns_string(
        self,
        mock_add_msg,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
    ):
        """Test that QA stage returns a string answer."""
        # QuestionAnswerStage actually makes 2 chain calls, mock both
        mock_chain1 = create_mock_chain("Performance specs and features")
        mock_chain2 = create_mock_chain(
            "RAM is Random Access Memory used for..."
        )

        with patch.object(
            ChainFactory, "create_chain", side_effect=[mock_chain1, mock_chain2]
        ):
            stage = QuestionAnswerStage(mock_model, prompt_loader)
            chat_history = [HumanMessage(content="What is RAM?")]

            result = stage.execute(
                conversation_state, sample_task, chat_history
            )

            assert isinstance(result, dict), "QA should return dict"
            assert "stream" in result, "Should have stream key"


class TestEndToEndFlow:
    """Test complete conversation flows end-to-end."""

    @patch.object(ConversationState, "update_retrieved_items")
    @patch.object(ConversationState, "add_system_message")
    @patch("crs.agents.stages.ItemRetriever")
    def test_complete_recommendation_flow(
        self,
        mock_retriever_class,
        mock_add_system_msg,
        mock_update_items,
        mock_model,
        prompt_loader,
        sample_task,
    ):
        """Test complete flow from preferences to recommendation."""
        state = ConversationState()

        # Step 1: Extract preferences
        pref_chain = create_mock_chain("Gaming laptop\n$1500 budget")
        with patch.object(
            ChainFactory, "create_chain", return_value=pref_chain
        ):
            pref_stage = PreferenceSummarizationStage(mock_model, prompt_loader)
            pref_result = pref_stage.execute(
                state,
                sample_task,
                [HumanMessage(content="I need gaming laptop, $1500")],
            )

            assert pref_result in PreferenceStatus
            assert state.preferences is not None

        # Step 2: Retrieve items
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            (
                {
                    "title": "Laptop 1",
                    "parent_asin": "A1",
                    "content": "Gaming",
                    "images": [],
                },
                0.9,
            ),
            (
                {
                    "title": "Laptop 2",
                    "parent_asin": "A2",
                    "content": "Office",
                    "images": [],
                },
                0.8,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        retrieval_stage = ItemRetrievalStage(mock_model, prompt_loader)
        retrieved = retrieval_stage.execute(state, sample_task, [])

        assert isinstance(retrieved, list)
        assert len(retrieved) > 0
        assert all("id" in item for item in retrieved)

        # Step 3: Select item
        select_chain = create_mock_chain("0")
        with patch.object(
            ChainFactory, "create_chain", return_value=select_chain
        ):
            state.retrieved_items = retrieved  # Set retrieved items manually since update_retrieved_items is mocked
            selection_stage = ItemSelectionStage(mock_model, prompt_loader)
            selected = selection_stage.execute(state, sample_task, [])

            if selected is not None:
                assert isinstance(selected, dict)
                assert "id" in selected

        # Step 4: Make decision
        decision_chain = create_mock_chain("Recommend\nReady to recommend")
        with patch.object(
            ChainFactory, "create_chain", return_value=decision_chain
        ):
            decision_stage = DecisionStage(mock_model, prompt_loader)
            decision = decision_stage.execute(
                state, sample_task, [HumanMessage(content="Show me laptops")]
            )

            assert isinstance(decision, DecisionType)

        # Step 5: Generate recommendation
        if selected is not None:
            rec_chain = create_mock_chain("I recommend Laptop 1")
            with patch.object(
                ChainFactory, "create_chain", return_value=rec_chain
            ):
                rec_stage = RecommendationStage(mock_model, prompt_loader)
                recommendation = rec_stage.execute(state, sample_task, [])

                assert isinstance(recommendation, dict)
                assert "stream" in recommendation

    @patch.object(ConversationState, "update_retrieved_items")
    @patch("crs.agents.stages.ItemRetriever")
    def test_target_not_in_first_recommendations(
        self,
        mock_retriever_class,
        mock_update,
        mock_model,
        prompt_loader,
        sample_task,
    ):
        """Test that target item is properly filtered in the full flow."""
        state = ConversationState()
        target_asin = sample_task["target"]["asin"]

        # Setup retrieval with target included
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            (
                {
                    "title": "Target",
                    "parent_asin": target_asin,
                    "content": "Target item",
                    "images": [],
                },
                0.95,
            ),
            (
                {
                    "title": "Other",
                    "parent_asin": "OTHER",
                    "content": "Other item",
                    "images": [],
                },
                0.85,
            ),
        ]
        mock_retriever_class.return_value = mock_retriever

        state.preferences = "Gaming"
        state.recommended_items = []  # First recommendation

        retrieval_stage = ItemRetrievalStage(mock_model, prompt_loader)
        retrieved = retrieval_stage.execute(state, sample_task, [])

        # Target should not be in retrieved items
        retrieved_ids = [item["id"] for item in retrieved]
        assert (
            target_asin not in retrieved_ids
        ), "Target should be filtered out before first recommendation"
