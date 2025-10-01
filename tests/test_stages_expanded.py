"""
Tests for the expanded conversation stages.

Tests the new modular stages:
1. PreferenceSummarizationStage
2. ItemRetrievalStage
3. Updated RecommendationStage
4. Integration with ConversationState
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from crs.agents.prompt_loader import PromptLoader
from crs.agents.stages import (
    ItemRetrievalStage,
    PreferenceSummarizationStage,
    RecommendationStage,
)
from crs.agents.state_manager import ConversationState


class TestExpandedStages:
    """Test cases for the expanded conversation stages."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        return model

    @pytest.fixture
    def prompt_loader(self):
        """Create a prompt loader for testing."""
        return PromptLoader()

    @pytest.fixture
    def conversation_state(self):
        """Create a conversation state for testing."""
        return ConversationState()

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return {"domain": "Movies", "target": {"name": "test_movie"}}

    @pytest.fixture
    def sample_chat_history(self):
        """Create sample chat history for testing."""
        return [
            HumanMessage("I like action movies with good special effects"),
            AIMessage("What about budget range?"),
            HumanMessage("Under $20 for rental, prefer recent movies"),
        ]

    def test_preference_summarization_stage(
        self,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
        sample_chat_history,
    ):
        """Test the PreferenceSummarizationStage."""
        # Mock the chain invoke to return preferences
        with patch.object(prompt_loader, "load_prompt") as mock_load:
            mock_load.return_value = "Test prompt template"

            stage = PreferenceSummarizationStage(mock_model, prompt_loader)

            # Mock the chain factory and chain
            with patch.object(
                stage.chain_factory, "create_chain"
            ) as mock_create:
                mock_chain = Mock()
                mock_chain.invoke.return_value = "Action movies, good special effects, under $20 rental, recent films"
                mock_create.return_value = mock_chain

                result = stage.execute(
                    conversation_state, sample_task, sample_chat_history
                )

                assert (
                    result
                    == "Action movies, good special effects, under $20 rental, recent films"
                )
                assert (
                    conversation_state.preferences
                    == "Action movies, good special effects, under $20 rental, recent films"
                )
                mock_chain.invoke.assert_called_once()

    def test_item_retrieval_stage(
        self,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
        sample_chat_history,
    ):
        """Test the ItemRetrievalStage."""
        # Set up preferences
        conversation_state.preferences = "Action movies, good special effects"

        stage = ItemRetrievalStage(mock_model, prompt_loader)
        result = stage.execute(
            conversation_state, sample_task, sample_chat_history
        )

        # Should return mock items
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(
            "id" in item and "title" in item and "score" in item
            for item in result
        )
        assert conversation_state.retrieved_items == result

    def test_updated_recommendation_stage(
        self,
        mock_model,
        prompt_loader,
        conversation_state,
        sample_task,
        sample_chat_history,
    ):
        """Test the updated RecommendationStage using retrieved items."""
        # Set up state with preferences and retrieved items
        conversation_state.preferences = "Action movies, good special effects"
        conversation_state.retrieved_items = [
            {"id": "movie1", "title": "Test Action Movie", "score": 0.9},
            {"id": "movie2", "title": "Another Action Film", "score": 0.8},
        ]

        with patch.object(prompt_loader, "load_prompt") as mock_load:
            mock_load.return_value = "Test recommendation prompt"

            stage = RecommendationStage(mock_model, prompt_loader)

            with patch.object(
                stage.chain_factory, "create_chain"
            ) as mock_create:
                mock_chain = Mock()
                mock_chain.invoke.return_value = "Based on your preferences, I recommend Test Action Movie and Another Action Film"
                mock_create.return_value = mock_chain

                result = stage.execute(
                    conversation_state, sample_task, sample_chat_history
                )

                # RecommendationStage now returns a dict with the model stream and image metadata
                assert isinstance(result, dict)
                assert (
                    "stream" in result
                    and "image_url" in result
                    and "item" in result
                )

                # The returned stream text should contain our mocked recommendation
                assert "Test Action Movie" in result["stream"]
                assert "Another Action Film" in result["stream"]

                # Ensure the chain stream was called (not invoke) and captured input data
                mock_chain.stream.assert_called_once()
                call_args = mock_chain.stream.call_args[0][0]
                assert "item_to_recommend" in call_args
                assert "preferences" in call_args
                # Image URL must not be passed into the agent input
                assert (
                    "item_image_url" not in call_args
                    and "image_url" not in call_args
                )

    def test_conversation_state_updates(self, conversation_state):
        """Test that ConversationState handles new fields correctly."""
        # Test preferences update
        test_preferences = "Action movies, sci-fi, under $15"
        conversation_state.update_preferences(test_preferences)
        assert conversation_state.preferences == test_preferences

        # Test retrieved items update
        test_items = [
            {"id": "1", "title": "Test Movie 1", "score": 0.9},
            {"id": "2", "title": "Test Movie 2", "score": 0.8},
        ]
        conversation_state.update_retrieved_items(test_items)
        assert conversation_state.retrieved_items == test_items

        # Check that system messages were added
        assert (
            len(conversation_state.chat_log) == 2
        )  # One for preferences, one for items


def demo_expanded_stages():
    """Demonstrate the new expanded stages workflow."""
    print("=== Expanded Stages Demo ===\n")

    from crs.agents.connectors import get_model_connector

    # Mock setup (since we don't want to actually call LLM in demo)
    model = Mock()
    prompt_loader = PromptLoader()
    state = ConversationState()

    task = {"domain": "Movies"}
    chat_history = [
        HumanMessage("I want action movies with great special effects"),
        AIMessage("What's your budget range?"),
        HumanMessage("Under $20 for rental, prefer recent films"),
    ]

    print("1. Initial State:")
    print(f"   Preferences: {state.preferences}")
    print(f"   Retrieved Items: {len(state.retrieved_items)}")
    print()

    # Stage 1: Preference Summarization
    print("2. Preference Summarization Stage:")
    pref_stage = PreferenceSummarizationStage(model, prompt_loader)

    # Mock the chain for demo
    with patch.object(pref_stage.chain_factory, "create_chain") as mock_create:
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Action movies, great special effects, under $20 rental, recent films"
        mock_create.return_value = mock_chain

        preferences = pref_stage.execute(state, task, chat_history)
        print(f"   Extracted Preferences: {preferences}")
    print()

    # Stage 2: Item Retrieval
    print("3. Item Retrieval Stage:")
    retrieval_stage = ItemRetrievalStage(model, prompt_loader)
    items = retrieval_stage.execute(state, task, chat_history)
    print(f"   Retrieved {len(items)} items:")
    for item in items:
        print(f"     - {item['title']} (Score: {item['score']})")
    print()

    # Stage 3: Recommendation with Retrieved Items
    print("4. Recommendation Stage (using retrieved items):")
    rec_stage = RecommendationStage(model, prompt_loader)

    with patch.object(rec_stage.chain_factory, "create_chain") as mock_create:
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Based on your preferences for action movies with great special effects under $20, I recommend Mock Movies Item 1 (Score: 0.9) - it has excellent action sequences and visual effects."
        mock_create.return_value = mock_chain

        recommendation = rec_stage.execute(state, task, chat_history)
        print(f"   Recommendation: {recommendation}")
    print()

    print("5. Final State:")
    print(f"   Preferences: {state.preferences}")
    print(f"   Retrieved Items: {len(state.retrieved_items)}")
    print(f"   Chat Log Messages: {len(state.chat_log)}")
    print()

    print("=== Demo Complete ===")


def test_response_stage_embeds_image(
    prompt_loader, conversation_state, sample_task, sample_chat_history
):
    """Test that ResponseStage embeds image above textual recommendation when provided."""
    from crs.agents.stages import ResponseStage

    model = Mock()
    stage = ResponseStage(model, prompt_loader)

    # Create a fake recommendation dict with a stream string and image_url
    recommendation = {
        "stream": "I recommend Movie X for its action.",
        "image_url": "http://example.com/image.jpg",
    }

    result = stage.execute(
        conversation_state,
        sample_task,
        sample_chat_history,
        decision="recommend",
        recommendation=recommendation,
    )

    # If the result is a string (non-stream), it should contain markdown image above the text
    assert isinstance(result, str)
    assert result.startswith(
        "![recommended item](http://example.com/image.jpg)"
    )
    assert "I recommend Movie X" in result


if __name__ == "__main__":
    # If run directly, show the demo
    demo_expanded_stages()
