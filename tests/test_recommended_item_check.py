"""Test the RecommendedItemCheckStage functionality."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from crs.agents.stages import RecommendedItemCheckStage
from crs.agents.state_manager import ConversationState
from crs.agents.prompt_loader import PromptLoader


def test_recommended_item_check_stage_no_previous_items():
    """Test that stage returns None when no previous recommendations exist."""
    # Setup
    model = Mock()
    prompt_loader = Mock(spec=PromptLoader)
    stage = RecommendedItemCheckStage(model, prompt_loader)
    
    state = ConversationState()
    task = {"domain": "laptop"}
    chat_history = [
        HumanMessage(content="I need a laptop with 16GB RAM"),
        AIMessage(content="What's your budget?"),
        HumanMessage(content="Under $1000"),
    ]
    
    # Execute
    result = stage.execute(state, task, chat_history)
    
    # Assert
    assert result is None


def test_recommended_item_check_stage_match_found():
    """Test that stage returns matching item when found."""
    # Setup
    model = Mock()
    prompt_loader = Mock(spec=PromptLoader)
    prompt_loader.load_prompt = Mock(return_value="test prompt {domain} {preferences} {recommended_items} {latest_chat_history}")
    
    # Mock the chain to return a MATCH response
    mock_chain = Mock()
    mock_chain.invoke = Mock(return_value="MATCH\nB0123456\nThe item matches all preferences including the new RAM requirement.")
    
    stage = RecommendedItemCheckStage(model, prompt_loader)
    stage.chain_factory = Mock()
    stage.chain_factory.create_chain = Mock(return_value=mock_chain)
    
    # Setup state with a recommended item
    state = ConversationState()
    state.preferences = "- 16GB RAM\n- Under $1000"
    state.recommended_items = [
        {
            "id": "B0123456",
            "title": "Test Laptop",
            "content": "A great laptop with 16GB RAM"
        }
    ]
    state.get_latest_chat_history = Mock(return_value=[
        HumanMessage(content="I need 16GB RAM"),
        AIMessage(content="Got it!")
    ])
    
    task = {"domain": "laptop"}
    chat_history = [
        HumanMessage(content="I need a laptop"),
        AIMessage(content="What specs?"),
        HumanMessage(content="16GB RAM"),
    ]
    
    # Execute
    result = stage.execute(state, task, chat_history)
    
    # Assert
    assert result is not None
    assert result["id"] == "B0123456"
    assert result["title"] == "Test Laptop"


def test_recommended_item_check_stage_no_match():
    """Test that stage returns None when no match is found."""
    # Setup
    model = Mock()
    prompt_loader = Mock(spec=PromptLoader)
    prompt_loader.load_prompt = Mock(return_value="test prompt {domain} {preferences} {recommended_items} {latest_chat_history}")
    
    # Mock the chain to return a NO_MATCH response
    mock_chain = Mock()
    mock_chain.invoke = Mock(return_value="NO_MATCH\n\nNone of the previously recommended items have the required GPU.")
    
    stage = RecommendedItemCheckStage(model, prompt_loader)
    stage.chain_factory = Mock()
    stage.chain_factory.create_chain = Mock(return_value=mock_chain)
    
    # Setup state with a recommended item
    state = ConversationState()
    state.preferences = "- 16GB RAM\n- Dedicated GPU"
    state.recommended_items = [
        {
            "id": "B0123456",
            "title": "Test Laptop",
            "content": "A laptop with 16GB RAM but integrated graphics"
        }
    ]
    state.get_latest_chat_history = Mock(return_value=[
        HumanMessage(content="I need a dedicated GPU"),
        AIMessage(content="Got it!")
    ])
    
    task = {"domain": "laptop"}
    chat_history = [
        HumanMessage(content="I need a laptop"),
        AIMessage(content="What specs?"),
        HumanMessage(content="Must have a dedicated GPU"),
    ]
    
    # Execute
    result = stage.execute(state, task, chat_history)
    
    # Assert
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
