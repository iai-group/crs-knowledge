# Expanded Conversation Stages

This document describes the expanded conversation stages system that separates preference summarization and item retrieval into distinct stages.

## New Architecture

The conversation flow has been expanded from 4 stages to 5 stages:

### 1. PreferenceSummarizationStage
- **Purpose**: Extract and summarize user preferences from conversation history
- **Input**: Chat history, current preferences
- **Output**: Updated user preferences string
- **Prompt**: `preference_summarization_prompt.txt`

### 2. ItemRetrievalStage  
- **Purpose**: Retrieve relevant items based on user preferences
- **Input**: User preferences, domain
- **Output**: List of retrieved items with metadata and scores
- **Implementation**: Uses `crs.retrieval.retrieval.ItemRetriever`
- **Fallback**: Mock items if retrieval fails

### 3. HistorySummarizationStage (Legacy)
- **Purpose**: Maintain backward compatibility
- **Status**: Kept for compatibility but not used in new flow

### 4. DecisionStage
- **Purpose**: Make decisions about next actions
- **No changes from original implementation**

### 5. RecommendationStage (Updated)
- **Purpose**: Generate recommendations using retrieved items
- **Input**: User preferences, retrieved items, chat history  
- **Output**: Natural language recommendations
- **Change**: Now uses actual retrieved items instead of generating from scratch

### 6. ResponseStage (Updated)
- **Purpose**: Generate final response to user
- **Input**: Preferences, retrieved items, decision, recommendation
- **Output**: Streaming response

## State Management

The `ConversationState` has been enhanced with:

```python
@dataclass
class ConversationState:
    preferences: str = "No preferences specified."
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    # ... existing fields
    
    def update_preferences(self, new_preferences: str)
    def update_retrieved_items(self, items: List[Dict[str, Any]])
```

## Configuration Integration

- Retrieval top_k parameter is configurable via `config.toml`
- Uses `config_loader.get('retrieval.top_k', 5)` with fallback

## Retrieval Integration

- **Primary**: Uses `crs.retrieval.retrieval.ItemRetriever`
- **Query**: Uses user preferences as search query
- **Fallback**: Mock items if retrieval system fails
- **Output Format**: 
  ```python
  {
      "id": "item_id",
      "title": "Item Title", 
      "metadata": {...},
      "score": 0.95
  }
  ```

## Error Handling

- Graceful fallback when retrieval system fails
- Proper error logging with Warning messages
- Mock data ensures system continues to function

## Testing

- Comprehensive test suite in `tests/test_stages_expanded.py`
- Demo function showing complete workflow
- Integration tests with mocked LLM calls

## Usage Example

```python
from crs.agents.orchestrator import create_orchestrator

orchestrator = create_orchestrator("gpt-4.1-nano")
response = orchestrator.process_conversation(task, chat_history)
```

## Benefits

1. **Separation of Concerns**: Preferences and retrieval are separate stages
2. **Actual Retrieval**: Uses real item retrieval instead of LLM generation
3. **Configurable**: Retrieval parameters configurable via config
4. **Robust**: Fallback system ensures reliability
5. **Testable**: Each stage can be tested independently
6. **Backward Compatible**: Legacy stages maintained

## Future Improvements

1. Fix dtype issue in retrieval system
2. Add domain-specific retrieval configurations
3. Implement sophisticated query generation from preferences
4. Add relevance filtering based on similarity thresholds
5. Support multiple retrieval backends
