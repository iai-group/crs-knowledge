# RecommendedItemCheckStage

## Overview

The `RecommendedItemCheckStage` is a new conversation stage that checks if any previously recommended items satisfy newly added user preferences before retrieving new items from the database. This optimization reduces unnecessary database queries and can help maintain conversation coherence by re-recommending items that already meet all requirements.

## Purpose

When users add new preferences during a conversation, the system typically retrieves new items from the database. However, items that were already recommended might satisfy these new preferences. This stage uses an LLM call to intelligently check if any previously recommended items match the complete set of preferences (including the new ones).

## Flow Integration

The stage is inserted between `PreferenceSummarizationStage` and `ItemRetrievalStage`:

1. **PreferenceSummarizationStage** - Extracts and summarizes user preferences
2. **RecommendedItemCheckStage** *(new)* - If new preferences detected, checks if previously recommended items satisfy them
3. **ItemRetrievalStage** - Retrieves new items from database (still runs, but matching item is prioritized)
4. **DecisionStage** - Decides next action
5. **RecommendationAnalyzerStage** - Analyzes selected item
6. **RecommendationStage** - Generates recommendation

## Implementation Details

### Stage Behavior

- **Input**: Current conversation state, task info, chat history
- **Output**: The matching previously recommended item (dict) or None
- **Trigger**: Only runs when `PreferenceStatus.NEW` is returned from preference summarization

### Prompt Template

Located at: `data/prompts/check_recommended_items_prompt.txt`

The prompt instructs the LLM to:
1. Analyze previously recommended items against ALL user preferences
2. Pay special attention to newly mentioned preferences
3. Return a structured response: `MATCH` or `NO_MATCH`
4. If match found, include the item ID and explanation

### Response Format

```
MATCH
B0123456789
The previously recommended laptop meets all requirements including the newly added 16GB RAM specification.
```

or

```
NO_MATCH

None of the previously recommended items have the required dedicated GPU that was just mentioned.
```

## Benefits

1. **Efficiency**: Reduces unnecessary database queries
2. **Coherence**: Maintains conversation flow by reconsidering items that fit all requirements
3. **User Experience**: Can quickly re-surface appropriate items when preferences are refined
4. **Cost Reduction**: Fewer database retrievals mean lower operational costs

## Example Scenario

**Turn 1:**
- User: "I need a laptop"
- System recommends: Laptop A (general purpose)

**Turn 2:**
- User: "It should have 16GB RAM and SSD"
- Old behavior: Retrieve new items from database
- New behavior: Check if Laptop A has 16GB RAM and SSD
  - If yes: Re-surface Laptop A (added to front of retrieved items)
  - If no: Proceed with normal database retrieval

## Testing

Unit tests are located in `tests/test_recommended_item_check.py` and cover:
- No previous recommendations scenario
- Match found scenario
- No match found scenario

Run tests with:
```bash
python -m pytest tests/test_recommended_item_check.py -v
```

## Configuration

No additional configuration is required. The stage uses the same model and prompt loader as other conversation stages.

## Future Enhancements

Potential improvements:
1. Add caching to avoid re-checking the same items multiple times
2. Support partial matches with similarity scores
3. Track how often re-recommendations occur for analytics
4. Add configuration to enable/disable this optimization
