# Conversational Recommender System (CRS)

A modular conversational recommender system built with LangChain and Streamlit.

## Architecture

```
crs/
├── main.py                 # Streamlit application entry point
├── config_loader.py        # Configuration management
├── agents/                 # LLM-based conversation agents
│   ├── orchestrator.py     # Main conversation flow orchestrator
│   ├── stages.py           # Individual conversation stages
│   ├── chains.py           # LangChain integration
│   ├── connectors.py       # LLM provider connectors (OpenAI, Google, Ollama)
│   ├── prompt_loader.py    # External prompt file management
│   ├── chain_factory.py    # Chain creation utilities
│   └── state_manager.py    # Conversation state management
├── components/             # Streamlit UI components
│   ├── pages.py            # Page routing and rendering
│   ├── chat_interface.py   # Chat UI with streaming
│   ├── questionaire.py     # Pre/post questionnaires
│   ├── task.py             # Task display and item tracking
│   └── introduction.py     # Study introduction page
└── retrieval/              # Item retrieval system
    ├── retrieval.py        # Semantic search using sentence transformers
    └── retrieval_utils.py  # Retrieval utilities and caching
```

## Conversation Flow

The CRS uses a multi-stage pipeline for each conversation turn:

1. **PreferenceSummarizationStage** - Extracts and tracks user preferences from conversation
2. **ItemRetrievalStage** - Retrieves matching items using semantic search (if new preferences)
3. **DecisionStage** - Determines next action (recommend, elicit, answer, redirect, confirm)
4. **ItemSelectionStage** - Selects best item from candidates (if recommending)
5. **RecommendationAnalyzerStage** - Analyzes selected item for explanation
6. **RecommendationStage / ResponseStage** - Generates final response

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
export OPENAI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### 3. Create config file

Create `config/config.toml` with your settings:

```toml
[models]
default_model = "gpt-4.1-mini"

[ui]
page_title = "Recommendation Game"

[retrieval]
# Retrieval settings
```

### 4. Prepare item data

Place item embeddings and metadata in `data/items/`:
- `{domain}_meta.jsonl` - Item metadata
- `{domain}_embedded.jsonl` - Pre-computed embeddings

### 5. Run the application

```bash
streamlit run crs/main.py
```

## Module Details

### Agents (`crs/agents/`)

The agents module handles the LLM-based conversation logic:

- **orchestrator.py** - `ConversationOrchestrator` manages the multi-stage pipeline
- **stages.py** - Individual stages (`PreferenceSummarizationStage`, `DecisionStage`, `RecommendationStage`, etc.)
- **connectors.py** - Model connectors for OpenAI, Google, and Ollama
- **state_manager.py** - `ConversationState` and `StreamlitStateManager` for session management
- **prompt_loader.py** - Loads prompts from `data/prompts/`

### Components (`crs/components/`)

Streamlit UI components:

- **pages.py** - Routes between study pages (screen, pre, chat, post, end)
- **chat_interface.py** - Chat UI with streaming responses and timer
- **questionaire.py** - Pre/post task questionnaires with domain assignment
- **task.py** - Task display and recommended items tracker

### Retrieval (`crs/retrieval/`)

Semantic search for item retrieval:

- **retrieval.py** - `ItemRetriever` using sentence transformers (Qwen3-Embedding-0.6B)
- **retrieval_utils.py** - Caching and error handling utilities
