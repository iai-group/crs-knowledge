"""Individual conversation stages."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from ..config_loader import get_config_loader
from ..retrieval.retrieval import ItemRetriever
from .chain_factory import ChainFactory
from .prompt_loader import PromptLoader
from .state_manager import ConversationState


class ConversationStage(ABC):
    """Abstract base class for conversation stages."""

    def __init__(self, model, prompt_loader: PromptLoader):
        self.model = model
        self.prompt_loader = prompt_loader
        self.chain_factory = ChainFactory(model)

    @abstractmethod
    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> Any:
        """Execute this stage of the conversation."""
        pass


class PreferenceSummarizationStage(ConversationStage):
    """Stage 1: Summarize user preferences from conversation history."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> str:
        """Summarize user preferences and update state."""
        prompt_template = self.prompt_loader.load_prompt(
            "preference_summarization_prompt_bullet.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        input_data = {
            "domain": task.get("domain"),
            "current_preferences": state.preferences,
            "latest_chat_history": latest_chat_history,
        }

        updated_preferences = chain.invoke(input_data).strip()
        state.update_preferences(updated_preferences)

        print(f"\nUpdated preferences:\n{updated_preferences}")
        print()

        number_of_preferences = len(
            [
                line
                for line in updated_preferences.split("\n")
                if line.strip() != ""
            ]
        )
        state.add_system_message(
            f"Updated preferences to {number_of_preferences} points."
        )

        return number_of_preferences


class ItemRetrievalStage(ConversationStage):
    """Stage 2: Retrieve items based on user preferences."""

    def __init__(
        self,
        model,
        prompt_loader: PromptLoader,
    ):
        super().__init__(model, prompt_loader)
        self.retriever: ItemRetriever = None

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> List[Dict[str, Any]]:
        """Retrieve items based on user preferences."""
        domain = task.get("domain")
        preferences = state.preferences

        # Get the number of items to retrieve from config or use default
        config = get_config_loader()
        top_k = config.get("retrieval.top_k", 10)
        min_turns_for_target = config.get("retrieval.min_turns_for_target", 3)

        # Initialize the retriever directly
        if self.retriever is None:
            self.retriever = self._get_retriever_for_domain(domain)

        # Use preferences as the query for retrieval
        if not preferences:
            return []

        # Retrieve items using the retriever
        results = self.retriever.retrieve(preferences, top_k=top_k)
        for res in results:
            print(
                f"Retrieved item: {res[0].get('title', 'Unknown Title')} (Score: {res[1]:.4f})"
            )

        # Convert retrieval results to our expected format
        retrieved_items = []
        target_asin = task.get("target", {}).get("asin")
        target_filtered = False

        for metadata, score in results:
            # metadata is now a structured dict instead of a string
            item_asin = metadata.get(
                "parent_asin", f"item_{len(retrieved_items)}"
            )

            # Filter out target item if we're under the minimum turn threshold
            if (
                item_asin == target_asin
                and state.turn_count < min_turns_for_target
            ):
                target_filtered = True
                continue

            item = {
                "id": item_asin,
                "title": metadata.get("title", "Unknown Item"),
                "content": metadata.get("features", ""),
                "images": metadata.get("images", []),
            }
            retrieved_items.append(item)

        if retrieved_items and retrieved_items[0].get("id") == target_asin:
            state.update_success_detection(
                {
                    "target_found": True,
                    "turn_found": state.turn_count,
                }
            )
            state.add_system_message(
                f"Target found in retrieval results (turn {state.turn_count})."
            )

        # Log target filtering if it occurred
        if target_filtered:
            state.add_system_message(
                f"Target item filtered out (turn {state.turn_count} < {min_turns_for_target})"
            )

        unrecommended = state.get_unrecommended_items(retrieved_items)

        filtered_out = len(retrieved_items) - len(unrecommended)
        if filtered_out > 0:
            state.add_system_message(
                f"Filtered out {filtered_out} previously recommended item(s) from retrieval results."
            )

        retrieved_items = unrecommended

        # Update state with retrieved items
        state.update_retrieved_items(retrieved_items)

        return retrieved_items

    def _get_retriever_for_domain(self, domain: str) -> ItemRetriever:
        """Get the appropriate retriever for the given domain."""
        # TODO: Make this configurable based on domain
        # For now, default to bikes dataset
        return ItemRetriever()


class DecisionStage(ConversationStage):
    """Stage 2: Make decision about next action."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> str:
        """Decide on the next action to take."""
        prompt_template = self.prompt_loader.load_prompt(
            "decision_stage_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        input_data = {
            "domain": task.get("domain"),
            "latest_chat_history": latest_chat_history,
            "number_of_preferences": len(
                [
                    line
                    for line in state.preferences.split("\n")
                    if line.strip() != ""
                ]
            ),
            "number_of_recommendations": len(state.retrieved_items),
        }

        raw_decision_output = chain.invoke(input_data)
        state.add_system_message(raw_decision_output)

        print()
        print(raw_decision_output)
        print()

        # Return the first non-empty line of the model output (strip whitespace)
        first_line = ""
        for ln in raw_decision_output.splitlines():
            if ln and ln.strip():
                first_line = ln.strip()
                break

        return first_line


class RecommendationAnalyzerStage(ConversationStage):
    """Stage 3: Analyze recommendations."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> str:
        """Generate explanations for the given recommendation."""
        prompt_template = self.prompt_loader.load_prompt(
            "explanation_stage_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)
        recommended_item_metadata = state.retrieved_items[0].get("content", {})
        target_item_metadata = task.get("target", {}).get("content", {})

        input_data = {
            "domain": task.get("domain"),
            "recommended_item_metadata": recommended_item_metadata,
            "target_item_metadata": target_item_metadata,
        }

        aspects = chain.invoke(input_data).strip()

        state.add_system_message(f"Comparison aspects found: \n{aspects}")

        return aspects


class RecommendationStage(ConversationStage):
    """Stage 4: Generate recommendations using retrieved items."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
        explanation_aspects: str = "",
    ) -> str:
        """Generate target-aware recommendations based on retrieved items."""

        # Check if target is found for guidance strategy
        success_info = state.get_success_info()
        target_found = success_info.get("target_found", False)

        if target_found:
            # Target is in results - use standard recommendation
            prompt_template = self.prompt_loader.load_prompt(
                "target_found_prompt.txt"
            )
        else:
            # Target not found - use target-aware guidance
            prompt_template = self.prompt_loader.load_prompt(
                "recommendation_with_target_guidance_prompt.txt"
            )

        state.add_recommended_item(state.retrieved_items[0].get("id"))

        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        # Prepare input data with target awareness
        input_data = {
            "preferences": state.preferences,
            "item_to_recommend": state.retrieved_items[0],
            "latest_chat_history": latest_chat_history,
            "domain": task.get("domain"),
            "explanation_aspects": explanation_aspects,
        }

        return {
            "stream": chain.stream(input_data),
            "image_url": self._get_item_image_url(state.retrieved_items[0]),
        }

    def _get_item_image_url(self, item: Dict[str, Any]) -> str:
        """Extract the best available image URL from item metadata."""
        images = item.get("images", [])

        if not images:
            return ""

        # Get the first image (usually the main product image)
        main_image = images[0]

        # Prefer large, then hi_res, then thumb
        if "large" in main_image and main_image["large"]:
            return main_image["large"]
        elif "hi_res" in main_image and main_image["hi_res"]:
            return main_image["hi_res"]
        elif "thumb" in main_image and main_image["thumb"]:
            return main_image["thumb"]

        return ""


class ResponseStage(ConversationStage):
    """Stage 5: Generate final user response."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
        decision: str = "",
        recommendation: str = "",
    ) -> Any:
        """Generate the final response to the user."""

        # Map actions to specific response templates
        template_map = {
            "answer": "response_stage_answer.txt",
            "recommend": "response_stage_recommend.txt",
            "elicit": "response_stage_elicit.txt",
            "redirect": "response_stage_redirect.txt",
        }

        # Use specific template or fall back to clarify_feature
        template_name = template_map.get(
            decision.lower(), "response_stage_redirect.txt"
        )

        print(
            f"Loading template for decision {decision.lower()}:", template_name
        )
        prompt_template = self.prompt_loader.load_prompt(template_name)
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        # Base input data for all response types
        input_data = {
            "domain": task.get("domain"),
            "latest_chat_history": latest_chat_history,
            "chat_history": chat_history,
            "preferences": state.preferences,
            "recommendation": recommendation,
        }

        return {"stream": chain.stream(input_data)}
