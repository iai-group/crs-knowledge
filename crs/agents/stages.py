"""Individual conversation stages."""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from ..retrieval.retrieval import ItemRetriever
from .chain_factory import ChainFactory
from .prompt_loader import PromptLoader
from .state_manager import ConversationState

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Enumeration of possible conversation decisions."""

    RECOMMEND = "recommend"
    ELICIT = "elicit"
    ANSWER = "answer"
    REDIRECT = "redirect"
    ASK_PRIORITIZE = "ask_prioritize"
    CONFIRM = "confirm"
    ANSWER_ABOUT_RECOMMENDATION = "answer_about_recommendation"

    @classmethod
    def from_string(
        cls, value: str, default: "DecisionType" = None
    ) -> "DecisionType":
        """Convert a string to DecisionType, handling case-insensitivity.

        Args:
            value: String representation of the decision
            default: Default DecisionType to return if value is invalid (defaults to ELICIT)

        Returns:
            DecisionType enum value, or default if value cannot be mapped
        """
        if default is None:
            default = cls.ELICIT

        if not value:
            logger.debug(
                f"Warning: Empty decision value. Defaulting to {default.value}."
            )
            return default

        normalized = value.strip().lower()

        for decision in cls:
            if decision.value == normalized:
                return decision

        for decision in cls:
            if decision.value in normalized or normalized in decision.value:
                logger.debug(
                    f"Warning: Partial match for '{value}', using {decision.value}."
                )
                return decision

        valid_options = ", ".join([d.value for d in cls])
        logger.debug(
            f"Warning: Invalid decision '{value}'. "
            f"Valid options: {valid_options}. "
            f"Defaulting to {default.value} to continue conversation."
        )
        return default


class PreferenceStatus(Enum):
    """Enumeration of preference summarization outcomes."""

    NEW = "new"
    OLD = "old"
    TOO_MANY = "too_many"


class ConversationStage(ABC):
    """Abstract base class for conversation stages."""

    def __init__(self, model, prompt_loader: PromptLoader):
        self.model = model
        self.prompt_loader = prompt_loader
        self.chain_factory = ChainFactory(model)
        self.turn_state: Optional[Dict[str, Any]] = None

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

    def _calculate_preference_similarity(
        self, old_preferences: str, new_preferences: str
    ) -> float:
        """Calculate Jaccard similarity between old and new preferences.

        Args:
            old_preferences: Previous preference text
            new_preferences: Updated preference text

        Returns:
            Jaccard similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        if not old_preferences and not new_preferences:
            return 1.0
        if not old_preferences or not new_preferences:
            return 0.0

        def tokenize(text: str) -> set:
            words = text.lower().split()
            return set(
                word.strip(".,!?;:()[]{}\"'")
                for word in words
                if word.strip(".,!?;:()[]{}\"'")
            )

        old_tokens = tokenize(old_preferences)
        new_tokens = tokenize(new_preferences)

        intersection = old_tokens & new_tokens
        union = old_tokens | new_tokens

        if not union:
            return 1.0

        similarity = len(intersection) / len(union)

        logger.debug(
            f"Preference similarity: {similarity:.3f} "
            f"(intersection: {len(intersection)}, union: {len(union)})"
        )

        return similarity

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> PreferenceStatus:
        """Summarize user preferences and update state.

        Returns:
            PreferenceStatus enum indicating the outcome
        """
        prompt_template = self.prompt_loader.load_prompt(
            "preference_summarization_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)
        old_preferences = state.get_preferences()
        num_old_pref = state.get_len_preferences()

        input_data = {
            "domain": task.get("domain"),
            "current_preferences": old_preferences,
            "latest_chat_history": latest_chat_history,
        }

        updated_preferences = chain.invoke(input_data).strip()
        number_of_new_preferences = len(
            [
                line
                for line in updated_preferences.split("\n")
                if line.strip() != ""
            ]
        )

        if number_of_new_preferences > num_old_pref + 4:
            return PreferenceStatus.TOO_MANY

        state.update_preferences(updated_preferences, number_of_new_preferences)

        logger.debug(f"\nUpdated preferences:\n{updated_preferences}\n")

        state.add_system_message(
            f"Updated to {number_of_new_preferences} preferences: {updated_preferences}"
        )

        has_preferences = number_of_new_preferences >= 1
        count_changed = number_of_new_preferences != num_old_pref

        if has_preferences and count_changed:
            logger.debug(
                f"Detected NEW preferences (count changed: {num_old_pref} → {number_of_new_preferences})"
            )
            return PreferenceStatus.NEW

        if has_preferences and number_of_new_preferences == num_old_pref:
            similarity = self._calculate_preference_similarity(
                old_preferences, updated_preferences
            )

            SIMILARITY_THRESHOLD = 0.95

            if similarity < SIMILARITY_THRESHOLD:
                logger.debug(
                    f"Detected NEW preferences (content changed with same count, similarity={similarity:.3f})"
                )
                return PreferenceStatus.NEW

        return PreferenceStatus.OLD


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

        if self.retriever is None:
            self.retriever = self._get_retriever_for_domain(domain)

        if not preferences:
            return []

        results = self.retriever.retrieve(preferences, top_k=100)
        logger.debug(f"\nRetrieved {len(results)} items from database")

        all_retrieved_items = []

        for metadata, score in results:
            item_asin = metadata.get(
                "parent_asin", f"item_{len(all_retrieved_items)}"
            )

            item = {
                "id": item_asin,
                "title": metadata.get("title", "Unknown Item"),
                "content": metadata.get("content", ""),
                "images": metadata.get("images", []),
                "score": score,
            }
            all_retrieved_items.append(item)

        unfiltered_items = state.get_unfiltered_items(all_retrieved_items)
        unrecommended_items = state.get_unrecommended_items(unfiltered_items)

        filtered_before_selection = len(all_retrieved_items) - len(
            unrecommended_items
        )
        if filtered_before_selection > 0:
            logger.debug(
                f"Filtered out {filtered_before_selection} items before selecting top 10"
            )

        retrieved_items = unrecommended_items

        target_asin = task.get("target", {}).get("id")
        if target_asin:
            target_found = any(
                item.get("id") == target_asin for item in retrieved_items
            )
            if target_found:
                logger.debug(
                    f"\n✓ Target item (asin: {target_asin}) found in top 10"
                )
            else:
                logger.debug(
                    f"\n✗ Target item (asin: {target_asin}) NOT found in top 10"
                )

            if len(state.recommended_items) <= 1:
                retrieved_items = [
                    item
                    for item in retrieved_items
                    if item.get("id") != target_asin
                ]
                logger.debug(
                    f"Filtered out target item {target_asin} (no recommendations yet)"
                )

        for item in retrieved_items[:5]:
            logger.debug(
                f"Retrieved item: {item.get('id', 'Unknown ID')} {item.get('title', 'Unknown Title')} (Score: {item.get('score', 0):.4f})"
            )

        state.update_retrieved_items(retrieved_items)

        return retrieved_items

    def _get_retriever_for_domain(self, domain: str) -> ItemRetriever:
        """Get the appropriate retriever for the given domain."""
        return ItemRetriever(domain=domain)


class ItemSelectionStage(ConversationStage):
    """Stage: Select the best item from retrieved items using LLM."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
        target: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the best item to recommend from retrieved items.

        Returns:
            The selected item dict, or None if no items available or selection fails
        """
        if not state.retrieved_items:
            logger.debug("Warning: No retrieved items available for selection.")
            return None

        items_to_select_from = state.retrieved_items + state.recommended_items

        if (
            target
            and len(state.recommended_items) >= 4
            and all(
                target.get("id") not in item.get("id")
                for item in items_to_select_from
            )
        ):
            logger.debug(
                "Target item not found in any items ready for recommendation."
            )
            return target

        if len(items_to_select_from) == 1:
            logger.debug("Only one item available, selecting it by default.")
            return items_to_select_from[0]

        prompt_template = self.prompt_loader.load_prompt(
            "item_selection_stage_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        items = deepcopy(state.retrieved_items)
        i = 0
        while True:
            i += 1
            if not items:
                break

            input_data = {
                "domain": task.get("domain"),
                "preferences": state.preferences,
                "latest_chat_history": latest_chat_history,
                "retrieved_items": self.format_items(items[:10]),
                "recommended_items": (
                    self.format_items(state.recommended_items)
                    if len(state.recommended_items) >= 4
                    else "None"
                ),
            }

            selected_id = chain.invoke(input_data).strip()
            logger.debug(
                f"\nItteration {i}. LLM selected item ID: {selected_id}"
            )

            selected_item = None

            for item in items_to_select_from:
                if item.get("id") == selected_id:
                    selected_item = item
                    logger.debug(f"✓ Selected item: {item.get('title')}")
                    break

            if selected_item is not None:
                break
            else:
                items = items[10:]

        if selected_item is None:
            logger.debug(
                f"Warning: Could not find item with ID '{selected_id}'. "
                f"Falling back to first item."
            )
            selected_item = state.retrieved_items[0]

        state.add_system_message(
            f"Selected item for recommendation: {selected_item.get('title')} "
            f"(ID: {selected_item.get('id')})"
        )

        return selected_item

    def format_items(self, items):
        formatted = []
        for idx, item in enumerate(items, 1):
            item_info = f"Item {idx}:\n"
            item_info += f"  ID: {item.get('id', 'N/A')}\n"
            item_info += f"  Title: {item.get('title', 'N/A')}\n"
            item_info += f"  Features: {item.get('content', 'N/A')}\n"
            formatted.append(item_info)
        return "\n".join(formatted)


class DecisionStage(ConversationStage):
    """Stage 2: Make decision about next action."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> DecisionType:
        """Decide on the next action to take.

        Returns:
            DecisionType enum indicating the next action
        """
        prompt_template = self.prompt_loader.load_prompt(
            "decision_stage_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        input_data = {
            "domain": task.get("domain"),
            "latest_chat_history": latest_chat_history,
            "number_of_preferences": state.get_len_preferences(),
            "number_of_recommendations": len(state.retrieved_items),
            "has_recommendation": ("yes" if state.recommended_items else "no"),
        }

        raw_decision_output = chain.invoke(input_data)
        state.add_system_message(raw_decision_output)

        logger.debug(raw_decision_output)

        first_line = ""
        for ln in raw_decision_output.splitlines():
            if ln and ln.strip():
                first_line = ln.strip()
                break

        decision = DecisionType.from_string(first_line)
        logger.debug(f"✓ Decision: {decision.value}")
        return decision


class RecommendationAnalyzerStage(ConversationStage):
    """Stage 3: Analyze recommendations."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
        selected_item: Dict[str, Any] = None,
    ) -> str:
        """Generate explanations for the given recommendation.

        Args:
            state: Conversation state
            task: Task information
            chat_history: Chat history
            selected_item: The specific item to analyze (if None, uses first retrieved item)

        Returns:
            String of comparison aspects, or empty string if no items available
        """
        if selected_item is None:
            if not state.retrieved_items:
                logger.debug(
                    "Warning: No retrieved items available for analysis."
                )
                return ""
            selected_item = state.retrieved_items[0]

        prompt_template = self.prompt_loader.load_prompt(
            "explanation_stage_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)
        recommended_item_metadata = selected_item.get("content", {})
        target_item_metadata = task.get("target", {}).get("content", {})

        input_data = {
            "domain": task.get("domain"),
            "recommended_item_metadata": recommended_item_metadata,
            "target_item_metadata": target_item_metadata,
        }

        aspects = chain.invoke(input_data).strip()

        state.add_system_message(f"Comparison aspects found: \n{aspects}")

        return aspects


class QuestionAnswerStage(ConversationStage):
    """Stage for answering questions about recommended items with nuanced comparison to target."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
    ) -> Any:
        """Answer a question about the recommended item with nuanced comparison to target.

        This stage is called when users ask questions about a previously recommended item
        (e.g., "Is this bike good for going uphill?"). Instead of absolute yes/no answers,
        it provides nuanced responses by comparing the recommended item with the target item.

        Args:
            state: Conversation state (should have last_recommended_item set)
            task: Task information including domain and target
            chat_history: Chat history

        Returns:
            Dict with stream of response, or falls back to ResponseStage if no recommended item
        """
        latest_chat_history = state.get_latest_chat_history(chat_history)

        target_item_metadata = task.get("target", {}).get("content", {})

        all_recommended_items_text = ""
        for idx, item in enumerate(state.recommended_items, 1):
            item_id = item.get("id", f"item_{idx}")
            title = item.get("title", "Unknown Item")
            content = item.get("content", "")
            all_recommended_items_text += (
                f"\n=== Recommended Item {idx} ===\n"
                f"ID: {item_id}\n"
                f"Title: {title}\n"
                f"Details: {content}\n"
            )

        explanation_prompt = self.prompt_loader.load_prompt(
            "elicit_relevant_aspects_prompt.txt"
        )
        explanation_chain = self.chain_factory.create_chain(explanation_prompt)

        explanation_input = {
            "domain": task.get("domain"),
            "latest_chat_history": latest_chat_history,
            "recommended_items": all_recommended_items_text,
            "target_item_metadata": target_item_metadata,
        }

        comparison_aspects = explanation_chain.invoke(explanation_input).strip()

        state.add_system_message(
            f"Question-relevant comparison aspects: \n{comparison_aspects}"
        )

        prompt_template = self.prompt_loader.load_prompt(
            "answer_about_recommendation_prompt.txt"
        )
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        recommended_items_summary = ""
        for idx, item in enumerate(state.recommended_items, 1):
            title = item.get("title", "Unknown Item")
            recommended_items_summary += f"{idx}. {title}\n"

        input_data = {
            "domain": task.get("domain"),
            "latest_chat_history": latest_chat_history,
            "chat_history": chat_history,
            "preferences": state.preferences,
            "recommended_items": recommended_items_summary.strip(),
            "recommended_items_details": all_recommended_items_text,
            "target_item_metadata": target_item_metadata,
            "comparison_aspects": comparison_aspects,
        }

        return {"stream": chain.stream(input_data)}


class RecommendationStage(ConversationStage):
    """Stage 4: Generate recommendations using retrieved items."""

    def execute(
        self,
        state: ConversationState,
        task: Dict[str, Any],
        chat_history: List[BaseMessage],
        explanation_aspects: str = "",
        selected_item: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Generate target-aware recommendations based on selected item.

        Args:
            state: Conversation state
            task: Task information
            chat_history: Chat history
            explanation_aspects: Aspects to explain in the recommendation
            selected_item: The specific item to recommend (if None, uses first retrieved item)

        Returns:
            Dict with stream and image_url, or None if no items available
        """
        if selected_item is None:
            if not state.retrieved_items:
                logger.debug(
                    "Warning: No retrieved items available for recommendation."
                )
                state.add_system_message(
                    "Cannot recommend: no items retrieved yet."
                )
                return None
            selected_item = state.retrieved_items[0]

        prompt_template = self.prompt_loader.load_prompt(
            "recommendation_with_target_guidance_prompt.txt"
        )

        selected_item_id = selected_item.get("id")
        has_been_recommended = any(
            item.get("id") == selected_item_id
            for item in state.recommended_items
        )

        state.add_recommended_item(selected_item)

        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_last_user_message(chat_history)

        input_data = {
            "preferences": state.preferences,
            "item_to_recommend": selected_item,
            "latest_chat_history": latest_chat_history,
            "domain": task.get("domain"),
            "explanation_aspects": explanation_aspects,
            "has_been_recommended": "yes" if has_been_recommended else "no",
        }

        return {
            "stream": chain.stream(input_data),
            "image_url": self._get_item_image_url(selected_item),
        }

    def _get_item_image_url(self, item: Dict[str, Any]) -> str:
        """Extract the best available image URL from item metadata."""
        images = item.get("images", [])

        if not images:
            return ""

        main_image = images[0]

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
        decision: DecisionType = None,
        recommendation: str = "",
    ) -> Any:
        """Generate the final response to the user.

        Args:
            state: Conversation state
            task: Task information
            chat_history: Chat history
            decision: DecisionType enum (optional, defaults to ELICIT)
            recommendation: Recommendation text (optional)

        Returns:
            Dict with stream of response
        """
        if decision is None:
            decision = DecisionType.ELICIT

        template_map = {
            DecisionType.ANSWER: "response_stage_answer.txt",
            DecisionType.RECOMMEND: "response_stage_recommend.txt",
            DecisionType.ELICIT: "response_stage_elicit.txt",
            DecisionType.REDIRECT: "response_stage_redirect.txt",
            DecisionType.ASK_PRIORITIZE: "response_stage_ask_prioritize.txt",
            DecisionType.CONFIRM: "response_stage_confirm.txt",
        }

        template_name = template_map[decision]

        logger.debug(
            f"Loading template for decision {decision.value}: {template_name}"
        )
        prompt_template = self.prompt_loader.load_prompt(template_name)
        chain = self.chain_factory.create_chain(prompt_template)

        latest_chat_history = state.get_latest_chat_history(chat_history)

        input_data = {
            "domain": task.get("domain"),
            "latest_chat_history": latest_chat_history,
            "chat_history": chat_history,
            "preferences": state.preferences,
            "recommended_items": state.recommended_items,
        }

        return {"stream": chain.stream(input_data)}
