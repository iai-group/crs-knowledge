"""State management for conversations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


@dataclass
class ConversationState:
    """Manages the state of a conversation."""

    preferences: str = field(default_factory=lambda: "")
    num_preferences: int = field(default=0)
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    chat_log: List[BaseMessage] = field(default_factory=list)
    turn_count: int = 0
    recommended_items: List[Dict[str, Any]] = field(default_factory=list)
    recommended_items_changed: bool = False
    filtered_out_items: List[Dict[str, Any]] = field(default_factory=list)

    def update_preferences(
        self, new_preferences: str, num_preferences: int = None
    ):
        """Update the user preferences."""
        self.preferences = new_preferences.strip()
        if num_preferences is not None:
            self.num_preferences = num_preferences
        else:
            self.num_preferences = len(
                [
                    line
                    for line in self.preferences.split("\n")
                    if line.strip() != ""
                ]
            )

    def get_preferences(self) -> str:
        """Get the current user preferences."""
        return self.preferences or "No preferences specified."

    def get_len_preferences(self) -> int:
        """Get the length of current user preferences."""
        return self.num_preferences

    def update_retrieved_items(self, items: List[Dict[str, Any]]):
        """Update the retrieved items list."""
        self.retrieved_items = items
        item_titles = [item.get("title", "Unknown") for item in items]
        self.add_system_message("Retrieved items:\n" + "\n".join(item_titles))

    def add_recommended_item(self, item: Dict[str, Any]):
        """Track an item that has been recommended to avoid repeating."""
        item_id = (
            item.get("id")
            or item.get("parent_asin")
            or item.get("metadata", {}).get("parent_asin")
        )
        if item_id and not self.is_item_already_recommended(item_id):
            self.recommended_items.append(item)
            item_title = (
                item.get("title") or item.get("content", {}).get("title")
                if isinstance(item.get("content"), dict)
                else f"Item {item_id}"
            )
            self.add_system_message(
                f"Added to recommended: {item_id} ({item_title})"
            )
            self.recommended_items_changed = True

    def is_item_already_recommended(self, item_id: str) -> bool:
        """Check if an item has already been recommended."""
        for recommended_item in self.recommended_items:
            rec_id = (
                recommended_item.get("id")
                or recommended_item.get("parent_asin")
                or recommended_item.get("metadata", {}).get("parent_asin")
            )
            if rec_id == item_id:
                return True
        return False

    def get_unrecommended_items(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out items that have already been recommended."""
        unrecommended = []
        for item in items:
            item_id = item.get("id") or item.get("metadata", {}).get(
                "parent_asin"
            )
            if item_id and not self.is_item_already_recommended(item_id):
                unrecommended.append(item)
        return unrecommended

    def add_filtered_out_item(self, item: Dict[str, Any]):
        """Add an item to the filtered out list to exclude from future retrievals."""
        item_id = (
            item.get("id")
            or item.get("parent_asin")
            or item.get("metadata", {}).get("parent_asin")
        )
        if item_id and not self.is_item_filtered_out(item_id):
            self.filtered_out_items.append(item)
            item_title = (
                item.get("title") or item.get("content", {}).get("title")
                if isinstance(item.get("content"), dict)
                else f"Item {item_id}"
            )
            self.add_system_message(
                f"Added to filter out: {item_id} ({item_title})"
            )

    def is_item_filtered_out(self, item_id: str) -> bool:
        """Check if an item has been filtered out."""
        for filtered_item in self.filtered_out_items:
            filt_id = (
                filtered_item.get("id")
                or filtered_item.get("parent_asin")
                or filtered_item.get("metadata", {}).get("parent_asin")
            )
            if filt_id == item_id:
                return True
        return False

    def get_unfiltered_items(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out items that have been marked as filtered out."""
        unfiltered = []
        for item in items:
            item_id = item.get("id") or item.get("metadata", {}).get(
                "parent_asin"
            )
            if item_id and not self.is_item_filtered_out(item_id):
                unfiltered.append(item)
        return unfiltered

    def add_system_message(self, content: str):
        """Add a system message to the chat log."""
        st.session_state.chat_log.append(SystemMessage(content))

    def increment_turn(self):
        """Increment the turn counter."""
        self.turn_count += 1

    def get_latest_chat_history(
        self, chat_history: List[BaseMessage], count: int = 2
    ) -> List[BaseMessage]:
        """Get the latest messages from chat history."""
        messages = []
        for msg in chat_history[-count:]:
            if isinstance(msg, HumanMessage):
                messages.append("User: " + msg.content)
            elif isinstance(msg, AIMessage):
                messages.append("Assistant: " + msg.content)
        return "\n".join(messages)

    def get_last_user_message(self, chat_history: List[BaseMessage]) -> str:
        """Return the most recent human message content or empty string.

        This helper centralizes extraction of the raw user text for stages
        that need the unformatted user question (rather than the
        formatted `get_latest_chat_history` output which includes prefixes).
        """
        for msg in reversed(chat_history):
            if isinstance(msg, HumanMessage):
                return msg.content.strip()
        return ""


class StateManager:
    """Manages conversation state with different backends."""

    def __init__(self, backend: str = "memory"):
        self.backend = backend
        self._states: Dict[str, ConversationState] = {}

    def get_state(self, session_id: str = "default") -> ConversationState:
        """Get or create conversation state for a session."""
        if session_id not in self._states:
            self._states[session_id] = ConversationState()
        return self._states[session_id]

    def update_state(self, session_id: str, state: ConversationState):
        """Update conversation state for a session."""
        self._states[session_id] = state


# Streamlit-specific state manager
class StreamlitStateManager(StateManager):
    """State manager that integrates with Streamlit session state."""

    def __init__(self):
        super().__init__("streamlit")

    def get_state(self, session_id: str = "default") -> ConversationState:
        """Get state from Streamlit session state."""
        import streamlit as st

        if "conversation_state" not in st.session_state:
            st.session_state.conversation_state = ConversationState()

        return st.session_state.conversation_state

    def update_state(self, session_id: str, state: ConversationState):
        """Update Streamlit session state."""
        import streamlit as st

        st.session_state.conversation_state = state
