"""State management for conversations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import BaseMessage, SystemMessage


@dataclass
class ConversationState:
    """Manages the state of a conversation."""

    preferences: str = field(
        default_factory=lambda: "No preferences specified."
    )
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    chat_log: List[BaseMessage] = field(default_factory=list)
    turn_count: int = 0
    success_detection: Optional[Dict[str, Any]] = field(default=None)
    recommended_items: List[str] = field(
        default_factory=list
    )  # Track recommended item IDs

    def update_preferences(self, new_preferences: str):
        """Update the user preferences."""
        self.preferences = new_preferences.strip()
        self.add_system_message(f"Updated preferences: {new_preferences}")

    def update_retrieved_items(self, items: List[Dict[str, Any]]):
        """Update the retrieved items list."""
        self.retrieved_items = items
        item_titles = [item.get("title", "Unknown") for item in items]
        self.add_system_message(f"Retrieved items: {', '.join(item_titles)}")

    def add_recommended_item(self, item_id: str, item_title: str = None):
        """Track an item that has been recommended to avoid repeating."""
        if item_id not in self.recommended_items:
            self.recommended_items.append(item_id)
            title_info = f" ({item_title})" if item_title else ""
            self.add_system_message(
                f"Added to recommended: {item_id}{title_info}"
            )

    def is_item_already_recommended(self, item_id: str) -> bool:
        """Check if an item has already been recommended."""
        return item_id in self.recommended_items

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

    def update_success_detection(self, detection_result: Dict[str, Any]):
        """Update the success detection result."""
        self.success_detection = detection_result

    def is_target_found(self) -> bool:
        """Check if the target has been found."""
        return self.success_detection and self.success_detection.get(
            "target_found", False
        )

    def get_success_info(self) -> Dict[str, Any]:
        """Get current success detection information."""
        return self.success_detection or {
            "success": False,
            "target_found": False,
        }

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
        return (
            chat_history[-count:]
            if len(chat_history) >= count
            else chat_history
        )


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
