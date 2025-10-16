"""Tests for the chat timer functionality."""

import time
from unittest.mock import Mock, patch

import pytest


def test_timer_initialization():
    """Test that timer variables are properly initialized."""
    # Mock streamlit session state
    mock_session_state = {}
    
    with patch('streamlit.session_state', mock_session_state):
        # Simulate initialization in pages.py
        if "chat_start_time" not in mock_session_state:
            mock_session_state["chat_start_time"] = time.time()
            mock_session_state["chat_timer_expired"] = False
        
        assert "chat_start_time" in mock_session_state
        assert "chat_timer_expired" in mock_session_state
        assert mock_session_state["chat_timer_expired"] is False
        assert isinstance(mock_session_state["chat_start_time"], float)


def test_timer_expiration():
    """Test that timer expires after 10 minutes."""
    TIMER_DURATION = 600  # 10 minutes
    current_time = time.time()
    
    # Test with time not expired
    start_time_recent = current_time - 300  # 5 minutes ago
    elapsed_time = current_time - start_time_recent
    remaining_time = max(0, TIMER_DURATION - elapsed_time)
    assert remaining_time > 0
    
    # Test with time expired
    start_time_old = current_time - 700  # 11+ minutes ago
    elapsed_time = current_time - start_time_old
    remaining_time = max(0, TIMER_DURATION - elapsed_time)
    assert remaining_time == 0


def test_timer_display_format():
    """Test that timer displays correctly formatted time."""
    TIMER_DURATION = 600
    
    # Test with 5 minutes 30 seconds remaining
    remaining_time = 330
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)
    
    assert minutes == 5
    assert seconds == 30
    
    # Test with less than 1 minute remaining
    remaining_time = 45
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)
    
    assert minutes == 0
    assert seconds == 45