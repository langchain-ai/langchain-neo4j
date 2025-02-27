import gc
from unittest.mock import MagicMock, patch

import pytest

from langchain_neo4j.chat_message_histories.neo4j import Neo4jChatMessageHistory


def test_init_without_session_id() -> None:
    """Test initializing without session_id raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        Neo4jChatMessageHistory(None)  # type: ignore[arg-type]
    assert "Please ensure that the session_id parameter is provided" in str(
        exc_info.value
    )


def test_messages_setter() -> None:
    """Test that assigning to messages raises NotImplementedError."""
    with patch("neo4j.GraphDatabase.driver", autospec=True):
        message_store = Neo4jChatMessageHistory(
            session_id="test_session",
            url="bolt://url",
            username="username",
            password="password",
        )

        with pytest.raises(NotImplementedError) as exc_info:
            message_store.messages = []
        assert "Direct assignment to 'messages' is not allowed." in str(exc_info.value)


def test_driver_closed_on_delete() -> None:
    """Test that the driver is closed when the object is deleted."""
    with patch("neo4j.GraphDatabase.driver", autospec=True):
        message_store = Neo4jChatMessageHistory(
            session_id="test_session",
            url="bolt://url",
            username="username",
            password="password",
        )
        mock_driver = message_store._driver
        assert isinstance(mock_driver.close, MagicMock)
        message_store.__del__()
        gc.collect()
        mock_driver.close.assert_called_once()
