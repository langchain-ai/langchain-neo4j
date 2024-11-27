from unittest.mock import MagicMock, patch

import pytest

from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph, value_sanitize


def test_value_sanitize_with_small_list():  # type: ignore[no-untyped-def]
    small_list = list(range(15))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "small_list": small_list}
    expected_output = {"key1": "value1", "small_list": small_list}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_oversized_list():  # type: ignore[no-untyped-def]
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": oversized_list}
    expected_output = {
        "key1": "value1"
        # oversized_list should not be included
    }
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_nested_oversized_list():  # type: ignore[no-untyped-def]
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": {"key": oversized_list}}
    expected_output = {"key1": "value1", "oversized_list": {}}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_dict_in_list():  # type: ignore[no-untyped-def]
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": [1, 2, {"key": oversized_list}]}
    expected_output = {"key1": "value1", "oversized_list": [1, 2, {}]}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_dict_in_nested_list():  # type: ignore[no-untyped-def]
    input_dict = {
        "key1": "value1",
        "deeply_nested_lists": [[[[{"final_nested_key": list(range(200))}]]]],
    }
    expected_output = {"key1": "value1", "deeply_nested_lists": [[[[{}]]]]}
    assert value_sanitize(input_dict) == expected_output


def test_driver_state_management():  # type: ignore[no-untyped-def]
    """Comprehensive test for driver state management."""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        # Setup mock driver
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.execute_query = MagicMock(return_value=([], None, None))

        # Create graph instance
        graph = Neo4jGraph(
            url="bolt://localhost:7687", username="neo4j", password="password"
        )

        # Store original driver
        original_driver = graph._driver
        original_driver.close = MagicMock()

        # Test initial state
        assert hasattr(graph, "_driver")

        # First close
        graph.close()
        original_driver.close.assert_called_once()
        assert not hasattr(graph, "_driver")

        # Verify methods raise error when driver is closed
        with pytest.raises(
            RuntimeError,
            match="Cannot perform operations - Neo4j connection has been closed",
        ):
            graph.query("RETURN 1")

        with pytest.raises(
            RuntimeError,
            match="Cannot perform operations - Neo4j connection has been closed",
        ):
            graph.refresh_schema()


def test_close_method_removes_driver():  # type: ignore[no-untyped-def]
    """Test that close method removes the _driver attribute."""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        # Configure mock to return a mock driver
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance

        # Configure mock execute_query to return empty result
        mock_driver_instance.execute_query = MagicMock(return_value=([], None, None))

        # Add a _closed attribute to simulate driver state
        mock_driver_instance._closed = False

        graph = Neo4jGraph(
            url="bolt://localhost:7687", username="neo4j", password="password"
        )

        # Store a reference to the original driver
        original_driver = graph._driver

        # Ensure driver's close method can be mocked
        original_driver.close = MagicMock()

        # Call close method
        graph.close()

        # Verify driver.close was called
        original_driver.close.assert_called_once()

        # Verify _driver attribute is removed
        assert not hasattr(graph, "_driver")

        # Verify second close does not raise an error
        graph.close()  # Should not raise any exception


def test_multiple_close_calls_safe():  # type: ignore[no-untyped-def]
    """Test that multiple close calls do not raise errors."""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        # Configure mock to return a mock driver
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance

        # Configure mock execute_query to return empty result
        mock_driver_instance.execute_query = MagicMock(return_value=([], None, None))

        # Add a _closed attribute to simulate driver state
        mock_driver_instance._closed = False

        graph = Neo4jGraph(
            url="bolt://localhost:7687", username="neo4j", password="password"
        )

        # Store a reference to the original driver
        original_driver = graph._driver

        # Mock the driver's close method
        original_driver.close = MagicMock()

        # First close
        graph.close()
        original_driver.close.assert_called_once()

        # Verify _driver attribute is removed
        assert not hasattr(graph, "_driver")

        # Second close should not raise an error
        graph.close()  # Should not raise any exception
