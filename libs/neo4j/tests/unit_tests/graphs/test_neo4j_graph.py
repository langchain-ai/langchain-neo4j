from types import ModuleType
from typing import Generator, Mapping, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest

from langchain_neo4j.graphs.neo4j_graph import (
    Neo4jGraph,
    _format_schema,
    value_sanitize,
)


@pytest.fixture
def mock_neo4j_driver() -> Generator[MagicMock, None, None]:
    with patch("neo4j.GraphDatabase.driver", autospec=True) as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None
        mock_driver_instance.execute_query = MagicMock(return_value=([], None, None))
        mock_driver_instance._closed = False
        yield mock_driver_instance


def test_value_sanitize_with_small_list() -> None:
    small_list = list(range(15))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "small_list": small_list}
    expected_output = {"key1": "value1", "small_list": small_list}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_oversized_list() -> None:
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": oversized_list}
    expected_output = {
        "key1": "value1"
        # oversized_list should not be included
    }
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_nested_oversized_list() -> None:
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": {"key": oversized_list}}
    expected_output = {"key1": "value1", "oversized_list": {}}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_dict_in_list() -> None:
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": [1, 2, {"key": oversized_list}]}
    expected_output = {"key1": "value1", "oversized_list": [1, 2, {}]}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_dict_in_nested_list() -> None:
    input_dict = {
        "key1": "value1",
        "deeply_nested_lists": [[[[{"final_nested_key": list(range(200))}]]]],
    }
    expected_output = {"key1": "value1", "deeply_nested_lists": [[[[{}]]]]}
    assert value_sanitize(input_dict) == expected_output


def test_driver_state_management(mock_neo4j_driver: MagicMock) -> None:
    """Comprehensive test for driver state management."""
    # Create graph instance
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )

    # Store original driver
    original_driver = graph._driver
    assert isinstance(original_driver.close, MagicMock)

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


def test_close_method_removes_driver(mock_neo4j_driver: MagicMock) -> None:
    """Test that close method removes the _driver attribute."""
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )

    # Store a reference to the original driver
    original_driver = graph._driver
    assert isinstance(original_driver.close, MagicMock)

    # Call close method
    graph.close()

    # Verify driver.close was called
    original_driver.close.assert_called_once()

    # Verify _driver attribute is removed
    assert not hasattr(graph, "_driver")

    # Verify second close does not raise an error
    graph.close()  # Should not raise any exception


def test_multiple_close_calls_safe(mock_neo4j_driver: MagicMock) -> None:
    """Test that multiple close calls do not raise errors."""
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )

    # Store a reference to the original driver
    original_driver = graph._driver
    assert isinstance(original_driver.close, MagicMock)

    # First close
    graph.close()
    original_driver.close.assert_called_once()

    # Verify _driver attribute is removed
    assert not hasattr(graph, "_driver")

    # Second close should not raise an error
    graph.close()  # Should not raise any exception


def test_import_error() -> None:
    """Test that ImportError is raised when neo4j package is not installed."""
    original_import = __import__

    def mock_import(
        name: str,
        globals: Union[Mapping[str, object], None] = None,
        locals: Union[Mapping[str, object], None] = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "neo4j":
            raise ImportError()
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError) as exc_info:
            Neo4jGraph()
        assert "Could not import neo4j python package." in str(exc_info.value)


# _format_schema tests


def test_format_schema_string_high_distinct_count() -> None:
    schema = {
        "node_props": {
            "Person": [
                {
                    "property": "name",
                    "type": "STRING",
                    "values": ["Alice", "Bob", "Charlie"],
                    "distinct_count": 11,  # Greater than DISTINCT_VALUE_LIMIT (10)
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Person**\n"
        '  - `name`: STRING Example: "Alice"\n'
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_string_low_distinct_count() -> None:
    schema = {
        "node_props": {
            "Animal": [
                {
                    "property": "species",
                    "type": "STRING",
                    "values": ["Cat", "Dog"],
                    "distinct_count": 2,  # Less than DISTINCT_VALUE_LIMIT (10)
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Animal**\n"
        "  - `species`: STRING Available options: ['Cat', 'Dog']\n"
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_numeric_with_min_max() -> None:
    schema = {
        "node_props": {
            "Person": [{"property": "age", "type": "INTEGER", "min": 20, "max": 70}]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Person**\n"
        "  - `age`: INTEGER Min: 20, Max: 70\n"
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_numeric_with_values() -> None:
    schema = {
        "node_props": {
            "Event": [
                {
                    "property": "date",
                    "type": "DATE",
                    "values": ["2021-01-01", "2021-01-02"],
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Event**\n"
        '  - `date`: DATE Example: "2021-01-01"\n'
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_list_skipped() -> None:
    schema = {
        "node_props": {
            "Document": [
                {
                    "property": "embedding",
                    "type": "LIST",
                    "min_size": 150,  # Greater than LIST_LIMIT (128)
                    "max_size": 200,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Document**\n"
        # 'embedding' property should be skipped
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_list_included() -> None:
    schema = {
        "node_props": {
            "Document": [
                {"property": "keywords", "type": "LIST", "min_size": 2, "max_size": 5}
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Document**\n"
        "  - `keywords`: LIST Min Size: 2, Max Size: 5\n"
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_rel_string_high_distinct_count() -> None:
    schema = {
        "node_props": {},
        "rel_props": {
            "KNOWS": [
                {
                    "property": "since",
                    "type": "STRING",
                    "values": ["2000", "2001", "2002"],
                    "distinct_count": 15,
                }
            ]
        },
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "\n"
        "Relationship properties:\n"
        "- **KNOWS**\n"
        '  - `since`: STRING Example: "2000"\n'
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_rel_string_low_distinct_count() -> None:
    schema = {
        "node_props": {},
        "rel_props": {
            "LIKES": [
                {
                    "property": "intensity",
                    "type": "STRING",
                    "values": ["High", "Medium", "Low"],
                    "distinct_count": 3,
                }
            ]
        },
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "\n"
        "Relationship properties:\n"
        "- **LIKES**\n"
        "  - `intensity`: STRING Available options: ['High', 'Medium', 'Low']\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_rel_numeric_with_min_max() -> None:
    schema = {
        "node_props": {},
        "rel_props": {
            "WORKS_WITH": [
                {"property": "since", "type": "INTEGER", "min": 1995, "max": 2020}
            ]
        },
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "\n"
        "Relationship properties:\n"
        "- **WORKS_WITH**\n"
        "  - `since`: INTEGER Min: 1995, Max: 2020\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_rel_list_skipped() -> None:
    schema = {
        "node_props": {},
        "rel_props": {
            "KNOWS": [
                {
                    "property": "embedding",
                    "type": "LIST",
                    "min_size": 150,
                    "max_size": 200,
                }
            ]
        },
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "\n"
        "Relationship properties:\n"
        "- **KNOWS**\n"
        # 'embedding' property should be skipped
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_rel_list_included() -> None:
    schema = {
        "node_props": {},
        "rel_props": {
            "KNOWS": [
                {"property": "messages", "type": "LIST", "min_size": 2, "max_size": 5}
            ]
        },
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "\n"
        "Relationship properties:\n"
        "- **KNOWS**\n"
        "  - `messages`: LIST Min Size: 2, Max Size: 5\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_rel_numeric_no_min_max() -> None:
    schema = {
        "node_props": {},
        "rel_props": {
            "OWES": [
                {
                    "property": "amount",
                    "type": "FLOAT",
                    # 'min' and 'max' are missing
                    "values": [3.14, 2.71],
                }
            ]
        },
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "\n"
        "Relationship properties:\n"
        "- **OWES**\n"
        '  - `amount`: FLOAT Example: "3.14"\n'
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_values_empty() -> None:
    schema = {
        "node_props": {
            "Person": [
                {
                    "property": "name",
                    "type": "STRING",
                    "values": [],
                    "distinct_count": 15,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Person**\n"
        "  - `name`: STRING \n"  # Example should be empty
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


def test_format_schema_values_none() -> None:
    schema = {
        "node_props": {
            "Person": [
                {
                    "property": "name",
                    "type": "STRING",
                    # 'values' is missing
                    "distinct_count": 15,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    expected_output = (
        "Node properties:\n"
        "- **Person**\n"
        "  - `name`: STRING \n"  # Example should be empty
        "Relationship properties:\n"
        "\n"
        "The relationships:\n"
    )
    result = _format_schema(schema, is_enhanced=True)
    assert result == expected_output


# _enhanced_schema_cypher tests


def test_enhanced_schema_cypher_integer_exhaustive_true(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )

    graph.structured_schema = {"metadata": {"index": []}}
    properties = [{"property": "age", "type": "INTEGER"}]
    query = graph._enhanced_schema_cypher("Person", properties, exhaustive=True)
    assert "min(n.`age`) AS `age_min`" in query
    assert "max(n.`age`) AS `age_max`" in query
    assert "count(distinct n.`age`) AS `age_distinct`" in query
    assert (
        "min: toString(`age_min`), max: toString(`age_max`), "
        "distinct_count: `age_distinct`" in query
    )


def test_enhanced_schema_cypher_list_exhaustive_true(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    graph.structured_schema = {"metadata": {"index": []}}
    properties = [{"property": "tags", "type": "LIST"}]
    query = graph._enhanced_schema_cypher("Article", properties, exhaustive=True)
    assert "min(size(n.`tags`)) AS `tags_size_min`" in query
    assert "max(size(n.`tags`)) AS `tags_size_max`" in query
    assert "min_size: `tags_size_min`, max_size: `tags_size_max`" in query


def test_enhanced_schema_cypher_boolean_exhaustive_true(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    properties = [{"property": "active", "type": "BOOLEAN"}]
    query = graph._enhanced_schema_cypher("User", properties, exhaustive=True)
    # BOOLEAN types should be skipped, so their properties should not be in the query
    assert "n.`active`" not in query


def test_enhanced_schema_cypher_integer_exhaustive_false_no_index(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    graph.structured_schema = {"metadata": {"index": []}}
    properties = [{"property": "age", "type": "INTEGER"}]
    query = graph._enhanced_schema_cypher("Person", properties, exhaustive=False)
    assert "collect(distinct toString(n.`age`)) AS `age_values`" in query
    assert "values: `age_values`" in query


def test_enhanced_schema_cypher_integer_exhaustive_false_with_index(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    graph.structured_schema = {
        "metadata": {
            "index": [
                {
                    "label": "Person",
                    "properties": ["age"],
                    "type": "RANGE",
                }
            ]
        }
    }
    properties = [{"property": "age", "type": "INTEGER"}]
    query = graph._enhanced_schema_cypher("Person", properties, exhaustive=False)
    assert "min(n.`age`) AS `age_min`" in query
    assert "max(n.`age`) AS `age_max`" in query
    assert "count(distinct n.`age`) AS `age_distinct`" in query
    assert (
        "min: toString(`age_min`), max: toString(`age_max`), "
        "distinct_count: `age_distinct`" in query
    )


def test_enhanced_schema_cypher_list_exhaustive_false(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    properties = [{"property": "tags", "type": "LIST"}]
    query = graph._enhanced_schema_cypher("Article", properties, exhaustive=False)
    assert "min(size(n.`tags`)) AS `tags_size_min`" in query
    assert "max(size(n.`tags`)) AS `tags_size_max`" in query
    assert "min_size: `tags_size_min`, max_size: `tags_size_max`" in query


def test_enhanced_schema_cypher_boolean_exhaustive_false(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    properties = [{"property": "active", "type": "BOOLEAN"}]
    query = graph._enhanced_schema_cypher("User", properties, exhaustive=False)
    # BOOLEAN types should be skipped, so their properties should not be in the query
    assert "n.`active`" not in query


def test_enhanced_schema_cypher_string_exhaustive_false_with_index(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    graph.structured_schema = {
        "metadata": {
            "index": [
                {
                    "label": "Person",
                    "properties": ["status"],
                    "type": "RANGE",
                    "size": 5,
                    "distinctValues": 5,
                }
            ]
        }
    }
    graph.query = MagicMock(return_value=[{"value": ["Single", "Married", "Divorced"]}])
    properties = [{"property": "status", "type": "STRING"}]
    query = graph._enhanced_schema_cypher("Person", properties, exhaustive=False)
    assert "values: ['Single', 'Married', 'Divorced'], distinct_count: 3" in query


def test_enhanced_schema_cypher_string_exhaustive_false_no_index(
    mock_neo4j_driver: MagicMock,
) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    graph.structured_schema = {"metadata": {"index": []}}
    properties = [{"property": "status", "type": "STRING"}]
    query = graph._enhanced_schema_cypher("Person", properties, exhaustive=False)
    assert (
        "collect(distinct substring(toString(n.`status`), 0, 50)) AS `status_values`"
        in query
    )
    assert "values: `status_values`" in query


def test_enhanced_schema_cypher_point_type(mock_neo4j_driver: MagicMock) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    properties = [{"property": "location", "type": "POINT"}]
    query = graph._enhanced_schema_cypher("Place", properties, exhaustive=True)
    # POINT types should be skipped
    assert "n.`location`" not in query


def test_enhanced_schema_cypher_duration_type(mock_neo4j_driver: MagicMock) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    properties = [{"property": "duration", "type": "DURATION"}]
    query = graph._enhanced_schema_cypher("Event", properties, exhaustive=False)
    # DURATION types should be skipped
    assert "n.`duration`" not in query


def test_enhanced_schema_cypher_relationship(mock_neo4j_driver: MagicMock) -> None:
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="password"
    )
    properties = [{"property": "since", "type": "INTEGER"}]

    query = graph._enhanced_schema_cypher(
        label_or_type="FRIENDS_WITH",
        properties=properties,
        exhaustive=True,
        is_relationship=True,
    )

    assert query.startswith("MATCH ()-[n:`FRIENDS_WITH`]->()")
    assert "min(n.`since`) AS `since_min`" in query
    assert "max(n.`since`) AS `since_max`" in query
    assert "count(distinct n.`since`) AS `since_distinct`" in query
    expected_return_clause = (
        "`since`: {min: toString(`since_min`), max: toString(`since_max`), "
        "distinct_count: `since_distinct`}"
    )
    assert expected_return_clause in query
