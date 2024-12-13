from types import ModuleType
from typing import Any, Dict, Generator, Mapping, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ClientError, Neo4jError

from langchain_neo4j.graphs.neo4j_graph import (
    LIST_LIMIT,
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


@pytest.mark.parametrize(
    "description, input_value, expected_output",
    [
        (
            "Small list",
            {"key1": "value1", "small_list": list(range(15))},
            {"key1": "value1", "small_list": list(range(15))},
        ),
        (
            "Oversized list",
            {"key1": "value1", "oversized_list": list(range(LIST_LIMIT + 1))},
            {"key1": "value1"},
        ),
        (
            "Nested oversized list",
            {"key1": "value1", "oversized_list": {"key": list(range(150))}},
            {"key1": "value1", "oversized_list": {}},
        ),
        (
            "Dict in list",
            {
                "key1": "value1",
                "oversized_list": [1, 2, {"key": list(range(LIST_LIMIT + 1))}],
            },
            {"key1": "value1", "oversized_list": [1, 2, {}]},
        ),
        (
            "Dict in nested list",
            {
                "key1": "value1",
                "deeply_nested_lists": [
                    [[[{"final_nested_key": list(range(LIST_LIMIT + 1))}]]]
                ],
            },
            {"key1": "value1", "deeply_nested_lists": [[[[{}]]]]},
        ),
        (
            "Bare oversized list",
            list(range(LIST_LIMIT + 1)),
            None,
        ),
        (
            "None value",
            None,
            None,
        ),
    ],
)
def test_value_sanitize(
    description: str, input_value: Dict[str, Any], expected_output: Any
) -> None:
    """Test the value_sanitize function."""
    assert (
        value_sanitize(input_value) == expected_output
    ), f"Failed test case: {description}"


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


def test_neo4j_graph_del_method(mock_neo4j_driver: MagicMock) -> None:
    """Test the __del__ method."""
    with patch.object(Neo4jGraph, "close") as mock_close:
        graph = Neo4jGraph(
            url="bolt://localhost:7687", username="neo4j", password="password"
        )
        # Ensure exceptions are suppressed when the graph's destructor is called
        mock_close.side_effect = Exception()
        mock_close.assert_not_called()
        graph.__del__()
        mock_close.assert_called_once()


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


def test_neo4j_graph_init_with_empty_credentials() -> None:
    """Test the __init__ method when no credentials have been provided."""
    with patch("neo4j.GraphDatabase.driver", autospec=True) as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None
        Neo4jGraph(
            url="bolt://localhost:7687", username="", password="", refresh_schema=False
        )
        mock_driver.assert_called_with("bolt://localhost:7687", auth=None)


def test_init_apoc_procedure_not_found(
    mock_neo4j_driver: MagicMock,
) -> None:
    """Test an error is raised when APOC is not installed."""
    with patch("langchain_neo4j.Neo4jGraph.refresh_schema") as mock_refresh_schema:
        err = ClientError()
        err.code = "Neo.ClientError.Procedure.ProcedureNotFound"
        mock_refresh_schema.side_effect = err
        with pytest.raises(ValueError) as exc_info:
            Neo4jGraph(url="bolt://localhost:7687", username="", password="")
        assert "Could not use APOC procedures." in str(exc_info.value)


def test_init_refresh_schema_other_err(
    mock_neo4j_driver: MagicMock,
) -> None:
    """Test any other ClientErrors raised when calling refresh_schema in __init__ are
    re-raised."""
    with patch("langchain_neo4j.Neo4jGraph.refresh_schema") as mock_refresh_schema:
        err = ClientError()
        err.code = "other_error"
        mock_refresh_schema.side_effect = err
        with pytest.raises(ClientError) as exc_info:
            Neo4jGraph(url="bolt://localhost:7687", username="", password="")
        assert exc_info.value == err


def test_query_fallback_execution(mock_neo4j_driver: MagicMock) -> None:
    """Test the fallback to allow for implicit transactions in query."""
    err = Neo4jError()
    err.code = "Neo.DatabaseError.Statement.ExecutionFailed"
    err.message = "in an implicit transaction"
    mock_neo4j_driver.execute_query.side_effect = err
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="test_db",
        sanitize=True,
    )
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = {
        "key1": "value1",
        "oversized_list": list(range(LIST_LIMIT + 1)),
    }
    mock_session.run.return_value = [mock_result]
    mock_neo4j_driver.session.return_value.__enter__.return_value = mock_session
    mock_neo4j_driver.session.return_value.__exit__.return_value = None
    query = "MATCH (n) RETURN n;"
    params = {"param1": "value1"}
    json_data = graph.query(query, params)
    mock_neo4j_driver.session.assert_called_with(database="test_db")
    called_args, _ = mock_session.run.call_args
    called_query = called_args[0]
    assert called_query.text == query
    assert called_query.timeout == graph.timeout
    assert called_args[1] == params
    assert json_data == [{"key1": "value1"}]


def test_refresh_schema_handles_client_error(mock_neo4j_driver: MagicMock) -> None:
    """Test refresh schema handles a client error which might arise due to a user
    not having access to schema information"""

    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="test_db",
    )
    node_properties = [
        {
            "output": {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "labels": "LabelA",
            }
        }
    ]
    relationships_properties = [
        {
            "output": {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        }
    ]
    relationships = [
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
    ]

    # Mock the query method to raise ClientError for constraint and index queries
    graph.query = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            node_properties,
            relationships_properties,
            relationships,
            ClientError("Mock ClientError"),
        ]
    )
    graph.refresh_schema()

    # Assertions
    # Ensure constraints and indexes are empty due to the ClientError
    assert graph.structured_schema["metadata"]["constraint"] == []
    assert graph.structured_schema["metadata"]["index"] == []

    # Ensure the query method was called as expected
    assert graph.query.call_count == 4
    graph.query.assert_any_call("SHOW CONSTRAINTS")


def test_get_schema(mock_neo4j_driver: MagicMock) -> None:
    """Tests the get_schema property."""
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        refresh_schema=False,
    )
    graph.schema = "test"
    assert graph.get_schema == "test"


@pytest.mark.parametrize(
    "description, schema, is_enhanced, expected_output",
    [
        (
            "Enhanced, string property with high distinct count",
            {
                "node_props": {
                    "Person": [
                        {
                            "property": "name",
                            "type": "STRING",
                            "values": ["Alice", "Bob", "Charlie"],
                            "distinct_count": 11,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                '  - `name`: STRING Example: "Alice"\n'
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, string property with low distinct count",
            {
                "node_props": {
                    "Animal": [
                        {
                            "property": "species",
                            "type": "STRING",
                            "values": ["Cat", "Dog"],
                            "distinct_count": 2,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Animal**\n"
                "  - `species`: STRING Available options: ['Cat', 'Dog']\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, numeric property with min and max",
            {
                "node_props": {
                    "Person": [
                        {"property": "age", "type": "INTEGER", "min": 20, "max": 70}
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                "  - `age`: INTEGER Min: 20, Max: 70\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, numeric property with values",
            {
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
            },
            True,
            (
                "Node properties:\n"
                "- **Event**\n"
                '  - `date`: DATE Example: "2021-01-01"\n'
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, list property that should be skipped",
            {
                "node_props": {
                    "Document": [
                        {
                            "property": "embedding",
                            "type": "LIST",
                            "min_size": 150,
                            "max_size": 200,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Document**\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, list property that should be included",
            {
                "node_props": {
                    "Document": [
                        {
                            "property": "keywords",
                            "type": "LIST",
                            "min_size": 2,
                            "max_size": 5,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Document**\n"
                "  - `keywords`: LIST Min Size: 2, Max Size: 5\n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship string property with high distinct count",
            {
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
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **KNOWS**\n"
                '  - `since`: STRING Example: "2000"\n'
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship string property with low distinct count",
            {
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
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **LIKES**\n"
                "  - `intensity`: STRING Available options: ['High', 'Medium', 'Low']\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship numeric property with min and max",
            {
                "node_props": {},
                "rel_props": {
                    "WORKS_WITH": [
                        {
                            "property": "since",
                            "type": "INTEGER",
                            "min": 1995,
                            "max": 2020,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **WORKS_WITH**\n"
                "  - `since`: INTEGER Min: 1995, Max: 2020\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship list property that should be skipped",
            {
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
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **KNOWS**\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship list property that should be included",
            {
                "node_props": {},
                "rel_props": {
                    "KNOWS": [
                        {
                            "property": "messages",
                            "type": "LIST",
                            "min_size": 2,
                            "max_size": 5,
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **KNOWS**\n"
                "  - `messages`: LIST Min Size: 2, Max Size: 5\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, relationship numeric property without min and max",
            {
                "node_props": {},
                "rel_props": {
                    "OWES": [
                        {
                            "property": "amount",
                            "type": "FLOAT",
                            "values": [3.14, 2.71],
                        }
                    ]
                },
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "\n"
                "Relationship properties:\n"
                "- **OWES**\n"
                '  - `amount`: FLOAT Example: "3.14"\n'
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, property with empty values list",
            {
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
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                "  - `name`: STRING \n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
        (
            "Enhanced, property with missing values",
            {
                "node_props": {
                    "Person": [
                        {
                            "property": "name",
                            "type": "STRING",
                            "distinct_count": 15,
                        }
                    ]
                },
                "rel_props": {},
                "relationships": [],
            },
            True,
            (
                "Node properties:\n"
                "- **Person**\n"
                "  - `name`: STRING \n"
                "Relationship properties:\n"
                "\n"
                "The relationships:\n"
            ),
        ),
    ],
)
def test_format_schema(
    description: str, schema: Dict, is_enhanced: bool, expected_output: str
) -> None:
    result = _format_schema(schema, is_enhanced)
    assert result == expected_output, f"Failed test case: {description}"


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
    graph.query = MagicMock(return_value=[{"value": ["Single", "Married", "Divorced"]}])  # type: ignore[method-assign]
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
