from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Record, bearer_auth
from neo4j._work.summary import ResultSummary
from neo4j.exceptions import ClientError, ConfigurationError, Neo4jError
from neo4j_graphrag.schema import LIST_LIMIT

from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph


@pytest.fixture
def mock_neo4j_driver() -> Generator[MagicMock, None, None]:
    with patch("neo4j.GraphDatabase.driver", autospec=True) as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None
        mock_driver_instance.execute_query = MagicMock(
            return_value=MagicMock(
                records=[], summary=MagicMock(spec=ResultSummary), keys=[]
            )
        )
        mock_driver_instance._closed = False
        yield mock_driver_instance


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


def test_neo4j_graph_init_with_token() -> None:
    """Test the __init__ method when only token is provided (no username/password)."""
    with patch("neo4j.GraphDatabase.driver", autospec=True) as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None
        Neo4jGraph(
            url="bolt://localhost:7687",
            token="my-bearer-token",
            refresh_schema=False,
        )
        mock_driver.assert_called_with(
            "bolt://localhost:7687", auth=bearer_auth("my-bearer-token")
        )


def test_neo4j_graph_init_without_credentials() -> None:
    """Test the __init__ method raises error when no credentials are provided."""
    with patch("neo4j.GraphDatabase.driver", autospec=True) as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None
        with pytest.raises(ValueError, match="pass `username` as a named parameter."):
            Neo4jGraph(
                url="bolt://localhost:7687",
                refresh_schema=False,
            )


def test_neo4j_graph_init_driver_config_err() -> None:
    """Test the __init__ method with an incorrect driver config."""
    with patch("neo4j.GraphDatabase.driver", autospec=True) as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance
        err = ConfigurationError()
        mock_driver_instance.verify_connectivity.side_effect = err
        with pytest.raises(ValueError) as exc_info:
            Neo4jGraph(
                url="bolt://localhost:7687",
                username="username",
                password="password",
                refresh_schema=False,
            )
        assert "Please ensure that the driver config is correct" in str(exc_info.value)


def test_init_apoc_procedure_not_found(
    mock_neo4j_driver: MagicMock,
) -> None:
    """Test an error is raised when APOC is not installed."""
    with patch("langchain_neo4j.Neo4jGraph.refresh_schema") as mock_refresh_schema:
        err = Neo4jError._basic_hydrate(
            neo4j_code="Neo.ClientError.Procedure.ProcedureNotFound",
            message="Procedure not found",
        )
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
        err = Neo4jError._basic_hydrate(
            neo4j_code="Neo.ClientError.General.OtherError",
            message="Other error",
        )
        mock_refresh_schema.side_effect = err
        with pytest.raises(ClientError) as exc_info:
            Neo4jGraph(url="bolt://localhost:7687", username="", password="")
        assert exc_info.value == err


def test_query_fallback_execution(mock_neo4j_driver: MagicMock) -> None:
    """Test the fallback to allow for implicit transactions in query."""
    err = Neo4jError._basic_hydrate(
        neo4j_code="Neo.DatabaseError.Statement.ExecutionFailed",
        message="in an implicit transaction",
    )
    mock_neo4j_driver.execute_query.side_effect = err
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="test_db",
        sanitize=True,
        refresh_schema=False,
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
        refresh_schema=False,
    )
    node_properties = [
        Record(
            {
                "output": {
                    "properties": [{"property": "property_a", "type": "STRING"}],
                    "label": "LabelA",
                }
            }
        )
    ]
    relationships_properties = [
        Record(
            {
                "output": {
                    "type": "REL_TYPE",
                    "properties": [{"property": "rel_prop", "type": "STRING"}],
                }
            }
        )
    ]
    relationships = [
        Record({"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}}),
        Record({"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}}),
    ]

    mock_neo4j_driver.execute_query.side_effect = [
        MagicMock(
            records=node_properties, summary=MagicMock(spec=ResultSummary), keys=[]
        ),
        MagicMock(
            records=relationships_properties,
            summary=MagicMock(spec=ResultSummary),
            keys=[],
        ),
        MagicMock(
            records=relationships, summary=MagicMock(spec=ResultSummary), keys=[]
        ),
        ClientError("Mock ClientError"),
    ]
    graph.refresh_schema()

    # Assertions
    # Ensure constraints and indexes are empty due to the ClientError
    assert graph.structured_schema["metadata"]["constraint"] == []
    assert graph.structured_schema["metadata"]["index"] == []

    # Ensure the query method was called as expected
    assert mock_neo4j_driver.execute_query.call_count == 4
    calls = mock_neo4j_driver.execute_query.call_args_list
    assert any(call.args[0].text == "SHOW CONSTRAINTS" for call in calls)


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


def test_add_graph_docs_inc_src_err(mock_neo4j_driver: MagicMock) -> None:
    """Tests an error is raised when using add_graph_documents with include_source set
    to True and a document is missing a source."""
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        refresh_schema=False,
    )
    node_1 = Node(id=1)
    node_2 = Node(id=2)
    rel = Relationship(source=node_1, target=node_2, type="REL")

    graph_doc = GraphDocument(
        nodes=[node_1, node_2],
        relationships=[rel],
    )
    with pytest.raises(TypeError) as exc_info:
        graph.add_graph_documents(graph_documents=[graph_doc], include_source=True)

    assert (
        "include_source is set to True, but at least one document has no `source`."
        in str(exc_info.value)
    )
