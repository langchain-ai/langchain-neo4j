import os
import urllib

import pytest
from langchain_core.documents import Document
from neo4j_graphrag.schema import NODE_PROPERTIES_QUERY, REL_PROPERTIES_QUERY, REL_QUERY

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_neo4j.graphs.neo4j_graph import BASE_ENTITY_LABEL
from tests.integration_tests.utils import Neo4jCredentials

test_data = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo"), Node(id="bar", type="bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
                properties={"key": "val"},
            )
        ],
        source=Document(page_content="source document"),
    )
]

test_data_backticks = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo`"), Node(id="bar", type="`bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="f`oo"),
                target=Node(id="bar", type="ba`r"),
                type="`REL`",
            )
        ],
        source=Document(page_content="source document"),
    )
]


@pytest.mark.usefixtures("clear_neo4j_database")
def test_connect_neo4j(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that Neo4j database is correctly instantiated and connected."""
    graph = Neo4jGraph(**neo4j_credentials)

    output = graph.query('RETURN "test" AS output')
    expected_output = [{"output": "test"}]
    assert output == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_connect_neo4j_env() -> None:
    """Test that Neo4j database environment variables."""
    assert os.environ.get("NEO4J_URI") is not None
    assert os.environ.get("NEO4J_USERNAME") is not None
    assert os.environ.get("NEO4J_PASSWORD") is not None
    graph = Neo4jGraph()

    output = graph.query('RETURN "test" AS output')
    expected_output = [{"output": "test"}]
    assert output == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_cypher_return_correct_schema(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that chain returns direct results."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )
    # Refresh schema information
    graph.refresh_schema()

    node_properties = graph.query(
        NODE_PROPERTIES_QUERY,
        params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL], "SAMPLE": 1000},
    )
    relationships_properties = graph.query(
        REL_PROPERTIES_QUERY,
        params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL], "SAMPLE": 1000},
    )
    relationships = graph.query(
        REL_QUERY, params={"EXCLUDED_LABELS": [BASE_ENTITY_LABEL], "SAMPLE": 1000}
    )

    expected_node_properties = [
        {
            "output": {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "label": "LabelA",
            }
        }
    ]
    expected_relationships_properties = [
        {
            "output": {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        }
    ]
    expected_relationships = [
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"}},
        {"output": {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"}},
    ]

    assert node_properties == expected_node_properties
    assert relationships_properties == expected_relationships_properties
    # Order is not guaranteed with Neo4j returns
    assert (
        sorted(relationships, key=lambda x: x["output"]["end"])
        == expected_relationships
    )


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_timeout(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j uses the timeout correctly."""
    graph = Neo4jGraph(timeout=0.1, **neo4j_credentials)
    try:
        graph.query("UNWIND range(0,100000,1) AS i MERGE (:Foo {id:i})")
    except Exception as e:
        assert hasattr(e, "code")
        assert (
            e.code
            == "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration"
        )


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_sanitize_values(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that lists with more than `128` elements are removed from the results."""
    graph = Neo4jGraph(sanitize=True, **neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        """
        CREATE (la:LabelA {property_a: 'a'})
        CREATE (lb:LabelB)
        CREATE (lc:LabelC)
        MERGE (la)-[:REL_TYPE]-> (lb)
        MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
        """
    )
    graph.refresh_schema()

    output = graph.query("RETURN range(0,130,1) AS result")
    assert output == [{}]


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_add_data(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j correctly import graph document."""
    graph = Neo4jGraph(sanitize=True, **neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data)
    output = graph.query(
        "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY label"
    )
    assert output == [{"label": ["bar"], "count": 1}, {"label": ["foo"], "count": 1}]
    assert graph.structured_schema["metadata"]["constraint"] == []


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_add_data_source(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j correctly import graph document with source."""
    graph = Neo4jGraph(sanitize=True, **neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data, include_source=True)
    output = graph.query(
        "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY label"
    )
    assert output == [
        {"label": ["Document"], "count": 1},
        {"label": ["bar"], "count": 1},
        {"label": ["foo"], "count": 1},
    ]
    assert graph.structured_schema["metadata"]["constraint"] == []


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_add_data_base(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j correctly import graph document with base_entity."""
    graph = Neo4jGraph(sanitize=True, **neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data, baseEntityLabel=True)
    output = graph.query(
        "MATCH (n) RETURN apoc.coll.sort(labels(n)) AS label, "
        "count(*) AS count ORDER BY label"
    )
    assert output == [
        {"label": [BASE_ENTITY_LABEL, "bar"], "count": 1},
        {"label": [BASE_ENTITY_LABEL, "foo"], "count": 1},
    ]
    assert graph.structured_schema["metadata"]["constraint"] != []


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_add_data_base_source(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j correctly import graph document with base_entity and source."""
    graph = Neo4jGraph(sanitize=True, **neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.refresh_schema()
    # Create two nodes and a relationship
    graph.add_graph_documents(test_data, baseEntityLabel=True, include_source=True)
    output = graph.query(
        "MATCH (n) RETURN apoc.coll.sort(labels(n)) AS label, "
        "count(*) AS count ORDER BY label"
    )
    assert output == [
        {"label": ["Document"], "count": 1},
        {"label": [BASE_ENTITY_LABEL, "bar"], "count": 1},
        {"label": [BASE_ENTITY_LABEL, "foo"], "count": 1},
    ]
    assert graph.structured_schema["metadata"]["constraint"] != []


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_filtering_labels(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j correctly filters excluded labels."""
    graph = Neo4jGraph(sanitize=True, **neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Remove all constraints
    graph.query("CALL apoc.schema.assert({}, {})")
    graph.query(
        """
        CREATE (:_Bloom_Scene_ {property_a: 'a'})
        -[:_Bloom_HAS_SCENE_ {property_b: 'b'}]
        ->(:_Bloom_Perspective_)
        """
    )
    graph.refresh_schema()

    # Assert all are empty
    assert graph.structured_schema["node_props"] == {}
    assert graph.structured_schema["rel_props"] == {}
    assert graph.structured_schema["relationships"] == []


@pytest.mark.usefixtures("clear_neo4j_database")
def test_driver_config(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j works with driver config."""
    graph = Neo4jGraph(
        driver_config={"max_connection_pool_size": 1}, **neo4j_credentials
    )
    graph.query("RETURN 'foo'")


@pytest.mark.usefixtures("clear_neo4j_database")
def test_enhanced_schema(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that neo4j works with driver config."""
    graph = Neo4jGraph(enhanced_schema=True, **neo4j_credentials)
    graph.query("MATCH (n) DETACH DELETE n")
    graph.add_graph_documents(test_data)
    graph.refresh_schema()
    expected_output = {
        "node_props": {
            "foo": [
                {
                    "property": "id",
                    "type": "STRING",
                    "values": ["foo"],
                    "distinct_count": 1,
                }
            ],
            "bar": [
                {
                    "property": "id",
                    "type": "STRING",
                    "values": ["bar"],
                    "distinct_count": 1,
                }
            ],
        },
        "rel_props": {
            "REL": [
                {
                    "distinct_count": 1,
                    "property": "key",
                    "type": "STRING",
                    "values": ["val"],
                }
            ]
        },
        "relationships": [{"start": "foo", "type": "REL", "end": "bar"}],
    }
    # remove metadata portion of schema
    del graph.structured_schema["metadata"]
    assert graph.structured_schema == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_enhanced_schema_exception(neo4j_credentials: Neo4jCredentials) -> None:
    """Test no error with weird schema."""
    graph = Neo4jGraph(enhanced_schema=True, refresh_schema=False, **neo4j_credentials)
    graph.query("MATCH (n) DETACH DELETE n")
    graph.query(
        "CREATE (:Node {foo: 'bar'}), (:Node {foo: 1}), (:Node {foo: [1,2]}), "
        "(: EmptyNode)"
    )
    graph.query(
        "MATCH (a:Node {foo: 'bar'}), (b:Node {foo: 1}), "
        "(c:Node {foo: [1,2]}), (d: EmptyNode) "
        "CREATE (a)-[:REL {foo: 'bar'}]->(b), (b)-[:REL {foo: 1}]->(c), "
        "(c)-[:REL {foo: [1,2]}]->(a), (d)-[:EMPTY_REL {}]->(d)"
    )
    graph.refresh_schema()
    expected_output = {
        "node_props": {"Node": [{"property": "foo", "type": "STRING"}]},
        "rel_props": {"REL": [{"property": "foo", "type": "STRING"}]},
        "relationships": [
            {
                "end": "Node",
                "start": "Node",
                "type": "REL",
            },
            {"end": "EmptyNode", "start": "EmptyNode", "type": "EMPTY_REL"},
        ],
    }

    # remove metadata portion of schema
    del graph.structured_schema["metadata"]
    assert graph.structured_schema == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_backticks(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that backticks are correctly removed."""
    graph = Neo4jGraph(**neo4j_credentials)
    graph.query("MATCH (n) DETACH DELETE n")
    graph.add_graph_documents(test_data_backticks)
    nodes = graph.query("MATCH (n) RETURN labels(n) AS labels ORDER BY n.id")
    rels = graph.query("MATCH ()-[r]->() RETURN type(r) AS type")
    expected_nodes = [{"labels": ["bar"]}, {"labels": ["foo"]}]
    expected_rels = [{"type": "REL"}]

    assert nodes == expected_nodes
    assert rels == expected_rels


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_context_manager(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that Neo4jGraph works correctly with context manager."""
    with Neo4jGraph(**neo4j_credentials) as graph:
        # Test that the connection is working
        graph.query("RETURN 1 as n")

    # Test that the connection is closed after exiting context
    try:
        graph.query("RETURN 1 as n")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_explicit_close(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that Neo4jGraph can be explicitly closed."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Test that the connection is working
    graph.query("RETURN 1 as n")

    # Close the connection
    graph.close()

    # Test that the connection is closed
    try:
        graph.query("RETURN 1 as n")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_error_after_close(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that Neo4jGraph operations raise proper errors after closing."""
    graph = Neo4jGraph(**neo4j_credentials)
    graph.query("RETURN 1")  # Should work
    graph.close()

    # Test various operations after close
    try:
        graph.refresh_schema()
        assert (
            False
        ), "Expected RuntimeError when refreshing schema on closed connection"
    except RuntimeError as e:
        assert "connection has been closed" in str(e)

    try:
        graph.query("RETURN 1")
        assert False, "Expected RuntimeError when querying closed connection"
    except RuntimeError as e:
        assert "connection has been closed" in str(e)

    try:
        graph.add_graph_documents([test_data[0]])
        assert False, "Expected RuntimeError when adding documents to closed connection"
    except RuntimeError as e:
        assert "connection has been closed" in str(e)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_concurrent_connections(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that multiple Neo4jGraph instances can be used independently."""
    graph1 = Neo4jGraph(**neo4j_credentials)
    graph2 = Neo4jGraph(**neo4j_credentials)

    # Both connections should work independently
    assert graph1.query("RETURN 1 as n") == [{"n": 1}]
    assert graph2.query("RETURN 2 as n") == [{"n": 2}]

    # Closing one shouldn't affect the other
    graph1.close()
    try:
        graph1.query("RETURN 1")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass
    assert graph2.query("RETURN 2 as n") == [{"n": 2}]

    graph2.close()


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_nested_context_managers(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that nested context managers work correctly."""
    with Neo4jGraph(**neo4j_credentials) as graph1:
        with Neo4jGraph(**neo4j_credentials) as graph2:
            # Both connections should work
            assert graph1.query("RETURN 1 as n") == [{"n": 1}]
            assert graph2.query("RETURN 2 as n") == [{"n": 2}]

        # Inner connection should be closed, outer still works
        try:
            graph2.query("RETURN 2")
            assert False, "Expected RuntimeError when using closed connection"
        except RuntimeError:
            pass
        assert graph1.query("RETURN 1 as n") == [{"n": 1}]

    # Both connections should be closed
    try:
        graph1.query("RETURN 1")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass
    try:
        graph2.query("RETURN 2")
        assert False, "Expected RuntimeError when using closed connection"
    except RuntimeError:
        pass


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_multiple_close(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that Neo4jGraph can be closed multiple times without error."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Test that multiple closes don't raise errors
    graph.close()
    graph.close()  # This should not raise an error


@pytest.mark.usefixtures("clear_neo4j_database")
def test_invalid_url(neo4j_credentials: Neo4jCredentials) -> None:
    """Test initializing with invalid credentials raises ValueError."""
    # Parse the original URL
    parsed_url = urllib.parse.urlparse(neo4j_credentials["url"])
    # Increment the port number by 1 and wrap around if necessary
    original_port = parsed_url.port or 7687
    new_port = (original_port + 1) % 65535 or 1
    # Reconstruct the netloc (hostname:port)
    new_netloc = f"{parsed_url.hostname}:{new_port}"
    # Rebuild the URL with the new netloc
    new_url = parsed_url._replace(netloc=new_netloc).geturl()

    with pytest.raises(ValueError) as exc_info:
        Neo4jGraph(
            url=new_url,
            username=neo4j_credentials["username"],
            password=neo4j_credentials["password"],
        )
    assert "Please ensure that the url is correct" in str(exc_info.value)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_invalid_credentials(neo4j_credentials: Neo4jCredentials) -> None:
    """Test initializing with invalid credentials raises ValueError."""

    with pytest.raises(ValueError) as exc_info:
        Neo4jGraph(
            url=neo4j_credentials["url"],
            username="invalid_username",
            password="invalid_password",
        )
    assert "Please ensure that the credentials are correct" in str(exc_info.value)
