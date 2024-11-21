"""Test Neo4j functionality."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_neo4j.vectorstores.neo4j_vector import (
    IndexType,
    Neo4jVector,
    SearchType,
    _get_search_index_query,
    dict_to_yaml_str,
    remove_lucene_chars,
)


@pytest.fixture
def mock_vector_store() -> Neo4jVector:
    mock_neo4j = MagicMock()
    mock_driver_instance = MagicMock()
    mock_driver_instance.verify_connectivity.return_value = None
    mock_driver_instance.execute_query.return_value = ([], None, None)
    mock_neo4j.GraphDatabase.driver.return_value = mock_driver_instance
    mock_neo4j.exceptions.ServiceUnavailable = Exception
    mock_neo4j.exceptions.AuthError = Exception

    with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
        with patch.object(
            Neo4jVector,
            "query",
            return_value=[{"versions": ["5.23.0"], "edition": "enterprise"}],
        ):
            vector_store = Neo4jVector(
                embedding=MagicMock(),
                url="bolt://localhost:7687",
                username="neo4j",
                password="password",
            )

            vector_store.node_label = "Chunk"
            vector_store.embedding_node_property = "embedding"
            vector_store.text_node_property = "text"

            return vector_store


def test_escaping_lucene() -> None:
    """Test escaping lucene characters"""
    assert remove_lucene_chars("Hello+World") == "Hello World"
    assert remove_lucene_chars("Hello World\\") == "Hello World"
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter!")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter&&")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("Bill&&Melinda Gates Foundation")
        == "Bill  Melinda Gates Foundation"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter(&&)")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter??")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter^")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter+")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter-")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter~")
        == "It is the end of the world. Take shelter"
    )


def test_converting_to_yaml() -> None:
    example_dict = {
        "name": "John Doe",
        "age": 30,
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "location": {"city": "Ljubljana", "country": "Slovenia"},
    }

    yaml_str = dict_to_yaml_str(example_dict)

    expected_output = (
        "name: John Doe\nage: 30\nskills:\n- Python\n- "
        "Data Analysis\n- Machine Learning\nlocation:\n  city: Ljubljana\n"
        "  country: Slovenia\n"
    )

    assert yaml_str == expected_output


def test_get_search_index_query_hybrid_node_neo4j_5_23_above() -> None:
    expected_query = (
        "CALL () { "
        "CALL db.index.vector.queryNodes($index, $k, $embedding) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score UNION "
        "CALL db.index.fulltext.queryNodes($keyword_index, $query, "
        "{limit: $k}) YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $k "
    )

    actual_query = _get_search_index_query(SearchType.HYBRID, IndexType.NODE, True)

    assert actual_query == expected_query


def test_get_search_index_query_hybrid_node_neo4j_5_23_below() -> None:
    expected_query = (
        "CALL { "
        "CALL db.index.vector.queryNodes($index, $k, $embedding) "
        "YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score UNION "
        "CALL db.index.fulltext.queryNodes($keyword_index, $query, "
        "{limit: $k}) YIELD node, score "
        "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
        "UNWIND nodes AS n "
        "RETURN n.node AS node, (n.score / max) AS score "
        "} "
        "WITH node, max(score) AS score ORDER BY score DESC LIMIT $k "
    )

    actual_query = _get_search_index_query(SearchType.HYBRID, IndexType.NODE, False)

    assert actual_query == expected_query


def test_build_import_query_version_is_or_above_5_23(
    mock_vector_store: Neo4jVector,
) -> None:
    mock_vector_store.neo4j_version_is_5_23_or_above = True

    expected_query = (
        "UNWIND $data AS row "
        "CALL (row) { "
        "MERGE (c:`Chunk` {id: row.id}) "
        "WITH c, row "
        "CALL db.create.setNodeVectorProperty(c, "
        "'embedding', row.embedding) "
        "SET c.`text` = row.text "
        "SET c += row.metadata "
        "} IN TRANSACTIONS OF 1000 ROWS "
    )

    actual_query = mock_vector_store._build_import_query()

    assert actual_query == expected_query


def test_build_import_query_version_below_5_23(mock_vector_store: Neo4jVector) -> None:
    mock_vector_store.neo4j_version_is_5_23_or_above = False

    expected_query = (
        "UNWIND $data AS row "
        "CALL { WITH row "
        "MERGE (c:`Chunk` {id: row.id}) "
        "WITH c, row "
        "CALL db.create.setNodeVectorProperty(c, "
        "'embedding', row.embedding) "
        "SET c.`text` = row.text "
        "SET c += row.metadata "
        "} IN TRANSACTIONS OF 1000 ROWS "
    )

    actual_query = mock_vector_store._build_import_query()

    assert actual_query == expected_query


def test_build_delete_query_version_is_or_above_5_23(
    mock_vector_store: Neo4jVector,
) -> None:
    mock_vector_store.neo4j_version_is_5_23_or_above = True
    expected_query = (
        f"MATCH (n:`{mock_vector_store.node_label}`) "
        "CALL (n) { DETACH DELETE n } "
        "IN TRANSACTIONS OF 10000 ROWS;"
    )

    actual_query = mock_vector_store._build_delete_query()

    assert actual_query == expected_query


def test_build_delete_query_version_below_5_23(mock_vector_store: Neo4jVector) -> None:
    mock_vector_store.neo4j_version_is_5_23_or_above = False
    expected_query = (
        f"MATCH (n:`{mock_vector_store.node_label}`) "
        "CALL { WITH n DETACH DELETE n } "
        "IN TRANSACTIONS OF 10000 ROWS;"
    )

    actual_query = mock_vector_store._build_delete_query()

    assert actual_query == expected_query
