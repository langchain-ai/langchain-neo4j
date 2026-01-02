"""Test Neo4j functionality."""

from typing import Any, Optional, Type
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.types import SearchType

from langchain_neo4j.vectorstores.neo4j_vector import (
    Neo4jVector,
    check_if_not_null,
    dict_to_yaml_str,
    remove_lucene_chars,
)
from langchain_neo4j.vectorstores.utils import DistanceStrategy


@pytest.fixture
def mock_vector_store() -> Neo4jVector:
    mock_neo4j = MagicMock()
    mock_driver_instance = MagicMock()
    mock_driver_instance.verify_connectivity.return_value = None
    mock_driver_instance.execute_query.return_value = (
        [{"versions": ["5.23.0"], "edition": "enterprise"}],
        None,
        None,
    )
    mock_neo4j.GraphDatabase.driver.return_value = mock_driver_instance
    mock_neo4j.exceptions.ServiceUnavailable = Exception
    mock_neo4j.exceptions.AuthError = Exception

    with patch(
        "langchain_neo4j.vectorstores.neo4j_vector.neo4j",
        new=mock_neo4j,
    ):
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


@pytest.fixture
def neo4j_vector_factory() -> Any:
    def _create_vector_store(
        method: Optional[str] = None,
        texts: Optional[list[str]] = None,
        text_embeddings: Optional[list[tuple[str, list[float]]]] = None,
        query_return_value: Optional[dict] = None,
        verify_connectivity_side_effect: Optional[Exception] = None,
        auth_error_class: Type[Exception] = Exception,
        service_unavailable_class: Type[Exception] = Exception,
        search_type: SearchType = SearchType.VECTOR,
        **kwargs: Any,
    ) -> Any:
        mock_neo4j = MagicMock()
        mock_driver_instance = MagicMock()

        # Configure verify_connectivity
        if verify_connectivity_side_effect:
            mock_driver_instance.verify_connectivity.side_effect = (
                verify_connectivity_side_effect
            )
        else:
            mock_driver_instance.verify_connectivity.return_value = None

        # Configure execute_query
        if query_return_value is not None:
            mock_driver_instance.execute_query.return_value = (
                [query_return_value],
                None,
                None,
            )
        else:
            mock_driver_instance.execute_query.return_value = (
                [{"versions": ["5.23.0"], "edition": "enterprise"}],
                None,
                None,
            )

        # Assign the mocked driver to GraphDatabase.driver
        mock_neo4j.GraphDatabase.driver.return_value = mock_driver_instance
        mock_neo4j.exceptions.ServiceUnavailable = service_unavailable_class
        mock_neo4j.exceptions.AuthError = auth_error_class

        with patch(
            "langchain_neo4j.vectorstores.neo4j_vector.neo4j",
            new=mock_neo4j,
        ):
            query_return = (
                [query_return_value]
                if query_return_value
                else [{"versions": ["5.23.0"], "edition": "enterprise"}]
            )
            with patch.object(Neo4jVector, "query", return_value=query_return):
                embedding = kwargs.pop("embedding", MagicMock())
                common_kwargs = {
                    "embedding": embedding,
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "search_type": search_type,
                    **kwargs,
                }

                if texts and method == "from_texts":
                    vector_store = Neo4jVector.from_texts(texts=texts, **common_kwargs)
                elif text_embeddings and method == "from_embeddings":
                    vector_store = Neo4jVector.from_embeddings(
                        text_embeddings=text_embeddings, **common_kwargs
                    )
                elif method == "from_existing_index":
                    vector_store = Neo4jVector.from_existing_index(**common_kwargs)
                elif method == "from_existing_relationship_index":
                    vector_store = Neo4jVector.from_existing_relationship_index(
                        **common_kwargs
                    )
                elif method == "from_existing_graph":
                    vector_store = Neo4jVector.from_existing_graph(**common_kwargs)
                else:
                    vector_store = Neo4jVector(**common_kwargs)

                vector_store.node_label = "Chunk"
                vector_store.embedding_node_property = "embedding"
                vector_store.text_node_property = "text"
                return vector_store

    return _create_vector_store


@pytest.mark.parametrize(
    "description, version, is_5_23_or_above",
    [
        ("SemVer, < 5.23", (5, 22, 0), False),
        ("SemVer, >= 5.23", (5, 23, 0), True),
        ("CalVer", (2025, 1, 0), True),
    ],
)
@patch("langchain_neo4j.vectorstores.neo4j_vector.get_version")
def test_versioning_check(
    mock_get_version: MagicMock,
    mock_vector_store: Neo4jVector,
    description: str,
    version: tuple[int, int, int],
    is_5_23_or_above: bool,
) -> None:
    mock_get_version.return_value = version, False, False
    mock_vector_store.verify_version()
    assert (
        mock_vector_store.neo4j_version_is_5_23_or_above is is_5_23_or_above
    ), f"Failed test case: {description}"


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
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter/")
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


def test_check_if_not_null_happy_case() -> None:
    props = ["prop1", "prop2", "prop3"]
    values = ["value1", 123, True]
    check_if_not_null(props, values)


def test_check_if_not_null_with_empty_string() -> None:
    props = ["prop1", "prop2", "prop3"]
    values = ["valid", "valid", ""]

    with pytest.raises(ValueError) as exc_info:
        check_if_not_null(props, values)

    assert "must not be None or empty string" in str(exc_info.value)


def test_check_if_not_null_with_none_value() -> None:
    props = ["prop1", "prop2", "prop3"]
    values = ["valid", None, "valid"]

    with pytest.raises(ValueError) as exc_info:
        check_if_not_null(props, values)

    assert "must not be None or empty string" in str(exc_info.value)


def test_neo4jvector_invalid_distance_strategy() -> None:
    with pytest.raises(ValueError) as exc_info:
        Neo4jVector(
            embedding=MagicMock(),
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",
            distance_strategy="INVALID_STRATEGY",  # type: ignore[arg-type]
        )
    assert "distance_strategy must be either 'EUCLIDEAN_DISTANCE' or 'COSINE'" in str(
        exc_info.value
    )


def test_neo4jvector_service_unavailable() -> None:
    mock_driver_instance = MagicMock()
    mock_driver_instance.verify_connectivity.side_effect = (
        neo4j.exceptions.ServiceUnavailable
    )

    with patch(
        "langchain_neo4j.vectorstores.neo4j_vector.neo4j.GraphDatabase.driver",
        return_value=mock_driver_instance,
    ):
        with pytest.raises(ValueError) as exc_info:
            Neo4jVector(
                embedding=MagicMock(),
                url="bolt://invalid_host:7687",
                username="neo4j",
                password="password",
            )
    assert (
        "Could not connect to Neo4j database. Please ensure that the url is correct"
        in str(exc_info.value)
    )


def test_neo4jvector_auth_error(neo4j_vector_factory: Any) -> None:
    class MockAuthError(Exception):
        pass

    class MockServiceUnavailable(Exception):
        pass

    with pytest.raises(ValueError) as exc_info:
        neo4j_vector_factory(
            verify_connectivity_side_effect=MockAuthError("Authentication Failed"),
            auth_error_class=MockAuthError,
            service_unavailable_class=MockServiceUnavailable,
        )

    assert (
        "Could not connect to Neo4j database. Please ensure that the username "
        "and password are correct" in str(exc_info.value)
    )


def test_neo4jvector_version_with_aura(neo4j_vector_factory: Any) -> None:
    aura_version_response = {"versions": ["5.11.0-aura"], "edition": "enterprise"}
    vector_store = neo4j_vector_factory(query_return_value=aura_version_response)
    assert not vector_store.neo4j_version_is_5_23_or_above


def test_neo4jvector_version_too_low(neo4j_vector_factory: Any) -> None:
    low_version_response = {"versions": ["5.10.0"], "edition": "enterprise"}
    with pytest.raises(ValueError) as exc_info:
        neo4j_vector_factory(query_return_value=low_version_response)
    assert "Vector index is only supported in Neo4j version 5.11 or greater" in str(
        exc_info.value
    )


def test_neo4jvector_metadata_filter_version(neo4j_vector_factory: Any) -> None:
    version_response = {"versions": ["5.17.0"], "edition": "enterprise"}
    vector_store = neo4j_vector_factory(query_return_value=version_response)
    assert vector_store.support_metadata_filter is False


def test_neo4jvector_relationship_index_error(neo4j_vector_factory: Any) -> None:
    texts = ["text1", "text2"]

    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(None, "RELATIONSHIP")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_texts", texts=texts, search_type=SearchType.VECTOR
            )
    assert "Data ingestion is not supported with relationship vector index." in str(
        exc_info.value
    )


def test_neo4jvector_embedding_dimension_mismatch(neo4j_vector_factory: Any) -> None:
    texts = ["text1", "text2"]

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64

    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(128, "NODE")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_texts",
                texts=texts,
                embedding=mock_embedding,
                search_type=SearchType.VECTOR,
            )
    assert (
        "The provided embedding function and vector index dimensions do not match."
        in str(exc_info.value)
    )


def test_neo4jvector_fts_vector_node_label_mismatch(neo4j_vector_factory: Any) -> None:
    texts = ["text1", "text2"]
    embedding_dimension = 64

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * embedding_dimension

    with (
        patch.object(
            Neo4jVector,
            "retrieve_existing_index",
            return_value=(embedding_dimension, "NODE"),
        ),
        patch.object(
            Neo4jVector,
            "retrieve_existing_fts_index",
            return_value="DifferentNodeLabel",
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_texts",
                texts=texts,
                embedding=mock_embedding,
                search_type=SearchType.HYBRID,
                node_label="TestLabel",
                keyword_index_name="keyword_index",
            )
    assert "Vector and keyword index don't index the same node label" in str(
        exc_info.value
    )


def test_similarity_search_by_vector_metadata_filter_unsupported(
    neo4j_vector_factory: Any,
) -> None:
    """
    Test that similarity_search_by_vector raises ValueError when metadata
    filtering is unsupported.
    """
    vector_store = neo4j_vector_factory()
    vector_store.support_metadata_filter = False
    vector_store.search_type = SearchType.VECTOR
    vector_store.embedding_dimension = 64

    with pytest.raises(ValueError) as exc_info:
        vector_store.similarity_search_by_vector(
            embedding=[0] * 64,
            filter={"field": "value"},
        )
    assert (
        "Metadata filtering is only supported in Neo4j version 5.18 or greater"
        in str(exc_info.value)
    )


def test_similarity_search_by_vector_metadata_filter_hybrid(
    neo4j_vector_factory: Any,
) -> None:
    vector_store = neo4j_vector_factory()

    vector_store.support_metadata_filter = True
    vector_store.search_type = SearchType.HYBRID
    vector_store.embedding_dimension = 64

    with pytest.raises(Exception) as exc_info:
        vector_store.similarity_search_by_vector(
            embedding=[0] * 64,
            filter={"field": "value"},
        )
    assert "Filters are not supported with hybrid search" in str(exc_info.value)


def test_from_existing_index_relationship_index_error(
    neo4j_vector_factory: Any,
) -> None:
    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(64, "RELATIONSHIP")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_index",
                index_name="test_index",
                search_type=SearchType.VECTOR,
            )
    assert (
        "Relationship vector index is not supported with `from_existing_index` "
        "method." in str(exc_info.value)
    )


def test_from_existing_index_index_not_found(neo4j_vector_factory: Any) -> None:
    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(None, None)
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_index",
                embedding=MagicMock(),
                index_name="non_existent_index",
            )
    assert "The specified vector index name does not exist." in str(exc_info.value)


def test_from_existing_index_fts_vector_node_label_mismatch(
    neo4j_vector_factory: Any,
) -> None:
    embedding_dimension = 64

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * embedding_dimension

    with (
        patch.object(
            Neo4jVector,
            "retrieve_existing_index",
            return_value=(embedding_dimension, "NODE"),
        ),
        patch.object(
            Neo4jVector,
            "retrieve_existing_fts_index",
            return_value="DifferentNodeLabel",
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_index",
                embedding=mock_embedding,
                index_name="test_index",
                search_type=SearchType.HYBRID,
                keyword_index_name="keyword_index",
            )

    assert "Vector and keyword index don't index the same node label" in str(
        exc_info.value
    )


def test_from_existing_relationship_index_hybrid_not_supported() -> None:
    with pytest.raises(ValueError) as exc_info:
        Neo4jVector.from_existing_relationship_index(
            embedding=MagicMock(),
            index_name="test_index",
            search_type=SearchType.HYBRID,
        )
    assert (
        "Hybrid search is not supported in combination with relationship vector index"
        in str(exc_info.value)
    )


def test_from_existing_relationship_index_index_not_found(
    neo4j_vector_factory: Any,
) -> None:
    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(None, None)
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_relationship_index",
                index_name="non_existent_index",
            )
    assert "The specified vector index name does not exist" in str(exc_info.value)


def test_from_existing_relationship_index_node_index_error() -> None:
    with patch.object(Neo4jVector, "__init__", return_value=None):
        with patch.object(
            Neo4jVector, "retrieve_existing_index", return_value=(64, "NODE")
        ):
            with pytest.raises(ValueError) as exc_info:
                Neo4jVector.from_existing_relationship_index(
                    embedding=MagicMock(),
                    index_name="test_index",
                )
            assert (
                "Node vector index is not supported with "
                "`from_existing_relationship_index` method" in str(exc_info.value)
            )


def test_from_existing_relationship_index_embedding_dimension_mismatch(
    neo4j_vector_factory: Any,
) -> None:
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64
    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(128, "RELATIONSHIP")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_relationship_index",
                embedding=mock_embedding,
                index_name="test_index",
                search_type=SearchType.VECTOR,
            )

    assert (
        "The provided embedding function and vector index dimensions do not match"
        in str(exc_info.value)
    )


def test_from_existing_graph_empty_text_node_properties() -> None:
    with pytest.raises(ValueError) as exc_info:
        Neo4jVector.from_existing_graph(
            embedding=MagicMock(),
            node_label="TestLabel",
            embedding_node_property="embedding",
            text_node_properties=[],
        )
    assert "Parameter `text_node_properties` must not be an empty list" in str(
        exc_info.value
    )


def test_from_existing_graph_relationship_index_error(
    neo4j_vector_factory: Any,
) -> None:
    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(64, "RELATIONSHIP")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_graph",
                embedding=MagicMock(),
                node_label="TestLabel",
                embedding_node_property="embedding",
                text_node_properties=["text_property"],
                search_type=SearchType.HYBRID,
                keyword_index_name="keyword_index",
            )

        assert (
            "`from_existing_graph` method does not support  existing relationship "
            "vector index. Please use `from_existing_relationship_index` method"
            in str(exc_info.value)
        )


def test_from_existing_graph_embedding_dimension_mismatch(
    neo4j_vector_factory: Any,
) -> None:
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64

    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(128, "NODE")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_graph",
                embedding=mock_embedding,
                node_label="TestLabel",
                embedding_node_property="embedding",
                text_node_properties=["text_property"],
                search_type=SearchType.VECTOR,
            )

    assert (
        "The provided embedding function and vector index dimensions do not match"
        in str(exc_info.value)
    )


def test_from_existing_graph_fts_vector_node_label_mismatch(
    neo4j_vector_factory: Any,
) -> None:
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64
    with (
        patch.object(Neo4jVector, "retrieve_existing_index", return_value=(64, "NODE")),
        patch.object(
            Neo4jVector,
            "retrieve_existing_fts_index",
            return_value="DifferentNodeLabel",
        ),
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_graph",
                embedding=mock_embedding,
                node_label="TestLabel",
                embedding_node_property="embedding",
                text_node_properties=["text_property"],
                search_type=SearchType.HYBRID,
                keyword_index_name="keyword_index",
            )

    assert "Vector and keyword index don't index the same node label" in str(
        exc_info.value
    )


def test_select_relevance_score_fn_override(neo4j_vector_factory: Any) -> None:
    def override_fn(x: int) -> int:
        return x * 2

    vector_store = neo4j_vector_factory(
        embedding=MagicMock(),
        search_type=SearchType.VECTOR,
        relevance_score_fn=override_fn,
    )
    fn = vector_store._select_relevance_score_fn()

    assert fn(2) == 4


def test_select_relevance_score_fn_invalid_distance_strategy(
    neo4j_vector_factory: Any,
) -> None:
    vector_store = neo4j_vector_factory(
        embedding=MagicMock(), search_type=SearchType.VECTOR
    )
    vector_store._distance_strategy = "INVALID_STRATEGY"

    with pytest.raises(ValueError) as exc_info:
        vector_store._select_relevance_score_fn()

    assert (
        "No supported normalization function for distance_strategy of INVALID_STRATEGY"
        in str(exc_info.value)
    )


def test_select_relevance_score_fn_euclidean_distance(
    neo4j_vector_factory: Any,
) -> None:
    vector_store = neo4j_vector_factory(
        embedding=MagicMock(), distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    assert vector_store._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE


def test_select_relevance_score_fn_cosine(neo4j_vector_factory: Any) -> None:
    vector_store = neo4j_vector_factory(
        embedding=MagicMock(), distance_strategy=DistanceStrategy.COSINE
    )

    assert vector_store._distance_strategy == DistanceStrategy.COSINE


def test_from_existing_index_keyword_index_not_exist(neo4j_vector_factory: Any) -> None:
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64

    with (
        patch.object(Neo4jVector, "retrieve_existing_index", return_value=(64, "NODE")),
        patch.object(Neo4jVector, "retrieve_existing_fts_index", return_value=None),
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_index",
                embedding=mock_embedding,
                index_name="vector_index",
                search_type=SearchType.HYBRID,
                keyword_index_name="nonexistent_keyword_index",
            )
    expected_message = (
        "The specified keyword index name does not exist. "
        "Make sure to check if you spelled it correctly"
    )
    assert expected_message in str(exc_info.value)


def test_select_relevance_score_fn_unsupported_strategy(
    neo4j_vector_factory: Any,
) -> None:
    vector_store = neo4j_vector_factory(
        embedding=MagicMock(), distance_strategy=DistanceStrategy.COSINE
    )

    vector_store._distance_strategy = "UNSUPPORTED_STRATEGY"

    with pytest.raises(ValueError) as exc_info:
        vector_store._select_relevance_score_fn()

    expected_message = (
        "No supported normalization function for distance_strategy "
        "of UNSUPPORTED_STRATEGY."
        "Consider providing relevance_score_fn to PGVector constructor."
    )

    assert expected_message in str(exc_info.value), (
        f"Expected error message to contain '{expected_message}' "
        f"but got '{str(exc_info.value)}'"
    )


def test_embedding_dimension_inconsistent_raises_value_error(
    neo4j_vector_factory: Any,
) -> None:
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64

    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(128, "NODE")
    ):
        with pytest.raises(ValueError) as exc_info:
            neo4j_vector_factory(
                method="from_existing_index",
                embedding=mock_embedding,
                index_name="test_index",
            )
    assert (
        "The provided embedding function and vector index dimensions do not match."
        in str(exc_info.value)
    )


def test_similarity_search_with_missing_metadata(
    neo4j_vector_factory: Any,
) -> None:
    """Test that search works when both metadata and score are missing."""
    vector_store = neo4j_vector_factory()
    vector_store.support_metadata_filter = True
    vector_store.search_type = SearchType.VECTOR
    vector_store.embedding_dimension = 64
    vector_store.retrieval_query = "RETURN node.text AS text"

    mock_results = [
        {"text": "test content 1", "score": 0.95},
        {"text": "test content 2", "score": 0.85},
    ]

    with patch.object(Neo4jVector, "query", return_value=mock_results):
        docs = vector_store.similarity_search_with_score_by_vector(
            embedding=[0.1] * 64,
            k=2,
            query="test query",
        )

    assert len(docs) == 2
    assert docs[0][0].page_content == "test content 1"
    assert docs[0][0].metadata == {}
    assert docs[0][1] == 0.95
    assert docs[1][0].page_content == "test content 2"
    assert docs[1][0].metadata == {}
    assert docs[1][1] == 0.85


def test_similarity_search_with_all_fields_present(neo4j_vector_factory: Any) -> None:
    """Test that search works normally when all fields are present."""
    vector_store = neo4j_vector_factory()
    vector_store.support_metadata_filter = True
    vector_store.search_type = SearchType.VECTOR
    vector_store.embedding_dimension = 64
    vector_store.retrieval_query = (
        "RETURN node.text AS text, node {.*} AS metadata, score"
    )

    mock_results = [
        {"text": "test content 1", "metadata": {"key": "value1"}, "score": 0.95},
        {"text": "test content 2", "metadata": {"key": "value2"}, "score": 0.85},
    ]

    with patch.object(Neo4jVector, "query", return_value=mock_results):
        docs = vector_store.similarity_search_with_score_by_vector(
            embedding=[0.1] * 64,
            k=2,
            query="test query",
        )

    assert len(docs) == 2
    assert docs[0][0].page_content == "test content 1"
    assert docs[0][0].metadata == {"key": "value1"}
    assert docs[0][1] == 0.95
    assert docs[1][0].page_content == "test content 2"
    assert docs[1][0].metadata == {"key": "value2"}
    assert docs[1][1] == 0.85


def test_from_existing_index_with_text_node_properties(
    neo4j_vector_factory: Any,
) -> None:
    """Test that from_existing_index correctly constructs retrieval query
    when text_node_properties is provided, using embedding_node_property
    retrieved from the existing index."""
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64

    def mock_retrieve_existing_index(self: Neo4jVector) -> tuple:
        # Simulate what the real method does: set properties from the index
        self.embedding_node_property = "custom_embedding_prop"
        self.node_label = "TestNode"
        return (64, "NODE")

    with patch.object(
        Neo4jVector, "retrieve_existing_index", mock_retrieve_existing_index
    ):
        vector_store = neo4j_vector_factory(
            method="from_existing_index",
            embedding=mock_embedding,
            index_name="test_index",
            text_node_properties=["name", "description", "status"],
        )

    # Verify the retrieval query contains the concatenated text properties
    assert vector_store.retrieval_query is not None
    assert "['name', 'description', 'status']" in vector_store.retrieval_query
    assert "reduce(str='', k IN" in vector_store.retrieval_query
    assert "`name`: Null" in vector_store.retrieval_query
    assert "`description`: Null" in vector_store.retrieval_query
    assert "`status`: Null" in vector_store.retrieval_query
    # Verify that embedding_node_property from the index is used in the query
    assert "`custom_embedding_prop`: Null" in vector_store.retrieval_query


def test_from_existing_index_without_text_node_properties(
    neo4j_vector_factory: Any,
) -> None:
    """Test that from_existing_index works without text_node_properties
    (backward compatibility)."""
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 64

    with patch.object(
        Neo4jVector, "retrieve_existing_index", return_value=(64, "NODE")
    ):
        vector_store = neo4j_vector_factory(
            method="from_existing_index",
            embedding=mock_embedding,
            index_name="test_index",
        )

    # Without text_node_properties, retrieval_query should be empty (uses default)
    assert vector_store.retrieval_query == ""
