"""Test Neo4jVector functionality."""

import os
from typing import Any, Dict, List, cast

import pytest
from langchain_core.documents import Document
from neo4j_graphrag.neo4j_queries import _get_hybrid_query
from neo4j_graphrag.types import SearchType
from yaml import safe_load

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_neo4j.vectorstores.utils import DistanceStrategy
from tests.integration_tests.utils import Neo4jCredentials
from tests.integration_tests.vectorstores.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    FakeEmbeddings,
)
from tests.integration_tests.vectorstores.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
)

OS_TOKEN_COUNT = 1536

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!"]

"""
cd tests/integration_tests/docker-compose
docker-compose -f neo4j.yml up
"""


def drop_vector_indexes(store: Neo4jVector) -> None:
    """Cleanup all vector indexes"""
    all_indexes = store.query(
        """
            SHOW INDEXES YIELD name, type
            WHERE type IN ["VECTOR", "FULLTEXT"]
            RETURN name
                              """
    )
    for index in all_indexes:
        store.query(f"DROP INDEX `{index['name']}`")

    store.query("MATCH (n) DETACH DELETE n;")


class FakeEmbeddingsWithOsDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        embedding = [
            [float(1.0)] * (OS_TOKEN_COUNT - 1) + [100 * float(i + 1)]
            for i in range(len(embedding_texts))
        ]
        return embedding

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        embedding = [float(1.0)] * (OS_TOKEN_COUNT - 1) + [
            100 * float(texts.index(text) + 1)
        ]
        return embedding


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector() -> None:
    """Test end to end construction and search with environment variable credentials."""
    assert os.environ.get("NEO4J_URI") is not None
    assert os.environ.get("NEO4J_USERNAME") is not None
    assert os.environ.get("NEO4J_PASSWORD") is not None
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_euclidean(neo4j_credentials: Neo4jCredentials) -> None:
    """Test euclidean distance"""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_embeddings(neo4j_credentials: Neo4jCredentials) -> None:
    """Test end to end construction with embeddings and search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_catch_wrong_index_name(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test if index name is misspelled, but node label and property are correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="test",
        **neo4j_credentials,
    )
    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_catch_wrong_node_label(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test if node label is misspelled, but index name is correct."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="vector",
        node_label="test",
        **neo4j_credentials,
    )
    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_with_metadatas(neo4j_credentials: Neo4jCredentials) -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_with_metadatas_with_scores(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test end to end construction and search."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    output = [
        (doc, round(score, 1))
        for doc, score in docsearch.similarity_search_with_score("foo", k=1)
    ]
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_relevance_score(neo4j_credentials: Neo4jCredentials) -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    output_texts = [doc.page_content for doc, _ in output]

    expected_order = ["foo", "It is the end of the world. Take shelter!", "baz"]
    assert output_texts == expected_order
    relevance_scores = [score for _, score in output]
    assert all(
        earlier >= later
        for earlier, later in zip(relevance_scores, relevance_scores[1:])
    )

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_retriever_search_threshold(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test using retriever for searching with threshold."""
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.invoke("foo")

    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
    ]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_custom_return_neo4jvector(neo4j_credentials: Neo4jCredentials) -> None:
    """Test end to end construction and search."""
    docsearch = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        retrieval_query="RETURN 'foo' AS text, score, {test: 'test'} AS metadata",
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_prefer_indexname(neo4j_credentials: Neo4jCredentials) -> None:
    """Test using when two indexes are found, prefer by index_name."""
    Neo4jVector.from_texts(
        texts=["foo"],
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    Neo4jVector.from_texts(
        texts=["bar"],
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        node_label="Test",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    existing_index = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        text_node_property="info",
        **neo4j_credentials,
    )

    output = existing_index.similarity_search("bar", k=1)
    assert output == [Document(page_content="bar", metadata={})]
    drop_vector_indexes(existing_index)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_prefer_indexname_insert(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test using when two indexes are found, prefer by index_name."""
    Neo4jVector.from_texts(
        texts=["baz"],
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    Neo4jVector.from_texts(
        texts=["foo"],
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        node_label="Test",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    existing_index = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        text_node_property="info",
        **neo4j_credentials,
    )

    existing_index.add_documents([Document(page_content="bar", metadata={})])

    output = existing_index.similarity_search("bar", k=2)
    assert output == [
        Document(page_content="bar", metadata={}),
        Document(page_content="foo", metadata={}),
    ]
    drop_vector_indexes(existing_index)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_hybrid(neo4j_credentials: Neo4jCredentials) -> None:
    """Test end to end construction with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_hybrid_deduplicate(neo4j_credentials: Neo4jCredentials) -> None:
    """Test result deduplication with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=3)

    assert output == [
        Document(page_content="foo"),
        Document(page_content="It is the end of the world. Take shelter!"),
        Document(page_content="baz"),
    ]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_hybrid_retrieval_query(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN 'moo' AS text, score, {test: 'test'} AS metadata",
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="moo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_hybrid_retrieval_query2(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test custom retrieval_query with hybrid search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        retrieval_query="RETURN node.text AS text, score, {test: 'test'} AS metadata",
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"test": "test"})]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_missing_keyword(neo4j_credentials: Neo4jCredentials) -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    try:
        Neo4jVector.from_existing_index(
            embedding=FakeEmbeddingsWithOsDimension(),
            index_name="vector",
            search_type=SearchType.HYBRID,
            **neo4j_credentials,
        )
    except ValueError as e:
        assert str(e) == (
            "keyword_index name has to be specified when " "using hybrid search option"
        )
    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_hybrid_from_existing(neo4j_credentials: Neo4jCredentials) -> None:
    """Test hybrid search with missing keyword_index_search."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )
    existing = Neo4jVector.from_existing_index(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="vector",
        keyword_index_name="keyword",
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_from_existing_graph(neo4j_credentials: Neo4jCredentials) -> None:
    """Test from_existing_graph with a single property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'Foo'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="vector",
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
        **neo4j_credentials,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: Foo")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_from_existing_graph_hybrid(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test from_existing_graph hybrid with a single property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'foo'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="vector",
        node_label="Test",
        text_node_properties=["name"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: foo")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_from_existing_graph_multiple_properties(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test from_existing_graph with a two property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'Foo', name2: 'Fooz'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="vector",
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
        **neo4j_credentials,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_from_existing_graph_multiple_properties_hybrid(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test from_existing_graph with a two property."""
    graph = Neo4jVector.from_texts(
        texts=["test"],
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="foo",
        node_label="Foo",
        embedding_node_property="vector",
        text_node_property="info",
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    graph.query("MATCH (n) DETACH DELETE n")

    graph.query("CREATE (:Test {name:'Foo', name2: 'Fooz'})," "(:Test {name:'Bar'})")

    existing = Neo4jVector.from_existing_graph(
        embedding=FakeEmbeddingsWithOsDimension(),
        index_name="vector",
        node_label="Test",
        text_node_properties=["name", "name2"],
        embedding_node_property="embedding",
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )

    output = existing.similarity_search("foo", k=1)
    assert output == [Document(page_content="\nname: Foo\nname2: Fooz")]

    drop_vector_indexes(existing)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_special_character(neo4j_credentials: Neo4jCredentials) -> None:
    """Test removing lucene."""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search(
        "It is the end of the world. Take shelter!",
        k=1,
    )

    assert output == [
        Document(page_content="It is the end of the world. Take shelter!", metadata={})
    ]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_hybrid_score_normalization(neo4j_credentials: Neo4jCredentials) -> None:
    """Test if we can get two 1.0 documents with RRF"""
    text_embeddings = FakeEmbeddingsWithOsDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(["foo"], text_embeddings))
    docsearch = Neo4jVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        search_type=SearchType.HYBRID,
        **neo4j_credentials,
    )
    # Remove deduplication part of the query
    rrf_query = (
        _get_hybrid_query(neo4j_version_is_5_23_or_above=False)
        .rstrip("WITH node, max(score) AS score ORDER BY score DESC LIMIT $top_k")
        .replace("UNION", "UNION ALL")
        + "RETURN node.text AS text, score LIMIT 2"
    )

    output = docsearch.query(
        rrf_query,
        params={
            "vector_index_name": "vector",
            "top_k": 1,
            "query_vector": FakeEmbeddingsWithOsDimension().embed_query("foo"),
            "effective_search_ratio": 1,
            "query_text": "foo",
            "fulltext_index_name": "keyword",
        },
    )
    # Both FT and Vector must return 1.0 score
    assert output == [{"text": "foo", "score": 1.0}, {"text": "foo", "score": 1.0}]
    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_index_fetching(neo4j_credentials: Neo4jCredentials) -> None:
    """testing correct index creation and fetching"""
    embeddings = FakeEmbeddings()

    def create_store(
        node_label: str, index: str, text_properties: List[str]
    ) -> Neo4jVector:
        return Neo4jVector.from_existing_graph(
            embedding=embeddings,
            index_name=index,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property="embedding",
            **neo4j_credentials,
        )

    def fetch_store(index_name: str) -> Neo4jVector:
        store = Neo4jVector.from_existing_index(
            embedding=embeddings,
            index_name=index_name,
            **neo4j_credentials,
        )
        return store

    # create index 0
    index_0_str = "index0"
    create_store("label0", index_0_str, ["text"])

    # create index 1
    index_1_str = "index1"
    create_store("label1", index_1_str, ["text"])

    index_1_store = fetch_store(index_1_str)
    assert index_1_store.index_name == index_1_str

    index_0_store = fetch_store(index_0_str)
    assert index_0_store.index_name == index_0_str
    drop_vector_indexes(index_1_store)
    drop_vector_indexes(index_0_store)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_retrieval_params(neo4j_credentials: Neo4jCredentials) -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        retrieval_query="""
        RETURN $test as text, score, {test: $test1} AS metadata
        """,
        **neo4j_credentials,
    )

    output = docsearch.similarity_search(
        "Foo", k=2, params={"test": "test", "test1": "test1"}
    )
    assert output == [
        Document(page_content="test", metadata={"test": "test1"}),
        Document(page_content="test", metadata={"test": "test1"}),
    ]
    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_retrieval_dictionary(neo4j_credentials: Neo4jCredentials) -> None:
    """Test if we use parameters in retrieval query"""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        retrieval_query="""
        RETURN {
            name:'John', 
            age: 30,
            skills: ["Python", "Data Analysis", "Machine Learning"]} as text, 
            score, {} AS metadata
        """,
        **neo4j_credentials,
    )
    expected_output = [
        Document(
            page_content=(
                "skills:\n- Python\n- Data Analysis\n- "
                "Machine Learning\nage: 30\nname: John\n"
            )
        )
    ]

    output = docsearch.similarity_search("Foo", k=1)

    def parse_document(doc: Document) -> Any:
        return safe_load(doc.page_content)

    parsed_expected = [parse_document(doc) for doc in expected_output]
    parsed_output = [parse_document(doc) for doc in output]

    assert parsed_output == parsed_expected
    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_metadata_filters_type1(neo4j_credentials: Neo4jCredentials) -> None:
    """Test metadata filters"""
    docsearch = Neo4jVector.from_documents(
        DOCUMENTS,
        embedding=FakeEmbeddings(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    # We don't test type 5, because LIKE has very SQL specific examples
    for example in (
        TYPE_1_FILTERING_TEST_CASES
        + TYPE_2_FILTERING_TEST_CASES
        + TYPE_3_FILTERING_TEST_CASES
        + TYPE_4_FILTERING_TEST_CASES
    ):
        filter_dict = cast(Dict[str, Any], example[0])
        output = docsearch.similarity_search("Foo", filter=filter_dict)
        indices = cast(List[int], example[1])
        adjusted_indices = [index - 1 for index in indices]
        expected_output = [DOCUMENTS[index] for index in adjusted_indices]
        # We don't return id properties from similarity search by default
        # Also remove any key where the value is None
        for doc in expected_output:
            if "id" in doc.metadata:
                del doc.metadata["id"]
            keys_with_none = [
                key for key, value in doc.metadata.items() if value is None
            ]
            for key in keys_with_none:
                del doc.metadata[key]

        assert output == expected_output
    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_relationship_index(neo4j_credentials: Neo4jCredentials) -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=embeddings,
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    # Ingest data
    docsearch.query(
        (
            "CREATE ()-[:REL {text: 'foo', embedding: $e1}]->()"
            ", ()-[:REL {text: 'far', embedding: $e2}]->()"
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.query(
        """CREATE VECTOR INDEX `relationship`
FOR ()-[r:REL]-() ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
"""
    )
    relationship_index = Neo4jVector.from_existing_relationship_index(
        embeddings,
        index_name="relationship",
        **neo4j_credentials,
    )

    output = relationship_index.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_relationship_index_retrieval(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test end to end construction and search."""
    embeddings = FakeEmbeddingsWithOsDimension()
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=embeddings,
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    # Ingest data
    docsearch.query(
        (
            "CREATE ({node:'text'})-[:REL {text: 'foo', embedding: $e1}]->()"
            ", ({node:'text'})-[:REL {text: 'far', embedding: $e2}]->()"
        ),
        params={
            "e1": embeddings.embed_query("foo"),
            "e2": embeddings.embed_query("bar"),
        },
    )
    # Create relationship index
    docsearch.query(
        """CREATE VECTOR INDEX `relationship`
FOR ()-[r:REL]-() ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
"""
    )
    retrieval_query = (
        "RETURN relationship.text + '-' + startNode(relationship).node "
        "AS text, score, {foo:'bar'} AS metadata"
    )
    relationship_index = Neo4jVector.from_existing_relationship_index(
        embeddings,
        index_name="relationship",
        retrieval_query=retrieval_query,
        **neo4j_credentials,
    )

    output = relationship_index.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo-text", metadata={"foo": "bar"})]

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_max_marginal_relevance_search(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    texts = ["-0.124", "+0.127", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Neo4jVector.from_texts(
        texts,
        metadatas=metadatas,
        embedding=AngularTwoDimensionalEmbeddings(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )

    expected_set = {
        ("+0.25", 2),
        ("-0.124", 0),
    }

    output = docsearch.max_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == expected_set

    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_effective_search_ratio(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    """Test effective search parameter."""
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=2, effective_search_ratio=2)
    assert len(output) == 2

    output1 = docsearch.similarity_search_with_score(
        "foo", k=2, effective_search_ratio=2
    )
    assert len(output1) == 2
    # Assert ordered by score
    assert output1[0][1] > output1[1][1]
    drop_vector_indexes(docsearch)


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4jvector_passing_graph_object(neo4j_credentials: Neo4jCredentials) -> None:
    """Test end to end construction and search with passing graph object."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Rewrite env vars to make sure it fails if env is used
    old_url = os.environ["NEO4J_URI"]
    os.environ["NEO4J_URI"] = "foo"
    docsearch = Neo4jVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        graph=graph,
        pre_delete_collection=True,
        **neo4j_credentials,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    drop_vector_indexes(docsearch)
    os.environ["NEO4J_URI"] = old_url
