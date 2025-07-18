from __future__ import annotations

import logging
import os
from hashlib import md5
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
)

import neo4j
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from neo4j_graphrag.indexes import (
    create_fulltext_index,
    create_vector_index,
    retrieve_fulltext_index_info,
    retrieve_vector_index_info,
)
from neo4j_graphrag.neo4j_queries import get_search_query
from neo4j_graphrag.types import EntityType as IndexType
from neo4j_graphrag.types import SearchType
from neo4j_graphrag.utils.version_utils import (
    get_version,
    has_metadata_filtering_support,
    has_vector_index_support,
    is_version_5_23_or_above,
)

from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.utils import DistanceStrategy

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING: Final[dict[DistanceStrategy, Literal["euclidean", "cosine"]]] = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "euclidean",
    DistanceStrategy.COSINE: "cosine",
}
DEFAULT_SEARCH_TYPE = SearchType.VECTOR
DEFAULT_INDEX_TYPE = IndexType.NODE


def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if the values are not None or empty string"""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")


def remove_lucene_chars(text: str) -> str:
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
        "/",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


def dict_to_yaml_str(input_dict: Dict, indent: int = 0) -> str:
    """
    Convert a dictionary to a YAML-like string without using external libraries.

    Parameters:
    - input_dict (dict): The dictionary to convert.
    - indent (int): The current indentation level.

    Returns:
    - str: The YAML-like string representation of the input dictionary.
    """
    yaml_str = ""
    for key, value in input_dict.items():
        padding = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{padding}{key}:\n{dict_to_yaml_str(value, indent + 1)}"
        elif isinstance(value, list):
            yaml_str += f"{padding}{key}:\n"
            for item in value:
                yaml_str += f"{padding}- {item}\n"
        else:
            yaml_str += f"{padding}{key}: {value}\n"
    return yaml_str


class Neo4jVector(VectorStore):
    """`Neo4j` vector index.

    To use, you should have the ``neo4j`` python package installed.

    Args:
        url: Neo4j connection url
        username: Neo4j username.
        password: Neo4j password
        database: Optionally provide Neo4j database
                  Defaults to "neo4j"
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        distance_strategy: The distance strategy to use. (default: COSINE)
        search_type: The type of search to be performed, either
            'vector' or 'hybrid'
        node_label: The label used for nodes in the Neo4j database.
            (default: "Chunk")
        embedding_node_property: The property name in Neo4j to store embeddings.
            (default: "embedding")
        text_node_property: The property name in Neo4j to store the text.
            (default: "text")
        retrieval_query: The Cypher query to be used for customizing retrieval.
            If empty, a default query will be used.
        index_type: The type of index to be used, either
            'NODE' or 'RELATIONSHIP'
        pre_delete_collection: If True, will delete existing data if it exists.
            (default: False). Useful for testing.
        effective_search_ratio: Controls the candidate pool size by multiplying $k
            to balance query accuracy and performance.
        embedding_dimension: The dimension of the embeddings. If not provided,
            will query the embedding model to calculate the dimension.

    Example:
        .. code-block:: python

            from langchain_neo4j import Neo4jVector
            from langchain_openai import OpenAIEmbeddings

            url="bolt://localhost:7687"
            username="neo4j"
            password="password"
            embeddings = OpenAIEmbeddings()
            vectorestore = Neo4jVector.from_documents(
                embedding=embeddings,
                documents=docs,
                url=url
                username=username,
                password=password,
            )


    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        search_type: SearchType = SearchType.VECTOR,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        keyword_index_name: Optional[str] = "keyword",
        database: Optional[str] = None,
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        logger: Optional[logging.Logger] = None,
        pre_delete_collection: bool = False,
        retrieval_query: str = "",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        index_type: IndexType = DEFAULT_INDEX_TYPE,
        graph: Optional[Neo4jGraph] = None,
        embedding_dimension: Optional[int] = None,
    ) -> None:
        # Allow only cosine and euclidean distance strategies
        if distance_strategy not in [
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.COSINE,
        ]:
            raise ValueError(
                "distance_strategy must be either 'EUCLIDEAN_DISTANCE' or 'COSINE'"
            )

        # Graph object takes precedent over env or input params
        if graph:
            self._driver = graph._driver
            self._database = graph._database
        else:
            # Handle if the credentials are environment variables
            # Support URL for backwards compatibility
            if not url:
                url = os.environ.get("NEO4J_URL")

            url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
            username = get_from_dict_or_env(
                {"username": username}, "username", "NEO4J_USERNAME"
            )
            password = get_from_dict_or_env(
                {"password": password}, "password", "NEO4J_PASSWORD"
            )
            database = get_from_dict_or_env(
                {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
            )

            self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
            self._database = database
            # Verify connection
            try:
                self._driver.verify_connectivity()
            except neo4j.exceptions.ServiceUnavailable:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the url is correct"
                )
            except neo4j.exceptions.AuthError:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the username and password are correct"
                )

        self.schema = ""
        # Verify if the version support vector index
        self._is_enterprise = False
        self.verify_version()

        # Verify that required values are not null
        check_if_not_null(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        )

        self.embedding = embedding
        self._distance_strategy = distance_strategy
        self.index_name = index_name
        self.keyword_index_name = keyword_index_name
        self.node_label = node_label
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.retrieval_query = retrieval_query
        self.search_type = search_type
        self._index_type = index_type

        if embedding_dimension:
            self.embedding_dimension = embedding_dimension
        else:
            # Calculate embedding dimension
            self.embedding_dimension = len(embedding.embed_query("foo"))

        # Delete existing data if flagged
        if pre_delete_collection:
            from neo4j.exceptions import DatabaseError

            delete_query = self._build_delete_query()
            self.query(delete_query)
            # Delete index
            try:
                self.query(f"DROP INDEX {self.index_name}")
            except DatabaseError:  # Index didn't exist yet
                pass

    def _build_delete_query(self) -> str:
        if self.neo4j_version_is_5_23_or_above:
            call_prefix = "CALL (n) {"
        else:
            call_prefix = "CALL { WITH n"
        return (
            f"MATCH (n:`{self.node_label}`) "
            f"{call_prefix} DETACH DELETE n "
            "} IN TRANSACTIONS OF 10000 ROWS;"
        )

    def query(
        self,
        query: str,
        *,
        params: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database with retries and exponential backoff.

        Args:
            query (str): The Cypher query to execute.
            params (dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.
        """
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        params = params or {}
        try:
            data, _, _ = self._driver.execute_query(
                query, database_=self._database, parameters_=params
            )
            return [r.data() for r in data]
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and e.message is not None
                    and "in an implicit transaction" in e.message
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and e.message is not None
                    and (
                        "in an open transaction is not possible" in e.message
                        or "tried to execute in an explicit transaction" in e.message
                    )
                )
            ):
                raise
        # Fallback to allow implicit transactions
        with self._driver.session(database=self._database) as session:
            result = session.run(Query(text=query), params)
            return [r.data() for r in result]

    def verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        version_tuple, is_aura, is_enterprise = get_version(
            self._driver, self._database
        )
        self._is_enterprise = is_enterprise
        self.neo4j_version_is_5_23_or_above = is_version_5_23_or_above(version_tuple)
        if not has_vector_index_support(version_tuple):
            raise ValueError(
                "Vector index is only supported in Neo4j version 5.11 or greater"
            )
        self.support_metadata_filter = has_metadata_filtering_support(
            version_tuple, is_aura
        )

    def retrieve_existing_index(self) -> Optional[Tuple[Optional[int], str]]:
        """
        Check if the vector index exists in the Neo4j database
        and returns its embedding dimension.

        This method queries the Neo4j database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """
        index_information = retrieve_vector_index_info(
            driver=self._driver,
            index_name=self.index_name,
            label_or_type=self.node_label,
            embedding_property=self.embedding_node_property,
            neo4j_database=self._database,
        )
        if index_information:
            try:
                self.index_name = index_information["name"]
                self.node_label = index_information["labelsOrTypes"][0]
                self.embedding_node_property = index_information["properties"][0]
                self._index_type = index_information["entityType"]
                embedding_dimension = None
                index_config = index_information["options"]["indexConfig"]
                if "vector.dimensions" in index_config:
                    embedding_dimension = index_config["vector.dimensions"]
                return embedding_dimension, index_information["entityType"]
            except IndexError:
                return None
        else:
            return None

    def retrieve_existing_fts_index(
        self, text_node_properties: List[str] = []
    ) -> Optional[str]:
        """
        Check if the fulltext index exists in the Neo4j database

        This method queries the Neo4j database for existing fts indexes
        with the specified name.

        Returns:
            (Tuple): keyword index information
        """
        if self.keyword_index_name:
            index_information = retrieve_fulltext_index_info(
                driver=self._driver,
                index_name=self.keyword_index_name,
                label_or_type=self.node_label,
                text_properties=text_node_properties or [self.text_node_property],
                neo4j_database=self._database,
            )
        else:
            raise ValueError("keyword_index_name is not set.")
        if index_information:
            try:
                self.keyword_index_name = index_information["name"]
                self.text_node_property = index_information["properties"][0]
                node_label = index_information["labelsOrTypes"][0]
                return node_label
            except IndexError:
                return None
        else:
            return None

    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
        similarity_fn = DISTANCE_MAPPING[self._distance_strategy]
        create_vector_index(
            driver=self._driver,
            name=self.index_name,
            label=self.node_label,
            embedding_property=self.embedding_node_property,
            dimensions=self.embedding_dimension,
            similarity_fn=similarity_fn,
            fail_if_exists=False,
            neo4j_database=self._database,
        )

    def create_new_keyword_index(self, text_node_properties: List[str] = []) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in Neo4j.
        """
        if self.keyword_index_name:
            create_fulltext_index(
                driver=self._driver,
                name=self.keyword_index_name,
                label=self.node_label,
                node_properties=text_node_properties or [self.text_node_property],
                fail_if_exists=False,
                neo4j_database=self._database,
            )
        else:
            raise ValueError("keyword_index_name is not set.")

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        create_id_index: bool = True,
        search_type: SearchType = SearchType.VECTOR,
        **kwargs: Any,
    ) -> Neo4jVector:
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            embedding=embedding,
            search_type=search_type,
            **kwargs,
        )

        # Check if the vector index already exists
        existing_index_info = store.retrieve_existing_index()
        if existing_index_info:
            embedding_dimension, index_type = existing_index_info
        else:
            embedding_dimension = None
            index_type = None

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Data ingestion is not supported with relationship vector index."
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists. "
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Create unique constraint for faster import
        if create_id_index:
            store.query(
                "CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:`{store.node_label}`) REQUIRE n.id IS UNIQUE;"
            )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        import_query = self._build_import_query()

        parameters = {
            "data": [
                {"text": text, "metadata": metadata, "embedding": embedding, "id": id}
                for text, metadata, embedding, id in zip(
                    texts, metadatas, embeddings, ids
                )
            ]
        }

        self.query(import_query, params=parameters)

        return ids

    def _build_import_query(self) -> str:
        """
        Build the Cypher import query string based on the Neo4j version.

        Returns:
            str: The constructed Cypher query string.
        """
        if self.neo4j_version_is_5_23_or_above:
            call_prefix = "CALL (row) { "
        else:
            call_prefix = "CALL { WITH row "

        import_query = (
            "UNWIND $data AS row "
            f"{call_prefix}"
            f"MERGE (c:`{self.node_label}` {{id: row.id}}) "
            "WITH c, row "
            f"CALL db.create.setNodeVectorProperty(c, "
            f"'{self.embedding_node_property}', row.embedding) "
            f"SET c.`{self.text_node_property}` = row.text "
            "SET c += row.metadata "
            "} IN TRANSACTIONS OF 1000 ROWS "
        )

        return import_query

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        params: Dict[str, Any] = {},
        filter: Optional[Dict[str, Any]] = None,
        effective_search_ratio: int = 1,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Neo4jVector.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            effective_search_ratio (int): Controls the candidate pool size
               by multiplying $k to balance query accuracy and performance.
               Defaults to 1.
        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        params: Dict[str, Any] = {},
        filter: Optional[Dict[str, Any]] = None,
        effective_search_ratio: int = 1,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            effective_search_ratio (int): Controls the candidate pool size
               by multiplying $k to balance query accuracy and performance.
               Defaults to 1.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            query=query,
            params=params,
            filter=filter,
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        effective_search_ratio: int = 1,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the Neo4j database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the Neo4j database. The results are returned
        as a list of tuples, each containing a Document object and
        its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.
            effective_search_ratio (int): Controls the candidate pool size
               by multiplying $k to balance query accuracy and performance.
               Defaults to 1.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """
        if filter and not self.support_metadata_filter:
            raise ValueError(
                "Metadata filtering is only supported in "
                "Neo4j version 5.18 or greater"
            )
        entity_prefix = (
            "relationship" if self._index_type == IndexType.RELATIONSHIP else "node"
        )
        default_retrieval = (
            f"RETURN {entity_prefix}.`{self.text_node_property}` AS text, score, "
            f"{entity_prefix} "
            "{.*, "
            f"`{self.text_node_property}`: Null, "
            f"`{self.embedding_node_property}`: Null, id: Null "
        )
        if kwargs.get("return_embeddings"):
            default_retrieval += (
                f", _embedding_: {entity_prefix}.`{self.embedding_node_property}` "
            )
        default_retrieval += "} AS metadata"
        retrieval_query = (
            self.retrieval_query if self.retrieval_query else default_retrieval
        )

        read_query, filter_params = get_search_query(
            search_type=self.search_type,
            entity_type=self._index_type,
            retrieval_query=retrieval_query,
            node_label=self.node_label,
            embedding_node_property=self.embedding_node_property,
            embedding_dimension=self.embedding_dimension,
            filters=filter,
            neo4j_version_is_5_23_or_above=self.neo4j_version_is_5_23_or_above,
            use_parallel_runtime=self._is_enterprise,
        )
        parameters = {
            "vector_index_name": self.index_name,
            "top_k": k,
            "query_vector": embedding,
            "fulltext_index_name": self.keyword_index_name,
            "query_text": remove_lucene_chars(kwargs["query"]),
            "effective_search_ratio": effective_search_ratio,
            **params,
            **filter_params,
        }

        results = self.query(read_query, params=parameters)

        if any(result.get("text") is None for result in results):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.text_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `text` column"
                )
        if kwargs.get("return_embeddings") and any(
            result["metadata"]["_embedding_"] is None for result in results
        ):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.embedding_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `_embedding_` metadata column"
                )

        docs = [
            (
                Document(
                    page_content=dict_to_yaml_str(result["text"])
                    if isinstance(result["text"], dict)
                    else result["text"],
                    metadata={
                        k: v for k, v in result["metadata"].items() if v is not None
                    },
                ),
                result["score"],
            )
            for result in results
        ]
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        effective_search_ratio: int = 1,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Dictionary of argument(s) to
                    filter on metadata.
                Defaults to None.
            params (Dict[str, Any]): The search params for the index type.
                Defaults to empty dict.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            params=params,
            effective_search_ratio=effective_search_ratio,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[Neo4jVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Return Neo4jVector initialized from texts and embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Neo4jVector:
        """Construct Neo4jVector wrapper from raw documents and pre-
        generated embeddings.

        Return Neo4jVector initialized from documents and embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters.

        Example:
            .. code-block:: python

                from langchain_neo4j import Neo4jVector
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = Neo4jVector.from_embeddings(
                    text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        index_name: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        keyword_index_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Get instance of an existing Neo4j vector index. This method will
        return the instance of the store without inserting any new
        embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters along with
        the `index_name` definition.
        """

        if search_type == SearchType.HYBRID and not keyword_index_name:
            raise ValueError(
                "keyword_index name has to be specified "
                "when using hybrid search option"
            )

        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            embedding_dimension=embedding_dimension,
            **kwargs,
        )

        # Check if the vector index already exists
        existing_index_info = store.retrieve_existing_index()
        if existing_index_info:
            embedding_dimension_from_existing, index_type = existing_index_info
        else:
            embedding_dimension_from_existing = None
            index_type = None

        if embedding_dimension:
            if embedding_dimension_from_existing != embedding_dimension:
                raise ValueError(
                    "The provided embedding function and vector index "
                    "dimensions do not match.\n"
                    f"Embedding function dimension: {embedding_dimension}\n"
                    f"Vector index dimension: {embedding_dimension_from_existing}"
                )
        else:
            embedding_dimension = embedding_dimension_from_existing

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "Relationship vector index is not supported with "
                "`from_existing_index` method. Please use the "
                "`from_existing_relationship_index` method."
            )

        if not index_type:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )

        # Check if embedding function and vector index dimensions match
        if embedding_dimension and not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                raise ValueError(
                    "The specified keyword index name does not exist. "
                    "Make sure to check if you spelled it correctly"
                )
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        return store

    @classmethod
    def from_existing_relationship_index(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        index_name: str,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        embedding_dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Get instance of an existing Neo4j relationship vector index.
        This method will return the instance of the store without
        inserting any new embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters along with
        the `index_name` definition.
        """

        if search_type == SearchType.HYBRID:
            raise ValueError(
                "Hybrid search is not supported in combination "
                "with relationship vector index"
            )

        store = cls(
            embedding=embedding,
            index_name=index_name,
            embedding_dimension=embedding_dimension,
            **kwargs,
        )

        # Check if the vector index already exists
        existing_index_info = store.retrieve_existing_index()
        if existing_index_info:
            embedding_dimension_from_existing, index_type = existing_index_info
        else:
            embedding_dimension_from_existing = None
            index_type = None

        if embedding_dimension:
            if embedding_dimension_from_existing != embedding_dimension:
                raise ValueError(
                    "The provided embedding function and vector index "
                    "dimensions do not match.\n"
                    f"Embedding function dimension: {embedding_dimension}\n"
                    f"Vector index dimension: {embedding_dimension_from_existing}"
                )
        else:
            embedding_dimension = embedding_dimension_from_existing

        if not index_type:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )
        # Raise error if relationship index type
        if index_type == "NODE":
            raise ValueError(
                "Node vector index is not supported with "
                "`from_existing_relationship_index` method. Please use the "
                "`from_existing_index` method."
            )

        # Check if embedding function and vector index dimensions match
        if embedding_dimension and not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        return store

    @classmethod
    def from_documents(
        cls: Type[Neo4jVector],
        documents: List[Document],
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Return Neo4jVector initialized from documents and embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_existing_graph(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Initialize and return a Neo4jVector instance from an existing graph.

        This method initializes a Neo4jVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        Neo4jVector: An instance of Neo4jVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> neo4j_vector = Neo4jVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        Neo4j credentials are required in the form of `url`, `username`, and `password`,
        and optional `database` parameters passed as additional keyword arguments.
        """
        # Validate the list is not empty
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )
        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                f"RETURN reduce(str='', k IN {text_node_properties} |"
                " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, "
                "node {.*, `"
                + embedding_node_property
                + "`: Null, id: Null, "
                + ", ".join([f"`{prop}`: Null" for prop in text_node_properties])
                + "} AS metadata, score"
            )
        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            **kwargs,
        )

        # Check if the vector index already exists
        existing_index_info = store.retrieve_existing_index()
        if existing_index_info:
            embedding_dimension, index_type = existing_index_info
        else:
            embedding_dimension = None
            index_type = None

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "`from_existing_graph` method does not support "
                " existing relationship vector index. "
                "Please use `from_existing_relationship_index` method"
            )

        # If the vector index doesn't exist yet
        if not index_type:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif (
            embedding_dimension and not store.embedding_dimension == embedding_dimension
        ):
            raise ValueError(
                f"Index with name {store.index_name} already exists. "
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index(text_node_properties)
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index(text_node_properties)
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Populate embeddings
        while True:
            fetch_query = (
                f"MATCH (n:`{node_label}`) "
                f"WHERE n.{embedding_node_property} IS null "
                "AND any(k in $props WHERE n[k] IS NOT null) "
                f"RETURN elementId(n) AS id, reduce(str='',"
                "k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text "
                "LIMIT 1000"
            )
            data = store.query(fetch_query, params={"props": text_node_properties})
            if not data:
                break
            text_embeddings = embedding.embed_documents([el["text"] for el in data])

            params = {
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            }

            store.query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE elementId(n) = row.id "
                f"CALL db.create.setNodeVectorProperty(n, "
                f"'{embedding_node_property}', row.embedding) "
                "RETURN count(*)",
                params=params,
            )
            # If embedding calculation should be stopped
            if len(data) < 1000:
                break
        return store

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        try:
            import numpy as np
        except ImportError as e:
            msg = (
                "max_marginal_relevance_search requires numpy to be installed. "
                "Please install numpy with `pip install langchain-neo4j[mmr]` "
                "or `pip install numpy`."
            )
            raise ImportError(msg) from e

        # Embed the query
        query_embedding = self.embedding.embed_query(query)

        # Fetch the initial documents
        got_docs = self.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            query=query,
            k=fetch_k,
            return_embeddings=True,
            filter=filter,
            **kwargs,
        )

        # Get the embeddings for the fetched documents
        got_embeddings = [doc.metadata["_embedding_"] for doc, _ in got_docs]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), got_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        # Remove embedding values from metadata
        for doc in selected_docs:
            del doc.metadata["_embedding_"]

        return selected_docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return lambda x: x
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return lambda x: x
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
