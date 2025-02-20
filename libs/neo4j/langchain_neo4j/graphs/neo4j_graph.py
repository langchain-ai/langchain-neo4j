from hashlib import md5
from typing import Any, Dict, List, Optional, Type

import neo4j
from langchain_core.utils import get_from_dict_or_env
from neo4j_graphrag.schema import (
    BASE_ENTITY_LABEL,
    _value_sanitize,
    format_schema,
    get_structured_schema,
)

from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.graph_store import GraphStore

include_docs_query = (
    "MERGE (d:Document {id:$document.metadata.id}) "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)


def _get_node_import_query(baseEntityLabel: bool, include_source: bool) -> str:
    if baseEntityLabel:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id}}) "
            "SET source += row.properties "
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            "WITH source, row "
            "CALL apoc.create.addLabels( source, [row.type] ) YIELD node "
            "RETURN distinct 'done' AS result"
        )
    else:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.type], {id: row.id}, "
            "row.properties, {}) YIELD node "
            f"{'MERGE (d)-[:MENTIONS]->(node) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )


def _get_rel_import_query(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return (
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source}}) "
            f"MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target}}) "
            "WITH source, target, row "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )
    else:
        return (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.source_label], {id: row.source},"
            "{}, {}) YIELD node as source "
            "CALL apoc.merge.node([row.target_label], {id: row.target},"
            "{}, {}) YIELD node as target "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )


def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


class Neo4jGraph(GraphStore):
    """Neo4j database wrapper for various graph operations.

    Parameters:
    url (Optional[str]): The URL of the Neo4j database server.
    username (Optional[str]): The username for database authentication.
    password (Optional[str]): The password for database authentication.
    database (str): The name of the database to connect to. Default is 'neo4j'.
    timeout (Optional[float]): The timeout for transactions in seconds.
            Useful for terminating long-running queries.
            By default, there is no timeout set.
    sanitize (bool): A flag to indicate whether to remove lists with
            more than 128 elements from results. Useful for removing
            embedding-like properties from database responses. Default is False.
    refresh_schema (bool): A flag whether to refresh schema information
            at initialization. Default is True.
    enhanced_schema (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is False.
    driver_config (Dict): Configuration passed to Neo4j Driver.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
        refresh_schema: bool = True,
        *,
        driver_config: Optional[Dict] = None,
        enhanced_schema: bool = False,
    ) -> None:
        """Create a new Neo4j graph wrapper instance."""

        url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
        # if username and password are "", assume Neo4j auth is disabled
        if username == "" and password == "":
            auth = None
        else:
            username = get_from_dict_or_env(
                {"username": username},
                "username",
                "NEO4J_USERNAME",
            )
            password = get_from_dict_or_env(
                {"password": password},
                "password",
                "NEO4J_PASSWORD",
            )
            auth = (username, password)
        database = get_from_dict_or_env(
            {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
        )

        self._driver = neo4j.GraphDatabase.driver(
            url, auth=auth, **(driver_config or {})
        )
        self._database = database
        self.timeout = timeout
        self.sanitize = sanitize
        self._enhanced_schema = enhanced_schema
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ConfigurationError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the driver config is correct"
            )
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
        # Set schema
        if refresh_schema:
            try:
                self.refresh_schema()
            except neo4j.exceptions.ClientError as e:
                if e.code == "Neo.ClientError.Procedure.ProcedureNotFound":
                    raise ValueError(
                        "Could not use APOC procedures. "
                        "Please ensure the APOC plugin is installed in Neo4j and that "
                        "'apoc.meta.data()' is allowed in Neo4j configuration "
                    )
                raise e

    def _check_driver_state(self) -> None:
        """
        Check if the driver is available and ready for operations.

        Raises:
            RuntimeError: If the driver has been closed or is not initialized.
        """
        if not hasattr(self, "_driver"):
            raise RuntimeError(
                "Cannot perform operations - Neo4j connection has been closed"
            )

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

    def query(
        self,
        query: str,
        params: dict = {},
        session_params: dict = {},
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.
            session_params (dict): Parameters to pass to the session used for executing
                the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        if not session_params:
            try:
                data, _, _ = self._driver.execute_query(
                    Query(text=query, timeout=self.timeout),
                    database_=self._database,
                    parameters_=params,
                )
                json_data = [r.data() for r in data]
                if self.sanitize:
                    json_data = [_value_sanitize(el) for el in json_data]
                return json_data
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
                            or "tried to execute in an explicit transaction"
                            in e.message
                        )
                    )
                ):
                    raise
        # fallback to allow implicit transactions
        session_params.setdefault("database", self._database)
        with self._driver.session(**session_params) as session:
            result = session.run(Query(text=query, timeout=self.timeout), params)
            json_data = [r.data() for r in result]
            if self.sanitize:
                json_data = [_value_sanitize(el) for el in json_data]
            return json_data

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        self.structured_schema = get_structured_schema(
            driver=self._driver,
            is_enhanced=self._enhanced_schema,
            database=self._database,
            timeout=self.timeout,
            sanitize=self.sanitize,
        )
        self.schema = format_schema(
            schema=self.structured_schema, is_enhanced=self._enhanced_schema
        )

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """
        This method constructs nodes and relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and optionally the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the MENTIONS relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document metadata
        if available; otherwise it calculates the MD5 hash of `page_content`
        for merging process. Defaults to False.
        - baseEntityLabel (bool, optional): If True, each newly created node
        gets a secondary __Entity__ label, which is indexed and improves import
        speed and performance. Defaults to False.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        if baseEntityLabel:  # Check if constraint already exists
            constraint_exists = any(
                [
                    el["labelsOrTypes"] == [BASE_ENTITY_LABEL]
                    and el["properties"] == ["id"]
                    for el in self.structured_schema.get("metadata", {}).get(
                        "constraint", []
                    )
                ]
            )

            if not constraint_exists:
                # Create constraint
                self.query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (b:{BASE_ENTITY_LABEL}) "
                    "REQUIRE b.id IS UNIQUE;"
                )
                self.refresh_schema()  # Refresh constraint information

        # Check each graph_document has a source when include_source is true
        if include_source:
            for doc in graph_documents:
                if doc.source is None:
                    raise TypeError(
                        "include_source is set to True, "
                        "but at least one document has no `source`."
                    )

        node_import_query = _get_node_import_query(baseEntityLabel, include_source)
        rel_import_query = _get_rel_import_query(baseEntityLabel)
        for document in graph_documents:
            node_import_query_params: dict[str, Any] = {
                "data": [el.__dict__ for el in document.nodes]
            }
            if include_source and document.source:
                if not document.source.metadata.get("id"):
                    document.source.metadata["id"] = md5(
                        document.source.page_content.encode("utf-8")
                    ).hexdigest()
                node_import_query_params["document"] = document.source.__dict__

            # Remove backticks from node types
            for node in document.nodes:
                node.type = _remove_backticks(node.type)
            # Import nodes
            self.query(node_import_query, node_import_query_params)
            # Import relationships
            self.query(
                rel_import_query,
                {
                    "data": [
                        {
                            "source": el.source.id,
                            "source_label": _remove_backticks(el.source.type),
                            "target": el.target.id,
                            "target_label": _remove_backticks(el.target.type),
                            "type": _remove_backticks(
                                el.type.replace(" ", "_").upper()
                            ),
                            "properties": el.properties,
                        }
                        for el in document.relationships
                    ]
                },
            )

    def close(self) -> None:
        """
        Explicitly close the Neo4j driver connection.

        Delegates connection management to the Neo4j driver.
        """
        if hasattr(self, "_driver"):
            self._driver.close()
            # Remove the driver attribute to indicate closure
            delattr(self, "_driver")

    def __enter__(self) -> "Neo4jGraph":
        """
        Enter the runtime context for the Neo4j graph connection.

        Enables use of the graph connection with the 'with' statement.
        This method allows for automatic resource management and ensures
        that the connection is properly handled.

        Returns:
            Neo4jGraph: The current graph connection instance

        Example:
            with Neo4jGraph(...) as graph:
                graph.query(...)  # Connection automatically managed
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Exit the runtime context for the Neo4j graph connection.

        This method is automatically called when exiting a 'with' statement.
        It ensures that the database connection is closed, regardless of
        whether an exception occurred during the context's execution.

        Args:
            exc_type: The type of exception that caused the context to exit
                      (None if no exception occurred)
            exc_val: The exception instance that caused the context to exit
                     (None if no exception occurred)
            exc_tb: The traceback for the exception (None if no exception occurred)

        Note:
            Any exception is re-raised after the connection is closed.
        """
        self.close()

    def __del__(self) -> None:
        """
        Destructor for the Neo4j graph connection.

        This method is called during garbage collection to ensure that
        database resources are released if not explicitly closed.

        Caution:
            - Do not rely on this method for deterministic resource cleanup
            - Always prefer explicit .close() or context manager

        Best practices:
            1. Use context manager:
               with Neo4jGraph(...) as graph:
                   ...
            2. Explicitly close:
               graph = Neo4jGraph(...)
               try:
                   ...
               finally:
                   graph.close()
        """
        try:
            self.close()
        except Exception:
            # Suppress any exceptions during garbage collection
            pass
