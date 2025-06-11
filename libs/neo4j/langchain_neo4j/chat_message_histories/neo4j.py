from typing import List, Optional, Union

import neo4j
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.utils import get_from_dict_or_env
from neo4j_graphrag.message_history import (
    ADD_MESSAGE_QUERY,
    CREATE_SESSION_NODE_QUERY,
    DELETE_MESSAGES_QUERY,
    DELETE_SESSION_AND_MESSAGES_QUERY,
    GET_MESSAGES_QUERY,
)

from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph


class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Neo4j database."""

    def __init__(
        self,
        session_id: Union[str, int],
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        node_label: str = "Session",
        window: int = 3,
        *,
        graph: Optional[Neo4jGraph] = None,
    ):
        # Make sure session id is not null
        if not session_id:
            raise ValueError("Please ensure that the session_id parameter is provided")

        # Graph object takes precedent over env or input params
        if graph:
            self._driver = graph._driver
            self._database = graph._database
        else:
            # Handle if the credentials are environment variables
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
        self._session_id = session_id
        self._node_label = node_label
        self._window = window
        # Create session node
        self._driver.execute_query(
            CREATE_SESSION_NODE_QUERY.format(node_label=self._node_label),
            {"session_id": self._session_id},
            database=self._database,
        )

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Neo4j"""
        records, _, _ = self._driver.execute_query(
            GET_MESSAGES_QUERY.format(
                node_label=self._node_label, window=self._window * 2
            ),
            {"session_id": self._session_id},
            database=self._database,
        )
        messages = [
            {
                "data": el["result"]["data"],
                "type": el["result"]["role"],
            }
            for el in records
        ]
        return messages_from_dict(messages)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Neo4j"""
        self._driver.execute_query(
            ADD_MESSAGE_QUERY.format(node_label=self._node_label),
            {
                "role": message.type,
                "content": message.content,
                "session_id": self._session_id,
            },
            database=self._database,
        )

    def clear(self, delete_session_node: bool = False) -> None:
        """Clear session memory from Neo4j

        Args:
            delete_session_node (bool): Whether to delete the session node.
                Defaults to False.
        """
        if delete_session_node:
            self._driver.execute_query(
                query_=DELETE_SESSION_AND_MESSAGES_QUERY.format(
                    node_label=self._node_label
                ),
                parameters_={"session_id": self._session_id},
                database=self._database,
            )
        else:
            self._driver.execute_query(
                query_=DELETE_MESSAGES_QUERY.format(node_label=self._node_label),
                parameters_={"session_id": self._session_id},
                database=self._database,
            )

    def __del__(self) -> None:
        if self._driver:
            self._driver.close()
