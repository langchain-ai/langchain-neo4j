import os
import urllib.parse

import neo4j
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_neo4j.chat_message_histories.neo4j import Neo4jChatMessageHistory
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from tests.integration_tests.utils import Neo4jCredentials


@pytest.mark.usefixtures("clear_neo4j_database")
def test_add_messages() -> None:
    """Basic testing: adding messages to the Neo4jChatMessageHistory."""
    assert os.environ.get("NEO4J_URI") is not None
    assert os.environ.get("NEO4J_USERNAME") is not None
    assert os.environ.get("NEO4J_PASSWORD") is not None
    message_store = Neo4jChatMessageHistory("23334")
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")

    # create another message store to check if the messages are stored correctly
    message_store_another = Neo4jChatMessageHistory("46666")
    message_store_another.clear()
    assert len(message_store_another.messages) == 0
    message_store_another.add_user_message("Hello! Bot!")
    message_store_another.add_ai_message("Hi there!")
    message_store_another.add_user_message("How's this pr going?")

    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2
    assert isinstance(message_store.messages[0], HumanMessage)
    assert isinstance(message_store.messages[1], AIMessage)
    assert message_store.messages[0].content == "Hello! Language Chain!"
    assert message_store.messages[1].content == "Hi Guys!"

    assert len(message_store_another.messages) == 3
    assert isinstance(message_store_another.messages[0], HumanMessage)
    assert isinstance(message_store_another.messages[1], AIMessage)
    assert isinstance(message_store_another.messages[2], HumanMessage)
    assert message_store_another.messages[0].content == "Hello! Bot!"
    assert message_store_another.messages[1].content == "Hi there!"
    assert message_store_another.messages[2].content == "How's this pr going?"

    # Now clear the first history
    message_store.clear()
    assert len(message_store.messages) == 0
    assert len(message_store_another.messages) == 3
    message_store_another.clear()
    assert len(message_store.messages) == 0
    assert len(message_store_another.messages) == 0


@pytest.mark.usefixtures("clear_neo4j_database")
def test_add_messages_graph_object(neo4j_credentials: Neo4jCredentials) -> None:
    """Basic testing: Passing driver through graph object."""
    graph = Neo4jGraph(**neo4j_credentials)
    # rewrite env for testing
    old_username = os.environ["NEO4J_USERNAME"]
    os.environ["NEO4J_USERNAME"] = "foo"
    message_store = Neo4jChatMessageHistory("23334", graph=graph)
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")
    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2
    os.environ["NEO4J_USERNAME"] = old_username


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
        Neo4jChatMessageHistory(
            "test_session",
            url=new_url,
            username=neo4j_credentials["username"],
            password=neo4j_credentials["password"],
        )
    assert "Please ensure that the url is correct" in str(exc_info.value)


def test_invalid_credentials(neo4j_credentials: Neo4jCredentials) -> None:
    """Test initializing with invalid credentials raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        Neo4jChatMessageHistory(
            "test_session",
            url=neo4j_credentials["url"],
            username="invalid_username",
            password="invalid_password",
        )
    assert "Please ensure that the username and password are correct" in str(
        exc_info.value
    )


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_message_history_clear_messages(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    message_history = Neo4jChatMessageHistory(session_id="123", **neo4j_credentials)
    message_history.add_messages(
        [
            HumanMessage(content="You are a helpful assistant."),
            AIMessage(content="Hello"),
        ]
    )
    assert len(message_history.messages) == 2
    message_history.clear()
    assert len(message_history.messages) == 0
    # Test that the session node is not deleted
    results = message_history._driver.execute_query(
        query_="MATCH (s:`Session`) WHERE s.id = '123' RETURN s"
    )
    assert len(results.records) == 1
    assert results.records[0]["s"]["id"] == "123"
    assert list(results.records[0]["s"].labels) == ["Session"]


@pytest.mark.usefixtures("clear_neo4j_database")
def test_neo4j_message_history_clear_session_and_messages(
    neo4j_credentials: Neo4jCredentials,
) -> None:
    message_history = Neo4jChatMessageHistory(session_id="123", **neo4j_credentials)
    message_history.add_messages(
        [
            HumanMessage(content="You are a helpful assistant."),
            AIMessage(content="Hello"),
        ]
    )
    assert len(message_history.messages) == 2
    message_history.clear(delete_session_node=True)
    assert len(message_history.messages) == 0
    # Test that the session node is deleted
    results = message_history._driver.execute_query(
        query_="MATCH (s:`Session`) WHERE s.id = '123' RETURN s"
    )
    assert results.records == []


@pytest.mark.usefixtures("clear_neo4j_database")
def test_database_parameter(neo4j_credentials: Neo4jCredentials) -> None:
    custom_db = "testdb"
    driver = neo4j.GraphDatabase.driver(
        neo4j_credentials["url"],
        auth=(neo4j_credentials["username"], neo4j_credentials["password"]),
    )

    driver.execute_query(f"CREATE DATABASE `{custom_db}` IF NOT EXISTS")
    message_store = Neo4jChatMessageHistory(
        "session_custom_db",
        database=custom_db,
        **neo4j_credentials,
    )
    message_store.clear()
    assert message_store._database == custom_db
    message_store.add_user_message("Hello on custom DB")
    message_store.add_ai_message("Custom DB OK")
    assert len(message_store.messages) == 2
    driver.execute_query(f"DROP DATABASE `{custom_db}` IF EXISTS")
    driver.close()
