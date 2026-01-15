import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase

from langchain_neo4j import AsyncNeo4jSaver, Neo4jSaver
from tests.integration_tests.utils import Neo4jCredentials

url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD", "password")
os.environ["NEO4J_URI"] = url
os.environ["NEO4J_USERNAME"] = username
os.environ["NEO4J_PASSWORD"] = password


@pytest.fixture
def clear_neo4j_database() -> None:
    driver = GraphDatabase.driver(url, auth=(username, password))
    driver.execute_query("MATCH (n) DETACH DELETE n;")
    driver.close()


@pytest.fixture(scope="session")
def neo4j_credentials() -> Neo4jCredentials:
    return {
        "url": url,
        "username": username,
        "password": password,
    }


@pytest.fixture
def neo4j_driver() -> Generator[Driver, None, None]:
    """Create a Neo4j driver for testing."""
    driver = GraphDatabase.driver(url, auth=(username, password))
    yield driver
    driver.close()


@pytest.fixture
def neo4j_saver(neo4j_driver: Driver) -> Generator[Neo4jSaver, None, None]:
    """Create a Neo4jSaver for testing."""
    saver = Neo4jSaver(neo4j_driver)
    saver.setup()
    yield saver


@pytest.fixture
def clean_neo4j_saver(neo4j_saver: Neo4jSaver) -> Generator[Neo4jSaver, None, None]:
    """Create a Neo4jSaver and clean up test data after each test."""
    yield neo4j_saver
    with neo4j_saver._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest_asyncio.fixture
async def async_neo4j_driver() -> AsyncGenerator[AsyncDriver, None]:
    """Create an async Neo4j driver for testing."""
    driver = AsyncGraphDatabase.driver(url, auth=(username, password))
    yield driver
    await driver.close()


@pytest_asyncio.fixture
async def async_neo4j_saver(async_neo4j_driver: AsyncDriver) -> AsyncNeo4jSaver:
    """Create an AsyncNeo4jSaver for testing."""
    saver = AsyncNeo4jSaver(async_neo4j_driver)
    await saver.setup()
    return saver


@pytest_asyncio.fixture
async def clean_async_neo4j_saver(
    async_neo4j_saver: AsyncNeo4jSaver,
) -> AsyncGenerator[AsyncNeo4jSaver, None]:
    """Create an AsyncNeo4jSaver and clean up test data after each test."""
    yield async_neo4j_saver
    async with async_neo4j_saver._driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def sample_checkpoint() -> dict[str, Any]:
    """Create a sample checkpoint for testing."""
    return {
        "v": 1,
        "id": "checkpoint-1",
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        "channel_versions": {
            "messages": "00000000000000000001.0000000000000000",
        },
        "versions_seen": {},
        "pending_sends": [],
    }


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Create sample checkpoint metadata for testing."""
    return {
        "source": "input",
        "step": 0,
        "writes": {},
        "parents": {},
    }


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Create a sample config for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread-1",
            "checkpoint_ns": "",
        }
    }
