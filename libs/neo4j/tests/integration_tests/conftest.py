import os

import neo4j
import pytest

from tests.integration_tests.utils import Neo4jCredentials

url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")
os.environ["NEO4J_URI"] = url
os.environ["NEO4J_USERNAME"] = username
os.environ["NEO4J_PASSWORD"] = password


@pytest.fixture
def clear_neo4j_database() -> None:
    driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
    driver.execute_query("MATCH (n) DETACH DELETE n;")
    driver.close()


@pytest.fixture(scope="session")
def neo4j_credentials() -> Neo4jCredentials:
    return {
        "url": url,
        "username": username,
        "password": password,
    }
