import os

import neo4j
import pytest

url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD", "pleaseletmein")


@pytest.fixture
def clear_neo4j_database() -> None:
    driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
    driver.execute_query("MATCH (n) DETACH DELETE n;")
    driver.close()
