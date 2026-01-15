"""Synchronous connection utilities for Neo4j checkpoint saver."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Union

from neo4j import Driver, GraphDatabase, Session

# Type alias for connection types
Conn = Union[Driver, str]


@contextmanager
def get_connection(conn: Conn) -> Iterator[Driver]:
    """Get a Neo4j driver from various connection types.

    Supports:
    - Direct Driver instance (passed through)
    - Connection string URI (creates new driver)

    Args:
        conn: Either a Driver instance or a connection string URI.

    Yields:
        A Neo4j Driver instance.

    Example:
        >>> with get_connection("bolt://localhost:7687") as driver:
        ...     with driver.session() as session:
        ...         session.run("MATCH (n) RETURN n")
    """
    if isinstance(conn, str):
        # Connection string provided, create driver
        driver = GraphDatabase.driver(conn)
        try:
            yield driver
        finally:
            driver.close()
    else:
        # Driver instance provided, use directly
        yield conn


def create_driver(
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    **kwargs: object,
) -> Driver:
    """Create a Neo4j driver with authentication.

    Args:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687").
        user: Neo4j username.
        password: Neo4j password.
        database: Optional database name (defaults to Neo4j default).
        **kwargs: Additional driver configuration options.

    Returns:
        A configured Neo4j Driver instance.

    Example:
        >>> driver = create_driver(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
    """
    return GraphDatabase.driver(uri, auth=(user, password), **kwargs)


@contextmanager
def get_session(driver: Driver, database: str | None = None) -> Iterator[Session]:
    """Get a Neo4j session from a driver.

    Args:
        driver: The Neo4j driver instance.
        database: Optional database name.

    Yields:
        A Neo4j Session instance.
    """
    session_kwargs = {}
    if database:
        session_kwargs["database"] = database

    with driver.session(**session_kwargs) as session:
        yield session
