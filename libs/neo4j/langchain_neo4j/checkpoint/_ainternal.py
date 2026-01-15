"""Asynchronous connection utilities for Neo4j checkpoint saver."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Union

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

# Type alias for async connection types
AsyncConn = Union[AsyncDriver, str]


@asynccontextmanager
async def get_async_connection(conn: AsyncConn) -> AsyncIterator[AsyncDriver]:
    """Get an async Neo4j driver from various connection types.

    Supports:
    - Direct AsyncDriver instance (passed through)
    - Connection string URI (creates new driver)

    Args:
        conn: Either an AsyncDriver instance or a connection string URI.

    Yields:
        An async Neo4j Driver instance.

    Example:
        >>> async with get_async_connection("bolt://localhost:7687") as driver:
        ...     async with driver.session() as session:
        ...         await session.run("MATCH (n) RETURN n")
    """
    if isinstance(conn, str):
        # Connection string provided, create driver
        driver = AsyncGraphDatabase.driver(conn)
        try:
            yield driver
        finally:
            await driver.close()
    else:
        # Driver instance provided, use directly
        yield conn


async def create_async_driver(
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    **kwargs: object,
) -> AsyncDriver:
    """Create an async Neo4j driver with authentication.

    Args:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687").
        user: Neo4j username.
        password: Neo4j password.
        database: Optional database name (defaults to Neo4j default).
        **kwargs: Additional driver configuration options.

    Returns:
        A configured async Neo4j Driver instance.

    Example:
        >>> driver = await create_async_driver(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
    """
    return AsyncGraphDatabase.driver(uri, auth=(user, password), **kwargs)


@asynccontextmanager
async def get_async_session(
    driver: AsyncDriver, database: str | None = None
) -> AsyncIterator[AsyncSession]:
    """Get an async Neo4j session from a driver.

    Args:
        driver: The async Neo4j driver instance.
        database: Optional database name.

    Yields:
        An async Neo4j Session instance.
    """
    session_kwargs = {}
    if database:
        session_kwargs["database"] = database

    async with driver.session(**session_kwargs) as session:
        yield session
