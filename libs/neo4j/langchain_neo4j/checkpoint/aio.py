"""Asynchronous Neo4j implementation of LangGraph checkpoint saver."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from neo4j import AsyncDriver

from langchain_neo4j.checkpoint._ainternal import create_async_driver, get_async_session
from langchain_neo4j.checkpoint.base import (
    CYPHER_CREATE_BRANCH_CONSTRAINT,
    CYPHER_CREATE_BRANCH_NAME_INDEX,
    CYPHER_CREATE_CHANNEL_STATE_CONSTRAINT,
    CYPHER_CREATE_CHECKPOINT_CONSTRAINT,
    CYPHER_CREATE_CHECKPOINT_CREATED_INDEX,
    CYPHER_CREATE_CHECKPOINT_ID_INDEX,
    CYPHER_CREATE_MAIN_BRANCH,
    CYPHER_CREATE_THREAD_CONSTRAINT,
    CYPHER_DELETE_ORPHAN_CHANNEL_STATES,
    CYPHER_DELETE_THREAD,
    CYPHER_GET_ACTIVE_BRANCH_HEAD,
    CYPHER_GET_CHANNEL_STATES,
    CYPHER_GET_CHECKPOINT_BY_ID,
    CYPHER_GET_LATEST_CHECKPOINT,
    CYPHER_GET_WRITES,
    CYPHER_LINK_PARENT_CHECKPOINT,
    CYPHER_LIST_CHECKPOINTS,
    CYPHER_THREAD_HAS_BRANCHES,
    CYPHER_UPDATE_BRANCH_HEAD,
    CYPHER_UPSERT_CHANNEL_STATE,
    CYPHER_UPSERT_CHECKPOINT_SIMPLE,
    CYPHER_UPSERT_WRITE,
    BaseNeo4jSaver,
)

__all__ = ["AsyncNeo4jSaver"]


class AsyncNeo4jSaver(BaseNeo4jSaver):
    """Asynchronous Neo4j checkpoint saver for LangGraph.

    This class implements the BaseCheckpointSaver interface using Neo4j
    as the persistence backend with async support and a proper graph model.
    It supports storing checkpoints, channel states, and pending writes
    using relationships for efficient traversal.

    Graph Model:
        (:Thread)-[:HAS_CHECKPOINT]->(:Checkpoint)-[:PREVIOUS]->(:Checkpoint)
        (:Checkpoint)-[:HAS_CHANNEL]->(:ChannelState)
        (:Checkpoint)-[:HAS_WRITE]->(:PendingWrite)

    Example:
        >>> # Using from_conn_string (recommended)
        >>> async with await AsyncNeo4jSaver.from_conn_string(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... ) as checkpointer:
        ...     await checkpointer.setup()  # Create indexes (run once)
        ...     graph = builder.compile(checkpointer=checkpointer)
        ...     result = await graph.ainvoke({"messages": [...]}, config)

        >>> # Using existing async driver
        >>> driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        >>> checkpointer = AsyncNeo4jSaver(driver)
        >>> await checkpointer.setup()
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str | None = None,
    ) -> None:
        """Initialize the async Neo4j checkpoint saver.

        Args:
            driver: An async Neo4j Driver instance.
            database: Optional database name (defaults to Neo4j default).
        """
        super().__init__()
        self._driver = driver
        self._database = database
        self._lock = asyncio.Lock()
        self._owns_driver = False

    @classmethod
    async def from_conn_string(
        cls,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
    ) -> AsyncNeo4jSaver:
        """Create an AsyncNeo4jSaver from connection parameters.

        This is the recommended way to create an AsyncNeo4jSaver as it
        properly manages the driver lifecycle.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687").
            user: Neo4j username.
            password: Neo4j password.
            database: Optional database name.

        Returns:
            A configured AsyncNeo4jSaver instance.

        Example:
            >>> checkpointer = await AsyncNeo4jSaver.from_conn_string(
            ...     uri="bolt://localhost:7687",
            ...     user="neo4j",
            ...     password="password"
            ... )
            >>> try:
            ...     await checkpointer.setup()
            ...     # Use checkpointer...
            ... finally:
            ...     await checkpointer.close()
        """
        driver = await create_async_driver(uri, user, password)
        saver = cls(driver, database)
        saver._owns_driver = True
        return saver

    async def close(self) -> None:
        """Close the driver connection if owned by this instance."""
        if self._owns_driver and self._driver:
            await self._driver.close()

    async def __aenter__(self) -> AsyncNeo4jSaver:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def setup(self) -> None:
        """Create indexes and constraints in Neo4j.

        This method should be called once before using the checkpointer.
        It creates the necessary indexes and constraints for the graph model,
        including branch-related constraints for time-travel functionality.
        """
        async with self._lock:
            async with get_async_session(self._driver, self._database) as session:
                # Create constraints for graph model
                await session.run(CYPHER_CREATE_THREAD_CONSTRAINT)
                await session.run(CYPHER_CREATE_CHECKPOINT_CONSTRAINT)
                await session.run(CYPHER_CREATE_CHANNEL_STATE_CONSTRAINT)
                # Create branch-related constraints
                await session.run(CYPHER_CREATE_BRANCH_CONSTRAINT)
                # Create indexes
                await session.run(CYPHER_CREATE_CHECKPOINT_ID_INDEX)
                await session.run(CYPHER_CREATE_CHECKPOINT_CREATED_INDEX)
                await session.run(CYPHER_CREATE_BRANCH_NAME_INDEX)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata.

        Creates Thread and Checkpoint nodes with HAS_CHECKPOINT relationship,
        links to parent via PREVIOUS relationship, and stores channel states
        with HAS_CHANNEL relationships. Also manages branch tracking for
        time-travel functionality.

        Args:
            config: The runnable configuration containing thread_id.
            checkpoint: The checkpoint data to store.
            metadata: Metadata about the checkpoint (source, step, etc.).
            new_versions: Channel version information.

        Returns:
            Updated configuration with the new checkpoint_id.
        """
        thread_id, checkpoint_ns, parent_checkpoint_id = self._parse_config(config)
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint and metadata
        type_, serialized_checkpoint = self._dump_checkpoint(checkpoint)
        _, serialized_metadata = self._dump_metadata(metadata)

        # Get channel values and versions from checkpoint
        channel_values = checkpoint.get("channel_values", {})
        channel_versions = checkpoint.get("channel_versions", {})

        async with self._lock:
            async with get_async_session(self._driver, self._database) as session:
                # Step 1: Create Thread and Checkpoint with HAS_CHECKPOINT relationship
                await session.run(
                    CYPHER_UPSERT_CHECKPOINT_SIMPLE,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "type": type_,
                        "checkpoint": serialized_checkpoint,
                        "metadata": serialized_metadata,
                    },
                )

                # Step 2: Link to parent checkpoint if exists
                if parent_checkpoint_id:
                    await session.run(
                        CYPHER_LINK_PARENT_CHECKPOINT,
                        {
                            "checkpoint_id": checkpoint_id,
                            "parent_checkpoint_id": parent_checkpoint_id,
                        },
                    )

                # Step 3: Create ChannelState nodes and HAS_CHANNEL relationships
                blobs = self._dump_blobs(channel_values, channel_versions)
                for blob in blobs:
                    await session.run(
                        CYPHER_UPSERT_CHANNEL_STATE,
                        {
                            "checkpoint_id": checkpoint_id,
                            "channel": blob["channel"],
                            "version": blob["version"],
                            "type": blob["type"],
                            "blob": blob["blob"],
                        },
                    )

                # Step 4: Handle branch management
                # Check if thread has any branches
                branch_result = await session.run(
                    CYPHER_THREAD_HAS_BRANCHES,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                    },
                )
                branch_record = await branch_result.single()

                if branch_record and branch_record["branch_count"] == 0:
                    # No branches exist, create main branch
                    await session.run(
                        CYPHER_CREATE_MAIN_BRANCH,
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "branch_id": str(uuid.uuid4()),
                        },
                    )

                # Update the active branch HEAD to point to this checkpoint
                await session.run(
                    CYPHER_UPDATE_BRANCH_HEAD,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    },
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes for fault tolerance.

        Creates PendingWrite nodes and links them to the Checkpoint
        via HAS_WRITE relationships.

        Args:
            config: The runnable configuration.
            writes: Sequence of (channel, value) tuples to store.
            task_id: Identifier for the task that produced these writes.
            task_path: Path information for the task (default: "").
        """
        thread_id, checkpoint_ns, checkpoint_id = self._parse_config(config)

        if not checkpoint_id:
            raise ValueError("checkpoint_id is required for put_writes")

        write_records = self._dump_writes(list(writes), task_id, task_path)

        async with self._lock:
            async with get_async_session(self._driver, self._database) as session:
                for record in write_records:
                    await session.run(
                        CYPHER_UPSERT_WRITE,
                        {
                            "checkpoint_id": checkpoint_id,
                            "task_id": record["task_id"],
                            "task_path": record["task_path"],
                            "idx": record["idx"],
                            "channel": record["channel"],
                            "type": record["type"],
                            "blob": record["blob"],
                        },
                    )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint tuple by configuration.

        Traverses the graph model from Thread -> Checkpoint -> ChannelStates
        to retrieve all data. When no checkpoint_id is specified, retrieves
        the HEAD of the active branch (instead of lexicographic latest).

        Args:
            config: The runnable configuration with thread_id and optional checkpoint_id

        Returns:
            CheckpointTuple if found, None otherwise.
        """
        thread_id, checkpoint_ns, checkpoint_id = self._parse_config(config)

        async with self._lock:
            async with get_async_session(self._driver, self._database) as session:
                # Get checkpoint record via relationship traversal
                if checkpoint_id:
                    result = await session.run(
                        CYPHER_GET_CHECKPOINT_BY_ID,
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                else:
                    # Try to get active branch HEAD first
                    result = await session.run(
                        CYPHER_GET_ACTIVE_BRANCH_HEAD,
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                        },
                    )
                    record = await result.single()
                    # Fallback to latest checkpoint if no branches exist yet
                    if not record:
                        result = await session.run(
                            CYPHER_GET_LATEST_CHECKPOINT,
                            {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                            },
                        )

                if checkpoint_id:
                    record = await result.single()

                if not record:
                    return None

                checkpoint_record = dict(record)
                checkpoint_id = checkpoint_record["checkpoint_id"]

                # Load checkpoint to get channel info
                checkpoint = self._load_checkpoint(
                    checkpoint_record["type"], checkpoint_record["checkpoint"]
                )

                # Get channel states via HAS_CHANNEL relationships
                channel_versions = checkpoint.get("channel_versions", {})
                if channel_versions:
                    channels = list(channel_versions.keys())
                    versions = list(channel_versions.values())
                    blob_result = await session.run(
                        CYPHER_GET_CHANNEL_STATES,
                        {
                            "checkpoint_id": checkpoint_id,
                            "channels": channels,
                            "versions": versions,
                        },
                    )
                    blob_records = [dict(r) async for r in blob_result]
                    channel_values = self._load_blobs(blob_records)
                else:
                    channel_values = {}

                # Get pending writes via HAS_WRITE relationships
                write_result = await session.run(
                    CYPHER_GET_WRITES,
                    {"checkpoint_id": checkpoint_id},
                )
                write_records = [dict(r) async for r in write_result]
                pending_writes = self._load_writes(write_records)

                return self._make_checkpoint_tuple(
                    checkpoint_record,
                    channel_values,
                    pending_writes,
                )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints matching the given criteria.

        Traverses from Thread to Checkpoints via HAS_CHECKPOINT relationships.

        Args:
            config: Configuration with thread_id to filter by.
            filter: Optional metadata filters (not yet implemented).
            before: Optional config to list checkpoints before.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple instances in descending order by checkpoint_id.
        """
        if not config:
            return

        thread_id, checkpoint_ns, _ = self._parse_config(config)

        before_id: str | None = None
        if before:
            _, _, before_id = self._parse_config(before)

        effective_limit = limit or 100  # Default limit

        async with self._lock:
            async with get_async_session(self._driver, self._database) as session:
                result = await session.run(
                    CYPHER_LIST_CHECKPOINTS,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "before_id": before_id,
                        "limit": effective_limit,
                    },
                )

                async for record in result:
                    checkpoint_record = dict(record)

                    # For list, we don't load full channel values to save memory
                    # Pending writes are also not loaded for list operations
                    yield self._make_checkpoint_tuple(
                        checkpoint_record,
                        channel_values={},
                        pending_writes=[],
                    )

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, channel states, and writes for a thread.

        Uses DETACH DELETE to cascade through relationships.

        Args:
            thread_id: The thread identifier to delete.
        """
        async with self._lock:
            async with get_async_session(self._driver, self._database) as session:
                # Delete thread and all connected checkpoints/writes
                await session.run(CYPHER_DELETE_THREAD, {"thread_id": thread_id})
                # Clean up any orphaned ChannelState nodes
                await session.run(CYPHER_DELETE_ORPHAN_CHANNEL_STATES)

    # Sync method implementations that delegate to async methods
    # These are required by the BaseCheckpointSaver interface

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Synchronous wrapper for aput."""
        return asyncio.get_event_loop().run_until_complete(
            self.aput(config, checkpoint, metadata, new_versions)
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Synchronous wrapper for aput_writes."""
        asyncio.get_event_loop().run_until_complete(
            self.aput_writes(config, writes, task_id, task_path)
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Synchronous wrapper for aget_tuple."""
        return asyncio.get_event_loop().run_until_complete(self.aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Synchronous wrapper for alist."""

        async def _collect() -> list[CheckpointTuple]:
            results = []
            async for item in self.alist(
                config, filter=filter, before=before, limit=limit
            ):
                results.append(item)
            return results

        yield from asyncio.get_event_loop().run_until_complete(_collect())

    def delete_thread(self, thread_id: str) -> None:
        """Synchronous wrapper for adelete_thread."""
        asyncio.get_event_loop().run_until_complete(self.adelete_thread(thread_id))
