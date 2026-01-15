"""Tests for asynchronous AsyncNeo4jSaver."""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langchain_neo4j import AsyncNeo4jSaver


class TestAsyncNeo4jSaver:
    """Test cases for asynchronous Neo4j checkpoint saver."""

    @pytest.mark.asyncio
    async def test_aput_and_aget_tuple(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test async storing and retrieving a checkpoint."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )

        # Store checkpoint
        result_config = await clean_async_neo4j_saver.aput(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Verify returned config has checkpoint_id
        assert result_config["configurable"]["checkpoint_id"] == sample_checkpoint["id"]
        assert result_config["configurable"]["thread_id"] == thread_id

        # Retrieve checkpoint
        tuple_ = await clean_async_neo4j_saver.aget_tuple(result_config)

        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == sample_checkpoint["id"]
        assert tuple_.metadata["source"] == "input"
        assert tuple_.config["configurable"]["thread_id"] == thread_id

    @pytest.mark.asyncio
    async def test_aget_latest_checkpoint(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
    ) -> None:
        """Test async retrieving latest checkpoint when no checkpoint_id specified."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )

        # Store multiple checkpoints
        for i in range(3):
            checkpoint: dict[str, Any] = {
                "v": 1,
                "id": f"checkpoint-{i}",
                "ts": f"2024-01-0{i + 1}T00:00:00Z",
                "channel_values": {"counter": i},
                "channel_versions": {"counter": f"{i:032}.{0:016}"},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: dict[str, Any] = {
                "source": "loop",
                "step": i,
                "writes": {},
                "parents": {},
            }

            result = await clean_async_neo4j_saver.aput(
                config,
                cast(Checkpoint, checkpoint),
                cast(CheckpointMetadata, metadata),
                {},
            )
            config = cast(
                RunnableConfig,
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                        "checkpoint_id": result["configurable"]["checkpoint_id"],
                    }
                },
            )

        # Get without checkpoint_id should return latest
        latest_config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )
        tuple_ = await clean_async_neo4j_saver.aget_tuple(latest_config)

        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == "checkpoint-2"

    @pytest.mark.asyncio
    async def test_aput_writes(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test async storing pending writes."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )

        # First create a checkpoint
        result_config = await clean_async_neo4j_saver.aput(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Store writes
        writes: list[tuple[str, Any]] = [
            ("messages", {"role": "user", "content": "Hello again"}),
            ("counter", 42),
        ]
        await clean_async_neo4j_saver.aput_writes(
            result_config, writes, task_id="task-1"
        )

        # Verify writes are retrieved with checkpoint
        tuple_ = await clean_async_neo4j_saver.aget_tuple(result_config)

        assert tuple_ is not None
        assert tuple_.pending_writes is not None
        assert len(tuple_.pending_writes) == 2

        # Check writes content
        write_channels = [w[1] for w in tuple_.pending_writes]
        assert "messages" in write_channels
        assert "counter" in write_channels

    @pytest.mark.asyncio
    async def test_alist_checkpoints(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
    ) -> None:
        """Test async listing checkpoints with filtering and pagination."""
        thread_id = f"test-thread-{uuid.uuid4()}"

        # Store multiple checkpoints
        for i in range(5):
            config = cast(
                RunnableConfig,
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            )
            checkpoint: dict[str, Any] = {
                "v": 1,
                "id": f"cp-{i:03d}",
                "ts": f"2024-01-0{i + 1}T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: dict[str, Any] = {
                "source": "loop",
                "step": i,
                "writes": {},
                "parents": {},
            }
            await clean_async_neo4j_saver.aput(
                config,
                cast(Checkpoint, checkpoint),
                cast(CheckpointMetadata, metadata),
                {},
            )

        # List all
        list_config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )
        all_checkpoints = []
        async for cp in clean_async_neo4j_saver.alist(list_config):
            all_checkpoints.append(cp)

        assert len(all_checkpoints) == 5

        # Verify ordering (newest first)
        checkpoint_ids = [c.checkpoint["id"] for c in all_checkpoints]
        assert checkpoint_ids == ["cp-004", "cp-003", "cp-002", "cp-001", "cp-000"]

    @pytest.mark.asyncio
    async def test_adelete_thread(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test async deleting all checkpoints for a thread."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        )

        # Store checkpoint
        await clean_async_neo4j_saver.aput(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Verify exists
        tuple_ = await clean_async_neo4j_saver.aget_tuple(config)
        assert tuple_ is not None

        # Delete
        await clean_async_neo4j_saver.adelete_thread(thread_id)

        # Verify deleted
        tuple_ = await clean_async_neo4j_saver.aget_tuple(config)
        assert tuple_ is None

    @pytest.mark.asyncio
    async def test_aget_nonexistent_checkpoint(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
    ) -> None:
        """Test async retrieving a non-existent checkpoint returns None."""
        config = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "nonexistent-thread",
                    "checkpoint_ns": "",
                }
            },
        )

        tuple_ = await clean_async_neo4j_saver.aget_tuple(config)
        assert tuple_ is None

    @pytest.mark.asyncio
    async def test_async_multiple_threads(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
    ) -> None:
        """Test that async checkpoints from different threads are isolated."""
        thread_id_1 = f"test-thread-{uuid.uuid4()}"
        thread_id_2 = f"test-thread-{uuid.uuid4()}"

        # Store checkpoint for thread 1
        config_1 = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id_1, "checkpoint_ns": ""}},
        )
        checkpoint_1: dict[str, Any] = {
            "v": 1,
            "id": "thread1-cp",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"data": "thread1"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: dict[str, Any] = {
            "source": "input",
            "step": 0,
            "writes": {},
            "parents": {},
        }
        await clean_async_neo4j_saver.aput(
            config_1,
            cast(Checkpoint, checkpoint_1),
            cast(CheckpointMetadata, metadata),
            {},
        )

        # Store checkpoint for thread 2
        config_2 = cast(
            RunnableConfig,
            {"configurable": {"thread_id": thread_id_2, "checkpoint_ns": ""}},
        )
        checkpoint_2: dict[str, Any] = {
            "v": 1,
            "id": "thread2-cp",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"data": "thread2"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        await clean_async_neo4j_saver.aput(
            config_2,
            cast(Checkpoint, checkpoint_2),
            cast(CheckpointMetadata, metadata),
            {},
        )

        # Verify threads are isolated
        tuple_1 = await clean_async_neo4j_saver.aget_tuple(config_1)
        tuple_2 = await clean_async_neo4j_saver.aget_tuple(config_2)

        assert tuple_1 is not None
        assert tuple_2 is not None
        assert tuple_1.checkpoint["id"] == "thread1-cp"
        assert tuple_2.checkpoint["id"] == "thread2-cp"

    @pytest.mark.asyncio
    async def test_async_config_missing_thread_id_raises(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that async missing thread_id raises ValueError."""
        config = cast(RunnableConfig, {"configurable": {}})

        with pytest.raises(ValueError, match="thread_id is required"):
            await clean_async_neo4j_saver.aput(
                config,
                cast(Checkpoint, sample_checkpoint),
                cast(CheckpointMetadata, sample_metadata),
                {},
            )

    @pytest.mark.asyncio
    async def test_aput_writes_without_checkpoint_id_raises(
        self,
        clean_async_neo4j_saver: AsyncNeo4jSaver,
    ) -> None:
        """Test that async put_writes without checkpoint_id raises ValueError."""
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": "test", "checkpoint_ns": ""}},
        )

        with pytest.raises(ValueError, match="checkpoint_id is required"):
            await clean_async_neo4j_saver.aput_writes(
                config, [("channel", "value")], "task-1"
            )
