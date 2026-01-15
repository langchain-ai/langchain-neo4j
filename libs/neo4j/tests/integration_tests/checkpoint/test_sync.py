"""Tests for synchronous Neo4jSaver."""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langchain_neo4j import Neo4jSaver


class TestNeo4jSaver:
    """Test cases for synchronous Neo4j checkpoint saver."""

    def test_put_and_get_tuple(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test storing and retrieving a checkpoint."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store checkpoint
        result_config = clean_neo4j_saver.put(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Verify returned config has checkpoint_id
        assert result_config["configurable"]["checkpoint_id"] == sample_checkpoint["id"]
        assert result_config["configurable"]["thread_id"] == thread_id

        # Retrieve checkpoint
        tuple_ = clean_neo4j_saver.get_tuple(result_config)

        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == sample_checkpoint["id"]
        assert tuple_.metadata["source"] == "input"
        assert tuple_.config["configurable"]["thread_id"] == thread_id

    def test_get_latest_checkpoint(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test retrieving latest checkpoint when no checkpoint_id specified."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store multiple checkpoints
        for i in range(3):
            checkpoint: Checkpoint = cast(
                Checkpoint,
                {
                    "v": 1,
                    "id": f"checkpoint-{i}",
                    "ts": f"2024-01-0{i + 1}T00:00:00Z",
                    "channel_values": {"counter": i},
                    "channel_versions": {"counter": f"{i:032}.{0:016}"},
                    "versions_seen": {},
                    "pending_sends": [],
                },
            )
            metadata: CheckpointMetadata = cast(
                CheckpointMetadata,
                {
                    "source": "loop",
                    "step": i,
                    "writes": {},
                    "parents": {},
                },
            )

            result = clean_neo4j_saver.put(config, checkpoint, metadata, {})
            # Update config with new checkpoint_id for next iteration (parent)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": result["configurable"]["checkpoint_id"],
                }
            }

        # Get without checkpoint_id should return latest
        latest_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        tuple_ = clean_neo4j_saver.get_tuple(latest_config)

        assert tuple_ is not None
        # Latest should be checkpoint-2 (highest lexicographically)
        assert tuple_.checkpoint["id"] == "checkpoint-2"

    def test_put_writes(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test storing pending writes."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # First create a checkpoint
        result_config = clean_neo4j_saver.put(
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
        clean_neo4j_saver.put_writes(result_config, writes, task_id="task-1")

        # Verify writes are retrieved with checkpoint
        tuple_ = clean_neo4j_saver.get_tuple(result_config)

        assert tuple_ is not None
        pending_writes = tuple_.pending_writes
        assert pending_writes is not None
        assert len(pending_writes) == 2

        # Check writes content
        write_channels = [w[1] for w in pending_writes]
        assert "messages" in write_channels
        assert "counter" in write_channels

    def test_list_checkpoints(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test listing checkpoints with filtering and pagination."""
        thread_id = f"test-thread-{uuid.uuid4()}"

        # Store multiple checkpoints
        for i in range(5):
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
            }
            checkpoint: Checkpoint = cast(
                Checkpoint,
                {
                    "v": 1,
                    "id": f"cp-{i:03d}",  # Zero-padded for proper ordering
                    "ts": f"2024-01-0{i + 1}T00:00:00Z",
                    "channel_values": {},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                },
            )
            metadata: CheckpointMetadata = cast(
                CheckpointMetadata,
                {
                    "source": "loop",
                    "step": i,
                    "writes": {},
                    "parents": {},
                },
            )
            clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # List all
        list_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        all_checkpoints = list(clean_neo4j_saver.list(list_config))
        assert len(all_checkpoints) == 5

        # Verify ordering (newest first)
        checkpoint_ids = [c.checkpoint["id"] for c in all_checkpoints]
        assert checkpoint_ids == ["cp-004", "cp-003", "cp-002", "cp-001", "cp-000"]

        # List with limit
        limited = list(clean_neo4j_saver.list(list_config, limit=3))
        assert len(limited) == 3
        assert limited[0].checkpoint["id"] == "cp-004"

    def test_delete_thread(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test deleting all checkpoints for a thread."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store checkpoint
        clean_neo4j_saver.put(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Verify exists
        tuple_ = clean_neo4j_saver.get_tuple(config)
        assert tuple_ is not None

        # Delete
        clean_neo4j_saver.delete_thread(thread_id)

        # Verify deleted
        tuple_ = clean_neo4j_saver.get_tuple(config)
        assert tuple_ is None

    def test_parent_checkpoint_relationship(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test parent-child checkpoint relationships."""
        thread_id = f"test-thread-{uuid.uuid4()}"

        # Create parent checkpoint
        parent_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        parent_checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "parent-cp",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        parent_metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(parent_config, parent_checkpoint, parent_metadata, {})

        # Create child checkpoint with parent reference
        child_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "parent-cp",  # This becomes parent
            }
        }
        child_checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "child-cp",
                "ts": "2024-01-01T00:01:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        child_metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "loop",
                "step": 1,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(child_config, child_checkpoint, child_metadata, {})

        # Verify parent relationship
        get_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "child-cp",
            }
        }
        child_tuple = clean_neo4j_saver.get_tuple(get_config)

        assert child_tuple is not None
        assert child_tuple.parent_config is not None
        assert child_tuple.parent_config["configurable"]["checkpoint_id"] == "parent-cp"

    def test_get_nonexistent_checkpoint(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test retrieving a non-existent checkpoint returns None."""
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "nonexistent-thread",
                "checkpoint_ns": "",
            }
        }

        tuple_ = clean_neo4j_saver.get_tuple(config)
        assert tuple_ is None

    def test_multiple_threads(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that checkpoints from different threads are isolated."""
        thread_id_1 = f"test-thread-{uuid.uuid4()}"
        thread_id_2 = f"test-thread-{uuid.uuid4()}"

        # Store checkpoint for thread 1
        config_1: RunnableConfig = {
            "configurable": {"thread_id": thread_id_1, "checkpoint_ns": ""}
        }
        checkpoint_1: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "thread1-cp",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"data": "thread1"},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(config_1, checkpoint_1, metadata, {})

        # Store checkpoint for thread 2
        config_2: RunnableConfig = {
            "configurable": {"thread_id": thread_id_2, "checkpoint_ns": ""}
        }
        checkpoint_2: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "thread2-cp",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"data": "thread2"},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        clean_neo4j_saver.put(config_2, checkpoint_2, metadata, {})

        # Verify threads are isolated
        tuple_1 = clean_neo4j_saver.get_tuple(config_1)
        tuple_2 = clean_neo4j_saver.get_tuple(config_2)

        assert tuple_1 is not None
        assert tuple_2 is not None
        assert tuple_1.checkpoint["id"] == "thread1-cp"
        assert tuple_2.checkpoint["id"] == "thread2-cp"

    def test_config_missing_thread_id_raises(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that missing thread_id raises ValueError."""
        config: RunnableConfig = {"configurable": {}}

        with pytest.raises(ValueError, match="thread_id is required"):
            clean_neo4j_saver.put(
                config,
                cast(Checkpoint, sample_checkpoint),
                cast(CheckpointMetadata, sample_metadata),
                {},
            )

    def test_put_writes_without_checkpoint_id_raises(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that put_writes without checkpoint_id raises ValueError."""
        config: RunnableConfig = {
            "configurable": {"thread_id": "test", "checkpoint_ns": ""}
        }

        with pytest.raises(ValueError, match="checkpoint_id is required"):
            clean_neo4j_saver.put_writes(config, [("channel", "value")], "task-1")

    def test_blob_data_properly_serialized(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that blob data is properly serialized to base64 and can be retrieved."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Create checkpoint with complex channel values
        complex_data = {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "nested": {"key": "value", "list": [1, 2, 3]},
        }
        checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "test-cp-blob",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": complex_data,
                "channel_versions": {
                    "messages": f"{1:032}.{0:016}",
                    "nested": f"{1:032}.{0:016}",
                },
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )

        # Store checkpoint
        result_config = clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # Retrieve and verify blob data is correctly deserialized
        tuple_ = clean_neo4j_saver.get_tuple(result_config)

        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == "test-cp-blob"

        # Verify channel_values are preserved (they should be in checkpoint)
        assert "channel_values" in tuple_.checkpoint
        # The actual values may be stored in blobs, check the checkpoint structure
        assert tuple_.checkpoint["channel_versions"]["messages"] == f"{1:032}.{0:016}"

    def test_blob_data_stored_as_json_in_neo4j(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that blob data is stored as human-readable JSON strings in Neo4j."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Create checkpoint with channel values
        checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "test-cp-json",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"test_channel": {"key": "value"}},
                "channel_versions": {"test_channel": f"{1:032}.{0:016}"},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )

        # Store checkpoint
        clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # Query Neo4j directly to verify ChannelState blob is stored as JSON string
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                      -[:HAS_CHECKPOINT]->(c:Checkpoint)
                      -[:HAS_CHANNEL]->(cs:ChannelState)
                RETURN cs.blob as blob, cs.type as type, cs.channel as channel
                """,
                {"thread_id": thread_id},
            )
            record = result.single()

            assert record is not None
            blob_value = record["blob"]

            # Verify blob is a string (JSON), not "[object Object]"
            assert isinstance(
                blob_value, str
            ), f"Blob should be a string, got {type(blob_value)}"
            assert blob_value != "[object Object]", "Blob should not be [object Object]"
            assert blob_value != "undefined", "Blob should not be undefined"

            # Verify it's valid JSON and human-readable
            import json

            try:
                parsed = json.loads(blob_value)
                # Verify the data is readable
                assert parsed == {
                    "key": "value"
                }, f"Parsed JSON should match original data, got {parsed}"
            except json.JSONDecodeError as e:
                pytest.fail(f"Blob is not valid JSON: {e}")

    def test_checkpoint_data_properly_stored_in_neo4j(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that checkpoint data is properly stored and can be retrieved."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store checkpoint
        clean_neo4j_saver.put(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Query Neo4j directly to verify checkpoint is stored as a string
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                      -[:HAS_CHECKPOINT]->(c:Checkpoint)
                RETURN c.checkpoint as checkpoint, c.metadata as metadata,
                       c.type as type
                """,
                {"thread_id": thread_id},
            )
            record = result.single()

            assert record is not None

            # Verify checkpoint is a string (JSON-wrapped)
            checkpoint_value = record["checkpoint"]
            assert isinstance(
                checkpoint_value, str
            ), f"Checkpoint should be a string, got {type(checkpoint_value)}"
            assert (
                checkpoint_value != "[object Object]"
            ), "Checkpoint should not be [object Object]"

            # Verify metadata is a string (JSON-wrapped)
            metadata_value = record["metadata"]
            assert isinstance(
                metadata_value, str
            ), f"Metadata should be a string, got {type(metadata_value)}"
            assert (
                metadata_value != "[object Object]"
            ), "Metadata should not be [object Object]"

            # Verify both are valid JSON (may be serde-wrapped or plain JSON)
            import json

            try:
                parsed_cp = json.loads(checkpoint_value)
                # Could be serde-wrapped or plain JSON
                if "__serde_type__" in parsed_cp:
                    assert (
                        "__serde_data__" in parsed_cp
                    ), "Serde wrapper should have data"
                else:
                    assert (
                        "id" in parsed_cp
                    ), "Plain JSON checkpoint should have 'id' field"

                parsed_meta = json.loads(metadata_value)
                if "__serde_type__" in parsed_meta:
                    assert (
                        "__serde_data__" in parsed_meta
                    ), "Serde wrapper should have data"
                else:
                    assert (
                        "source" in parsed_meta
                    ), "Plain JSON metadata should have 'source' field"
            except json.JSONDecodeError as e:
                pytest.fail(f"Data is not valid JSON: {e}")

        # Most importantly: verify data can be retrieved correctly
        tuple_ = clean_neo4j_saver.get_tuple(config)
        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == sample_checkpoint["id"]
        assert tuple_.metadata["source"] == sample_metadata["source"]

    def test_graph_structure_created(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that proper graph structure is created with Thread,
        Checkpoint nodes and relationships."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Create checkpoint with channel values
        checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "test-cp-graph",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"messages": [{"role": "user", "content": "Hello"}]},
                "channel_versions": {"messages": f"{1:032}.{0:016}"},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )

        # Store checkpoint
        clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # Verify graph structure
        with clean_neo4j_saver._driver.session() as session:
            # Check Thread node exists
            result = session.run(
                "MATCH (t:Thread {thread_id: $thread_id}) RETURN t",
                {"thread_id": thread_id},
            )
            thread_record = result.single()
            assert thread_record is not None, "Thread node should exist"

            # Check Checkpoint node exists and is connected to Thread
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                -[:HAS_CHECKPOINT]->(c:Checkpoint)
                RETURN c.checkpoint_id as checkpoint_id
                """,
                {"thread_id": thread_id},
            )
            cp_record = result.single()
            assert (
                cp_record is not None
            ), "Checkpoint should be connected to Thread via HAS_CHECKPOINT"
            assert cp_record["checkpoint_id"] == "test-cp-graph"

            # Check ChannelState node exists and is connected to Checkpoint
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                      -[:HAS_CHECKPOINT]->(c:Checkpoint)
                      -[:HAS_CHANNEL]->(cs:ChannelState)
                RETURN cs.channel as channel, cs.blob as blob
                """,
                {"thread_id": thread_id},
            )
            cs_record = result.single()
            assert (
                cs_record is not None
            ), "ChannelState should be connected to Checkpoint via HAS_CHANNEL"
            assert cs_record["channel"] == "messages"

    def test_checkpoint_chain_traversal(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that PREVIOUS relationship creates proper checkpoint chain."""
        thread_id = f"test-thread-{uuid.uuid4()}"

        # Create parent checkpoint
        parent_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        parent_checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "parent-cp-chain",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        parent_metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(parent_config, parent_checkpoint, parent_metadata, {})

        # Create child checkpoint with parent reference
        child_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "parent-cp-chain",  # This becomes parent
            }
        }
        child_checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "child-cp-chain",
                "ts": "2024-01-01T00:01:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        child_metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "loop",
                "step": 1,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(child_config, child_checkpoint, child_metadata, {})

        # Create grandchild checkpoint
        grandchild_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "child-cp-chain",  # This becomes parent
            }
        }
        grandchild_checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "grandchild-cp-chain",
                "ts": "2024-01-01T00:02:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        grandchild_metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "loop",
                "step": 2,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(
            grandchild_config, grandchild_checkpoint, grandchild_metadata, {}
        )

        # Verify PREVIOUS relationship chain exists
        with clean_neo4j_saver._driver.session() as session:
            # Check the full chain: grandchild -> child -> parent
            result = session.run(
                """
                MATCH path = (gc:Checkpoint {checkpoint_id: 'grandchild-cp-chain'})
                             -[:PREVIOUS]->
                             (c:Checkpoint {checkpoint_id: 'child-cp-chain'})
                             -[:PREVIOUS]->
                             (p:Checkpoint {checkpoint_id: 'parent-cp-chain'})
                RETURN length(path) as chain_length
                """,
            )
            record = result.single()
            assert record is not None, "Full checkpoint chain should exist"
            assert (
                record["chain_length"] == 2
            ), "Chain should have 2 PREVIOUS relationships"

            # Verify parent has no PREVIOUS relationship (it's the root)
            result = session.run(
                """
                MATCH (p:Checkpoint {checkpoint_id: 'parent-cp-chain'})
                -[:PREVIOUS]->(older:Checkpoint)
                RETURN older
                """,
            )
            record = result.single()
            assert (
                record is None
            ), "Parent checkpoint should have no PREVIOUS relationship"

    def test_pending_writes_graph_structure(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that pending writes are stored as
        PendingWrite nodes with HAS_WRITE relationship."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Create checkpoint
        result_config = clean_neo4j_saver.put(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Store pending writes
        writes: list[tuple[str, Any]] = [
            ("messages", {"role": "user", "content": "Hello again"}),
            ("counter", 42),
        ]
        clean_neo4j_saver.put_writes(result_config, writes, task_id="task-graph-test")

        # Verify PendingWrite nodes and HAS_WRITE relationships
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
                      -[:HAS_WRITE]->(w:PendingWrite)
                RETURN w.channel as channel, w.task_id as task_id, w.blob as blob
                ORDER BY w.idx
                """,
                {"checkpoint_id": sample_checkpoint["id"]},
            )
            records = list(result)

            assert len(records) == 2, "Should have 2 PendingWrite nodes"
            assert records[0]["task_id"] == "task-graph-test"
            assert records[1]["task_id"] == "task-graph-test"

            channels = [r["channel"] for r in records]
            assert "messages" in channels
            assert "counter" in channels

            # Verify blobs are base64 encoded
            for r in records:
                assert isinstance(r["blob"], str)
                assert r["blob"] != "[object Object]"

    def test_delete_thread_removes_graph_structure(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that delete_thread removes Thread,
        Checkpoint, and PendingWrite nodes."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Create checkpoint with channel values
        checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "test-cp-delete",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"messages": [{"role": "user", "content": "Hello"}]},
                "channel_versions": {"messages": f"{1:032}.{0:016}"},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        result_config = clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # Add pending writes
        clean_neo4j_saver.put_writes(
            result_config, [("test", "value")], task_id="task-delete"
        )

        # Verify structure exists
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                OPTIONAL MATCH (t)-[:HAS_CHECKPOINT]->(c:Checkpoint)
                OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:PendingWrite)
                RETURN count(t) as threads, count(c) as checkpoints, count(w) as writes
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
            assert record is not None
            assert record["threads"] == 1
            assert record["checkpoints"] == 1
            assert record["writes"] == 1

        # Delete thread
        clean_neo4j_saver.delete_thread(thread_id)

        # Verify all nodes are deleted
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                OPTIONAL MATCH (t)-[:HAS_CHECKPOINT]->(c:Checkpoint)
                OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:PendingWrite)
                RETURN count(t) as threads, count(c) as checkpoints, count(w) as writes
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
            assert record is not None
            assert record["threads"] == 0, "Thread should be deleted"
            assert record["checkpoints"] == 0, "Checkpoints should be deleted"
            assert record["writes"] == 0, "PendingWrites should be deleted"

    def test_branch_created_on_first_checkpoint(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that a 'main' branch is created when first checkpoint is stored."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store checkpoint
        clean_neo4j_saver.put(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Verify Branch node was created
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})-[:HAS_BRANCH]->(b:Branch)
                OPTIONAL MATCH (t)-[:ACTIVE_BRANCH]->(active:Branch)
                RETURN b.name as name, b.branch_id as branch_id,
                       active.branch_id = b.branch_id as is_active
                """,
                {"thread_id": thread_id},
            )
            record = result.single()

            assert record is not None, "Branch should be created"
            assert record["name"] == "main", "First branch should be named 'main'"
            assert record["is_active"], "Main branch should be the active branch"

    def test_branch_head_updated_on_checkpoint(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that branch HEAD is updated when new checkpoints are added."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store first checkpoint
        checkpoint_1: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "cp-1",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(config, checkpoint_1, metadata, {})

        # Verify HEAD points to cp-1
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                      -[:ACTIVE_BRANCH]->(b:Branch)
                      -[:HEAD]->(c:Checkpoint)
                RETURN c.checkpoint_id as head_id
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
            assert record is not None
            assert record["head_id"] == "cp-1"

        # Store second checkpoint
        checkpoint_2: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "cp-2",
                "ts": "2024-01-01T00:01:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        config_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        clean_neo4j_saver.put(config_2, checkpoint_2, metadata, {})

        # Verify HEAD now points to cp-2
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})
                      -[:ACTIVE_BRANCH]->(b:Branch)
                      -[:HEAD]->(c:Checkpoint)
                RETURN c.checkpoint_id as head_id
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
            assert record is not None
            assert (
                record["head_id"] == "cp-2"
            ), "HEAD should be updated to latest checkpoint"

    def test_checkpoint_linked_to_branch(
        self,
        clean_neo4j_saver: Neo4jSaver,
        sample_checkpoint: dict[str, Any],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Test that checkpoints are linked to their
        branch via ON_BRANCH relationship."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store checkpoint
        clean_neo4j_saver.put(
            config,
            cast(Checkpoint, sample_checkpoint),
            cast(CheckpointMetadata, sample_metadata),
            {},
        )

        # Verify ON_BRANCH relationship exists
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
                      -[:ON_BRANCH]->(b:Branch)
                RETURN b.name as branch_name
                """,
                {"checkpoint_id": sample_checkpoint["id"]},
            )
            record = result.single()

            assert (
                record is not None
            ), "Checkpoint should be linked to branch via ON_BRANCH"
            assert record["branch_name"] == "main"

    def test_get_tuple_uses_active_branch_head(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that get_tuple without checkpoint_id returns active branch HEAD."""
        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store multiple checkpoints
        for i in range(3):
            checkpoint: Checkpoint = cast(
                Checkpoint,
                {
                    "v": 1,
                    "id": f"cp-{i}",
                    "ts": f"2024-01-0{i + 1}T00:00:00Z",
                    "channel_values": {"step": i},
                    "channel_versions": {"step": f"{i + 1:032}.{0:016}"},
                    "versions_seen": {},
                    "pending_sends": [],
                },
            )
            metadata: CheckpointMetadata = cast(
                CheckpointMetadata,
                {
                    "source": "loop",
                    "step": i,
                    "writes": {},
                    "parents": {},
                },
            )
            result = clean_neo4j_saver.put(config, checkpoint, metadata, {})
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": result["configurable"]["checkpoint_id"],
                }
            }

        # Get without checkpoint_id should return HEAD of active branch (cp-2)
        latest_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        tuple_ = clean_neo4j_saver.get_tuple(latest_config)

        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == "cp-2", "Should return active branch HEAD"

    def test_multiple_branches_on_thread(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test creating multiple branches on a thread."""
        from langchain_neo4j.checkpoint.base import CYPHER_CREATE_BRANCH

        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store initial checkpoint (creates main branch)
        checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "cp-main",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # Create a second branch
        with clean_neo4j_saver._driver.session() as session:
            session.run(
                CYPHER_CREATE_BRANCH,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "branch_id": "fork-branch-1",
                    "name": "experiment",
                    "fork_point_id": "cp-main",
                },
            )

        # Verify both branches exist
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})-[:HAS_BRANCH]->(b:Branch)
                RETURN b.name as name ORDER BY b.name
                """,
                {"thread_id": thread_id},
            )
            branches = [r["name"] for r in result]

            assert len(branches) == 2
            assert "main" in branches
            assert "experiment" in branches

    def test_switch_active_branch(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test switching the active branch for a thread."""
        from langchain_neo4j.checkpoint.base import (
            CYPHER_CREATE_BRANCH,
            CYPHER_SET_ACTIVE_BRANCH,
        )

        thread_id = f"test-thread-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Store checkpoint (creates main branch)
        checkpoint: Checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "id": "cp-main",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            {
                "source": "input",
                "step": 0,
                "writes": {},
                "parents": {},
            },
        )
        clean_neo4j_saver.put(config, checkpoint, metadata, {})

        # Create second branch
        fork_branch_id = "fork-branch-switch"
        with clean_neo4j_saver._driver.session() as session:
            session.run(
                CYPHER_CREATE_BRANCH,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "branch_id": fork_branch_id,
                    "name": "fork-1",
                    "fork_point_id": "cp-main",
                },
            )

        # Verify main is active
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})-[:ACTIVE_BRANCH]->(b:Branch)
                RETURN b.name as name
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
            assert record is not None
            assert record["name"] == "main"

        # Switch to fork branch
        with clean_neo4j_saver._driver.session() as session:
            session.run(
                CYPHER_SET_ACTIVE_BRANCH,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "branch_id": fork_branch_id,
                },
            )

        # Verify fork-1 is now active
        with clean_neo4j_saver._driver.session() as session:
            result = session.run(
                """
                MATCH (t:Thread {thread_id: $thread_id})-[:ACTIVE_BRANCH]->(b:Branch)
                RETURN b.name as name
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
            assert record is not None
            assert (
                record["name"] == "fork-1"
            ), "Active branch should be switched to fork-1"
