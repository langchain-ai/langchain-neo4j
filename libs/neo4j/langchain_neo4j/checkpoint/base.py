"""Base class for Neo4j checkpoint savers."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# =============================================================================
# Cypher queries for schema setup - Graph Model
# =============================================================================

CYPHER_CREATE_THREAD_CONSTRAINT = """
CREATE CONSTRAINT thread_unique IF NOT EXISTS
FOR (t:Thread) REQUIRE (t.thread_id, t.checkpoint_ns) IS UNIQUE
"""

CYPHER_CREATE_CHECKPOINT_CONSTRAINT = """
CREATE CONSTRAINT checkpoint_id_unique IF NOT EXISTS
FOR (c:Checkpoint) REQUIRE c.checkpoint_id IS UNIQUE
"""

CYPHER_CREATE_CHANNEL_STATE_CONSTRAINT = """
CREATE CONSTRAINT channel_state_unique IF NOT EXISTS
FOR (cs:ChannelState) REQUIRE (cs.channel, cs.version) IS UNIQUE
"""

CYPHER_CREATE_CHECKPOINT_ID_INDEX = """
CREATE INDEX checkpoint_id_idx IF NOT EXISTS
FOR (c:Checkpoint) ON (c.checkpoint_id)
"""

CYPHER_CREATE_CHECKPOINT_CREATED_INDEX = """
CREATE INDEX checkpoint_created_idx IF NOT EXISTS
FOR (c:Checkpoint) ON (c.created_at)
"""

# =============================================================================
# Branch-related constraints and indexes
# =============================================================================

CYPHER_CREATE_BRANCH_CONSTRAINT = """
CREATE CONSTRAINT branch_id_unique IF NOT EXISTS
FOR (b:Branch) REQUIRE b.branch_id IS UNIQUE
"""

CYPHER_CREATE_BRANCH_NAME_INDEX = """
CREATE INDEX branch_name_idx IF NOT EXISTS
FOR (b:Branch) ON (b.name)
"""

# =============================================================================
# Cypher queries for checkpoint operations - Graph Model
# =============================================================================

# Create/update Thread and Checkpoint with HAS_CHECKPOINT relationship
CYPHER_UPSERT_CHECKPOINT = """
MERGE (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
CREATE (c:Checkpoint {
    checkpoint_id: $checkpoint_id,
    type: $type,
    checkpoint: $checkpoint,
    metadata: $metadata,
    created_at: datetime()
})
CREATE (t)-[:HAS_CHECKPOINT]->(c)
WITH c
CALL {
    WITH c
    MATCH (parent:Checkpoint {checkpoint_id: $parent_checkpoint_id})
    CREATE (c)-[:PREVIOUS]->(parent)
    RETURN parent
    UNION ALL
    WITH c
    WHERE $parent_checkpoint_id IS NULL
    RETURN null as parent
}
RETURN c.checkpoint_id as checkpoint_id
"""

# Simpler version without CALL subquery for Neo4j < 5.0 compatibility
CYPHER_UPSERT_CHECKPOINT_SIMPLE = """
MERGE (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
CREATE (c:Checkpoint {
    checkpoint_id: $checkpoint_id,
    type: $type,
    checkpoint: $checkpoint,
    metadata: $metadata,
    created_at: datetime()
})
CREATE (t)-[:HAS_CHECKPOINT]->(c)
RETURN c.checkpoint_id as checkpoint_id
"""

# Link checkpoint to parent (separate query for compatibility)
CYPHER_LINK_PARENT_CHECKPOINT = """
MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
MATCH (parent:Checkpoint {checkpoint_id: $parent_checkpoint_id})
MERGE (c)-[:PREVIOUS]->(parent)
"""

# Create ChannelState and link to Checkpoint with HAS_CHANNEL relationship
CYPHER_UPSERT_CHANNEL_STATE = """
MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
MERGE (cs:ChannelState {channel: $channel, version: $version})
ON CREATE SET cs.type = $type, cs.blob = $blob
CREATE (c)-[:HAS_CHANNEL]->(cs)
"""

# Create PendingWrite and link to Checkpoint with HAS_WRITE relationship
CYPHER_UPSERT_WRITE = """
MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
CREATE (w:PendingWrite {
    task_id: $task_id,
    task_path: $task_path,
    idx: $idx,
    channel: $channel,
    type: $type,
    blob: $blob
})
CREATE (c)-[:HAS_WRITE]->(w)
"""

# Get checkpoint by ID with relationships
CYPHER_GET_CHECKPOINT_BY_ID = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:HAS_CHECKPOINT]->(c:Checkpoint {checkpoint_id: $checkpoint_id})
OPTIONAL MATCH (c)-[:PREVIOUS]->(parent:Checkpoint)
RETURN t.thread_id as thread_id,
       t.checkpoint_ns as checkpoint_ns,
       c.checkpoint_id as checkpoint_id,
       parent.checkpoint_id as parent_checkpoint_id,
       c.type as type,
       c.checkpoint as checkpoint,
       c.metadata as metadata
"""

# Get latest checkpoint for thread
CYPHER_GET_LATEST_CHECKPOINT = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:HAS_CHECKPOINT]->(c:Checkpoint)
WITH t, c ORDER BY c.checkpoint_id DESC LIMIT 1
OPTIONAL MATCH (c)-[:PREVIOUS]->(parent:Checkpoint)
RETURN t.thread_id as thread_id,
       t.checkpoint_ns as checkpoint_ns,
       c.checkpoint_id as checkpoint_id,
       parent.checkpoint_id as parent_checkpoint_id,
       c.type as type,
       c.checkpoint as checkpoint,
       c.metadata as metadata
"""

# Get channel states for a checkpoint
CYPHER_GET_CHANNEL_STATES = """
MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
      -[:HAS_CHANNEL]->(cs:ChannelState)
WHERE cs.channel IN $channels AND cs.version IN $versions
RETURN cs.channel as channel,
       cs.type as type,
       cs.blob as blob,
       cs.version as version
"""

# Get pending writes for a checkpoint
CYPHER_GET_WRITES = """
MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
      -[:HAS_WRITE]->(w:PendingWrite)
RETURN w.task_id as task_id,
       w.task_path as task_path,
       w.channel as channel,
       w.type as type,
       w.blob as blob,
       w.idx as idx
ORDER BY w.idx
"""

# List checkpoints for a thread
CYPHER_LIST_CHECKPOINTS = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:HAS_CHECKPOINT]->(c:Checkpoint)
WHERE $before_id IS NULL OR c.checkpoint_id < $before_id
WITH t, c ORDER BY c.checkpoint_id DESC LIMIT $limit
OPTIONAL MATCH (c)-[:PREVIOUS]->(parent:Checkpoint)
RETURN t.thread_id as thread_id,
       t.checkpoint_ns as checkpoint_ns,
       c.checkpoint_id as checkpoint_id,
       parent.checkpoint_id as parent_checkpoint_id,
       c.type as type,
       c.checkpoint as checkpoint,
       c.metadata as metadata
"""

# Delete thread and all related data (cascade via relationships)
CYPHER_DELETE_THREAD = """
MATCH (t:Thread {thread_id: $thread_id})
OPTIONAL MATCH (t)-[:HAS_CHECKPOINT]->(c:Checkpoint)
OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:PendingWrite)
DETACH DELETE t, c, w
"""

# Clean up orphaned ChannelState nodes (not connected to any checkpoint)
CYPHER_DELETE_ORPHAN_CHANNEL_STATES = """
MATCH (cs:ChannelState)
WHERE NOT (cs)<-[:HAS_CHANNEL]-()
DELETE cs
"""

# =============================================================================
# Branch-related Cypher queries
# =============================================================================

# Create the main branch for a thread (called on first checkpoint)
CYPHER_CREATE_MAIN_BRANCH = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
WHERE NOT (t)-[:HAS_BRANCH]->()
CREATE (b:Branch {
    branch_id: $branch_id,
    name: 'main',
    created_at: datetime(),
    fork_point_id: null
})
CREATE (t)-[:HAS_BRANCH]->(b)
CREATE (t)-[:ACTIVE_BRANCH]->(b)
RETURN b.branch_id as branch_id
"""

# Create a new branch (fork) from a checkpoint
CYPHER_CREATE_BRANCH = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
CREATE (b:Branch {
    branch_id: $branch_id,
    name: $name,
    created_at: datetime(),
    fork_point_id: $fork_point_id
})
CREATE (t)-[:HAS_BRANCH]->(b)
WITH t, b
MATCH (c:Checkpoint {checkpoint_id: $fork_point_id})
CREATE (b)-[:HEAD]->(c)
RETURN b.branch_id as branch_id
"""

# Set active branch for a thread
CYPHER_SET_ACTIVE_BRANCH = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
OPTIONAL MATCH (t)-[old:ACTIVE_BRANCH]->()
DELETE old
WITH t
MATCH (t)-[:HAS_BRANCH]->(b:Branch {branch_id: $branch_id})
CREATE (t)-[:ACTIVE_BRANCH]->(b)
RETURN b.branch_id as branch_id
"""

# Update branch HEAD and link checkpoint to branch
CYPHER_UPDATE_BRANCH_HEAD = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:ACTIVE_BRANCH]->(b:Branch)
OPTIONAL MATCH (b)-[old:HEAD]->()
DELETE old
WITH b
MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})
CREATE (b)-[:HEAD]->(c)
MERGE (c)-[:ON_BRANCH]->(b)
RETURN b.branch_id as branch_id
"""

# Get active branch HEAD checkpoint (replaces simple "latest" query)
CYPHER_GET_ACTIVE_BRANCH_HEAD = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:ACTIVE_BRANCH]->(b:Branch)
      -[:HEAD]->(c:Checkpoint)
OPTIONAL MATCH (c)-[:PREVIOUS]->(parent:Checkpoint)
RETURN t.thread_id as thread_id,
       t.checkpoint_ns as checkpoint_ns,
       c.checkpoint_id as checkpoint_id,
       parent.checkpoint_id as parent_checkpoint_id,
       c.type as type,
       c.checkpoint as checkpoint,
       c.metadata as metadata,
       b.branch_id as branch_id,
       b.name as branch_name
"""

# List all branches for a thread
CYPHER_LIST_BRANCHES = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:HAS_BRANCH]->(b:Branch)
OPTIONAL MATCH (t)-[active:ACTIVE_BRANCH]->(b)
OPTIONAL MATCH (b)-[:HEAD]->(head:Checkpoint)
RETURN b.branch_id as branch_id,
       b.name as name,
       b.created_at as created_at,
       b.fork_point_id as fork_point_id,
       active IS NOT NULL as is_active,
       head.checkpoint_id as head_checkpoint_id
ORDER BY b.created_at
"""

# List checkpoints on a specific branch
CYPHER_LIST_BRANCH_CHECKPOINTS = """
MATCH (b:Branch {branch_id: $branch_id})<-[:ON_BRANCH]-(c:Checkpoint)
OPTIONAL MATCH (c)-[:PREVIOUS]->(parent:Checkpoint)
RETURN c.checkpoint_id as checkpoint_id,
       parent.checkpoint_id as parent_checkpoint_id,
       c.type as type,
       c.checkpoint as checkpoint,
       c.metadata as metadata
ORDER BY c.checkpoint_id DESC
LIMIT $limit
"""

# Get checkpoint tree for visualization
CYPHER_GET_CHECKPOINT_TREE = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:HAS_CHECKPOINT]->(c:Checkpoint)
OPTIONAL MATCH (c)-[:PREVIOUS]->(parent:Checkpoint)
OPTIONAL MATCH (c)-[:ON_BRANCH]->(b:Branch)
RETURN c.checkpoint_id as checkpoint_id,
       parent.checkpoint_id as parent_id,
       b.branch_id as branch_id,
       b.name as branch_name
ORDER BY c.checkpoint_id
"""

# Check if thread has any branches (for migration/setup)
CYPHER_THREAD_HAS_BRANCHES = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
OPTIONAL MATCH (t)-[:HAS_BRANCH]->(b:Branch)
RETURN t.thread_id as thread_id, count(b) as branch_count
"""

# Get active branch info
CYPHER_GET_ACTIVE_BRANCH = """
MATCH (t:Thread {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
      -[:ACTIVE_BRANCH]->(b:Branch)
RETURN b.branch_id as branch_id, b.name as name
"""

# Delete a branch (but not its checkpoints - they may be shared)
CYPHER_DELETE_BRANCH = """
MATCH (b:Branch {branch_id: $branch_id})
DETACH DELETE b
"""


class BaseNeo4jSaver(BaseCheckpointSaver):
    """Base class for Neo4j checkpoint savers with shared logic."""

    serde: SerializerProtocol = JsonPlusSerializer()

    @property
    def config_specs(self) -> list[dict[str, Any]]:
        """Return configuration specifications."""
        return [
            {
                "id": "thread_id",
                "annotation": str,
                "name": "Thread ID",
                "description": "Unique identifier for the conversation thread.",
            },
            {
                "id": "checkpoint_ns",
                "annotation": str,
                "name": "Checkpoint Namespace",
                "description": "Namespace for organizing checkpoints.",
                "default": "",
            },
            {
                "id": "checkpoint_id",
                "annotation": Optional[str],
                "name": "Checkpoint ID",
                "description": "Unique identifier for a specific checkpoint.",
                "default": None,
            },
        ]

    @staticmethod
    def _parse_config(config: RunnableConfig) -> tuple[str, str, str | None]:
        """Extract thread_id, checkpoint_ns, and checkpoint_id from config.

        Args:
            config: The runnable configuration dictionary.

        Returns:
            Tuple of (thread_id, checkpoint_ns, checkpoint_id).

        Raises:
            ValueError: If thread_id is not provided.
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id is required in config['configurable']")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")
        return thread_id, checkpoint_ns, checkpoint_id

    def get_next_version(self, current: str | None, channel: ChannelVersions) -> str:
        """Generate next version ID using monotonic versioning.

        Follows the same pattern as the Postgres checkpointer.

        Args:
            current: The current version string, or None if first version.
            channel: The channel versions dictionary.

        Returns:
            A new version string in the format '{version:032}.{hash:016}'.
        """
        if current is None:
            return f"{1:032}.{0:016}"

        # Parse current version
        version_str, hash_str = current.split(".")
        next_version = int(version_str) + 1
        next_hash = int(hash_str)

        return f"{next_version:032}.{next_hash:016}"

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> tuple[str, str]:
        """Serialize checkpoint data.

        Uses serde to properly serialize complex objects like LangChain messages.
        The data is stored as hex-encoded msgpack wrapped in JSON for storage.

        Args:
            checkpoint: The checkpoint to serialize.

        Returns:
            Tuple of (type, serialized data string).
        """
        type_, data = self.serde.dumps_typed(checkpoint)
        if isinstance(data, bytes):
            # Store as JSON with serde wrapper
            return "serde", json.dumps({"__serde_type__": type_, "__serde_data__": data.hex()})
        return "json", json.dumps(checkpoint, default=str)

    def _load_checkpoint(self, type_: str, data: str) -> Checkpoint:
        """Deserialize checkpoint data.

        Args:
            type_: The serialization type.
            data: The serialized data string.

        Returns:
            The deserialized checkpoint.
        """
        parsed = json.loads(data)
        # Check if it's a serde-wrapped value
        if isinstance(parsed, dict) and "__serde_type__" in parsed:
            serde_type = parsed["__serde_type__"]
            data_bytes = bytes.fromhex(parsed["__serde_data__"])
            return self.serde.loads_typed((serde_type, data_bytes))
        return parsed

    def _dump_metadata(self, metadata: CheckpointMetadata) -> tuple[str, str]:
        """Serialize checkpoint metadata.

        Uses serde to properly serialize any complex objects in metadata.

        Args:
            metadata: The metadata to serialize.

        Returns:
            Tuple of (type, serialized data string).
        """
        type_, data = self.serde.dumps_typed(metadata)
        if isinstance(data, bytes):
            return "serde", json.dumps({"__serde_type__": type_, "__serde_data__": data.hex()})
        return "json", json.dumps(metadata, default=str)

    def _load_metadata(self, type_: str, data: str) -> CheckpointMetadata:
        """Deserialize checkpoint metadata.

        Args:
            type_: The serialization type.
            data: The serialized data string.

        Returns:
            The deserialized metadata.
        """
        parsed = json.loads(data)
        # Check if it's a serde-wrapped value
        if isinstance(parsed, dict) and "__serde_type__" in parsed:
            serde_type = parsed["__serde_type__"]
            data_bytes = bytes.fromhex(parsed["__serde_data__"])
            return self.serde.loads_typed((serde_type, data_bytes))
        return parsed

    def _is_simple_json_serializable(self, value: Any) -> bool:
        """Check if a value can be serialized to simple JSON without data loss.

        Returns False for complex objects like LangChain messages that need
        proper serialization to preserve their type information.
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return True
        if isinstance(value, (list, tuple)):
            return all(self._is_simple_json_serializable(item) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(k, str) and self._is_simple_json_serializable(v)
                for k, v in value.items()
            )
        return False

    def _dump_blobs(
        self, channel_values: dict[str, Any], channel_versions: ChannelVersions
    ) -> list[dict[str, Any]]:
        """Serialize channel values to human-readable JSON for blob storage.

        Uses the serde serializer for complex objects (like LangChain messages)
        to preserve type information, while storing simple values directly as JSON.

        Args:
            channel_values: Dictionary of channel names to values.
            channel_versions: Dictionary of channel names to versions.

        Returns:
            List of blob dictionaries ready for storage.
        """
        blobs = []
        for channel, value in channel_values.items():
            version = channel_versions.get(channel, "")

            # For simple JSON-serializable values, store directly
            if self._is_simple_json_serializable(value):
                blob_data = json.dumps(value)
                blob_type = "json"
            else:
                # For complex objects (LangChain messages, etc.), use serde
                # This preserves type information for proper deserialization
                type_, data = self.serde.dumps_typed(value)
                if isinstance(data, bytes):
                    # Store as JSON with serde wrapper for human readability of the format
                    blob_data = json.dumps({"__serde_type__": type_, "__serde_data__": data.hex()})
                    blob_type = "serde"
                else:
                    blob_data = json.dumps(value)
                    blob_type = "json"

            blobs.append(
                {
                    "channel": channel,
                    "version": version,
                    "type": blob_type,
                    "blob": blob_data,
                }
            )
        return blobs

    def _load_blobs(self, blob_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Deserialize blob records from JSON to channel values.

        Args:
            blob_records: List of blob record dictionaries.

        Returns:
            Dictionary of channel names to deserialized values.
        """
        channel_values = {}
        for record in blob_records:
            channel = record["channel"]
            data = record["blob"]
            blob_type = record.get("type", "json")

            # Parse JSON
            parsed = json.loads(data)

            # Check if it's a serde-wrapped value
            if blob_type == "serde" or (isinstance(parsed, dict) and "__serde_type__" in parsed):
                type_ = parsed["__serde_type__"]
                data_bytes = bytes.fromhex(parsed["__serde_data__"])
                channel_values[channel] = self.serde.loads_typed((type_, data_bytes))
            else:
                channel_values[channel] = parsed
        return channel_values

    def _dump_writes(
        self, writes: list[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> list[dict[str, Any]]:
        """Serialize pending writes to human-readable JSON for storage.

        Args:
            writes: List of (channel, value) tuples.
            task_id: The task identifier.
            task_path: The task path (default: "").

        Returns:
            List of write dictionaries ready for storage.
        """
        write_records = []
        for idx, (channel, value) in enumerate(writes):
            # For simple JSON-serializable values, store directly
            if self._is_simple_json_serializable(value):
                blob_data = json.dumps(value)
                blob_type = "json"
            else:
                # For complex objects, use serde
                type_, data = self.serde.dumps_typed(value)
                if isinstance(data, bytes):
                    blob_data = json.dumps({"__serde_type__": type_, "__serde_data__": data.hex()})
                    blob_type = "serde"
                else:
                    blob_data = json.dumps(value)
                    blob_type = "json"

            write_records.append(
                {
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": idx,
                    "channel": channel,
                    "type": blob_type,
                    "blob": blob_data,
                }
            )
        return write_records

    def _load_writes(self, write_records: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
        """Deserialize write records from JSON to pending writes.

        Args:
            write_records: List of write record dictionaries.

        Returns:
            List of (task_id, channel, value) tuples.
        """
        pending_writes = []
        for record in write_records:
            task_id = record["task_id"]
            channel = record["channel"]
            data = record["blob"]
            blob_type = record.get("type", "json")

            # Parse JSON
            parsed = json.loads(data)

            # Check if it's a serde-wrapped value
            if blob_type == "serde" or (isinstance(parsed, dict) and "__serde_type__" in parsed):
                type_ = parsed["__serde_type__"]
                data_bytes = bytes.fromhex(parsed["__serde_data__"])
                value = self.serde.loads_typed((type_, data_bytes))
            else:
                value = parsed
            pending_writes.append((task_id, channel, value))
        return pending_writes

    def _make_checkpoint_tuple(
        self,
        checkpoint_record: dict[str, Any],
        channel_values: dict[str, Any],
        pending_writes: list[tuple[str, str, Any]],
    ) -> CheckpointTuple:
        """Create a CheckpointTuple from database records.

        Args:
            checkpoint_record: The checkpoint record from Neo4j.
            channel_values: The deserialized channel values.
            pending_writes: The deserialized pending writes.

        Returns:
            A CheckpointTuple instance.
        """
        thread_id = checkpoint_record["thread_id"]
        checkpoint_ns = checkpoint_record["checkpoint_ns"]
        checkpoint_id = checkpoint_record["checkpoint_id"]
        parent_checkpoint_id = checkpoint_record.get("parent_checkpoint_id")

        # Deserialize checkpoint and metadata
        checkpoint = self._load_checkpoint(
            checkpoint_record["type"], checkpoint_record["checkpoint"]
        )
        metadata = self._load_metadata(checkpoint_record["type"], checkpoint_record["metadata"])

        # Build config
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Build parent config if exists
        parent_config: RunnableConfig | None = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=parent_config,
        )
