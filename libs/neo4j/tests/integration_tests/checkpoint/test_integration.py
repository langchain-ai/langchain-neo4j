"""Integration tests with LangGraph."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from operator import add
from typing import Annotated, Any, cast

import pytest
from langchain_core.runnables import RunnableConfig

from langchain_neo4j import Neo4jSaver


class TestLangGraphIntegration:
    """Integration tests with LangGraph."""

    def test_graph_with_neo4j_checkpointer(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test running a LangGraph with Neo4j persistence."""
        try:
            from typing import TypedDict

            from langgraph.graph import StateGraph
        except ImportError:
            pytest.skip("langgraph not installed")

        class State(TypedDict):
            messages: Annotated[Sequence[dict], add]

        def chat_node(state: State) -> State:
            # Simple echo node
            last_msg = state["messages"][-1] if state["messages"] else {}
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Echo: {last_msg.get('content', '')}",
                    }
                ]
            }

        builder = StateGraph(State)
        builder.add_node("chat", chat_node)
        builder.set_entry_point("chat")
        builder.set_finish_point("chat")

        graph = builder.compile(checkpointer=clean_neo4j_saver)

        thread_id = f"integration-test-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Run the graph
        result = graph.invoke(
            cast(Any, {"messages": [{"role": "user", "content": "Hello"}]}),
            config,
        )

        assert len(result["messages"]) == 2
        assert result["messages"][1]["content"] == "Echo: Hello"

        # Verify checkpoint was saved
        tuple_ = clean_neo4j_saver.get_tuple(config)
        assert tuple_ is not None

    def test_conversation_continuity(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that conversation state persists across invocations."""
        try:
            from typing import TypedDict

            from langgraph.graph import StateGraph
        except ImportError:
            pytest.skip("langgraph not installed")

        class State(TypedDict):
            messages: Annotated[Sequence[dict], add]
            count: int

        def counter_node(state: State) -> State:
            new_count = state.get("count", 0) + 1
            return {
                "messages": [{"role": "assistant", "content": f"Count: {new_count}"}],
                "count": new_count,
            }

        builder = StateGraph(State)
        builder.add_node("counter", counter_node)
        builder.set_entry_point("counter")
        builder.set_finish_point("counter")

        graph = builder.compile(checkpointer=clean_neo4j_saver)

        thread_id = f"continuity-test-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # First invocation
        result1 = graph.invoke(
            cast(
                Any,
                {"messages": [{"role": "user", "content": "increment"}], "count": 0},
            ),
            config,
        )
        assert "Count: 1" in result1["messages"][-1]["content"]

        # Second invocation - should continue from checkpoint
        result2 = graph.invoke(
            cast(Any, {"messages": [{"role": "user", "content": "increment again"}]}),
            config,
        )
        # Count should be 2 because state was persisted
        assert "Count: 2" in result2["messages"][-1]["content"]

    def test_list_checkpoints_after_multiple_runs(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test listing checkpoints after multiple graph runs."""
        try:
            from typing import TypedDict

            from langgraph.graph import StateGraph
        except ImportError:
            pytest.skip("langgraph not installed")

        class State(TypedDict):
            value: int

        def increment_node(state: State) -> State:
            return {"value": state.get("value", 0) + 1}

        builder = StateGraph(State)
        builder.add_node("increment", increment_node)
        builder.set_entry_point("increment")
        builder.set_finish_point("increment")

        graph = builder.compile(checkpointer=clean_neo4j_saver)

        thread_id = f"list-test-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Run multiple times
        for i in range(3):
            graph.invoke(cast(Any, {"value": i}), config)

        # List all checkpoints
        checkpoints = list(clean_neo4j_saver.list(config))

        # Should have multiple checkpoints (input + loop for each run)
        assert len(checkpoints) >= 3

    def test_delete_thread_cleans_all_data(
        self,
        clean_neo4j_saver: Neo4jSaver,
    ) -> None:
        """Test that delete_thread removes all associated data."""
        try:
            from typing import TypedDict

            from langgraph.graph import StateGraph
        except ImportError:
            pytest.skip("langgraph not installed")

        class State(TypedDict):
            messages: Annotated[Sequence[dict], add]

        def echo_node(state: State) -> State:
            return {"messages": [{"role": "assistant", "content": "response"}]}

        builder = StateGraph(State)
        builder.add_node("echo", echo_node)
        builder.set_entry_point("echo")
        builder.set_finish_point("echo")

        graph = builder.compile(checkpointer=clean_neo4j_saver)

        thread_id = f"delete-test-{uuid.uuid4()}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Run graph to create checkpoints
        graph.invoke(
            cast(Any, {"messages": [{"role": "user", "content": "test"}]}),
            config,
        )

        # Verify checkpoint exists
        assert clean_neo4j_saver.get_tuple(config) is not None

        # Delete thread
        clean_neo4j_saver.delete_thread(thread_id)

        # Verify all data is gone
        assert clean_neo4j_saver.get_tuple(config) is None
        assert list(clean_neo4j_saver.list(config)) == []
