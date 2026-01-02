import pathlib
from csv import DictReader
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_classic.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from langchain_neo4j.chains.graph_qa.cypher import (
    GraphCypherQAChain,
    construct_schema,
    get_function_response,
)
from langchain_neo4j.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_neo4j.chains.graph_qa.prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_QA_PROMPT,
)
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.graph_store import GraphStore
from tests.llms.fake_llm import FakeLLM


class FakeGraphStore:
    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph database"""
        return ""

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph database"""
        return {}

    @property
    def _enhanced_schema(self) -> bool:
        """Whether the schema is enhanced"""
        return False

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query the graph."""
        return []

    def refresh_schema(self) -> None:
        """Refreshes the graph schema information."""
        pass

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        pass


def test_graph_store() -> None:
    """Tests that FakeGraphStore satisfies the GraphStore protocol requirements."""
    graph = FakeGraphStore()
    assert isinstance(graph, GraphStore)


def test_graph_cypher_qa_chain_prompt_selection_1() -> None:
    # Pass prompts directly. No kwargs is specified.
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        qa_prompt=qa_prompt,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True,
    )
    assert hasattr(chain.qa_chain, "first")
    assert chain.qa_chain.first == qa_prompt
    assert hasattr(chain.cypher_generation_chain, "first")
    assert chain.cypher_generation_chain.first == cypher_prompt


def test_graph_cypher_qa_chain_prompt_selection_2() -> None:
    # Default case. Pass nothing
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        allow_dangerous_requests=True,
    )
    assert hasattr(chain.qa_chain, "first")
    assert chain.qa_chain.first == CYPHER_QA_PROMPT
    assert hasattr(chain.cypher_generation_chain, "first")
    assert chain.cypher_generation_chain.first == CYPHER_GENERATION_PROMPT


def test_graph_cypher_qa_chain_prompt_selection_3() -> None:
    # Pass non-prompt args only to sub-chains via kwargs
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"memory": readonlymemory},
        qa_llm_kwargs={"memory": readonlymemory},
        allow_dangerous_requests=True,
    )
    assert hasattr(chain.qa_chain, "first")
    assert chain.qa_chain.first == CYPHER_QA_PROMPT
    assert hasattr(chain.cypher_generation_chain, "first")
    assert chain.cypher_generation_chain.first == CYPHER_GENERATION_PROMPT


def test_graph_cypher_qa_chain_prompt_selection_4() -> None:
    # Pass prompt, non-prompt args to subchains via kwargs
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"prompt": cypher_prompt, "memory": readonlymemory},
        qa_llm_kwargs={"prompt": qa_prompt, "memory": readonlymemory},
        allow_dangerous_requests=True,
    )
    assert hasattr(chain.qa_chain, "first")
    assert chain.qa_chain.first == qa_prompt
    assert hasattr(chain.cypher_generation_chain, "first")
    assert chain.cypher_generation_chain.first == cypher_prompt


def test_graph_cypher_qa_chain_prompt_selection_5() -> None:
    # Can't pass both prompt and kwargs at the same time
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            verbose=True,
            return_intermediate_steps=False,
            cypher_prompt=cypher_prompt,
            cypher_llm_kwargs={"memory": readonlymemory},
            allow_dangerous_requests=True,
        )
    assert (
        "Specifying cypher_prompt and cypher_llm_kwargs together is"
        " not allowed. Please pass prompt via cypher_llm_kwargs."
    ) == str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            verbose=True,
            return_intermediate_steps=False,
            qa_prompt=qa_prompt,
            qa_llm_kwargs={"memory": readonlymemory},
            allow_dangerous_requests=True,
        )
    assert (
        "Specifying qa_prompt and qa_llm_kwargs together is"
        " not allowed. Please pass prompt via qa_llm_kwargs."
    ) == str(exc_info.value)


def test_graph_cypher_qa_chain_prompt_selection_6() -> None:
    # Test function response prompt
    function_response_system = "Respond as a pirate!"
    response_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=function_response_system),
            HumanMessagePromptTemplate.from_template("{question}"),
            MessagesPlaceholder(variable_name="function_response"),
        ]
    )
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        use_function_response=True,
        function_response_system=function_response_system,
        allow_dangerous_requests=True,
    )
    assert hasattr(chain.qa_chain, "first")
    assert chain.qa_chain.first == response_prompt
    assert hasattr(chain.cypher_generation_chain, "first")
    assert chain.cypher_generation_chain.first == CYPHER_GENERATION_PROMPT


def test_graph_cypher_qa_chain_prompt_selection_7() -> None:
    # Pass prompts which do not inherit from BasePromptTemplate
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            cypher_llm_kwargs={"prompt": None},
            allow_dangerous_requests=True,
        )
    assert "The cypher_llm_kwargs `prompt` must inherit from BasePromptTemplate" == str(
        exc_info.value
    )
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            qa_llm_kwargs={"prompt": None},
            allow_dangerous_requests=True,
        )
    assert "The qa_llm_kwargs `prompt` must inherit from BasePromptTemplate" == str(
        exc_info.value
    )


def test_validate_cypher() -> None:
    with patch(
        "langchain_neo4j.chains.graph_qa.cypher.CypherQueryCorrector",
        autospec=True,
    ) as cypher_query_corrector_mock:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            validate_cypher=True,
            allow_dangerous_requests=True,
        )
        cypher_query_corrector_mock.assert_called_once_with([])


def test_chain_type() -> None:
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    assert chain._chain_type == "graph_cypher_chain"


def test_graph_cypher_qa_chain_without_examples() -> None:
    template = """You are a nice chatbot having a conversation with a human.

    Schema:
    {schema}

    Previous conversation:
    {chat_history}

    Examples (optional):
    {examples}

    New human question: {question}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["schema", "question", "examples", "chat_history"],
        template=template,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    prompt1 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties:\n\nRelationship "
        "properties:\n\nThe relationships"
        ":\n\n\n    "
        "Previous conversation:\n    \n\n    Examples (optional):\n"
        "    None\n\n    New human question: Test question\n    Response:"
    )

    prompt2 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties:\n\nRelationship "
        "properties:\n\nThe relationships"
        ":\n\n\n    "
        "Previous conversation:\n    Human: Test question\nAI: foo\n\n    "
        "Examples (optional):\n    None\n\n    New human question: "
        "Test new question\n    Response:"
    )

    llm = FakeLLM(queries={prompt1: "answer1", prompt2: "answer2"})
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"prompt": prompt, "memory": readonlymemory},
        memory=memory,
        allow_dangerous_requests=True,
    )
    chain.run("Test question")
    chain.run("Test new question")
    # If we get here without a key error, that means memory
    # was used properly to create prompts.
    assert True


def test_graph_cypher_qa_chain_with_examples() -> None:
    template = """You are a nice chatbot having a conversation with a human.

    Schema:
    {schema}

    Previous conversation:
    {chat_history}

    Examples (optional):
    {examples}

    New human question: {question}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["schema", "question", "examples", "chat_history"],
        template=template,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    prompt1 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties:\n\nRelationship "
        "properties:\n\nThe relationships"
        ":\n\n\n    "
        "Previous conversation:\n    \n\n    Examples (optional):\n"
        "    Test examples\n\n    New human question: "
        "Test question\n    Response:"
    )

    prompt2 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties:\n\nRelationship "
        "properties:\n\nThe relationships"
        ":\n\n\n    "
        "Previous conversation:\n    Human: Test question\nAI: foo\n\n    "
        "Examples (optional):\n    Test new examples\n\n    "
        "New human question: Test new question\n    Response:"
    )

    llm = FakeLLM(queries={prompt1: "answer1", prompt2: "answer2"})
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"prompt": prompt, "memory": readonlymemory},
        memory=memory,
        allow_dangerous_requests=True,
    )
    chain.run(query="Test question", examples="Test examples")
    chain.run(query="Test new question", examples="Test new examples")
    # If we get here without a key error, that means memory
    # and examples were used properly to create prompts.
    assert True


def test_cypher_generation_failure() -> None:
    """Test the chain doesn't fail if the Cypher query fails to be generated."""
    llm = FakeLLM(queries={"query": ""}, sequential_responses=True)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
        return_direct=True,
    )
    response = chain.run("Test question")
    assert response == []


def test_exclude_types() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {"ACTED_IN": [{"property": "role", "type": "STRING"}]},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    exclude_types = ["Person", "DIRECTED"]
    output = construct_schema(structured_schema, [], exclude_types, False)
    expected_schema = (
        "Node properties:\n"
        "Movie {title: STRING}\n"
        "Actor {name: STRING}\n"
        "Relationship properties:\n"
        "ACTED_IN {role: STRING}\n"
        "The relationships:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )
    assert output == expected_schema


def test_include_types() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {"ACTED_IN": [{"property": "role", "type": "STRING"}]},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor", "ACTED_IN"]
    output = construct_schema(structured_schema, include_types, [], False)
    expected_schema = (
        "Node properties:\n"
        "Movie {title: STRING}\n"
        "Actor {name: STRING}\n"
        "Relationship properties:\n"
        "ACTED_IN {role: STRING}\n"
        "The relationships:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )
    assert output == expected_schema


def test_include_types2() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {"ACTED_IN": [{"property": "role", "type": "STRING"}]},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor"]
    output = construct_schema(structured_schema, include_types, [], False)
    expected_schema = (
        "Node properties:\n"
        "Movie {title: STRING}\n"
        "Actor {name: STRING}\n"
        "Relationship properties:\n\n"
        "The relationships:\n"
    )
    assert output == expected_schema


def test_include_types3() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {"ACTED_IN": [{"property": "role", "type": "STRING"}]},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor", "ACTED_IN"]
    output = construct_schema(structured_schema, include_types, [], False)
    expected_schema = (
        "Node properties:\n"
        "Movie {title: STRING}\n"
        "Actor {name: STRING}\n"
        "Relationship properties:\n"
        "ACTED_IN {role: STRING}\n"
        "The relationships:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )
    assert output == expected_schema


def test_include_exclude_types_err() -> None:
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            include_types=["Movie", "Actor"],
            exclude_types=["Person", "DIRECTED"],
            allow_dangerous_requests=True,
        )
    assert (
        "Either `exclude_types` or `include_types` can be provided, but not both"
        == str(exc_info.value)
    )


def test_get_function_response() -> None:
    question = "Who directed Dune?"
    context = [{"director": "Denis Villeneuve"}]
    messages = get_function_response(question, context)
    assert len(messages) == 2
    # Validate AIMessage
    ai_message = messages[0]
    assert isinstance(ai_message, AIMessage)
    assert ai_message.content == ""
    assert "tool_calls" in ai_message.additional_kwargs
    tool_call = ai_message.additional_kwargs["tool_calls"][0]
    assert tool_call["function"]["arguments"] == f'{{"question":"{question}"}}'
    # Validate ToolMessage
    tool_message = messages[1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == str(context)


def test_allow_dangerous_requests_err() -> None:
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
        )
    assert (
        "In order to use this chain, you must acknowledge that it can make "
        "dangerous requests by setting `allow_dangerous_requests` to `True`."
    ) in str(exc_info.value)


def test_llm_arg_combinations() -> None:
    # No llm
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            graph=FakeGraphStore(), allow_dangerous_requests=True
        )
    assert "At least one LLM must be provided" == str(exc_info.value)
    # llm only
    GraphCypherQAChain.from_llm(
        llm=FakeLLM(), graph=FakeGraphStore(), allow_dangerous_requests=True
    )
    # qa_llm only
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            qa_llm=FakeLLM(), graph=FakeGraphStore(), allow_dangerous_requests=True
        )
    assert (
        "If `llm` is not provided, both `qa_llm` and `cypher_llm` must be provided."
        == str(exc_info.value)
    )
    # cypher_llm only
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            cypher_llm=FakeLLM(), graph=FakeGraphStore(), allow_dangerous_requests=True
        )
    assert (
        "If `llm` is not provided, both `qa_llm` and `cypher_llm` must be provided."
        == str(exc_info.value)
    )
    # llm + qa_llm
    GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        qa_llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    # llm + cypher_llm
    GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        cypher_llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    # qa_llm + cypher_llm
    GraphCypherQAChain.from_llm(
        qa_llm=FakeLLM(),
        cypher_llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    # llm + qa_llm + cypher_llm
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            qa_llm=FakeLLM(),
            cypher_llm=FakeLLM(),
            graph=FakeGraphStore(),
            allow_dangerous_requests=True,
        )
    assert (
        "You can specify up to two of 'cypher_llm', 'qa_llm'"
        ", and 'llm', but not all three simultaneously."
    ) == str(exc_info.value)


def test_use_function_response_err() -> None:
    llm = MagicMock(spec=LLM)
    with pytest.raises(ValueError) as exc_info:
        GraphCypherQAChain.from_llm(
            llm=llm,
            graph=FakeGraphStore(),
            allow_dangerous_requests=True,
            use_function_response=True,
        )
    assert "Provided LLM does not support native tools/functions" == str(exc_info.value)


HERE = pathlib.Path(__file__).parent

UNIT_TESTS_ROOT = HERE.parent


def test_validating_cypher_statements() -> None:
    cypher_file = str(UNIT_TESTS_ROOT / "data/cypher_corrector.csv")
    with open(cypher_file, newline="") as csvfile:
        csv_reader = DictReader(csvfile)
        for row in csv_reader:
            schema = load_schemas(row["schema"])
            corrector = CypherQueryCorrector(schema)
            assert corrector(row["statement"]) == row["correct_query"]


def load_schemas(str_schemas: str) -> List[Schema]:
    """
    Args:
        str_schemas: string of schemas
    """
    values = str_schemas.replace("(", "").replace(")", "").split(",")
    schemas = []
    for i in range(len(values) // 3):
        schemas.append(
            Schema(
                values[i * 3].strip(),
                values[i * 3 + 1].strip(),
                values[i * 3 + 2].strip(),
            )
        )
    return schemas
