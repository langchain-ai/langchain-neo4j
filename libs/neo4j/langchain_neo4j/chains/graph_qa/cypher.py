"""Question answering over a graph."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable
from pydantic import Field

from langchain_neo4j.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_neo4j.chains.graph_qa.prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_QA_PROMPT,
)
from langchain_neo4j.graphs.graph_store import GraphStore

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

FUNCTION_RESPONSE_SYSTEM = """You are an assistant that helps to form nice and human 
understandable answers based on the provided information from tools.
Do not add any other information that wasn't present in the tools, and use 
very concise style in interpreting results!
"""


def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text


def construct_schema(
    structured_schema: Dict[str, Any],
    include_types: List[str],
    exclude_types: List[str],
) -> str:
    """Filter the schema based on included or excluded types"""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema: Dict[str, Any] = {
        "node_props": {
            k: v
            for k, v in structured_schema.get("node_props", {}).items()
            if filter_func(k)
        },
        "rel_props": {
            k: v
            for k, v in structured_schema.get("rel_props", {}).items()
            if filter_func(k)
        },
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }

    # Format node properties
    formatted_node_props = []
    for label, properties in filtered_schema["node_props"].items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_node_props.append(f"{label} {{{props_str}}}")

    # Format relationship properties
    formatted_rel_props = []
    for rel_type, properties in filtered_schema["rel_props"].items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_rel_props.append(f"{rel_type} {{{props_str}}}")

    # Format relationships
    formatted_rels = [
        f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
        for el in filtered_schema["relationships"]
    ]

    return "\n".join(
        [
            "Node properties are the following:",
            ",".join(formatted_node_props),
            "Relationship properties are the following:",
            ",".join(formatted_rel_props),
            "The relationships are the following:",
            ",".join(formatted_rels),
        ]
    )


def get_function_response(
    question: str, context: List[Dict[str, Any]]
) -> List[BaseMessage]:
    TOOL_ID = "call_H7fABDuzEau48T10Qn0Lsh0D"
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(context), tool_call_id=TOOL_ID),
    ]
    return messages


class GraphCypherQAChain(Chain):
    """Chain for question-answering against a graph by generating Cypher statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    graph: GraphStore = Field(exclude=True)
    cypher_generation_chain: Runnable[Dict[str, Any], str]
    qa_chain: Runnable[Dict[str, Any], str]
    graph_schema: str
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""
    cypher_query_corrector: Optional[CypherQueryCorrector] = None
    """Optional cypher validation tool"""
    use_function_response: bool = False
    """Whether to wrap the database context as tool/function response"""
    allow_dangerous_requests: bool = False
    """Forced user opt-in to acknowledge that the chain can make dangerous requests.
    
    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: Optional[BasePromptTemplate] = None,
        cypher_prompt: Optional[BasePromptTemplate] = None,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        validate_cypher: bool = False,
        qa_llm_kwargs: Optional[Dict[str, Any]] = None,
        cypher_llm_kwargs: Optional[Dict[str, Any]] = None,
        use_function_response: bool = False,
        function_response_system: str = FUNCTION_RESPONSE_SYSTEM,
        **kwargs: Any,
    ) -> GraphCypherQAChain:
        """Initialize from LLM."""
        # Ensure at least one LLM is provided
        if llm is None and qa_llm is None and cypher_llm is None:
            raise ValueError("At least one LLM must be provided")

        # Prevent all three LLMs from being provided simultaneously
        if llm is not None and qa_llm is not None and cypher_llm is not None:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm', and 'llm', but "
                "not all three simultaneously."
            )

        # Assign default LLMs if specific ones are not provided
        if llm is not None:
            qa_llm = qa_llm or llm
            cypher_llm = cypher_llm or llm
        else:
            # If llm is None, both qa_llm and cypher_llm must be provided
            if qa_llm is None or cypher_llm is None:
                raise ValueError(
                    "If `llm` is not provided, both `qa_llm` and `cypher_llm` must be "
                    "provided."
                )
        if cypher_prompt:
            if cypher_llm_kwargs:
                raise ValueError(
                    "Specifying cypher_prompt and cypher_llm_kwargs together is"
                    " not allowed. Please pass prompt via cypher_llm_kwargs."
                )
        else:
            if cypher_llm_kwargs:
                cypher_prompt = cypher_llm_kwargs.pop(
                    "prompt", CYPHER_GENERATION_PROMPT
                )
                if not isinstance(cypher_prompt, BasePromptTemplate):
                    raise ValueError(
                        "The cypher_llm_kwargs `prompt` must inherit from "
                        "BasePromptTemplate"
                    )
            else:
                cypher_prompt = CYPHER_GENERATION_PROMPT
        if qa_prompt:
            if qa_llm_kwargs:
                raise ValueError(
                    "Specifying qa_prompt and qa_llm_kwargs together is"
                    " not allowed. Please pass prompt via qa_llm_kwargs."
                )
        else:
            if qa_llm_kwargs:
                qa_prompt = qa_llm_kwargs.pop("prompt", CYPHER_QA_PROMPT)
                if not isinstance(qa_prompt, BasePromptTemplate):
                    raise ValueError(
                        "The qa_llm_kwargs `prompt` must inherit from "
                        "BasePromptTemplate"
                    )
            else:
                qa_prompt = CYPHER_QA_PROMPT
        use_qa_llm_kwargs = qa_llm_kwargs if qa_llm_kwargs is not None else {}
        use_cypher_llm_kwargs = (
            cypher_llm_kwargs if cypher_llm_kwargs is not None else {}
        )

        if use_function_response:
            try:
                if hasattr(qa_llm, "bind_tools"):
                    qa_llm.bind_tools({})
                else:
                    raise AttributeError
                response_prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=function_response_system),
                        HumanMessagePromptTemplate.from_template("{question}"),
                        MessagesPlaceholder(variable_name="function_response"),
                    ]
                )
                qa_chain = response_prompt | qa_llm | StrOutputParser()
            except (NotImplementedError, AttributeError):
                raise ValueError("Provided LLM does not support native tools/functions")
        else:
            qa_chain = qa_prompt | qa_llm.bind(**use_qa_llm_kwargs) | StrOutputParser()
        cypher_generation_chain = (
            cypher_prompt | cypher_llm.bind(**use_cypher_llm_kwargs) | StrOutputParser()
        )

        if exclude_types and include_types:
            raise ValueError(
                "Either `exclude_types` or `include_types` "
                "can be provided, but not both"
            )
        graph_schema = construct_schema(
            kwargs["graph"].get_structured_schema, include_types, exclude_types
        )

        cypher_query_corrector = None
        if validate_cypher:
            corrector_schema = [
                Schema(el["start"], el["type"], el["end"])
                for el in kwargs["graph"].structured_schema.get("relationships")
            ]
            cypher_query_corrector = CypherQueryCorrector(corrector_schema)

        return cls(
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            cypher_query_corrector=cypher_query_corrector,
            use_function_response=use_function_response,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        args = {
            "question": question,
            "schema": self.graph_schema,
        }
        args.update(inputs)

        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.invoke(
            args, callbacks=callbacks
        )

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
        else:
            context = []

        final_result: Union[List[Dict[str, Any]], str]
        if self.return_direct:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"context": context})
            if self.use_function_response:
                function_response = get_function_response(question, context)
                final_result = self.qa_chain.invoke(
                    {"question": question, "function_response": function_response},
                )
            else:
                final_result = self.qa_chain.invoke(
                    {"question": question, "context": context},
                    callbacks=callbacks,
                )

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
