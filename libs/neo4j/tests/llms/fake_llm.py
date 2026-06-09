"""Fake LLM wrapper for testing purposes."""

from typing import Any, Dict, List, Mapping, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import validator


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @validator("queries", always=True)
    def check_queries_required(
        cls, queries: Optional[Mapping], values: Mapping[str, Any]
    ) -> Optional[Mapping]:
        if values.get("sequential_response") and not queries:
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return queries

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response

    def bind_tools(self, tools: Any) -> None:
        pass


class FakeStructuredChatModel(BaseChatModel):
    """Fake chat model that returns a pre-canned structured-output response.

    Designed for testing code that calls
    ``llm.with_structured_output(..., include_raw=True)`` (e.g.
    ``LLMGraphTransformer`` in function-calling mode). The ``response``
    attribute is yielded verbatim from the runnable returned by
    ``with_structured_output``.
    """

    response: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        response = self.response

        def _emit(_: Any) -> Any:
            if include_raw:
                return {
                    "raw": AIMessage(content=""),
                    "parsed": response,
                    "parsing_error": None,
                }
            return response

        return RunnableLambda(_emit)
