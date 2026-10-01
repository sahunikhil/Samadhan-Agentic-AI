"""Deterministic chat model for tests.

A ``ScriptedChatModel`` delegates every call to a *responder* function that sees the
messages, the names of bound tools and the requested structured-output schema. The
responder inspects the prompt to decide what to return, so tests are deterministic
even when specialists run in parallel (call order doesn't matter).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field

Responder = Callable[[list[BaseMessage], list[str], type[BaseModel] | None], Any]


def tool_call(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}]
    )


def text_of(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(m.text for m in messages)


class ScriptedChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Any
    tool_names: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tool_names": names})

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[LanguageModelInput, Any]:  # type: ignore[override]
        def run(inp: LanguageModelInput) -> Any:
            messages = self._convert_input(inp).to_messages()
            self.calls.append({"schema": schema.__name__, "messages": messages})
            out = self.responder(messages, [], schema)
            return out if isinstance(out, schema) else schema.model_validate(out)

        async def arun(inp: LanguageModelInput) -> Any:
            return run(inp)

        return RunnableLambda(run, afunc=arun)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append({"schema": None, "tools": list(self.tool_names), "messages": messages})
        out = self.responder(messages, list(self.tool_names), None)
        message = AIMessage(content=out) if isinstance(out, str) else out
        return ChatResult(generations=[ChatGeneration(message=message)])


def last_human(messages: Sequence[BaseMessage]) -> str:
    return next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")
