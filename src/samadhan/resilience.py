"""Circuit breakers for the agent's external dependencies (LLM providers, MCP servers).

Retries, timeouts and fallbacks already exist - why a breaker too?
    Retries help with *transient* blips. When a dependency is *down* (provider outage, a free
    tier's exhausted daily quota, a crashed MCP server), every call still waits for its timeout,
    retries, and only then falls back: each turn gets slower and the struggling dependency gets
    hammered. A breaker remembers: after ``failure_threshold`` consecutive failures it **opens**
    and callers fail *immediately* (-> the model fallback or a friendly "temporarily unavailable"
    tool message) for ``reset_timeout_s``; then one **half-open** probe decides whether to close.

        CLOSED --N consecutive failures--> OPEN --timeout--> HALF_OPEN --probe ok--> CLOSED
                                             ^------------------probe fails------------+

Where the breakers sit (order matters with LangChain middleware - first = outermost):
    * agent model calls:  ModelFallback -> **breaker** -> ModelRetry -> model
      (an open circuit is not retried with backoff; it hands straight to the fallback model)
    * agent tool calls:   ToolError -> **breaker** -> ToolRetry -> MCP tool
      (only *dependency* failures count - a tool's business error such as "order not found"
      arrives as ``ToolException`` and is not an outage)
    * graph-node LLM calls (``ModelRegistry.chat/structured``) and deterministic MCP calls
      (``MCPToolkit.call``) use :func:`guarded` / :meth:`CircuitBreaker.call`.

Breakers are per process. That is intended: each replica protects itself and learns about an
outage from its own traffic within ``failure_threshold`` calls - no shared state to fail.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from enum import IntEnum
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import ToolException
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from pydantic import ValidationError

from samadhan.observability import CIRCUIT_REJECTIONS, CIRCUIT_STATE, get_logger

log = get_logger(__name__)


class State(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitOpenError(RuntimeError):
    def __init__(self, name: str, retry_in_s: float) -> None:
        super().__init__(f"{name} is temporarily unavailable (circuit open, retry in {retry_in_s:.0f}s)")
        self.dependency = name


def is_dependency_failure(exc: BaseException) -> bool:
    """Outages count (timeouts, connection errors, 5xx, 408/429); business errors, malformed model
    output, other 4xx (e.g. Groq's 400 ``json_validate_failed``) and HITL control flow do not."""
    if not isinstance(exc, Exception):  # CancelledError, KeyboardInterrupt: the *caller* stopped
        return False
    if isinstance(exc, ToolException | ValidationError | OutputParserException | GraphBubbleUp | CircuitOpenError):
        return False
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status not in (408, 429))


class CircuitBreaker:
    def __init__(self, name: str, *, failure_threshold: int = 5, reset_timeout_s: float = 30.0) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.state = State.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._probe_in_flight = False
        CIRCUIT_STATE.labels(dependency=name).set(State.CLOSED)

    def _set(self, state: State) -> None:
        if state != self.state:
            log.warning("circuit_state", dependency=self.name, state=state.name, failures=self.failures)
        self.state = state
        CIRCUIT_STATE.labels(dependency=self.name).set(state)

    def before_call(self) -> None:
        """Raise ``CircuitOpenError`` instead of calling a dependency that is known to be down."""
        if self.state == State.OPEN:
            elapsed = time.monotonic() - self.opened_at
            if elapsed < self.reset_timeout_s:
                CIRCUIT_REJECTIONS.labels(dependency=self.name).inc()
                raise CircuitOpenError(self.name, self.reset_timeout_s - elapsed)
            self._set(State.HALF_OPEN)
        if self.state == State.HALF_OPEN:
            if self._probe_in_flight:  # exactly one probe; everyone else keeps failing fast
                CIRCUIT_REJECTIONS.labels(dependency=self.name).inc()
                raise CircuitOpenError(self.name, 1)
            self._probe_in_flight = True

    def record_success(self) -> None:
        self._probe_in_flight = False
        self.failures = 0
        self._set(State.CLOSED)

    def record_failure(self, exc: BaseException) -> None:
        self._probe_in_flight = False
        if not isinstance(exc, Exception):
            # Cancelled (client disconnect, node timeout, drain): we learned nothing about the
            # dependency - leave the state alone; the next call becomes the probe if half-open.
            return
        if not is_dependency_failure(exc):
            if self.state == State.HALF_OPEN:  # the dependency answered: it is up
                self.record_success()
            return
        self.failures += 1
        if self.state == State.HALF_OPEN or self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()
            self._set(State.OPEN)

    async def call[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        self.before_call()
        try:
            result = await fn()
        except BaseException as exc:
            self.record_failure(exc)
            raise
        self.record_success()
        return result


class BreakerRegistry:
    def __init__(self, *, failure_threshold: int = 5, reset_timeout_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name, failure_threshold=self.failure_threshold, reset_timeout_s=self.reset_timeout_s
            )
        return self._breakers[name]

    def snapshot(self) -> dict[str, str]:
        return {name: b.state.name.lower() for name, b in sorted(self._breakers.items())}


BREAKERS = BreakerRegistry()


def configure_breakers(failure_threshold: int, reset_timeout_s: float) -> None:
    BREAKERS.failure_threshold = failure_threshold
    BREAKERS.reset_timeout_s = reset_timeout_s


def model_key(model: Any) -> str:
    name = getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
    return f"llm:{name}"


def guarded(runnable: Runnable[Any, Any], name: str) -> Runnable[Any, Any]:
    """Wrap a runnable (a chat model or a structured-output chain) with the named breaker.

    Used inside ``with_fallbacks`` chains: an open circuit raises instantly, so the chain moves
    on to the next model without waiting out timeouts. Token streaming still works - the inner
    chat model streams through the callbacks carried in ``config``.
    """

    async def acall(value: Any, config: RunnableConfig) -> Any:
        return await BREAKERS.get(name).call(lambda: runnable.ainvoke(value, config))

    def call(value: Any, config: RunnableConfig) -> Any:
        breaker = BREAKERS.get(name)
        breaker.before_call()
        try:
            out = runnable.invoke(value, config)
        except BaseException as exc:
            breaker.record_failure(exc)
            raise
        breaker.record_success()
        return out

    return RunnableLambda(call, afunc=acall, name=f"guarded[{name}]")


class CircuitBreakerMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Breakers for an agent's model and tool calls (see module docstring for placement)."""

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await BREAKERS.get(model_key(request.model)).call(lambda: handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        server = ((request.tool.metadata if request.tool else None) or {}).get("mcp_server")
        if server is None:
            return await handler(request)
        return await BREAKERS.get(f"mcp:{server}").call(lambda: handler(request))
