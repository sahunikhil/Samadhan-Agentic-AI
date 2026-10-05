"""Circuit breaker state machine, failure classification, and fail-fast fallbacks."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import ToolException

import samadhan.resilience as resilience
from samadhan.resilience import BreakerRegistry, CircuitBreaker, CircuitOpenError, State, guarded, is_dependency_failure


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(resilience.time, "monotonic", c)
    return c


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(status_code)
        self.status_code = status_code


def test_only_outages_count_as_failures() -> None:
    assert is_dependency_failure(TimeoutError()) and is_dependency_failure(ConnectionError())
    assert is_dependency_failure(HTTPError(429)) and is_dependency_failure(HTTPError(503))
    assert not is_dependency_failure(HTTPError(400))  # e.g. Groq json_validate_failed: model output, not outage
    assert not is_dependency_failure(ToolException("order not found"))  # business error
    assert not is_dependency_failure(OutputParserException("bad json"))


async def test_opens_after_threshold_then_half_open_probe_closes(clock: Clock) -> None:
    breaker = CircuitBreaker("dep", failure_threshold=3, reset_timeout_s=30)
    calls = 0

    async def down() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError

    for _ in range(3):
        with pytest.raises(TimeoutError):
            await breaker.call(down)
    assert breaker.state == State.OPEN
    with pytest.raises(CircuitOpenError):  # fails fast, dependency not called
        await breaker.call(down)
    assert calls == 3

    clock.now += 31  # cool-down over -> one probe is let through

    async def up() -> str:
        return "ok"

    assert await breaker.call(up) == "ok"
    assert breaker.state == State.CLOSED and breaker.failures == 0


async def test_failed_probe_reopens_immediately(clock: Clock) -> None:
    breaker = CircuitBreaker("dep", failure_threshold=1, reset_timeout_s=10)

    async def down() -> None:
        raise ConnectionError

    with pytest.raises(ConnectionError):
        await breaker.call(down)
    clock.now += 11
    with pytest.raises(ConnectionError):
        await breaker.call(down)  # the probe
    assert breaker.state == State.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.call(down)


async def test_business_errors_never_open_the_circuit() -> None:
    breaker = CircuitBreaker("mcp:commerce", failure_threshold=2)

    async def not_found() -> None:
        raise ToolException("order not found")

    for _ in range(5):
        with pytest.raises(ToolException):
            await breaker.call(not_found)
    assert breaker.state == State.CLOSED


async def test_open_primary_is_skipped_by_fallback_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resilience, "BREAKERS", BreakerRegistry(failure_threshold=2, reset_timeout_s=60))
    primary_calls = 0

    async def primary(_: Any) -> str:
        nonlocal primary_calls
        primary_calls += 1
        raise TimeoutError("provider down")

    chain = guarded(RunnableLambda(primary), "llm:primary").with_fallbacks(
        [guarded(RunnableLambda(lambda _: "fallback answer"), "llm:fallback")]
    )
    for _ in range(5):
        assert await chain.ainvoke("hi") == "fallback answer"
    assert primary_calls == 2, "after 2 failures the primary is skipped without waiting for its timeout"
    assert resilience.BREAKERS.snapshot() == {"llm:fallback": "closed", "llm:primary": "open"}


async def test_cancellation_is_not_an_outage_and_never_closes_a_circuit(clock: Clock) -> None:
    import asyncio

    breaker = CircuitBreaker("dep", failure_threshold=2, reset_timeout_s=10)

    async def cancelled() -> None:
        raise asyncio.CancelledError  # e.g. the customer closed the browser mid-stream

    for _ in range(5):
        with pytest.raises(asyncio.CancelledError):
            await breaker.call(cancelled)
    assert breaker.state == State.CLOSED and breaker.failures == 0

    async def down() -> None:
        raise TimeoutError

    for _ in range(2):
        with pytest.raises(TimeoutError):
            await breaker.call(down)
    clock.now += 11
    with pytest.raises(asyncio.CancelledError):
        await breaker.call(cancelled)  # the half-open probe is cancelled ...
    assert breaker.state == State.HALF_OPEN  # ... which proves nothing: still not closed
    with pytest.raises(TimeoutError):
        await breaker.call(down)  # the next call is the probe
    assert breaker.state == State.OPEN


def test_http_status_on_the_response_is_classified() -> None:
    class Response:
        status_code = 401

    class HTTPStatusError(Exception):
        response = Response()

    assert not is_dependency_failure(HTTPStatusError())  # a client error, not an outage
