"""A2A: CaseFlow as a remote agent, driven by the official ``a2a-sdk`` client over real HTTP.

Covers discovery (agent card), transport auth (401), the task lifecycle mapping
(completed / input-required -> completed / staff wait), and per-customer task isolation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from a2a.client import Client, ClientConfig, create_client
from a2a.helpers import get_artifact_text, new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, Task, TaskState
from a2a.utils.errors import TaskNotFoundError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from caseflow.a2a_server import a2a_thread_id, parse_confirmation
from caseflow.api.app import create_app
from caseflow.api.security import issue_customer_token
from caseflow.config import Settings
from caseflow.llm import ModelRegistry
from tests.conftest import _free_port, _ServerThread
from tests.fakes import ScriptedChatModel
from tests.integration.test_support_graph import responder


@pytest.fixture(scope="module")
def api(mcp_servers: Settings) -> Iterator[Settings]:
    port = _free_port()
    settings = mcp_servers.model_copy(
        update={
            "retrieval": mcp_servers.retrieval.model_copy(update={"qdrant_path": ":memory:"}),
            "api": mcp_servers.api.model_copy(update={"public_url": f"http://127.0.0.1:{port}"}),
        }
    )
    models = ModelRegistry(settings.llm)
    fake = ScriptedChatModel(responder=responder)
    models.override("smart", fake)
    models.override("fast", fake)
    app = create_app(settings, models=models, checkpointer=InMemorySaver(), store=InMemoryStore())
    with _ServerThread(app, port):
        yield settings


async def _client(settings: Settings, customer: str | None) -> Client:
    headers = {"Authorization": f"Bearer {issue_customer_token(settings, customer)}"} if customer else {}
    http = httpx.AsyncClient(headers=headers, timeout=60)
    # Discovery: the client fetches /.well-known/agent-card.json and picks the JSON-RPC interface.
    return await create_client(settings.api.public_url, client_config=ClientConfig(httpx_client=http, streaming=True))


@pytest.fixture
async def cust1(api: Settings) -> AsyncIterator[Client]:
    client = await _client(api, "cust_001")  # owns VW-10003
    yield client
    await client.close()


@pytest.fixture
async def cust2(api: Settings) -> AsyncIterator[Client]:
    client = await _client(api, "cust_002")  # owns VW-10004 (refund) and VW-10005 (cancellable)
    yield client
    await client.close()


async def send(client: Client, text: str, *, task: Task | None = None) -> Task:
    """Send one message (continuing ``task`` if given) and return the final task snapshot."""
    message = new_text_message(
        text, role=Role.ROLE_USER, context_id=task.context_id if task else None, task_id=task.id if task else None
    )
    last: Task | None = None
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            last = event.task
    assert last is not None or task is not None
    return await client.get_task(GetTaskRequest(id=(last or task).id))  # type: ignore[union-attr]


def _text(task: Task) -> str:
    return "\n".join(get_artifact_text(a) for a in task.artifacts)


def test_parse_confirmation() -> None:
    assert parse_confirmation("Yes, go ahead") is True
    assert parse_confirmation("no, keep it") is False
    assert parse_confirmation("hmm, what does that mean?") is None
    assert parse_confirmation("yes... actually no") is None  # contradictory -> ask again, never guess
    assert a2a_thread_id("cust_1", "ctx") != a2a_thread_id("cust_2", "ctx")


async def test_agent_card_is_public_and_advertises_skills_and_auth(api: Settings) -> None:
    async with httpx.AsyncClient() as http:
        card = (await http.get(f"{api.api.public_url}/.well-known/agent-card.json")).json()
    assert {s["id"] for s in card["skills"]} == {"order-support", "returns-refunds", "product-policy-qa"}
    assert card["supportedInterfaces"][0]["url"].endswith("/a2a")
    assert card["securitySchemes"]["customerBearer"]["httpAuthSecurityScheme"]["scheme"] == "bearer"


async def test_jsonrpc_endpoint_requires_a_customer_token(api: Settings) -> None:
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{api.api.public_url}/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage"})
        forged = await http.post(
            f"{api.api.public_url}/a2a", json={}, headers={"Authorization": "Bearer not-a-real-token"}
        )
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer")
    assert forged.status_code == 401


async def test_simple_request_completes_with_a_reply_artifact(cust1: Client) -> None:
    task = await send(cust1, "Where is VW-10003? Also do you price match?")
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert "Tracking" in _text(task)


async def test_customer_confirmation_maps_to_input_required(cust2: Client) -> None:
    task = await send(cust2, "Please cancel order VW-10005")
    assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED  # MCP elicitation surfaced as A2A input

    unclear = await send(cust2, "what would that mean for my points?", task=task)
    assert unclear.status.state == TaskState.TASK_STATE_INPUT_REQUIRED  # never guesses a yes/no

    done = await send(cust2, "Yes, please go ahead", task=unclear)
    assert done.status.state == TaskState.TASK_STATE_COMPLETED
    assert '"cancelled":true' in _text(done).replace(" ", "")


async def test_staff_approval_wait_completes_with_a_status_note(cust2: Client) -> None:
    task = await send(cust2, "Where is my refund for my headphones return?")  # $242 > auto-approve limit
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert "Voltwise specialist" in _text(task)


async def test_customers_cannot_read_each_others_tasks(cust1: Client, cust2: Client) -> None:
    task = await send(cust1, "Where is VW-10003? Also do you price match?")
    with pytest.raises(TaskNotFoundError):  # "not found", not "forbidden": existence is not leaked
        await cust2.get_task(GetTaskRequest(id=task.id))  # the task store is scoped by the verified customer
