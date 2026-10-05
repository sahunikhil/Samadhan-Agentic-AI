"""Talk to Samadhan as another agent would - over A2A, knowing nothing about LangGraph.

    uv run samadhan serve api                       # terminal 1
    uv run python examples/a2a_client.py cust_002 "Please cancel order VW-10005"

What happens:
1. **Discovery** - the client fetches ``/.well-known/agent-card.json`` (skills, auth scheme, endpoint).
2. **Auth** - the card declares an HTTP bearer scheme; we send a customer token (demo: minted locally
   with the dev secret, in production issued by the identity provider for the delegating user).
3. **Task lifecycle** - streamed ``status_update`` / ``artifact_update`` events; when the task stops in
   ``input-required`` we answer on the same ``task_id`` / ``context_id`` (here: from stdin).
"""

from __future__ import annotations

import asyncio
import sys

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

from samadhan.api.security import issue_customer_token
from samadhan.config import get_settings

BASE_URL = "http://localhost:8000"


async def main(customer_id: str, text: str) -> None:
    token = issue_customer_token(get_settings(), customer_id)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=120) as http:
        client = await create_client(BASE_URL, client_config=ClientConfig(httpx_client=http, streaming=True))
        task_id: str | None = None
        context_id: str | None = None
        while True:
            message = new_text_message(text, role=Role.ROLE_USER, task_id=task_id, context_id=context_id)
            state = TaskState.TASK_STATE_UNSPECIFIED
            async for event in client.send_message(SendMessageRequest(message=message)):
                if event.HasField("task"):
                    task_id, context_id = event.task.id, event.task.context_id
                elif event.HasField("status_update"):
                    state = event.status_update.status.state
                    note = get_message_text(event.status_update.status.message)
                    print(f"[{TaskState.Name(state)}] {note}")
                elif event.HasField("artifact_update"):
                    print(f"\n{get_artifact_text(event.artifact_update.artifact)}\n")
            if state != TaskState.TASK_STATE_INPUT_REQUIRED:
                break
            # A real calling agent would ask its own user, or decide by policy.
            text = await asyncio.to_thread(input, "your answer> ")
        await client.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('usage: a2a_client.py <customer_id> "<message>"')
    asyncio.run(main(sys.argv[1], " ".join(sys.argv[2:])))
