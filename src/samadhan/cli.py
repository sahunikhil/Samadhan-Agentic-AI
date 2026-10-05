"""``samadhan`` command-line interface.

samadhan seed                    create + seed the demo commerce/helpdesk databases
samadhan ingest [--force]        (re)index the knowledge base
samadhan serve all               commerce + helpdesk MCP servers + API/UI in one process (dev)
samadhan serve api|commerce|helpdesk|knowledge   run one service (production: one per container)
samadhan chat --customer cust_002   interactive terminal chat, incl. approvals/confirmations
samadhan graph                   print the Mermaid diagram of the support graph
samadhan eval retrieval|rag|agent|redteam|all    run the evaluation suite
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from samadhan.config import get_settings

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Samadhan Agentic AI - multi-agent customer support."
)
eval_app = typer.Typer(
    no_args_is_help=True, help="Evaluation suite (retrieval IR metrics, RAGAS, agent metrics, red team)."
)
app.add_typer(eval_app, name="eval")
keys_app = typer.Typer(no_args_is_help=True, help="Service-token keys (Ed25519).")
app.add_typer(keys_app, name="keys")
feedback_app = typer.Typer(no_args_is_help=True, help="User feedback -> eval dataset flywheel.")
app.add_typer(feedback_app, name="feedback")
console = Console()

if sys.platform == "win32":  # psycopg async needs the selector loop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]


@app.command()
def seed() -> None:
    """Create and seed the demo databases (idempotent: resets demo data)."""
    from samadhan.mcp_servers.commerce.seed import seed_commerce
    from samadhan.mcp_servers.db import Database
    from samadhan.mcp_servers.helpdesk.server import HelpdeskBase

    s = get_settings()

    async def _run() -> dict[str, int]:
        commerce = Database(s.mcp.commerce_db_url)
        helpdesk = Database(s.mcp.helpdesk_db_url)
        out = await seed_commerce(commerce)
        await helpdesk.create_all(HelpdeskBase)
        await commerce.dispose()
        await helpdesk.dispose()
        return out

    console.print(f"[green]Seeded[/green] {asyncio.run(_run())}")


@app.command()
def ingest(force: Annotated[bool, typer.Option(help="Re-embed every document")] = False) -> None:
    """Index the knowledge base into the configured vector store (incremental by default)."""
    from samadhan.observability import configure_logging
    from samadhan.rag.embeddings import EmbeddingModels
    from samadhan.rag.ingest import ingest_knowledge_base
    from samadhan.rag.stores import create_vector_store

    s = get_settings()
    configure_logging(s)

    async def _run() -> None:
        store = create_vector_store(s)
        try:
            report = await ingest_knowledge_base(s, store, EmbeddingModels(s.retrieval), force=force)
            console.print_json(json.dumps(report.__dict__))
        finally:
            await store.close()

    asyncio.run(_run())


def _mcp_app(name: str) -> Any:
    from samadhan.telemetry import instrument_asgi, setup_tracing

    s = get_settings()
    setup_tracing(s, f"samadhan-mcp-{name}")
    return instrument_asgi(_mcp_asgi(name, s))


def _mcp_asgi(name: str, s: Any) -> Any:
    if name == "commerce":
        from samadhan.mcp_servers.commerce.server import create_commerce_server

        return create_commerce_server(s).http_app(stateless_http=True)
    if name == "helpdesk":
        from samadhan.mcp_servers.helpdesk.server import create_helpdesk_server

        return create_helpdesk_server(s).http_app(stateless_http=True)
    if name == "knowledge":
        from samadhan.mcp_servers.knowledge.server import create_knowledge_server
        from samadhan.rag.embeddings import EmbeddingModels
        from samadhan.rag.retriever import HybridRetriever
        from samadhan.rag.stores import create_vector_store

        holder: dict[str, HybridRetriever] = {}

        async def get_retriever() -> HybridRetriever:
            if "r" not in holder:
                models = EmbeddingModels(s.retrieval)
                holder["r"] = HybridRetriever(create_vector_store(s), models, s.retrieval)
            return holder["r"]

        return create_knowledge_server(get_retriever, s).http_app(stateless_http=True)
    raise typer.BadParameter(name)


@app.command()
def serve(
    service: Annotated[str, typer.Argument(help="api | commerce | helpdesk | knowledge | all")] = "all",
    port: Annotated[int | None, typer.Option(help="Override the port")] = None,
) -> None:
    """Run one service, or everything needed for local development (`all`)."""
    s = get_settings()
    if service == "api":
        uvicorn.run(
            "samadhan.api.app:create_app", factory=True, host=s.api.host, port=port or s.api.port, proxy_headers=True
        )
        return
    ports = {"commerce": s.mcp.commerce_port, "helpdesk": s.mcp.helpdesk_port, "knowledge": s.mcp.knowledge_port}
    if service in ports:
        uvicorn.run(_mcp_app(service), host=s.api.host, port=port or ports[service])  # type: ignore[arg-type]
        return
    if service != "all":
        raise typer.BadParameter("expected api, commerce, helpdesk, knowledge or all")

    from samadhan.api.app import create_app

    async def _all() -> None:
        servers = [
            uvicorn.Server(
                uvicorn.Config(_mcp_app("commerce"), host="127.0.0.1", port=s.mcp.commerce_port, log_level="warning")
            ),  # type: ignore[arg-type]
            uvicorn.Server(
                uvicorn.Config(_mcp_app("helpdesk"), host="127.0.0.1", port=s.mcp.helpdesk_port, log_level="warning")
            ),  # type: ignore[arg-type]
        ]
        await asyncio.gather(*(srv.serve() for srv in servers), _serve_api_when_mcp_ready(servers))

    async def _serve_api_when_mcp_ready(servers: list[uvicorn.Server]) -> None:
        while not all(srv.started for srv in servers):  # noqa: ASYNC110 - uvicorn exposes only a flag
            await asyncio.sleep(0.1)
        console.print(
            Panel.fit(
                f"Samadhan UI  ->  http://localhost:{port or s.api.port}\nAPI docs    ->  http://localhost:{port or s.api.port}/docs",
                title="ready",
            )
        )
        api = uvicorn.Server(uvicorn.Config(create_app(s), host=s.api.host, port=port or s.api.port))
        await api.serve()
        for srv in servers:
            srv.should_exit = True

    asyncio.run(_all())


@app.command()
def graph(xray: Annotated[bool, typer.Option(help="Expand subgraphs")] = True) -> None:
    """Print the support graph as a Mermaid diagram (paste into docs or mermaid.live)."""
    from samadhan.agents.graph import build_support_graph
    from samadhan.agents.specialists import GraphDeps
    from samadhan.agents.toolkit import MCPToolkit
    from samadhan.llm import ModelRegistry
    from samadhan.rag.graph import build_knowledge_graph

    s = get_settings()
    models = ModelRegistry(s.llm)
    kg = build_knowledge_graph(s, models, None)  # type: ignore[arg-type]  # structure only
    g = build_support_graph(GraphDeps(s, models, None, MCPToolkit(s), kg))  # type: ignore[arg-type]
    console.print(g.get_graph(xray=1 if xray else 0).draw_mermaid(), markup=False, highlight=False)
    console.print("\n%% knowledge_agent (Corrective RAG) subgraph", markup=False)
    console.print(kg.get_graph().draw_mermaid(), markup=False, highlight=False)


@app.command()
def chat(customer: Annotated[str, typer.Option(help="Demo customer id, e.g. cust_002")] = "cust_001") -> None:
    """Interactive terminal chat. You play the customer *and* the supervisor for approvals."""
    from samadhan.bootstrap import build_container
    from samadhan.service import SupportService

    async def _run() -> None:
        async with build_container() as container:
            svc = SupportService(container)
            thread_id: str | None = None
            console.print(
                Panel.fit(f"Chatting as [bold]{customer}[/bold]. Type 'exit' to quit, 'new' for a new thread.")
            )
            while True:
                text = Prompt.ask("[cyan]you[/cyan]").strip()
                if text.lower() in {"exit", "quit"}:
                    return
                if text.lower() == "new":
                    thread_id = None
                    continue
                events = svc.stream_turn(customer_id=customer, message=text, thread_id=thread_id, channel="cli")
                thread_id = await _render(events)
                while thread_id and (view := await svc.thread_view(thread_id))["pending"]:
                    decisions: dict[str, dict[str, object]] = {}
                    for p in view["pending"]:
                        if p["kind"] == "customer_confirmation":
                            decisions[p["interrupt_id"]] = {
                                "accept": Confirm.ask(f"[yellow]confirm[/yellow] {p['title']}")
                            }
                        elif p["kind"] == "refund_approval":
                            ok = Confirm.ask(f"[magenta]supervisor[/magenta] approve? {p['title']}")
                            decisions[p["interrupt_id"]] = {"decision": "approve" if ok else "reject"}
                        else:
                            reply = Prompt.ask(
                                "[magenta]human agent[/magenta] reply (empty = follow up by email)", default=""
                            )
                            decisions[p["interrupt_id"]] = (
                                {"action": "reply", "message": reply, "agent_name": "Jordan"}
                                if reply
                                else {"action": "defer"}
                            )
                    await _render(
                        svc.stream_resume(
                            thread_id=thread_id, decisions=decisions, actor="system", actor_name="cli", customer_id=None
                        )
                    )

    async def _render(events: object) -> str | None:
        thread: str | None = None
        streamed = False
        async for e in events:  # type: ignore[attr-defined]
            t = e["type"]
            if t == "run_started":
                thread = e["thread_id"]
            elif t == "status":
                console.print(f"  [dim]> {e['label']}[/dim]")
            elif t == "tool" and e.get("event") == "tool_start":
                console.print(f"  [yellow]tool[/yellow] {e['agent']} -> {e['tool']}({json.dumps(e.get('args', {}))})")
            elif t == "token":
                if not streamed:
                    console.print("[green]samadhan[/green]: ", end="")
                    streamed = True
                console.print(e["text"], end="", markup=False, highlight=False)
            elif t == "draft_reset":
                console.print("\n  [red](reviewer asked for a revision)[/red]")
                streamed = False
            elif t == "final":
                if not streamed:
                    console.print(f"[green]samadhan[/green]: {e['reply']}", markup=False)
                console.print(f"\n  [dim]outcome: {e['outcome']}[/dim]")
            elif t == "error":
                console.print(f"[red]{e['message']}[/red]")
        return thread

    asyncio.run(_run())


@app.command()
def token(customer_id: str) -> None:
    """Print a demo customer session token (for curl / API testing)."""
    from samadhan.api.security import issue_customer_token

    console.print(issue_customer_token(get_settings(), customer_id))


@keys_app.command("generate")
def keys_generate() -> None:
    """Print a fresh Ed25519 key pair as env lines: the PRIVATE key goes to the API only, the PUBLIC
    key (or SAMADHAN_MCP__JWT_JWKS_URI=<api>/.well-known/jwks.json) to the MCP servers."""
    from samadhan.mcp_servers.auth import generate_key_pair

    private, public = generate_key_pair()
    one_line = lambda pem: pem.strip().replace("\n", "\\n")  # noqa: E731 - env files need one line
    print(f'SAMADHAN_MCP__JWT_PRIVATE_KEY="{one_line(private)}"')
    print(f'SAMADHAN_MCP__JWT_PUBLIC_KEY="{one_line(public)}"')


@keys_app.command("ensure")
def keys_ensure() -> None:
    """Dev/Compose: create the shared local key pair if missing (no-op when keys are configured)."""
    from samadhan.mcp_servers.auth import key_id, verification_key

    console.print(f"service-token key ready (kid={key_id(verification_key(get_settings().mcp))})")


# ---- evaluation -------------------------------------------------------------------------------


def _ids(value: str | None) -> list[str] | None:
    """`--ids S01,S03` runs a subset (e.g. to resume after a free-tier quota ran out)."""
    return [x.strip() for x in value.split(",") if x.strip()] if value else None


@feedback_app.command("harvest")
def feedback_harvest(
    api: Annotated[str, typer.Option(help="Running Samadhan API")] = "http://localhost:8000",
    out: Path = Path("evals/datasets/candidates/feedback_candidates.jsonl"),
    limit: int = 200,
) -> None:
    """Turn thumbs-down replies into *unlabeled* eval candidates (label them, then promote to the golden set)."""
    import httpx

    from samadhan.feedback import to_candidate

    s = get_settings()
    r = httpx.get(
        f"{api}/v1/admin/feedback",
        params={"rating": "down", "limit": limit},
        headers={"X-Admin-Key": s.api.admin_api_key.get_secret_value()},
        timeout=30,
    )
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    new = [c for c in map(to_candidate, r.json()) if c["id"] not in seen]
    with out.open("a", encoding="utf-8") as fh:
        for c in new:
            fh.write(json.dumps(c) + "\n")
    console.print(f"[green]{len(new)} new candidate(s)[/green] -> {out} (already harvested: {len(seen)})")


@eval_app.command("retrieval")
def eval_retrieval(out: Path = Path("evals/reports")) -> None:
    """IR metrics (hit@k, recall@k, MRR, nDCG) for dense / sparse / hybrid / +rerank. No LLM needed."""
    from samadhan.evaluation.runner import run_retrieval_eval

    asyncio.run(run_retrieval_eval(out))


@eval_app.command("rag")
def eval_rag(out: Path = Path("evals/reports"), limit: int | None = None, ids: str | None = None) -> None:
    """RAGAS metrics on the knowledge specialist (faithfulness, relevancy, context precision/recall, correctness)."""
    from samadhan.evaluation.runner import run_rag_eval

    asyncio.run(run_rag_eval(out, limit=limit, ids=_ids(ids)))


@eval_app.command("agent")
def eval_agent(
    out: Path = Path("evals/reports"),
    limit: int | None = None,
    ids: str | None = None,
    judge: Annotated[bool, typer.Option(help="Score goal accuracy / topic adherence with the LLM judge")] = True,
    repeats: Annotated[int, typer.Option(min=1, help="Trials per scenario -> pass^k reliability")] = 1,
) -> None:
    """End-to-end scenarios: routing accuracy, tool-call F1, goal accuracy, topic adherence, HITL behavior."""
    from samadhan.evaluation.runner import run_agent_eval

    asyncio.run(run_agent_eval(out, limit=limit, ids=_ids(ids), use_judge=judge, repeats=repeats))


@eval_app.command("rejudge")
def eval_rejudge(rows: Path = Path("evals/reports/agent-rows.jsonl"), out: Path = Path("evals/reports")) -> None:
    """Re-score goal accuracy from persisted agent rows (new judge / fixed reference goals) - no agent re-run."""
    from samadhan.evaluation.runner import run_rejudge

    asyncio.run(run_rejudge(rows, out))


@eval_app.command("chunking")
def eval_chunking(out: Path = Path("evals/reports")) -> None:
    """Chunking ablation: chunk size, structure awareness, contextual headers - quality vs context cost. No LLM."""
    from samadhan.evaluation.runner import run_chunking_eval

    asyncio.run(run_chunking_eval(out))


@eval_app.command("cache")
def eval_cache(
    out: Path = Path("evals/reports"),
    verify: Annotated[bool, typer.Option(help="Run the LLM equivalence verifier on candidates")] = True,
) -> None:
    """Semantic cache safety: similarity sweep (paraphrases vs near misses) + verified false-hit rate."""
    from samadhan.evaluation.runner import run_cache_eval

    asyncio.run(run_cache_eval(out, verify=verify))


@eval_app.command("redteam")
def eval_redteam(out: Path = Path("evals/reports"), ids: str | None = None) -> None:
    """Prompt-injection / data-exfiltration attacks against the guardrails."""
    from samadhan.evaluation.runner import run_redteam_eval

    asyncio.run(run_redteam_eval(out, ids=_ids(ids)))


@eval_app.command("all")
def eval_all(
    out: Path = Path("evals/reports"),
    gate: Annotated[bool, typer.Option(help="Exit non-zero if a quality gate fails")] = True,
    only: Annotated[
        str, typer.Option(help="Comma list: retrieval,chunking,cache,rag,agent,redteam")
    ] = "retrieval,rag,agent,redteam",
) -> None:
    """Run every suite, write a combined report and enforce quality gates (CI)."""
    from samadhan.evaluation.runner import run_all

    suites = tuple(x.strip() for x in only.split(",") if x.strip())
    passed = asyncio.run(run_all(out, suites=suites))  # type: ignore[arg-type]
    if gate and not passed:
        raise typer.Exit(code=1)


Suite = Literal["retrieval", "chunking", "cache", "rag", "agent", "redteam"]

if __name__ == "__main__":
    app()
