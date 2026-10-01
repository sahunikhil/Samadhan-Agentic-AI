# Convenience targets (all wrap `uv run ...`; see README for the underlying commands).
.PHONY: install dev seed ingest test lint typecheck eval eval-retrieval eval-chunking eval-cache graph docker up down studio mcp

install:        ## install dependencies (incl. eval + dev tools)
	uv sync --extra eval

seed:           ## create + seed demo databases
	uv run caseflow seed

ingest:         ## index the knowledge base (incremental)
	uv run caseflow ingest

dev: seed       ## run MCP servers + API + UI locally on http://localhost:8000
	uv run caseflow serve all

test:           ## unit + integration tests (no API key needed)
	uv run pytest -q

lint:           ## ruff lint + format check
	uv run ruff check src tests && uv run ruff format --check src tests

typecheck:
	uv run mypy

eval-retrieval: ## retrieval ablation (no LLM)
	uv run caseflow eval retrieval

eval-chunking:  ## chunking ablation: quality vs context cost (no LLM)
	uv run caseflow eval chunking

eval-cache:     ## semantic-cache safety: similarity sweep + LLM-verified false-hit rate
	uv run caseflow eval cache

eval:           ## full evaluation suite with quality gates (needs an LLM key)
	uv run caseflow eval all

graph:          ## print the Mermaid diagram of the agent graph
	uv run caseflow graph

docker:
	docker build -t caseflow:latest .

up:             ## full stack: Postgres + Qdrant + MCP servers + API
	docker compose up -d --build

down:
	docker compose down

studio:         ## LangGraph Studio / Agent Server (start MCP servers first: make mcp)
	uv run --with "langgraph-cli[inmem]" langgraph dev --allow-blocking

mcp:            ## run commerce + helpdesk MCP servers in the background
	uv run caseflow serve commerce & uv run caseflow serve helpdesk &
