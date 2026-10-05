"""Typed, validated application settings.

Every tunable in Samadhan lives here, loaded from environment variables (and an
optional ``.env`` file) by ``pydantic-settings``. Nested groups use ``__`` as the
delimiter, so ``SAMADHAN_LLM__PROFILE=groq`` sets ``settings.llm.profile``.

Why one settings tree instead of scattered ``os.getenv`` calls?
  * Validation happens once, at startup, with readable errors ("fail fast").
  * Secrets are ``SecretStr`` so they never leak into logs or tracebacks.
  * Production guard-rails (``_check_production_safety``) refuse to boot with
    development defaults such as the demo JWT secret.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProfile = Literal["gemini", "groq", "ollama", "custom"]

# Free-tier defaults per provider (verified against provider docs, Sept 2026).
# Gemini: one free API key covers every role. Groq: very fast open-weight models.
# Ollama: fully local and offline - no key, no rate limits, no data leaves the box.
PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "gemini": {
        "smart": "google_genai:gemini-3.5-flash",
        "fast": "google_genai:gemini-3.5-flash-lite",
        "judge": "google_genai:gemini-3.8-flash",
    },
    "groq": {
        "smart": "groq:openai/gpt-oss-120b",
        "fast": "groq:openai/gpt-oss-20b",
        "judge": "groq:qwen/qwen3.8-27b",  # different model family than the system under test
    },
    "ollama": {
        "smart": "ollama:qwen3:8b",
        "fast": "ollama:qwen3:4b",
        "judge": "ollama:qwen3:8b",
    },
    "custom": {"smart": "", "fast": "", "judge": ""},
}


class LLMSettings(BaseModel):
    """Which models play which role.

    Samadhan never hard-codes a model. Each *role* maps to a
    ``provider:model`` string understood by ``langchain.chat_models.init_chat_model``:

    * ``smart`` - specialists and answer synthesis (tool calling + reasoning).
    * ``fast``  - triage, query rewriting, guard checks, memory extraction.
    * ``judge`` - offline evaluation (RAGAS). Kept separate so the grader is not
      the same model grading its own homework.

    A *profile* fills sensible free-tier defaults; any role can be overridden.
    """

    profile: LLMProfile = "gemini"
    smart_model: str | None = None
    fast_model: str | None = None
    judge_model: str | None = None
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Tried in order when the primary model errors (rate limit, outage).",
    )
    temperature: float = 0.0
    request_timeout_s: float = 60.0
    max_retries: int = 2
    ollama_base_url: str = "http://localhost:11434"

    def model_for(self, role: Literal["smart", "fast", "judge"]) -> str:
        explicit = {"smart": self.smart_model, "fast": self.fast_model, "judge": self.judge_model}[role]
        if explicit:
            return explicit
        default = PROFILE_DEFAULTS[self.profile][role]
        if not default:
            raise ValueError(f"LLM profile 'custom' requires SAMADHAN_LLM__{role.upper()}_MODEL to be set.")
        return default


class RetrievalSettings(BaseModel):
    """Knowledge-base indexing and hybrid retrieval."""

    backend: Literal["qdrant", "pgvector"] = "qdrant"
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL. Leave empty to use embedded (on-disk) mode - no server needed.",
    )
    qdrant_path: Path = Path(".data/qdrant")
    qdrant_api_key: SecretStr | None = None
    collection: str = "voltwise_kb"

    # Local ONNX models via FastEmbed: free, CPU-only, no API key.
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: Path | None = None

    hybrid: bool = True
    rerank: bool = True
    candidate_k: int = Field(
        default=12, ge=1, description="Candidates per retriever before fusion; also the reranker's input size."
    )
    top_k: int = Field(default=5, ge=1, description="Chunks handed to the LLM after reranking.")
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    rrf_k: int = 60
    relevance_threshold: float = Field(
        default=0.25,
        description="Min reranker probability for a chunk to count as relevant (CRAG evaluator).",
    )

    kb_dir: Path = Path("data/knowledge_base")
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Semantic answer cache for the knowledge specialist (see rag/semantic_cache.py).
    semantic_cache: bool = True
    cache_candidate_threshold: float = Field(
        default=0.80,
        description="Min embedding similarity for a cache *candidate*; an LLM verifier then confirms "
        "equivalence. Tuned with `samadhan eval cache`.",
    )
    cache_ttl_minutes: int = 24 * 60


class PersistenceSettings(BaseModel):
    """Where LangGraph checkpoints (short-term memory) and the Store (long-term memory) live."""

    database_url: SecretStr | None = Field(
        default=None,
        description="postgresql://... enables Postgres checkpointer + store + pgvector. Empty = local SQLite/in-memory.",
    )
    sqlite_path: Path = Path(".data/checkpoints.sqlite")
    pool_max_size: int = 10


class MCPSettings(BaseModel):
    """MCP server endpoints and the service-token contract between API and MCP servers."""

    commerce_url: str = "http://127.0.0.1:8101/mcp"
    helpdesk_url: str = "http://127.0.0.1:8102/mcp"
    knowledge_url: str = "http://127.0.0.1:8103/mcp"
    commerce_port: int = 8101
    helpdesk_port: int = 8102
    knowledge_port: int = 8103

    # Asymmetric (EdDSA / Ed25519) service tokens: only the API holds the private key; MCP servers verify
    # with the public key or the API's JWKS endpoint, so a compromised MCP server cannot mint tokens.
    # Unset everywhere (dev) -> one key pair is generated under jwt_key_dir and shared by local processes.
    jwt_private_key: SecretStr | None = Field(default=None, description="Ed25519 private key, PEM (API only)")
    jwt_public_key: str | None = Field(default=None, description="Ed25519 public key, PEM (MCP servers)")
    jwt_jwks_uri: str | None = Field(default=None, description="Alternative to jwt_public_key: the API's JWKS URL")
    jwt_key_dir: Path = Path(".data/keys")
    jwt_issuer: str = "samadhan-api"
    jwt_audience: str = "samadhan-mcp"
    token_ttl_s: int = Field(default=300, ge=30)
    tool_cache_ttl_s: int = Field(default=240, ge=0)

    commerce_db_url: str = "sqlite+aiosqlite:///.data/commerce.db"
    helpdesk_db_url: str = "sqlite+aiosqlite:///.data/helpdesk.db"


class AgentSettings(BaseModel):
    """Business rules and safety budgets for the agent graph."""

    company_name: str = "Voltwise"
    breaker_failure_threshold: int = Field(default=5, description="Consecutive failures that open a circuit")
    breaker_reset_timeout_s: float = Field(default=30.0, description="Open-circuit cool-down before a probe")
    turn_budget_usd: float = Field(
        default=0.05,
        description="Soft per-turn LLM spend budget (list prices); breaches are logged and counted.",
    )
    refund_auto_approve_limit: float = Field(
        default=100.0,
        description="Refunds at or below this amount are auto-approved; above it a human must approve.",
    )
    specialist_model_call_limit: int = 6
    specialist_tool_call_limit: int = 8
    max_revisions: int = 1
    history_window: int = 10
    summarize_after_messages: int = 16
    enable_output_guard: bool = True
    enable_long_term_memory: bool = True
    node_timeout_s: float = 90.0
    prompt_guard_model: str | None = Field(
        default="meta-llama/llama-prompt-guard-2-86m",
        description="Groq-hosted injection classifier (free tier). Used only when GROQ_API_KEY is set; None disables.",
    )
    prompt_guard_threshold: float = 0.9


class APISettings(BaseModel):
    host: str = "0.0.0.0"  # noqa: S104 - container default; bind is controlled by the platform
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    admin_api_key: SecretStr = SecretStr("dev-admin-key")
    token_secret: SecretStr = Field(
        default=SecretStr("dev-only-customer-token-secret-change-me-0123456789"),
        description="Signs customer session tokens. Separate from the MCP service-token key on purpose.",
    )
    demo_mode: bool = Field(
        default=True,
        description="Enables /v1/auth/demo-login which issues customer tokens without a password.",
    )
    customer_token_ttl_s: int = 3600
    customer_token_audience: str = "samadhan-api"  # noqa: S105 - JWT audience, not a secret
    rate_limit_per_minute: int = 30
    # Customer identity from your IdP (production): IdP-issued access tokens are accepted alongside the
    # API's own session tokens; the customer id comes from `customer_id_claim`.
    customer_oidc_issuer: str | None = None
    customer_oidc_audience: str | None = None
    customer_oidc_jwks_uri: str | None = None
    customer_oidc_public_key: str | None = None
    customer_id_claim: str = "sub"
    # Staff SSO (OIDC): staff present an IdP-issued JWT (Keycloak, Auth0, Entra ID, Okta ...); roles come
    # from `staff_roles_claim`: supervisor (approve refunds), agent (answer handoffs), admin (re-index).
    staff_oidc_issuer: str | None = None
    staff_oidc_audience: str | None = None
    staff_oidc_jwks_uri: str | None = None
    staff_oidc_public_key: str | None = Field(default=None, description="Static PEM alternative to the JWKS URI")
    staff_roles_claim: str = "roles"
    admin_key_enabled: bool = Field(
        default=True, description="Break-glass X-Admin-Key (all roles). Disable once SSO is configured."
    )
    max_body_bytes: int = Field(default=1_000_000, description="Larger request bodies are rejected with 413")
    ip_rate_limit_per_minute: int = Field(
        default=60, description="Per-IP limit for unauthenticated, CPU-heavy routes (/mcp/knowledge, demo login)"
    )
    public_url: str = Field(
        default="http://localhost:8000",
        description="Externally reachable base URL; advertised in the A2A agent card.",
    )
    a2a_enabled: bool = True


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    log_json: bool = False
    metrics_enabled: bool = True
    langfuse_enabled: bool = Field(
        default=False,
        description="Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST env vars.",
    )
    otel_enabled: bool = Field(
        default=False,
        description="OpenTelemetry traces (GenAI semconv) via OTLP; set OTEL_EXPORTER_OTLP_ENDPOINT.",
    )
    prices: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description='USD per 1M tokens, overrides samadhan.cost.DEFAULT_PRICES: {"model": {"input": .., "output": ..}}',
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAMADHAN_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def _check_production_safety(self) -> Settings:
        if self.environment != "prod":
            return self
        problems: list[str] = []
        m = self.mcp
        if not (m.jwt_private_key or m.jwt_public_key or m.jwt_jwks_uri):
            problems.append(
                "Service-token keys must be configured: SAMADHAN_MCP__JWT_PRIVATE_KEY (API) and "
                "SAMADHAN_MCP__JWT_PUBLIC_KEY or _JWKS_URI (MCP servers) - see `samadhan keys generate`"
            )
        if self.api.admin_api_key.get_secret_value() == "dev-admin-key":
            problems.append("SAMADHAN_API__ADMIN_API_KEY must be changed")
        if len(self.api.admin_api_key.get_secret_value()) < 24:
            problems.append("SAMADHAN_API__ADMIN_API_KEY must be at least 24 characters")
        if self.api.token_secret.get_secret_value().startswith("dev-only"):
            problems.append("SAMADHAN_API__TOKEN_SECRET must be set to a strong random value")
        # HS256 customer tokens are only as strong as this key: a short one can be brute-forced offline
        # from any captured token, and then *any* customer identity can be forged.
        if len(self.api.token_secret.get_secret_value()) < 32:
            problems.append("SAMADHAN_API__TOKEN_SECRET must be at least 32 characters")
        if self.api.demo_mode:
            problems.append("SAMADHAN_API__DEMO_MODE must be false (demo login issues tokens without a password)")
        if problems:
            raise ValueError("Unsafe production configuration:\n  - " + "\n  - ".join(problems))
        return self

    @property
    def data_dir(self) -> Path:
        return Path(".data")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached so env parsing happens once)."""
    return Settings()
