"""Role-based chat model registry.

Agents ask for a *role* ("smart", "fast"), never a concrete model. This single
indirection buys us:

* **Provider portability** - switch Gemini -> Groq -> local Ollama with one env var.
* **Cost control** - cheap/fast models do the high-volume routing and grading work;
  the stronger model is reserved for tool use and final answers (model routing).
* **Resilience** - every role has fallback models (by default the *other* role of the same
  profile, which on free tiers has its own separate quota). Agents get them through
  ``ModelFallbackMiddleware``; graph nodes through ``chat()`` / ``structured()``.
* **Testability** - tests inject deterministic fake models with ``override()``.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable

from caseflow.config import PROFILE_DEFAULTS, LLMSettings
from caseflow.resilience import guarded, model_key

Role = Literal["smart", "fast", "judge"]

# Free tiers are rate limited per minute. A client-side token bucket smooths bursts
# (parallel specialists!) instead of eating HTTP 429s and burning retries.
_RATE_LIMITS_RPS: dict[str, float] = {
    "google_genai": 0.25,  # ~15 requests/minute
    "groq": 0.5,  # ~30 requests/minute
}

_JSON_SCHEMA_PROVIDERS = {"groq", "google_genai", "ollama", "openai"}

# When a role's model fails (outage, 429, a free tier's daily token cap), use the profile's
# other model: different model => separate quota. Live evals hit exactly this: the fast
# model exhausted its 200K tokens/day while the smart model still had budget.
_CROSS_ROLE: dict[str, str] = {"fast": "smart", "smart": "fast"}


def _provider_of(model: str) -> str:
    return model.split(":", 1)[0] if ":" in model else "openai"


class ModelRegistry:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._cache: dict[str, BaseChatModel] = {}
        self._overrides: dict[str, BaseChatModel] = {}
        self._limiters: dict[str, InMemoryRateLimiter] = {}

    # -- public API -----------------------------------------------------------------
    def get(self, role: Role) -> BaseChatModel:
        if role in self._overrides:
            return self._overrides[role]
        return self._build(self._settings.model_for(role))

    def model_name(self, role: Role) -> str:
        if role in self._overrides:
            return f"override:{type(self._overrides[role]).__name__}"
        return self._settings.model_for(role)

    def fallback_names(self, role: Role = "smart") -> list[str]:
        """Explicit ``CASEFLOW_LLM__FALLBACK_MODELS`` win; otherwise the profile's other role."""
        if self._settings.fallback_models:
            return list(self._settings.fallback_models)
        other = _CROSS_ROLE.get(role)
        defaults = PROFILE_DEFAULTS.get(self._settings.profile, {})
        name = defaults.get(other or "", "") if other else ""
        return [name] if name and name != self._settings.model_for(role) else []

    def fallbacks(self, role: Role = "smart") -> list[BaseChatModel]:
        if self._overrides:
            return []  # deterministic tests never fall back to real providers
        return [self._build(m) for m in self.fallback_names(role)]

    def chat(self, role: Role) -> Runnable[Any, Any]:
        """The role's model with cross-model fallbacks, for plain generation in graph nodes."""
        model = self.get(role)
        fallbacks = self.fallbacks(role)
        if not fallbacks:
            return model
        # Each model behind its own circuit breaker: an open circuit skips straight to the next.
        return guarded(model, model_key(model)).with_fallbacks([guarded(f, model_key(f)) for f in fallbacks])

    def override(self, role: Role, model: BaseChatModel) -> None:
        """Replace a role with a specific model instance (tests, experiments)."""
        self._overrides[role] = model

    def structured(self, role: Role, schema: type[Any]) -> Runnable[Any, Any]:
        """Structured output with a *fallback strategy*.

        Neither method is 100% reliable on small open models - both failures were seen live:

        * ``function_calling`` (LangChain's default) forces ``tool_choice=required``; the model
          sometimes answers in prose -> "Tool choice is required, but model did not call a tool".
        * ``json_schema`` (native constrained decoding) occasionally returns an *empty*
          generation -> "json_validate_failed".

        So the chain is: native JSON-schema decoding -> function calling -> the same two on each
        fallback model (``Runnable.with_fallbacks``). Independent failure modes rarely coincide.
        """
        if role in self._overrides:
            return self.get(role).with_structured_output(schema)
        chain: list[Runnable[Any, Any]] = []
        for name in [self._settings.model_for(role), *self.fallback_names(role)]:
            model = self._build(name)
            key = model_key(model)
            if _provider_of(name) in _JSON_SCHEMA_PROVIDERS:
                chain.append(guarded(model.with_structured_output(schema, method="json_schema"), key))
            chain.append(guarded(model.with_structured_output(schema), key))
        return chain[0].with_fallbacks(chain[1:]) if len(chain) > 1 else chain[0]

    # -- internals ------------------------------------------------------------------
    def _limiter(self, provider: str) -> InMemoryRateLimiter | None:
        rps = _RATE_LIMITS_RPS.get(provider)
        if rps is None:
            return None
        if provider not in self._limiters:
            self._limiters[provider] = InMemoryRateLimiter(
                requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=4
            )
        return self._limiters[provider]

    def _build(self, model: str) -> BaseChatModel:
        if model in self._cache:
            return self._cache[model]
        provider = _provider_of(model)
        kwargs: dict[str, Any] = {"temperature": self._settings.temperature}
        if provider == "ollama":
            kwargs["base_url"] = self._settings.ollama_base_url
        else:
            kwargs["timeout"] = self._settings.request_timeout_s
            kwargs["max_retries"] = self._settings.max_retries
        if limiter := self._limiter(provider):
            kwargs["rate_limiter"] = limiter
        chat_model = init_chat_model(model, **kwargs)
        self._cache[model] = chat_model
        return chat_model
