from __future__ import annotations

from caseflow.config import LLMSettings
from caseflow.llm import ModelRegistry


def test_roles_fall_back_to_the_other_role_of_the_profile() -> None:
    registry = ModelRegistry(LLMSettings(profile="groq"))
    assert registry.fallback_names("fast") == ["groq:openai/gpt-oss-120b"]
    assert registry.fallback_names("smart") == ["groq:openai/gpt-oss-20b"]


def test_explicit_fallbacks_override_profile_defaults() -> None:
    registry = ModelRegistry(LLMSettings(profile="gemini", fallback_models=["groq:openai/gpt-oss-120b"]))
    assert registry.fallback_names("fast") == ["groq:openai/gpt-oss-120b"]


def test_custom_profile_role_resolution() -> None:
    settings = LLMSettings(profile="custom", smart_model="ollama:qwen3:8b", fast_model="ollama:qwen3:8b")
    registry = ModelRegistry(settings)
    assert registry.model_name("smart") == "ollama:qwen3:8b"
    assert registry.fallback_names("smart") == []  # nothing distinct to fall back to
