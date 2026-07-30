"""Ollama gateway — local URL provider via host registries."""

from __future__ import annotations

from typing import Any


def _normalize_secret(raw: str) -> str:
    from .ollama_url import normalize_ollama_base

    return normalize_ollama_base((raw or "").strip() or "http://localhost:11434")


def _fetch_models(api_key: str, **_kw: Any) -> Any:
    from .model_fetch import fetch_models

    return fetch_models(_normalize_secret(api_key))


def register(api) -> None:
    from .ollama_provider import OllamaProvider

    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "ollama",
        factory=lambda api_key, model, **kw: OllamaProvider(
            _normalize_secret(api_key), model, **kw
        ),
        fetch_models=_fetch_models,
        test_key_model="llama3.2",
        tool_schema="openai",
        key_optional=True,
        normalize_secret=_normalize_secret,
        clear_model_cache=clear_model_cache,
        cache_mode="local",
    )
    api.log("Ollama gateway contribution active (Providers)")
