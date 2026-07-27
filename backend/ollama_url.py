"""Ollama base URL normalization (stored in credentials as the ollama "key")."""

from __future__ import annotations

DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def normalize_ollama_base(url: str) -> str:
    """Canonical base without /v1 — e.g. http://localhost:11434."""
    u = (url or DEFAULT_OLLAMA_BASE).strip().rstrip("/")
    if not u:
        u = DEFAULT_OLLAMA_BASE
    if not u.startswith("http://") and not u.startswith("https://"):
        u = f"http://{u}"
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def ollama_openai_base_url(url: str) -> str:
    return f"{normalize_ollama_base(url)}/v1"
