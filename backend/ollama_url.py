"""Ollama base URL normalization (stored in credentials as the ollama "key")."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
# Windows Ollama binds 127.0.0.1 only. `localhost` resolves ::1 first, and that
# address returns WinError 10061 while the server is up.
_IPV4_LOOPBACK_HOSTS = frozenset({"localhost", "::1"})


def _ipv4_loopback(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host not in _IPV4_LOOPBACK_HOSTS:
        return url
    port = parts.port
    netloc = "127.0.0.1" if port is None else f"127.0.0.1:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def normalize_ollama_base(url: str) -> str:
    """Canonical base without /v1 — loopback is always 127.0.0.1."""
    u = (url or DEFAULT_OLLAMA_BASE).strip().rstrip("/")
    if not u:
        u = DEFAULT_OLLAMA_BASE
    if not u.startswith("http://") and not u.startswith("https://"):
        u = f"http://{u}"
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return _ipv4_loopback(u)


def ollama_openai_base_url(url: str) -> str:
    return f"{normalize_ollama_base(url)}/v1"
