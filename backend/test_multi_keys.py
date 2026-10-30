"""N Ollama "keys" (server URLs): each turn talks to the server core picked."""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for k in list(sys.modules):
    if k == "backend" or k.startswith("backend."):
        del sys.modules[k]
for p in _here.parents:
    cand = p / "UEFN-Ducky-Release" / "ducky_app"
    if (cand / "backend" / "agent").is_dir():
        sys.path.insert(0, str(_here))
        sys.path.insert(0, str(cand))
        break

from backend.agent.providers import _QUOTA_RE as CORE_QUOTA_RE
from ollama_provider import OllamaProvider
from ollama_url import normalize_ollama_base


def test_each_turn_uses_the_server_core_picked() -> None:
    # register()'s factory: OllamaProvider(_normalize_secret(api_key), model, **kw).
    a = OllamaProvider(normalize_ollama_base("http://pc-a:11434"), "llama3")._base_url
    b = OllamaProvider(normalize_ollama_base("http://pc-b:11434"), "llama3")._base_url
    assert "pc-a" in a and "pc-b" in b


def test_core_default_catches_ollama_cloud_limits_only() -> None:
    """Ollama Cloud's usage limit is a 429 (openai SDK str()); local errors never switch server."""
    cloud = ("Error code: 429 - {'error': {'message': \"you've reached your hourly usage limit, "
             "please upgrade to continue\", 'type': 'api_error'}}")
    assert CORE_QUOTA_RE.search(cloud)
    for local in ("Connection error.", "model requires more system memory (12.0 GiB) than is available",
                  "Error code: 404 - {'error': {'message': 'model \"llama3\" not found, try pulling it first'}}"):
        assert not CORE_QUOTA_RE.search(local), local
