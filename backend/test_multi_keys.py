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

from ollama_provider import OllamaProvider
from ollama_url import normalize_ollama_base


def test_each_turn_uses_the_server_core_picked() -> None:
    # register()'s factory: OllamaProvider(_normalize_secret(api_key), model, **kw).
    a = OllamaProvider(normalize_ollama_base("http://pc-a:11434"), "llama3")._base_url
    b = OllamaProvider(normalize_ollama_base("http://pc-b:11434"), "llama3")._base_url
    assert "pc-a" in a and "pc-b" in b
