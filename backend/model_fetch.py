"""Ollama model list fetch for this gateway plugin."""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.agent.model_fetch import ModelInfo, _cache_put

_log = logging.getLogger(__name__)
_CACHE_MAX = 512
_CACHE_TTL_S = 6 * 3600.0


_OLLAMA_INFO_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def clear_model_cache() -> None:
    _OLLAMA_INFO_CACHE.clear()


def fetch_models(base_url: str, **_kw: Any) -> list[ModelInfo]:
    return _fetch_ollama(base_url)

def ollama_model_info(base_url: str, model: str) -> dict[str, Any]:
    """Return {'context_length': int|None, 'capabilities': list[str]} via /api/show."""
    from .ollama_url import normalize_ollama_base

    base = normalize_ollama_base(base_url or "")
    name = (model or "").strip()
    if not name:
        return {"context_length": None, "capabilities": []}
    cache_key = (base, name)
    hit = _OLLAMA_INFO_CACHE.get(cache_key)
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]

    info: dict[str, Any] = {"context_length": None, "capabilities": []}
    try:
        import httpx

        r = httpx.post(f"{base}/api/show", json={"model": name}, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        caps = data.get("capabilities")
        if isinstance(caps, list):
            info["capabilities"] = [str(c) for c in caps]
        model_info = data.get("model_info") or {}
        for key, value in model_info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                info["context_length"] = value
                break
    except Exception:
        pass
    _cache_put(_OLLAMA_INFO_CACHE, cache_key, (time.time(), info))
    return info


def _fetch_ollama(base_url: str) -> list[ModelInfo]:
    """List pulled models from /api/tags only.

    /api/show per name used to run on Test & Save (10s each) and left the
    picker empty until that finished. Caps/context still come from
    ollama_model_info when a chat actually needs them.
    """
    import httpx

    from .ollama_url import normalize_ollama_base

    base = normalize_ollama_base(base_url)
    r = httpx.get(f"{base}/api/tags", timeout=15.0)
    r.raise_for_status()
    models: list[ModelInfo] = []
    for item in r.json().get("models", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        # ponytail: tags has no capabilities; assume tools so Settings pickers
        # show the row. Upgrade: fill from /api/show in the background.
        models.append(
            ModelInfo(
                id=name,
                display_name=name,
                supports_tools=True,
                price_in=0.0,
                price_out=0.0,
                is_local=True,
                supports_thinking_effort=True,
            )
        )
    models.sort(key=lambda m: m.id, reverse=True)
    return models

