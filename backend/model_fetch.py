"""Ollama model list fetch for this gateway plugin."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import fields
from typing import Any

from backend.agent.model_fetch import ModelInfo, _cache_put

_log = logging.getLogger(__name__)
_CACHE_MAX = 512
_CACHE_TTL_S = 6 * 3600.0
_MODEL_INFO_FIELDS = {f.name for f in fields(ModelInfo)}

THINKING_MENU = {
    "lo": "Faster",
    "hi": "Smarter",
    "levels": [
        {"id": "off", "label": "Off", "thinking_tokens": 0, "hint": "reasoning_effort=none"},
        {"id": "low", "label": "Low", "thinking_tokens": None, "hint": "reasoning_effort=low, no token cap"},
        {"id": "medium", "label": "Med", "thinking_tokens": None, "hint": "reasoning_effort=medium, no token cap"},
        {"id": "high", "label": "High", "thinking_tokens": None, "hint": "reasoning_effort=high, no token cap"},
    ],
}


def _model_info(**kw: Any) -> ModelInfo:
    """Drop unknown fields so an older host ModelInfo does not TypeError."""
    return ModelInfo(**{k: v for k, v in kw.items() if k in _MODEL_INFO_FIELDS})


_OLLAMA_INFO_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def thinking_menu_for(capabilities: list[str] | None) -> dict[str, Any] | None:
    caps = {str(c).strip().lower() for c in (capabilities or [])}
    return THINKING_MENU if "thinking" in caps else None


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


def model_from_show(name: str, info: dict[str, Any]) -> ModelInfo:
    caps = [str(c).lower() for c in (info.get("capabilities") or [])]
    menu = thinking_menu_for(caps)
    ctx = info.get("context_length")
    try:
        context_limit = int(ctx) if ctx else None
    except (TypeError, ValueError):
        context_limit = None
    return _model_info(
        id=name,
        display_name=name,
        supports_vision="vision" in caps,
        supports_tools="tools" in caps,
        context_limit=context_limit,
        price_in=0.0,
        price_out=0.0,
        is_local=True,
        supports_thinking_effort=bool(menu),
        thinking_menu=menu,
    )


def enrich_and_cache(base_url: str, names: list[str]) -> list[ModelInfo]:
    """Fill context + real capabilities from /api/show. Safe to run in a thread."""
    from .ollama_url import normalize_ollama_base

    base = normalize_ollama_base(base_url)
    models = [model_from_show(n, ollama_model_info(base, n)) for n in names if str(n).strip()]
    try:
        from backend.agent.model_fetch import _cache_provider_models

        _cache_provider_models("ollama", models)
    except Exception:
        pass
    return models


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
    names: list[str] = []
    for item in r.json().get("models", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        names.append(name)
        # ponytail: tags has no capabilities. Do not fake thinking/tools —
        # a background enrich fills /api/show. Upgrade: wait if tags grows caps.
        models.append(
            _model_info(
                id=name,
                display_name=name,
                price_in=0.0,
                price_out=0.0,
                is_local=True,
            )
        )
    models.sort(key=lambda m: m.id, reverse=True)
    if names and os.environ.get("DUCKY_OLLAMA_SKIP_ENRICH", "").strip() not in ("1", "true", "yes"):
        threading.Thread(target=enrich_and_cache, args=(base, names), daemon=True).start()
    return models
