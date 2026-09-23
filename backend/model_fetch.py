"""Ollama model list — /api/tags names + /api/show caps. Never invent vision/tools."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any

from backend.agent.model_fetch import ModelInfo, _cache_put

_log = logging.getLogger(__name__)
_CACHE_MAX = 512
_CACHE_TTL_S = 6 * 3600.0
_MODEL_INFO_FIELDS = {f.name for f in fields(ModelInfo)}

# /v1 reasoning_effort values Ollama documents — only attached when /api/show says thinking.
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

_OLLAMA_INFO_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_DISK_LOADED = False


def _model_info(**kw: Any) -> ModelInfo:
    """Drop unknown fields so an older host ModelInfo does not TypeError."""
    return ModelInfo(**{k: v for k, v in kw.items() if k in _MODEL_INFO_FIELDS})


def _disk_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TMP") or ".") / "UEFN-Ducky"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ollama_show_cache.json"


def _load_disk() -> None:
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    try:
        raw = json.loads(_disk_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return
    rows = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(rows, dict):
        return
    now = time.time()
    for key, row in rows.items():
        if not isinstance(row, dict) or "|" not in str(key):
            continue
        base, _, name = str(key).partition("|")
        ts = float(row.get("ts") or 0)
        info = row.get("info")
        if not base or not name or not isinstance(info, dict):
            continue
        if ts and now - ts > _CACHE_TTL_S:
            continue
        _OLLAMA_INFO_CACHE[(base, name)] = (ts or now, info)


def _save_disk() -> None:
    payload = {
        "models": {
            f"{base}|{name}": {"ts": ts, "info": info}
            for (base, name), (ts, info) in _OLLAMA_INFO_CACHE.items()
        }
    }
    path = _disk_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def thinking_menu_for(capabilities: list[str] | None) -> dict[str, Any] | None:
    caps = {str(c).strip().lower() for c in (capabilities or [])}
    return THINKING_MENU if "thinking" in caps else None


def clear_model_cache() -> None:
    _OLLAMA_INFO_CACHE.clear()
    try:
        _disk_path().unlink(missing_ok=True)
    except OSError:
        pass


def parse_modelfile_parameters(raw: str) -> dict[str, Any]:
    """Parse /api/show ``parameters`` (Modelfile PARAMETER lines)."""
    out: dict[str, Any] = {}
    for line in str(raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition(" ")
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        low = val.lower()
        if low in {"true", "false"}:
            out[key] = low == "true"
            continue
        try:
            if "." in val:
                out[key] = float(val)
            else:
                out[key] = int(val)
        except ValueError:
            out[key] = val
    return out


def parse_show_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Map a raw /api/show JSON body. Caps/context/params come only from that body."""
    caps: list[str] = []
    raw_caps = data.get("capabilities")
    if isinstance(raw_caps, list):
        caps = [str(c).strip().lower() for c in raw_caps if str(c).strip()]
    proj = data.get("projector_info")
    if isinstance(proj, dict) and proj and "vision" not in caps:
        caps.append("vision")
    ctx = None
    model_info = data.get("model_info") if isinstance(data.get("model_info"), dict) else {}
    for key, value in model_info.items():
        if str(key).endswith(".context_length"):
            try:
                ctx = int(value)
            except (TypeError, ValueError):
                ctx = None
            if ctx:
                break
    return {
        "capabilities": caps,
        "context_length": ctx,
        "parameters": parse_modelfile_parameters(str(data.get("parameters") or "")),
        "has_projector": bool(isinstance(proj, dict) and proj),
    }


def fetch_models(base_url: str, **_kw: Any) -> list[ModelInfo]:
    return _fetch_ollama(base_url)


def _peek_cache(base: str, name: str) -> dict[str, Any] | None:
    _load_disk()
    hit = _OLLAMA_INFO_CACHE.get((base, name))
    if hit is None:
        return None
    if (time.time() - hit[0]) >= _CACHE_TTL_S:
        return None
    return hit[1]


def ollama_model_info(base_url: str, model: str) -> dict[str, Any]:
    """Return caps/context/parameters via /api/show (cached)."""
    from .ollama_url import normalize_ollama_base

    base = normalize_ollama_base(base_url or "")
    name = (model or "").strip()
    if not name:
        return {"context_length": None, "capabilities": [], "parameters": {}}
    cached = _peek_cache(base, name)
    if cached is not None:
        return cached

    info: dict[str, Any] = {"context_length": None, "capabilities": [], "parameters": {}}
    try:
        import httpx

        r = httpx.post(f"{base}/api/show", json={"model": name}, timeout=10.0, trust_env=False)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            info = parse_show_payload(data)
    except Exception:
        pass
    _cache_put(_OLLAMA_INFO_CACHE, (base, name), (time.time(), info))
    try:
        _save_disk()
    except OSError:
        pass
    return info


def effective_context_limit(
    name: str,
    api_ctx: int | None,
    *,
    vram_bytes: int | None = None,
    apply_saved: bool = True,
) -> int | None:
    """Window Memory high-water should treat as the model max.

    ``/api/show`` can claim 256k; 8GB VRAM only holds ~32k. Advertise the
    smaller number so Settings → Memory folds old turns before Ollama fills.
    Full chat stays saved; the model sees summary + last N (same as OpenAI).
    """
    if api_ctx is None:
        return None
    try:
        limit = int(api_ctx)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    try:
        from .ollama_provider import vram_safe_ctx_cap

        limit = min(limit, int(vram_safe_ctx_cap(limit, vram_bytes=vram_bytes)))
    except Exception:
        pass
    if apply_saved:
        try:
            from .settings_store import saved_num_ctx

            saved = saved_num_ctx(name)
            if saved:
                limit = min(limit, int(saved))
        except Exception:
            pass
    return limit


def model_from_show(
    name: str,
    info: dict[str, Any],
    *,
    vram_bytes: int | None = None,
    apply_saved: bool = True,
) -> ModelInfo:
    caps = [str(c).lower() for c in (info.get("capabilities") or [])]
    menu = thinking_menu_for(caps)
    ctx = info.get("context_length")
    try:
        raw_ctx = int(ctx) if ctx else None
    except (TypeError, ValueError):
        raw_ctx = None
    context_limit = effective_context_limit(
        name, raw_ctx, vram_bytes=vram_bytes, apply_saved=apply_saved
    )
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


def _publish_host_catalog(models: list[ModelInfo]) -> None:
    try:
        from backend.agent.model_fetch import _cache_provider_models

        _cache_provider_models("ollama", models)
    except Exception:
        pass
    try:
        from frontend.ui_web import panel_api as _pa

        _pa._model_cache["ollama"] = list(models)
        _pa._save_model_cache_to_disk()
        _pa._notify_models_updated()
    except Exception:
        pass


def enrich_and_cache(base_url: str, names: list[str]) -> list[ModelInfo]:
    """Fill context + real capabilities from /api/show. Safe to run in a thread."""
    from .ollama_url import normalize_ollama_base

    base = normalize_ollama_base(base_url)
    models = [model_from_show(n, ollama_model_info(base, n)) for n in names if str(n).strip()]
    _publish_host_catalog(models)
    return models


def _show_many(base: str, names: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not names:
        return out
    workers = min(8, len(names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(ollama_model_info, base, n): n for n in names}
        for fut, name in futs.items():
            try:
                out[name] = fut.result()
            except Exception:
                out[name] = {"context_length": None, "capabilities": [], "parameters": {}}
    return out


def _fetch_ollama(base_url: str) -> list[ModelInfo]:
    """Names from /api/tags; vision/tools/thinking/ctx from /api/show."""
    import httpx

    from .ollama_url import normalize_ollama_base

    base = normalize_ollama_base(base_url)
    r = httpx.get(f"{base}/api/tags", timeout=15.0, trust_env=False)
    r.raise_for_status()
    names: list[str] = []
    for item in r.json().get("models", []):
        name = (item.get("name") or "").strip()
        if name:
            names.append(name)
    skip = os.environ.get("DUCKY_OLLAMA_SKIP_ENRICH", "").strip() in ("1", "true", "yes")
    models: list[ModelInfo] = []
    misses: list[str] = []
    for name in names:
        cached = _peek_cache(base, name)
        if cached is not None:
            models.append(model_from_show(name, cached))
        elif skip:
            models.append(_model_info(id=name, display_name=name, price_in=0.0, price_out=0.0, is_local=True))
        else:
            misses.append(name)
    if misses:
        shown = _show_many(base, misses)
        for name in misses:
            models.append(model_from_show(name, shown.get(name) or {}))
        _publish_host_catalog(models)
    models.sort(key=lambda m: m.id, reverse=True)
    return models
