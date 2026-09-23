"""Installed Ollama models: list, storage, delete, /api/ps usage windows."""

from __future__ import annotations

from typing import Any

try:
    from .ollama_url import normalize_ollama_base
except ImportError:
    from ollama_url import normalize_ollama_base


def format_bytes(n: int) -> str:
    raw = max(0, int(n or 0))
    if raw < 1024:
        return f"{raw} B"
    units = ("KB", "MB", "GB", "TB")
    value = float(raw)
    for unit in units:
        value /= 1024.0
        if value < 1024 or unit == "TB":
            if value >= 10:
                return f"{value:.1f} {unit}"
            return f"{value:.2f} {unit}"
    return f"{raw} B"


def _httpx():
    import httpx

    return httpx


def api_version_ok(base_url: str, timeout: float = 3.0) -> bool:
    base = normalize_ollama_base(base_url)
    try:
        r = _httpx().get(f"{base}/api/version", timeout=timeout, trust_env=False)
        return r.status_code < 400
    except Exception:
        return False


def _tags(base: str) -> list[dict[str, Any]]:
    r = _httpx().get(f"{base}/api/tags", timeout=15.0, trust_env=False)
    r.raise_for_status()
    rows = r.json().get("models") or []
    return [row for row in rows if isinstance(row, dict)]


def _ps(base: str) -> list[dict[str, Any]]:
    try:
        r = _httpx().get(f"{base}/api/ps", timeout=8.0, trust_env=False)
        r.raise_for_status()
        rows = r.json().get("models") or []
        return [row for row in rows if isinstance(row, dict)]
    except Exception:
        return []


def list_local(base_url: str) -> dict[str, Any]:
    """Yours: /api/tags + /api/ps + cached /api/show caps/context."""
    from .model_fetch import ollama_model_info

    base = normalize_ollama_base(base_url)
    try:
        tags = _tags(base)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "models": [], "storage_bytes": 0, "count": 0}
    loaded: dict[str, dict[str, Any]] = {}
    for row in _ps(base):
        name = str(row.get("name") or row.get("model") or "").strip()
        if name:
            loaded[name] = row
    models: list[dict[str, Any]] = []
    total = 0
    for item in tags:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        size = int(item.get("size") or 0)
        total += max(0, size)
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        info = ollama_model_info(base, name)
        caps = [str(c).lower() for c in (info.get("capabilities") or [])]
        live = loaded.get(name)
        models.append(
            {
                "name": name,
                "size": size,
                "size_label": format_bytes(size),
                "parameter_size": str(details.get("parameter_size") or ""),
                "quantization": str(details.get("quantization_level") or ""),
                "family": str(details.get("family") or ""),
                "modified_at": str(item.get("modified_at") or ""),
                "capabilities": caps,
                "context_length": info.get("context_length"),
                "loaded": live is not None,
                "size_vram": int((live or {}).get("size_vram") or 0),
                "size_vram_label": format_bytes(int((live or {}).get("size_vram") or 0)),
                "expires_at": str((live or {}).get("expires_at") or ""),
            }
        )
    models.sort(key=lambda m: str(m.get("name") or ""), reverse=True)
    return {
        "ok": True,
        "models": models,
        "storage_bytes": total,
        "storage_label": format_bytes(total),
        "count": len(models),
        "loaded_count": sum(1 for m in models if m.get("loaded")),
    }


def delete_model(base_url: str, name: str) -> dict[str, Any]:
    model = (name or "").strip()
    if not model:
        return {"ok": False, "error": "Model name required"}
    base = normalize_ollama_base(base_url)
    try:
        r = _httpx().request(
            "DELETE", f"{base}/api/delete", json={"name": model}, timeout=30.0, trust_env=False
        )
        r.raise_for_status()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "deleted": model}


def usage_windows(base_url: str) -> dict[str, Any]:
    """Picker footer bars: one window per loaded model (VRAM)."""
    base = normalize_ollama_base(base_url)
    rows = _ps(base)
    windows: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or row.get("model") or "").strip()
        if not name:
            continue
        vram = int(row.get("size_vram") or 0)
        size = int(row.get("size") or vram or 0)
        limit = max(vram, size, 1)
        used = min(max(vram, 0), limit)
        windows.append(
            {
                "id": name,
                "label": name,
                "used": float(used),
                "limit": float(limit),
                "unit": "bytes",
                "readout": f"{format_bytes(used)} VRAM",
            }
        )
    return {"windows": windows}
