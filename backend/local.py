"""Installed Ollama models: list, storage, delete, /api/ps usage windows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    from .ollama_url import normalize_ollama_base
except ImportError:
    from ollama_url import normalize_ollama_base

_SERVE_LOCK = threading.Lock()


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


def server_unreachable(exc: BaseException) -> bool:
    """True when nothing is listening. Timeouts are a live-but-stuck server."""
    text = f"{type(exc).__name__} {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return False
    if isinstance(exc, ConnectionRefusedError):
        return True
    return any(
        bit in text
        for bit in (
            "10061",
            "connection refused",
            "actively refused",
            "connecterror",
            "failed to establish",
            "all connection attempts failed",
        )
    )


def ollama_bin() -> str:
    found = shutil.which("ollama") or ""
    if found:
        return found
    if os.name == "nt":
        for cand in (
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ):
            if cand and os.path.isfile(cand):
                return cand
    return ""


def _loopback(base: str) -> bool:
    host = (urlsplit(base).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _spawn_serve(binary: str) -> None:
    kwargs: dict[str, Any] = {
        "args": [binary, "serve"],
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        extra = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["creationflags"] = flags | extra
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(**kwargs)
    except OSError:
        pass


def ensure_local_server(base_url: str, *, wait_s: float = 8.0) -> bool:
    """Start `ollama serve` when this machine's loopback daemon is down."""
    base = normalize_ollama_base(base_url)
    if api_version_ok(base, timeout=1.2):
        return True
    if not _loopback(base):
        return False
    binary = ollama_bin()
    if not binary:
        return False
    with _SERVE_LOCK:
        if not api_version_ok(base, timeout=0.8):
            _spawn_serve(binary)
    deadline = time.monotonic() + max(0.0, wait_s)
    while time.monotonic() < deadline:
        if api_version_ok(base, timeout=0.8):
            return True
        time.sleep(0.35)
    return False


def models_dir() -> Path:
    override = os.environ.get("OLLAMA_MODELS", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".ollama" / "models"


def _manifest_size(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return 0
    total = 0
    config = data.get("config") if isinstance(data, dict) else None
    if isinstance(config, dict):
        total += max(0, int(config.get("size") or 0))
    layers = data.get("layers") if isinstance(data, dict) else None
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, dict):
                total += max(0, int(layer.get("size") or 0))
    return total


def disk_models(root: Path | None = None) -> list[dict[str, Any]]:
    """Names from the local manifest tree. Works while the daemon is down."""
    manifests = (root or models_dir()) / "manifests"
    if not manifests.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in manifests.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(manifests).parts
        if len(rel) < 2:
            continue
        name, tag = rel[-2], rel[-1]
        size = _manifest_size(path)
        rows.append(
            {
                "name": f"{name}:{tag}",
                "size": size,
                "size_label": format_bytes(size),
                "parameter_size": "",
                "quantization": "",
                "family": "",
                "modified_at": "",
                "capabilities": [],
                "context_length": None,
                "loaded": False,
                "size_vram": 0,
                "size_vram_label": format_bytes(0),
                "expires_at": "",
            }
        )
    rows.sort(key=lambda m: str(m.get("name") or ""), reverse=True)
    return rows


def _tags(base: str) -> list[dict[str, Any]]:
    r = _httpx().get(f"{base}/api/tags", timeout=15.0, trust_env=False)
    r.raise_for_status()
    rows = r.json().get("models") or []
    return [row for row in rows if isinstance(row, dict)]


def read_tags(base: str) -> list[dict[str, Any]]:
    """GET /api/tags. If loopback refuses, start the daemon and try once more."""
    try:
        return _tags(base)
    except Exception as exc:
        if not server_unreachable(exc) or not ensure_local_server(base):
            raise
        return _tags(base)


def _ps(base: str) -> list[dict[str, Any]]:
    try:
        r = _httpx().get(f"{base}/api/ps", timeout=8.0, trust_env=False)
        r.raise_for_status()
        rows = r.json().get("models") or []
        return [row for row in rows if isinstance(row, dict)]
    except Exception:
        return []


def _pack(models: list[dict[str, Any]], *, ok: bool, error: str = "") -> dict[str, Any]:
    total = sum(max(0, int(m.get("size") or 0)) for m in models)
    out: dict[str, Any] = {
        "ok": ok,
        "models": models,
        "storage_bytes": total,
        "storage_label": format_bytes(total),
        "count": len(models),
        "loaded_count": sum(1 for m in models if m.get("loaded")),
    }
    if error:
        out["error"] = error
    return out


def list_local(base_url: str) -> dict[str, Any]:
    """Yours: /api/tags + /api/ps + cached /api/show caps/context."""
    base = normalize_ollama_base(base_url)
    try:
        tags = read_tags(base)
    except Exception as exc:
        disk = disk_models() if server_unreachable(exc) else []
        if disk:
            return _pack(
                disk,
                ok=False,
                error="Ollama isn't running. These models are already on disk; the server is starting.",
            )
        return {"ok": False, "error": str(exc), "models": [], "storage_bytes": 0, "count": 0}
    try:
        from .model_fetch import ollama_model_info
    except ImportError:
        from model_fetch import ollama_model_info
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
    return _pack(models, ok=True)


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
