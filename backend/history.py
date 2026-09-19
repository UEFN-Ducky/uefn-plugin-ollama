"""Last N Ollama call snapshots so sliders can be tuned against real runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

KEEP = 5


def _path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TMP") or ".") / "UEFN-Ducky"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ollama_call_history.json"


def load_history() -> list[dict[str, Any]]:
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    rows = raw.get("calls") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)][-KEEP:]


def record_call(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Append one call; keep only the last KEEP. Returns the new list."""
    calls = load_history()
    item = {k: v for k, v in row.items() if v is not None}
    item["ts"] = float(item.get("ts") or time.time())
    calls.append(item)
    calls = calls[-KEEP:]
    path = _path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"calls": calls}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return calls


def history_payload() -> dict[str, Any]:
    calls = load_history()
    return {"ok": True, "calls": calls, "count": len(calls), "keep": KEEP}


def get_call(index: int) -> dict[str, Any] | None:
    calls = load_history()
    if not calls:
        return None
    i = int(index)
    if i < 0:
        i = len(calls) + i
    if i < 0 or i >= len(calls):
        return None
    return calls[i]
