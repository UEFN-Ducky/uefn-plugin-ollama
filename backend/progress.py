"""Parse Ollama pull progress from /api/pull NDJSON or `ollama pull` terminal tail."""

from __future__ import annotations

import json
import re
from typing import Any

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DONE_RE = re.compile(r"\bsuccess\b|\bpulled\b|\bverifying sha256 digest\b", re.I)
_ERR_RE = re.compile(r"\berror\b[:\s]+(.+)", re.I)


def parse_pull_ndjson_line(line: str) -> dict[str, Any]:
    """One streamed /api/pull object → {status, percent, error, done}."""
    text = (line or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return parse_pull_tail(text)
    if not isinstance(data, dict):
        return {}
    err = str(data.get("error") or "").strip()
    status = str(data.get("status") or "").strip()
    completed = data.get("completed")
    total = data.get("total")
    percent: float | None = None
    try:
        if total and float(total) > 0 and completed is not None:
            percent = max(0.0, min(100.0, 100.0 * float(completed) / float(total)))
    except (TypeError, ValueError):
        percent = None
    done = (not err) and status.lower() in ("success", "complete", "completed")
    out: dict[str, Any] = {"status": status or ("error" if err else "")}
    if percent is not None:
        out["percent"] = percent
    if err:
        out["error"] = err
        out["done"] = True
    elif done:
        out["done"] = True
        out["percent"] = 100.0
    return out


def parse_pull_tail(text: str) -> dict[str, Any]:
    """Last useful status from `ollama pull` terminal output."""
    blob = text or ""
    err_m = _ERR_RE.search(blob)
    if err_m and not _DONE_RE.search(blob[-400:]):
        return {"status": "error", "error": err_m.group(1).strip()[:240], "done": True}
    pct = None
    matches = _PCT_RE.findall(blob)
    if matches:
        try:
            pct = max(0.0, min(100.0, float(matches[-1])))
        except ValueError:
            pct = None
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    status = lines[-1][:200] if lines else "pulling"
    done = bool(_DONE_RE.search(blob[-800:])) and not err_m
    out: dict[str, Any] = {"status": "success" if done else status}
    if pct is not None:
        out["percent"] = 100.0 if done else pct
    if done:
        out["done"] = True
        out["percent"] = 100.0
    return out
