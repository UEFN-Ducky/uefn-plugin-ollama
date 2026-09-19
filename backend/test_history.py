"""Last-5 call log + slider clamp (no host prefs)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

try:
    from .history import KEEP, get_call, history_payload, load_history, record_call
    from .settings_store import _clamp_opt, apply_history
except ImportError:
    from history import KEEP, get_call, history_payload, load_history, record_call
    from settings_store import _clamp_opt, apply_history


def _isolate() -> Path:
    root = Path(tempfile.mkdtemp(prefix="ollama-hist-"))
    os.environ["LOCALAPPDATA"] = str(root)
    return root


def test_keeps_last_five() -> None:
    _isolate()
    for i in range(7):
        record_call({"model": "llama3.2", "tokens_per_s": i, "num_ctx": 8192})
    rows = load_history()
    assert len(rows) == KEEP
    assert rows[0]["tokens_per_s"] == 2
    assert rows[-1]["tokens_per_s"] == 6
    payload = history_payload()
    assert payload["ok"] is True
    assert payload["count"] == 5
    assert get_call(0)["tokens_per_s"] == 2
    assert get_call(-1)["tokens_per_s"] == 6
    assert get_call(99) is None


def test_clamp_opt_bounds() -> None:
    assert _clamp_opt("temperature", 9) == 2.0
    assert _clamp_opt("temperature", -1) == 0.0
    assert _clamp_opt("top_k", 40.4) == 40
    assert _clamp_opt("f16_kv", 1) is True
    assert _clamp_opt("use_mmap", 0) is False


def test_apply_history_missing() -> None:
    _isolate()
    out = apply_history(0, "llama3.2")
    assert out["ok"] is False


if __name__ == "__main__":
    test_keeps_last_five()
    test_clamp_opt_bounds()
    test_apply_history_missing()
    print("ok")
