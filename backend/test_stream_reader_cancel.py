"""Regression: first thinking token must not abort the Ollama stream reader."""

from __future__ import annotations

from pathlib import Path


def test_reader_breaks_on_stop_reader_not_progress():
    src = Path(__file__).with_name("ollama_provider.py").read_text(encoding="utf-8")
    assert "if stop_reader.is_set():" in src
    # First-token `stop_progress.set()` used to `break` the chunk loop.
    assert "for chunk in stream:\n                    if stop_progress.is_set():" not in src
    assert '"reasoning_effort"' in src
    assert '{"think": False}' not in src
    assert '{"think": True}' not in src
