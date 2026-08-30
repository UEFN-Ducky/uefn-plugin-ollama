"""Ollama /v1 uses reasoning_effort, not native think bools."""

from __future__ import annotations

from .ollama_provider import _think_extra


def test_off_is_none_not_think_false():
    assert _think_extra("off") == {"reasoning_effort": "none"}
    assert _think_extra("") == {"reasoning_effort": "none"}
    assert "think" not in _think_extra("off")


def test_levels_map_to_v1_strings():
    assert _think_extra("low") == {"reasoning_effort": "low"}
    assert _think_extra("medium") == {"reasoning_effort": "medium"}
    assert _think_extra("high") == {"reasoning_effort": "high"}
