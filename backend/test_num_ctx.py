"""Unit tests for Ollama num_ctx bucketing + ratchet (no Ollama server needed)."""

from __future__ import annotations

from .ollama_provider import (
    _IMAGE_TOKENS_EACH,
    _attachment_tokens,
    _bucket_num_ctx,
    _estimate_prompt_tokens,
    _ratcheted_num_ctx,
    clear_num_ctx_ratchet,
)


class _Att:
    def __init__(self, kind: str, text: str = "") -> None:
        self.kind = kind
        self.text = text


class _Msg:
    def __init__(self, content: str = "", attachments=None, tool_calls=None) -> None:
        self.role = "user"
        self.content = content
        self.attachments = attachments or []
        self.tool_calls = tool_calls or []


def test_bucket_picks_smallest_fit():
    assert _bucket_num_ctx(262_144, 20_000) == 32_768
    assert _bucket_num_ctx(262_144, 40_000) == 65_536
    assert _bucket_num_ctx(262_144, 100_000) == 131_072
    assert _bucket_num_ctx(262_144, 200_000) == 262_144
    assert _bucket_num_ctx(32_768, 40_000) == 32_768


def test_ratchet_never_shrinks():
    clear_num_ctx_ratchet()
    a = _ratcheted_num_ctx("http://localhost:11434", "m", 262_144, 20_000)
    assert a == 32_768
    b = _ratcheted_num_ctx("http://localhost:11434", "m", 262_144, 1_000)
    assert b == 32_768  # still pinned
    c = _ratcheted_num_ctx("http://localhost:11434", "m", 262_144, 40_000)
    assert c == 65_536


def test_estimate_counts_images():
    clear_num_ctx_ratchet()
    msgs = [_Msg("hi", attachments=[_Att("image"), _Att("image")])]
    est = _estimate_prompt_tokens("sys", msgs, [])  # type: ignore[arg-type]
    assert est >= 2 * _IMAGE_TOKENS_EACH
    assert _attachment_tokens([_Att("image")]) == _IMAGE_TOKENS_EACH
