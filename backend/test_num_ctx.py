"""num_ctx is detected from the prompt + /api/show max — no slider ladder."""

from __future__ import annotations

from .ollama_provider import (
    _HEADROOM_TOKENS,
    _IMAGE_TOKENS_EACH,
    _attachment_tokens,
    _estimate_prompt_tokens,
    _needed_num_ctx,
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


def test_needed_is_prompt_plus_headroom_capped_at_model():
    assert _needed_num_ctx(262_144, 20_000) == 20_000 + _HEADROOM_TOKENS
    assert _needed_num_ctx(262_144, 40_000) == 40_000 + _HEADROOM_TOKENS
    assert _needed_num_ctx(32_768, 40_000) == 32_768
    assert _needed_num_ctx(262_144, 300_000) == 262_144


def test_ratchet_never_shrinks():
    clear_num_ctx_ratchet()
    a = _ratcheted_num_ctx("http://localhost:11434", "m", 262_144, 20_000)
    assert a == 20_000 + _HEADROOM_TOKENS
    b = _ratcheted_num_ctx("http://localhost:11434", "m", 262_144, 1_000)
    assert b == a
    c = _ratcheted_num_ctx("http://localhost:11434", "m", 262_144, 40_000)
    assert c == 40_000 + _HEADROOM_TOKENS


def test_estimate_counts_images():
    clear_num_ctx_ratchet()
    msgs = [_Msg("hi", attachments=[_Att("image"), _Att("image")])]
    est = _estimate_prompt_tokens("sys", msgs, [])  # type: ignore[arg-type]
    assert est >= 2 * _IMAGE_TOKENS_EACH
    assert _attachment_tokens([_Att("image")]) == _IMAGE_TOKENS_EACH
