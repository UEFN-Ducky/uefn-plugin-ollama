"""num_ctx is pinned once from host high-water + headroom — never resized mid-chat."""

from __future__ import annotations

from .ollama_provider import (
    _HEADROOM_TOKENS,
    _IMAGE_TOKENS_EACH,
    _attachment_tokens,
    _estimate_prompt_tokens,
    _pinned_num_ctx,
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


def test_pin_matches_host_high_water():
    from backend.agent.context_memory import epoch_num_ctx

    clear_num_ctx_ratchet()
    pinned = _pinned_num_ctx("http://localhost:11434", "m", 32_768)
    assert pinned == epoch_num_ctx(32_768)
    assert pinned <= 32_768
    assert pinned >= 1000 + _HEADROOM_TOKENS or pinned == 32_768


def test_pin_never_resizes():
    clear_num_ctx_ratchet()
    a = _pinned_num_ctx("http://localhost:11434", "m", 262_144)
    b = _pinned_num_ctx("http://localhost:11434", "m", 262_144)
    assert b == a
    c = _pinned_num_ctx("http://localhost:11434", "other", 8_192)
    assert c == 8_192 or c <= 8_192


def test_estimate_counts_images():
    clear_num_ctx_ratchet()
    msgs = [_Msg("hi", attachments=[_Att("image"), _Att("image")])]
    est = _estimate_prompt_tokens("sys", msgs, [])  # type: ignore[arg-type]
    assert est >= 2 * _IMAGE_TOKENS_EACH
    assert _attachment_tokens([_Att("image")]) == _IMAGE_TOKENS_EACH
