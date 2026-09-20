"""num_ctx is pinned once from host high-water + headroom — never resized mid-chat."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parents[1] / "UEFN-Ducky-Release" / "ducky_app"
if APP.is_dir() and str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "backend"))

try:
    from .ollama_provider import (
        _HEADROOM_TOKENS,
        _IMAGE_TOKENS_EACH,
        _attachment_tokens,
        _estimate_prompt_tokens,
        _num_ctx_ratchet,
        _pinned_num_ctx,
        clear_num_ctx_ratchet,
        vram_safe_ctx_cap,
    )
except ImportError:
    from ollama_provider import (
        _HEADROOM_TOKENS,
        _IMAGE_TOKENS_EACH,
        _attachment_tokens,
        _estimate_prompt_tokens,
        _num_ctx_ratchet,
        _pinned_num_ctx,
        clear_num_ctx_ratchet,
        vram_safe_ctx_cap,
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


def test_vram_cap_8gb_is_32k():
    eight = 8 * 1024 * 1024 * 1024
    assert vram_safe_ctx_cap(262_144, vram_bytes=eight) == 32_768
    assert vram_safe_ctx_cap(16_384, vram_bytes=eight) == 16_384
    assert vram_safe_ctx_cap(262_144, vram_bytes=24 * 1024 * 1024 * 1024) == 262_144
    assert vram_safe_ctx_cap(262_144, vram_bytes=0) == 32_768


def test_pin_snaps_down_to_vram():
    clear_num_ctx_ratchet()
    _num_ctx_ratchet["http://localhost:11434|fat"] = 65_536
    pinned = _pinned_num_ctx("http://localhost:11434", "fat", 262_144)
    assert pinned <= vram_safe_ctx_cap(262_144)
    assert pinned <= 32_768 or vram_safe_ctx_cap(262_144) >= 65_536


def test_estimate_counts_images():
    clear_num_ctx_ratchet()
    msgs = [_Msg("hi", attachments=[_Att("image"), _Att("image")])]
    est = _estimate_prompt_tokens("sys", msgs, [])  # type: ignore[arg-type]
    assert est >= 2 * _IMAGE_TOKENS_EACH
    assert _attachment_tokens([_Att("image")]) == _IMAGE_TOKENS_EACH


if __name__ == "__main__":
    test_pin_matches_host_high_water()
    test_pin_never_resizes()
    test_vram_cap_8gb_is_32k()
    test_pin_snaps_down_to_vram()
    test_estimate_counts_images()
    print("ok")
