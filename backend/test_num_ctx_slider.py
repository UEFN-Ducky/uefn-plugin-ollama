"""Context slider ticks never exceed the model's /api/show max."""

from __future__ import annotations

try:
    from .local import delete_model
    from .settings_store import clamp_num_ctx, ticks_for_max
except ImportError:
    from local import delete_model
    from settings_store import clamp_num_ctx, ticks_for_max


def test_ticks_stop_at_model_max() -> None:
    assert ticks_for_max(32_768) == [4096, 8192, 16384, 32768]
    assert 262144 not in ticks_for_max(32_768)


def test_256k_save_clamps_on_32k_model() -> None:
    assert clamp_num_ctx(262_144, 32_768) == 32_768
    assert clamp_num_ctx(8_000, 32_768) == 4096
    assert clamp_num_ctx(4_096, 32_768) == 4096


def test_unknown_max_keeps_positive() -> None:
    assert clamp_num_ctx(0, 0) == 1


def test_delete_refuses_empty_name() -> None:
    assert delete_model("http://localhost:11434", "")["ok"] is False
    assert delete_model("http://localhost:11434", "   ")["ok"] is False


if __name__ == "__main__":
    test_ticks_stop_at_model_max()
    test_256k_save_clamps_on_32k_model()
    test_unknown_max_keeps_positive()
    test_delete_refuses_empty_name()
    print("ok")
