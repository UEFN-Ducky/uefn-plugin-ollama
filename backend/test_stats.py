"""Live stats snapshot shape + thinking flag."""

from __future__ import annotations

import os
import subprocess

try:
    from . import stats
except ImportError:
    import stats


def test_live_stats_shape() -> None:
    stats._ps = lambda _base: [
        {"name": "llama3.2:latest", "size_vram": 1024 * 1024, "size": 2 * 1024 * 1024}
    ]
    stats.set_generating("")
    out = stats.live_stats("http://localhost:11434")
    assert out["models"][0]["name"] == "llama3.2:latest"
    stats.set_generating("llama3.2:latest")
    out = stats.live_stats("http://localhost:11434")
    stats.set_generating("")
    assert out["ok"] is True
    assert out["thinking"] is True
    assert out["generating_model"] == "llama3.2:latest"
    assert out["running"] is True
    assert "cpu_pct" in out
    assert "ram_pct" in out
    assert "ram_label" in out
    assert "disk_pct" in out
    assert "disk_label" in out
    assert "cpu_count" in out
    assert out["models"][0]["name"] == "llama3.2:latest"


def test_hidden_popen_hides_windows_console() -> None:
    kw = stats._hidden_popen_kwargs()
    if os.name == "nt":
        assert kw.get("creationflags") == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        assert kw == {}


def test_idle_when_not_generating() -> None:
    stats._ps = lambda _base: []
    stats.set_generating("")
    out = stats.live_stats("http://localhost:11434")
    assert out["ok"] is True
    assert out["thinking"] is False
    assert out["running"] is False
    assert out["models"] == []


def test_live_stats_skips_ps_while_generating() -> None:
    stats._PS_CACHE = [{"name": "qwen3.8:latest", "size_vram": 1024, "size": 2048}]
    calls = {"n": 0}

    def boom(_base):
        calls["n"] += 1
        raise AssertionError("/api/ps must not run while generating")

    stats._ps = boom
    stats.set_generating("qwen3.8:latest")
    try:
        out = stats.live_stats("http://localhost:11434")
    finally:
        stats.set_generating("")
    assert calls["n"] == 0
    assert out["thinking"] is True
    assert out["models"][0]["name"] == "qwen3.8:latest"


if __name__ == "__main__":
    test_live_stats_shape()
    test_hidden_popen_hides_windows_console()
    test_idle_when_not_generating()
    test_live_stats_skips_ps_while_generating()
    print("ok")
