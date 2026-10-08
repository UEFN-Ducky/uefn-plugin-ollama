"""Concurrent status requests must never open a burst of Ollama consoles."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

try:
    from . import local
except ImportError:
    import local


@pytest.fixture(autouse=True)
def reset_startup(monkeypatch):
    monkeypatch.setattr(local, "_SERVE_PROCESSES", {})
    monkeypatch.setattr(local, "_SERVE_NEXT_TRY", {})
    monkeypatch.setattr(local, "api_version_ok", lambda *a, **k: False)
    monkeypatch.setattr(local, "ollama_bin", lambda: "ollama")


def test_parallel_polls_share_one_starting_process(monkeypatch):
    launches = []
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(local, "_spawn_serve", lambda binary: launches.append(binary) or process)
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert not any(pool.map(lambda _: local.ensure_local_server("http://localhost:11434", wait_s=0), range(6)))
    assert launches == ["ollama"]
    monkeypatch.setattr(local.time, "monotonic", lambda: 10**12)
    assert not local.ensure_local_server("http://127.0.0.1:11434", wait_s=0)
    assert launches == ["ollama"]


@pytest.mark.parametrize("process", [None, SimpleNamespace(poll=lambda: 1)])
def test_failed_launch_has_retry_cooldown(monkeypatch, process):
    now = [100.0]
    launches = []
    monkeypatch.setattr(local.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(local, "_spawn_serve", lambda binary: launches.append(binary) or process)
    for _ in range(6):
        assert not local.ensure_local_server("http://127.0.0.1:11434", wait_s=0)
    assert launches == ["ollama"]
    now[0] += 31
    assert not local.ensure_local_server("http://127.0.0.1:11434", wait_s=0)
    assert launches == ["ollama", "ollama"]


def test_healthy_server_never_launches(monkeypatch):
    monkeypatch.setattr(local, "api_version_ok", lambda *a, **k: True)
    monkeypatch.setattr(local, "_spawn_serve", lambda *_: pytest.fail("healthy server was restarted"))
    assert local.ensure_local_server("http://127.0.0.1:11434", wait_s=0)


@pytest.mark.skipif(os.name != "nt", reason="Windows console flags")
def test_windows_server_launch_is_hidden(monkeypatch):
    launches = []
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(local.subprocess, "Popen", lambda **kwargs: launches.append(kwargs) or process)
    assert local._spawn_serve("ollama") is process
    launch = launches[0]
    assert launch["args"] == ["ollama", "serve"]
    assert launch["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert not launch["creationflags"] & subprocess.DETACHED_PROCESS
    assert launch["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
    assert launch["startupinfo"].wShowWindow == subprocess.SW_HIDE
    assert launch["stdin"] == launch["stdout"] == launch["stderr"] == subprocess.DEVNULL
