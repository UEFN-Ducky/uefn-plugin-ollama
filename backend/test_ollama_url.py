"""Loopback Ollama URLs must hit 127.0.0.1, not ::1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parent / "ollama_url.py"
    spec = importlib.util.spec_from_file_location("ollama_url_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_localhost_and_ipv6_loopback_become_ipv4():
    url = _load()
    assert url.normalize_ollama_base("http://localhost:11434") == "http://127.0.0.1:11434"
    assert url.normalize_ollama_base("http://localhost:11434/v1") == "http://127.0.0.1:11434"
    assert url.normalize_ollama_base("http://[::1]:11434") == "http://127.0.0.1:11434"
    assert url.normalize_ollama_base("") == "http://127.0.0.1:11434"


def test_other_hosts_stay():
    url = _load()
    assert url.normalize_ollama_base("http://192.168.1.20:11434") == "http://192.168.1.20:11434"
    assert url.normalize_ollama_base("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
