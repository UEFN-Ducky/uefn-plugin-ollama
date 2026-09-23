"""Yours lists on-disk models when the Ollama daemon refuses the port."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

try:
    from . import local
except ImportError:
    import local


def _manifest(root: Path, name: str, tag: str, size: int) -> None:
    path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": {"size": 5}, "layers": [{"size": size}]}),
        encoding="utf-8",
    )


def test_disk_models_use_manifest_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _manifest(root, "qwen3.8", "latest", 100)
        _manifest(root, "muse-glimmer", "latest", 40)
        names = [row["name"] for row in local.disk_models(root)]
    assert names == ["qwen3.8:latest", "muse-glimmer:latest"]


def test_list_local_keeps_disk_models_when_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _manifest(root, "qwen3.6", "latest", 80)
        old = os.environ.get("OLLAMA_MODELS")
        os.environ["OLLAMA_MODELS"] = str(root)
        orig_tags = local._tags
        orig_ensure = local.ensure_local_server
        try:

            def _boom(_base: str):
                raise ConnectionError(
                    "[WinError 10061] No connection could be made because the target machine actively refused it"
                )

            local._tags = _boom
            local.ensure_local_server = lambda *_a, **_k: False
            out = local.list_local("http://127.0.0.1:11434")
        finally:
            local._tags = orig_tags
            local.ensure_local_server = orig_ensure
            if old is None:
                os.environ.pop("OLLAMA_MODELS", None)
            else:
                os.environ["OLLAMA_MODELS"] = old
    assert out["ok"] is False
    assert [row["name"] for row in out["models"]] == ["qwen3.6:latest"]
    assert "on disk" in out["error"]


def test_remote_host_does_not_spawn() -> None:
    spawned: list[str] = []
    orig_ok = local.api_version_ok
    orig_spawn = local._spawn_serve
    try:
        local.api_version_ok = lambda *_a, **_k: False
        local._spawn_serve = lambda binary: spawned.append(binary)
        assert local.ensure_local_server("http://192.168.1.9:11434", wait_s=0) is False
    finally:
        local.api_version_ok = orig_ok
        local._spawn_serve = orig_spawn
    assert spawned == []


def test_loopback_spawns_when_down() -> None:
    spawned: list[str] = []
    probes = {"n": 0}
    orig_ok = local.api_version_ok
    orig_spawn = local._spawn_serve
    orig_bin = local.ollama_bin
    try:

        def _probe(*_a, **_k):
            probes["n"] += 1
            return probes["n"] >= 3

        local.api_version_ok = _probe
        local._spawn_serve = lambda binary: spawned.append(binary)
        local.ollama_bin = lambda: "ollama"
        assert local.ensure_local_server("http://127.0.0.1:11434", wait_s=2) is True
    finally:
        local.api_version_ok = orig_ok
        local._spawn_serve = orig_spawn
        local.ollama_bin = orig_bin
    assert spawned == ["ollama"]


if __name__ == "__main__":
    test_disk_models_use_manifest_names()
    test_list_local_keeps_disk_models_when_refused()
    test_remote_host_does_not_spawn()
    test_loopback_spawns_when_down()
    print("ok")
