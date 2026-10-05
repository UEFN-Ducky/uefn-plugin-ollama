"""Listing Ollama models must not call /api/show per name (that froze Test & Save)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(os.path.abspath(__file__)).parents[1]
APP = ROOT.parents[1] / "UEFN-Ducky-Release" / "ducky_app"
for _name in ("UEFN-Ducky-Release", "UEFN-Ducky-video", "UEFN-Ducky"):
    if (ROOT.parents[1] / _name / "ducky_app" / "backend" / "agent").is_dir():
        APP = ROOT.parents[1] / _name / "ducky_app"
        break
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def _load_plugin_fetch():
    import importlib.util
    import types

    pkg_name = "ollama_gw"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT / "backend")]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.model_fetch",
        ROOT / "backend" / "model_fetch.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    # The plugin dir is itself a ``backend`` package; the host's must win while importing.
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "backend" or k.startswith("backend.")}
    saved_path = list(sys.path)
    sys.path[:] = [str(APP)] + [p for p in sys.path if os.path.abspath(p) != str(ROOT)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "backend" or k.startswith("backend.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return mod


def test_fetch_models_uses_tags_only() -> None:
    model_fetch = _load_plugin_fetch()

    tags = MagicMock()
    tags.json.return_value = {
        "models": [
            {"name": "llama3.2:latest"},
            {"name": "qwen2.5:7b"},
            {"name": ""},
        ]
    }
    tags.raise_for_status = MagicMock()
    httpx = MagicMock()
    httpx.get.return_value = tags
    httpx.post.side_effect = AssertionError("/api/show must not run during list")

    sys.modules["httpx"] = httpx
    os.environ["DUCKY_OLLAMA_SKIP_ENRICH"] = "1"
    try:
        models = model_fetch.fetch_models("http://127.0.0.1:11434")
    finally:
        os.environ.pop("DUCKY_OLLAMA_SKIP_ENRICH", None)
        sys.modules.pop("httpx", None)

    assert [m.id for m in models] == ["qwen2.5:7b", "llama3.2:latest"]
    assert all(m.is_local for m in models)
    assert all(not m.thinking_menu for m in models)
    httpx.get.assert_called_once()
    assert httpx.get.call_args[0][0].endswith("/api/tags")
    httpx.post.assert_not_called()


def test_show_payload_vision_from_api_only() -> None:
    model_fetch = _load_plugin_fetch()
    qwen = model_fetch.parse_show_payload(
        {
            "capabilities": ["completion", "vision", "tools", "thinking"],
            "model_info": {"qwen35.context_length": 262144},
            "parameters": "temperature 0.6\ntop_p 0.95\nmin_p 0",
            "projector_info": {"clip.has_vision_encoder": True},
        }
    )
    assert "vision" in qwen["capabilities"]
    assert qwen["context_length"] == 262144
    assert qwen["parameters"]["temperature"] == 0.6
    row = model_fetch.model_from_show("qwen3.8:latest", qwen)
    assert row.supports_vision and row.supports_tools and row.thinking_menu
    projector_only = model_fetch.parse_show_payload(
        {"capabilities": ["completion"], "projector_info": {"clip.vision.block_count": 27}}
    )
    assert "vision" in projector_only["capabilities"]
    text_only = model_fetch.parse_show_payload({"capabilities": ["completion"]})
    assert "vision" not in text_only["capabilities"]
    assert not model_fetch.model_from_show("plain", text_only).supports_vision


def test_enrich_thinking_only_when_capability() -> None:
    model_fetch = _load_plugin_fetch()
    eight = 8 * 1024 * 1024 * 1024
    thinking = model_fetch.model_from_show(
        "qwen3.6:latest",
        {"context_length": 262144, "capabilities": ["completion", "thinking", "tools"]},
        vram_bytes=eight,
        apply_saved=False,
    )
    assert thinking.thinking_menu and thinking.supports_thinking_effort
    assert thinking.supports_tools and thinking.context_limit == 32768
    plain = model_fetch.model_from_show(
        "llama3.2:latest",
        {"context_length": 8192, "capabilities": ["completion"]},
        vram_bytes=eight,
        apply_saved=False,
    )
    assert plain.thinking_menu is None
    assert not plain.supports_thinking_effort
    assert plain.context_limit == 8192


def test_effective_context_limit_vram_caps_qwen() -> None:
    model_fetch = _load_plugin_fetch()
    eight = 8 * 1024 * 1024 * 1024
    fat = 24 * 1024 * 1024 * 1024
    assert (
        model_fetch.effective_context_limit(
            "qwen3.8:latest", 262144, vram_bytes=eight, apply_saved=False
        )
        == 32768
    )
    assert (
        model_fetch.effective_context_limit(
            "qwen3.8:latest", 262144, vram_bytes=fat, apply_saved=False
        )
        == 262144
    )
    assert (
        model_fetch.effective_context_limit(
            "llama3.2:latest", 8192, vram_bytes=eight, apply_saved=False
        )
        == 8192
    )
    assert model_fetch.effective_context_limit("q", None) is None
    # Host Memory high-water is 65% of this advertised window (not the 80k slider).
    assert int(32768 * 0.65) == 21299


def test_media_support_from_capabilities_words() -> None:
    model_fetch = _load_plugin_fetch()
    full = model_fetch.model_from_show("m:latest", {"capabilities": ["completion", "vision", "video", "audio"]})
    assert (full.supports_video, full.supports_audio, full.max_images) == (True, True, None)
    plain = model_fetch.model_from_show("m:latest", {"capabilities": ["completion", "vision"]})
    assert (plain.supports_video, plain.supports_audio) == (None, None)
    unknown = model_fetch.model_from_show("m:latest", {})
    assert (unknown.supports_video, unknown.supports_audio, unknown.max_images) == (None, None, None)


if __name__ == "__main__":
    test_fetch_models_uses_tags_only()
    test_show_payload_vision_from_api_only()
    test_enrich_thinking_only_when_capability()
    test_effective_context_limit_vram_caps_qwen()
    test_media_support_from_capabilities_words()
    print("ok")
