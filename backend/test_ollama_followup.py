from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for k in list(sys.modules):
    if k == "backend" or k.startswith("backend."):
        del sys.modules[k]
for p in _here.parents:
    cand = p / "UEFN-Ducky-Release" / "ducky_app"
    if (cand / "backend" / "agent").is_dir():
        sys.path.insert(0, str(_here))
        sys.path.insert(0, str(cand))
        break

from backend.agent.providers.base import ProviderMessage, ToolCallRequest
from ollama_provider import OllamaProvider


def test_followup_comes_after_previous_reply():
    msgs = [
        ProviderMessage(role="user", content="first"),
        ProviderMessage(
            role="assistant",
            content="Checking",
            tool_calls=[ToolCallRequest(id="t1", name="ping", arguments={})],
        ),
        ProviderMessage(role="tool", tool_call_id="t1", content='{"ok":true}'),
        ProviderMessage(role="assistant", content="Here is the answer."),
        ProviderMessage(role="user", content="follow up"),
    ]
    out = OllamaProvider("http://127.0.0.1:11434", "llama3")._to_openai_messages("sys", msgs)
    roles = [m["role"] for m in out if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assert out[-2]["content"] == "Here is the answer."
    assert out[-1]["content"] == "follow up"


def test_reasoning_replays_in_ollamas_reasoning_field():
    """Ollama's /v1 maps ``reasoning`` on an assistant message to the model's thinking;
    without it a DeepSeek-style model re-derives the same analysis on every tool step."""
    msgs = [
        ProviderMessage(role="user", content="wire the device"),
        ProviderMessage(
            role="assistant",
            content="",
            thinking="The spawner is not wired yet, so read status first.",
            tool_calls=[ToolCallRequest(id="t1", name="ducky_get_status", arguments={})],
        ),
        ProviderMessage(role="tool", tool_call_id="t1", content='{"ok":true}'),
        ProviderMessage(role="assistant", content="Done.", thinking="Status was fine."),
        ProviderMessage(role="assistant", content="No thoughts here."),
    ]
    out = OllamaProvider("http://127.0.0.1:11434", "deepseek-r1")._to_openai_messages("sys", msgs)
    step, final, plain = out[2], out[4], out[5]
    assert step["tool_calls"] and step["reasoning"] == "The spawner is not wired yet, so read status first."
    assert final["reasoning"] == "Status was fine."
    assert "reasoning" not in plain
    assert "reasoning" not in out[1] and "reasoning" not in out[3]  # user, tool
    assert all("reasoning_content" not in m for m in out)
