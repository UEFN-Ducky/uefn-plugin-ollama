"""Ollama provider — local Llama and other models via OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from backend.agent.multimodal_content import build_openai_user_content
from .ollama_url import ollama_openai_base_url
from backend.agent.providers.base import (
    ProviderMessage,
    StreamEvent,
    StreamEventKind,
    ToolCallRequest,
)
from backend.agent.providers.cache_utils import openai_system_messages, parse_openai_usage
from backend.agent.providers.thinking import ThinkSplitter, reasoning_from_delta


def _think_extra(thinking_effort: str) -> dict[str, Any]:
    """Map Ducky thinking_effort → Ollama ``think`` (qwen3 / thinking models)."""
    from backend.agent.thinking_effort import normalize_thinking_effort

    effort = normalize_thinking_effort(thinking_effort)
    if effort in ("", "off"):
        return {"think": False}
    if effort == "low":
        return {"think": "low"}
    if effort == "high":
        return {"think": "high"}
    # medium / on
    return {"think": True}


def _num_ctx(base_url: str, model: str) -> int | None:
    """Model context from Ollama ``/api/show`` (or catalog cache). Never invent a size."""
    reported = 0
    try:
        from .model_fetch import ollama_model_info

        reported = int(ollama_model_info(base_url, model).get("context_length") or 0)
    except Exception:
        reported = 0
    if reported <= 0:
        try:
            from backend.agent.model_fetch import get_model_info

            info = get_model_info("ollama", model)
            if info and info.context_limit:
                reported = int(info.context_limit)
        except Exception:
            reported = 0
    return reported if reported > 0 else None


def _friendly_ollama_error(exc: BaseException) -> str:
    err = str(exc)
    low = err.lower()
    if "exceed_context_size" in low or "exceeds the available context size" in low:
        return (
            "Prompt too large for Ollama's context window (system + tools + image). "
            "Try a shorter message, fewer skills, or a model with a larger context. "
            f"Detail: {err}"
        )
    return err


class OllamaProvider:
    def __init__(self, base_url: str, model: str, *, thinking_effort: str = "off") -> None:
        self._base_url = base_url
        self._model = model
        self._thinking_effort = thinking_effort or "off"

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key="ollama", base_url=ollama_openai_base_url(self._base_url))

    def _to_openai_messages(
        self,
        system: str,
        messages: list[ProviderMessage],
        *,
        cache: Any | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = list(openai_system_messages(cache, fallback_system=system))
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
                continue
            if m.role == "user" and m.attachments:
                out.append({"role": "user", "content": build_openai_user_content(m.content, m.attachments)})
                continue
            out.append({"role": m.role, "content": m.content})
        return out

    async def stream_turn(
        self,
        *,
        system: str,
        messages: list[ProviderMessage],
        tools: list[dict[str, Any]],
        cancel_event: Any | None = None,
        cache: Any | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self._client()
        collected_text = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        cancelled = False
        splitter = ThinkSplitter()

        options: dict[str, Any] = {"keep_alive": "15m"}
        num_ctx = _num_ctx(self._base_url, self._model)
        if num_ctx is not None:
            # Exact window from the model API — do not invent floors/ceilings.
            options["num_ctx"] = num_ctx
        extra_body: dict[str, Any] = {
            "keep_alive": "15m",
            "options": options,
            **_think_extra(self._thinking_effort),
        }
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(system, messages, cache=cache),
            "tools": tools if tools else None,
            "stream": True,
            # Keep the model resident (and its KV/prefix cache warm) between turns
            # instead of unloading after Ollama's 5-minute default. Sent both at
            # top level and nested under `options` since different Ollama
            # versions honor one or the other on the OpenAI-compat endpoint.
            # ``think`` disables/enables reasoning for qwen3-style thinking models
            # so the visible reply is not empty when effort is off.
            # ``num_ctx`` = model ``context_length`` from /api/show when known.
            "extra_body": extra_body,
        }

        try:
            stream = client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            yield StreamEvent(
                kind=StreamEventKind.ERROR,
                error=_friendly_ollama_error(exc),
            )
            return
        for chunk in stream:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                cancelled = True
                break
            if getattr(chunk, "usage", None):
                usage = parse_openai_usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = reasoning_from_delta(delta)
            if reasoning:
                yield StreamEvent(kind=StreamEventKind.THINKING, text=reasoning)
            if delta.content:
                for kind, seg in splitter.feed(delta.content):
                    if kind == "think":
                        yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
                    else:
                        collected_text += seg
                        yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    acc = tool_calls_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        if cancelled:
            return

        for kind, seg in splitter.flush():
            if kind == "think":
                yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
            else:
                collected_text += seg
                yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)

        rebuilt: list[ToolCallRequest] = []
        for acc in tool_calls_acc.values():
            try:
                args = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            rebuilt.append(
                ToolCallRequest(id=acc["id"] or "call", name=acc["name"], arguments=args)
            )
        if rebuilt:
            yield StreamEvent(kind=StreamEventKind.TOOL_CALLS, tool_calls=rebuilt, usage=usage)
        yield StreamEvent(
            kind=StreamEventKind.DONE,
            text=collected_text,
            stop_reason="tool_calls" if rebuilt else "stop",
            usage=usage,
        )

    async def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._client()
            r = client.chat.completions.create(
                model=self._model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
                extra_body={"think": False},
            )
            _ = r.choices[0].message.content
            return True, "Ollama OK"
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                return False, "Cannot reach Ollama — is it running? (ollama serve)"
            return False, err
