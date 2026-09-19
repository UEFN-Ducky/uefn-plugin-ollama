"""Ollama provider — local Llama and other models via OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator

from backend.agent.multimodal_content import build_openai_user_content
try:
    from .ollama_url import ollama_openai_base_url
except ImportError:
    from ollama_url import ollama_openai_base_url
from backend.agent.providers.base import (
    ProviderMessage,
    StreamEvent,
    StreamEventKind,
    ToolCallRequest,
)
from backend.agent.providers.cache_utils import openai_system_messages, parse_openai_usage
from backend.agent.providers.thinking import ThinkSplitter, reasoning_from_delta
from backend.agent.providers.wait_status import clamp_percent, format_wait_status

# OpenAI SDK defaults read=600s — first chunk only arrives AFTER prompt eval, so a
# fat agent prompt used to die as "Connection error" mid-eval. Keep read open-ended
# but fail connect/write quickly so real disconnects surface immediately.
def _http_timeout():
    import httpx

    return httpx.Timeout(connect=30.0, read=None, write=60.0, pool=30.0)


_PROGRESS_RE = re.compile(r"progress\s*=\s*(0\.\d+|1\.0+)")
# Progress % used to poll every 1s (and briefly spawned processes). That raced
# the inference workload for no real gain — Ollama already owns the GPU/CPU.
# Keep a one-shot Waiting status; optional slow log peek is off unless enabled.
_PROGRESS_POLL_SEC = 2.5
_ENABLE_PROGRESS_POLL = os.environ.get("DUCKY_OLLAMA_PROGRESS_POLL", "").strip() in (
    "1",
    "true",
    "yes",
)

# num_ctx = host epoch high-water + output headroom, capped at /api/show
# context_length. Pinned once per (url, model) — never resize mid-chat
# (a resize drops the KV cache). No slider, no 16/32/64/128/256k ladder.
_HEADROOM_TOKENS = 4_096
_IMAGE_TOKENS_EACH = 1_600
# Stay loaded while the Ollama server is up (15m idle used to cold-reload).
_KEEP_ALIVE = -1

_ratchet_lock = threading.Lock()
_num_ctx_ratchet: dict[str, int] = {}


def _think_extra(thinking_effort: str) -> dict[str, Any]:
    """Map Ducky thinking_effort → Ollama ``/v1`` ``reasoning_effort``.

    Native ``/api/chat`` uses ``think`` (bool or low|medium|high|max).
    ``/v1/chat/completions`` wants ``reasoning_effort``
    (none|low|medium|high|max). A bool ``think`` on /v1 is a 400.
    Default ``none`` so qwen3.6 tool-calls like OpenAI instead of thinking
    first and stopping.
    """
    from backend.agent.thinking_effort import normalize_thinking_effort

    effort = normalize_thinking_effort(thinking_effort)
    if effort in ("", "off"):
        return {"reasoning_effort": "none"}
    if effort == "low":
        return {"reasoning_effort": "low"}
    if effort == "high":
        return {"reasoning_effort": "high"}
    return {"reasoning_effort": "medium"}


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
    if "connection" in low or "connect" in low or "remoteprotocol" in low:
        return f"Connection error talking to Ollama — request ended. Detail: {err}"
    if "timed out" in low or "timeout" in low:
        return (
            "Ollama HTTP client timed out waiting for tokens (often during long prompt eval). "
            f"Detail: {err}"
        )
    return err


def _server_log_path() -> Path | None:
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if not local:
        return None
    path = Path(local) / "Ollama" / "server.log"
    return path if path.is_file() else None


_log_tail_lock = threading.Lock()
_log_tail_pos = 0
_log_tail_path: str | None = None


def _progress_from_server_log() -> float | None:
    """Exact prompt-eval fraction from Ollama's llama-server log (verbosity ≥4).

    Incremental read of new bytes only (no full-file scan, no process spawn).
    """
    global _log_tail_pos, _log_tail_path
    path = _server_log_path()
    if path is None:
        return None
    path_s = str(path)
    try:
        size = path.stat().st_size
    except Exception:
        return None
    with _log_tail_lock:
        if _log_tail_path != path_s or _log_tail_pos > size:
            _log_tail_path = path_s
            # First peek / truncate: small window from end.
            _log_tail_pos = max(0, size - 8192)
        if size <= _log_tail_pos:
            return None
        try:
            with path.open("rb") as fh:
                fh.seek(_log_tail_pos)
                chunk = fh.read(size - _log_tail_pos)
                _log_tail_pos = size
        except Exception:
            return None
    # Keep a little overlap so a split "progress=0.x" line isn't missed.
    if len(chunk) > 64:
        chunk = chunk[-min(len(chunk), 16384) :]
    text = chunk.decode("utf-8", "replace")
    matches = _PROGRESS_RE.findall(text)
    if not matches:
        return None
    try:
        return max(0.0, min(1.0, float(matches[-1])))
    except ValueError:
        return None


def _chars_to_tokens(chars: int, *, dense: bool = False) -> int:
    """Rough char→token. JSON/tools/code denser (~3) than prose (~4)."""
    if chars <= 0:
        return 0
    return max(1, chars // (3 if dense else 4))


def _attachment_tokens(attachments: Any) -> int:
    n = 0
    for att in attachments or []:
        kind = str(getattr(att, "kind", "") or "").strip().lower()
        if kind == "image":
            n += _IMAGE_TOKENS_EACH
            continue
        text = str(getattr(att, "text", "") or "")
        if text:
            n += _chars_to_tokens(len(text), dense=True)
    return n


def _estimate_prompt_tokens(
    system: str,
    messages: list[ProviderMessage],
    tools: list[dict[str, Any]],
) -> int:
    """Rough token estimate for progress + num_ctx (includes images)."""
    tokens = _chars_to_tokens(len(system or ""))
    for m in messages:
        tokens += _chars_to_tokens(len(m.content or ""))
        tokens += _attachment_tokens(getattr(m, "attachments", None))
        if m.tool_calls:
            try:
                blob = json.dumps([tc.__dict__ for tc in m.tool_calls], default=str)
            except Exception:
                blob = ""
            tokens += _chars_to_tokens(len(blob), dense=True)
    if tools:
        try:
            blob = json.dumps(tools, default=str)
        except Exception:
            blob = "x" * (2048 * len(tools))
        # Tool JSON tokenizes tighter than prose; //6 matches observed Ollama counts.
        tokens += max(1, len(blob) // 6)
    return max(1, tokens)


def _pinned_num_ctx(base_url: str, model: str, model_max: int) -> int:
    """Allocate num_ctx once from slider or host high-water; never resize mid-chat."""
    key = f"{(base_url or '').rstrip('/')}|{(model or '').strip()}"
    max_ctx = max(1, int(model_max))
    with _ratchet_lock:
        prev = int(_num_ctx_ratchet.get(key) or 0)
        if prev > 0:
            return min(prev, max_ctx)
        saved = None
        try:
            from .settings_store import clamp_num_ctx, saved_num_ctx

            raw = saved_num_ctx(model)
            if raw:
                saved = clamp_num_ctx(raw, max_ctx)
        except Exception:
            saved = None
        if saved:
            chosen = saved
        else:
            try:
                from backend.agent.context_memory import epoch_num_ctx

                chosen = epoch_num_ctx(max_ctx)
            except Exception:
                chosen = min(max_ctx, 80_000 + _HEADROOM_TOKENS)
        chosen = min(max(1, int(chosen)), max_ctx)
        _num_ctx_ratchet[key] = chosen
        return chosen


def clear_num_ctx_ratchet() -> None:
    """Test helper — reset process-local ratchet."""
    with _ratchet_lock:
        _num_ctx_ratchet.clear()


def _progress_snapshot(frac: float | None) -> tuple[str, float | None]:
    """Return (status_text, percent) from server.log only — never spawn processes."""
    pct = clamp_percent(frac)
    return format_wait_status(label="Waiting", percent=pct), pct


class OllamaProvider:
    def __init__(self, base_url: str, model: str, *, thinking_effort: str = "off") -> None:
        self._base_url = base_url
        self._model = model
        self._thinking_effort = thinking_effort or "off"

    def _client(self):
        from openai import OpenAI

        return OpenAI(
            api_key="ollama",
            base_url=ollama_openai_base_url(self._base_url),
            timeout=_http_timeout(),
        )

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

        model_max = _num_ctx(self._base_url, self._model)
        if model_max is None:
            yield StreamEvent(
                kind=StreamEventKind.ERROR,
                error=(
                    f"Ollama model {self._model!r} has no context_length from /api/show "
                    "(or model catalog). Refresh models in Settings → LLMs; refuse to guess a window."
                ),
            )
            return
        prompt_token_est = _estimate_prompt_tokens(system, messages, tools or [])
        num_ctx = _pinned_num_ctx(self._base_url, self._model, model_max)
        options: dict[str, Any] = {"num_ctx": num_ctx}
        keep_alive = _KEEP_ALIVE
        opt_snap: dict[str, Any] = {}
        try:
            from .settings_store import saved_keep_alive, saved_options

            saved_ka = saved_keep_alive(self._model)
            if saved_ka is not None:
                keep_alive = saved_ka
            opt_snap = saved_options(self._model)
            for key, val in opt_snap.items():
                if key == "num_thread" and int(val or 0) == 0:
                    continue
                if key == "num_predict" and int(val or 0) < 0:
                    continue
                if key == "seed" and int(val or 0) < 0:
                    continue
                if key == "num_batch" and int(val or 0) == 0:
                    continue
                if key == "min_p" and float(val or 0) <= 0:
                    continue
                options[key] = val
        except Exception:
            opt_snap = {}
        extra_body: dict[str, Any] = {
            # -1 = keep loaded for the life of the Ollama server (avoid 15m idle reload).
            "keep_alive": keep_alive,
            "options": options,
            **_think_extra(self._thinking_effort),
        }
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(system, messages, cache=cache),
            "tools": tools if tools else None,
            "stream": True,
            "stream_options": {"include_usage": True},
            # ``num_ctx`` pinned at chat start from host high-water + headroom.
            # ``reasoning_effort`` is the /v1 thinking control.
            "extra_body": extra_body,
        }
        cache_key = (getattr(cache, "prompt_cache_key", "") if cache else "") or ""
        if cache_key:
            create_kwargs["prompt_cache_key"] = cache_key

        yield StreamEvent(
            kind=StreamEventKind.STATUS,
            text=format_wait_status(
                label="Waiting",
                detail=f"~{prompt_token_est:,} tokens · ctx {num_ctx:,}",
            ),
            percent=None,
        )

        # Stream reader blocks until the first token (whole prompt eval).
        # Default: no progress poll thread — one Waiting status above is enough.
        # Set DUCKY_OLLAMA_PROGRESS_POLL=1 for slow incremental log % updates.
        event_q: queue.Queue[tuple[str, Any]] = queue.Queue()
        stop_progress = threading.Event()
        # First-token `stop_progress` only kills the optional % poller.
        # The reader used to watch that same event — qwen thinking's first
        # token ("The") aborted the HTTP stream and Ollama logged `cancel task`.
        stop_reader = threading.Event()

        def _reader() -> None:
            try:
                stream = client.chat.completions.create(**create_kwargs)
                for chunk in stream:
                    if stop_reader.is_set():
                        break
                    event_q.put(("chunk", chunk))
                event_q.put(("end", None))
            except Exception as exc:
                event_q.put(("err", exc))

        def _progress() -> None:
            last_key = ""
            misses = 0
            while not stop_progress.wait(_PROGRESS_POLL_SEC):
                if not reader_thread.is_alive():
                    return
                frac = _progress_from_server_log()
                if frac is None:
                    misses += 1
                    # Verbosity too low / no line yet — stop burning cycles.
                    if misses >= 3:
                        return
                    continue
                misses = 0
                text, pct = _progress_snapshot(frac)
                key = f"{text}|{pct}"
                if key != last_key:
                    last_key = key
                    event_q.put(("progress", (text, pct)))

        t0 = time.time()
        try:
            from .stats import set_generating

            set_generating(self._model)
        except Exception:
            pass
        reader_thread = threading.Thread(target=_reader, name="ollama-stream", daemon=True)
        reader_thread.start()
        progress_thread: threading.Thread | None = None
        if _ENABLE_PROGRESS_POLL:
            progress_thread = threading.Thread(
                target=_progress, name="ollama-progress", daemon=True
            )
            progress_thread.start()

        saw_content = False
        try:
            while True:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    cancelled = True
                    stop_reader.set()
                    stop_progress.set()
                    try:
                        client.close()
                    except Exception:
                        pass
                    break
                try:
                    kind, payload = event_q.get(timeout=0.25)
                except queue.Empty:
                    # Reader died without posting end/err — treat as connection loss.
                    if not reader_thread.is_alive() and not saw_content:
                        stop_progress.set()
                        yield StreamEvent(
                            kind=StreamEventKind.ERROR,
                            error="Connection error talking to Ollama — stream ended unexpectedly.",
                        )
                        return
                    continue
                if kind == "progress":
                    if not saw_content:
                        text, pct = payload
                        yield StreamEvent(
                            kind=StreamEventKind.STATUS,
                            text=str(text),
                            percent=pct,
                        )
                    continue
                if kind == "err":
                    stop_progress.set()
                    try:
                        client.close()
                    except Exception:
                        pass
                    yield StreamEvent(
                        kind=StreamEventKind.ERROR,
                        error=_friendly_ollama_error(payload),
                    )
                    return
                if kind == "end":
                    break
                # chunk
                chunk = payload
                if getattr(chunk, "usage", None):
                    usage = parse_openai_usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                reasoning = reasoning_from_delta(delta)
                if reasoning:
                    if not saw_content:
                        saw_content = True
                        stop_progress.set()
                    yield StreamEvent(kind=StreamEventKind.THINKING, text=reasoning)
                if delta.content:
                    if not saw_content:
                        saw_content = True
                        stop_progress.set()
                    for kind_seg, seg in splitter.feed(delta.content):
                        if kind_seg == "think":
                            yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
                        else:
                            collected_text += seg
                            yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)
                if delta.tool_calls:
                    if not saw_content:
                        saw_content = True
                        stop_progress.set()
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
        finally:
            stop_progress.set()
            try:
                from .stats import set_generating

                set_generating("")
            except Exception:
                pass

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
        elapsed = max(0.001, time.time() - t0)
        prompt_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        try:
            from .history import record_call

            record_call(
                {
                    "model": self._model,
                    "num_ctx": num_ctx,
                    "keep_alive": keep_alive,
                    "options": opt_snap,
                    "thinking": self._thinking_effort,
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": out_tok,
                    "elapsed_s": round(elapsed, 3),
                    "tokens_per_s": round(out_tok / elapsed, 2),
                    "ok": not cancelled,
                }
            )
        except Exception:
            pass
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
                extra_body={"reasoning_effort": "none"},
            )
            _ = r.choices[0].message.content
            return True, "Ollama OK"
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                return False, "Cannot reach Ollama — is it running? (ollama serve)"
            return False, err
