"""Ollama gateway — local URL provider via host registries."""

from __future__ import annotations

from typing import Any


def _normalize_secret(raw: str) -> str:
    from .ollama_url import normalize_ollama_base

    return normalize_ollama_base((raw or "").strip() or "http://localhost:11434")


def _fetch_models(api_key: str, **_kw: Any) -> Any:
    from .model_fetch import fetch_models

    return fetch_models(_normalize_secret(api_key))


def _fetch_usage(api_key: str, **_kw: Any) -> Any:
    from .local import usage_windows

    return usage_windows(_normalize_secret(api_key))


def _saved_base() -> str:
    try:
        from backend.agent.secrets import get_key

        raw = get_key("ollama") or ""
    except Exception:
        raw = ""
    return _normalize_secret(raw)


def _rpc_search(*, query: str = "", capabilities: Any = None, order: str = "popular", **_k: Any) -> Any:
    from .library import search_library

    caps = capabilities if isinstance(capabilities, list) else []
    return search_library(str(query or ""), caps, str(order or "popular"))


def _rpc_tags(*, slug: str = "", **_k: Any) -> Any:
    from .library import list_library_tags

    return list_library_tags(str(slug or ""))


def _rpc_local_list(**_k: Any) -> Any:
    from .local import list_local

    return list_local(_saved_base())


def _rpc_delete(*, name: str = "", **_k: Any) -> Any:
    from .local import delete_model

    return delete_model(_saved_base(), str(name or ""))


def _rpc_pull_start(*, name: str = "", **_k: Any) -> Any:
    from .pull import pull_start

    return pull_start(_saved_base(), str(name or ""))


def _rpc_pull_status(*, job_id: str = "", **_k: Any) -> Any:
    from .pull import pull_status

    return pull_status(str(job_id or ""))


def _rpc_pull_cancel(*, job_id: str = "", **_k: Any) -> Any:
    from .pull import pull_cancel

    return pull_cancel(str(job_id or ""))


def _rpc_settings_get(*, model: str = "", max_ctx: int = 0, **_k: Any) -> Any:
    from .settings_store import get_model_settings

    return get_model_settings(str(model or ""), max_ctx=int(max_ctx or 0))


def _rpc_settings_set(
    *,
    model: str = "",
    num_ctx: Any = None,
    keep_alive: Any = None,
    max_ctx: int = 0,
    options: Any = None,
    **opt_kw: Any,
) -> Any:
    from .settings_store import set_model_settings

    extra = {k: v for k, v in opt_kw.items() if k not in {"model", "num_ctx", "keep_alive", "max_ctx", "options"}}
    return set_model_settings(
        str(model or ""),
        num_ctx=None if num_ctx is None else int(num_ctx),
        keep_alive=None if keep_alive is None else int(keep_alive),
        max_ctx=int(max_ctx or 0),
        options=options if isinstance(options, dict) else None,
        **extra,
    )


def _rpc_live(**_k: Any) -> Any:
    from .stats import live_stats

    return live_stats(_saved_base())


def _rpc_history(**_k: Any) -> Any:
    from .history import history_payload

    return history_payload()


def _rpc_apply_history(*, index: int = 0, model: str = "", **_k: Any) -> Any:
    from .settings_store import apply_history

    return apply_history(int(index), str(model or ""))


def register(api) -> None:
    from .ollama_provider import OllamaProvider

    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "ollama",
        factory=lambda api_key, model, **kw: OllamaProvider(
            _normalize_secret(api_key), model, **kw
        ),
        fetch_models=_fetch_models,
        fetch_usage=_fetch_usage,
        test_key_model="",
        tool_schema="openai",
        key_optional=True,
        normalize_secret=_normalize_secret,
        clear_model_cache=clear_model_cache,
        cache_mode="local",
        shows_thinking_effort=True,
    )
    api.register_panel_rpc("library.search", _rpc_search)
    api.register_panel_rpc("library.tags", _rpc_tags)
    api.register_panel_rpc("local.list", _rpc_local_list)
    api.register_panel_rpc("local.delete", _rpc_delete)
    api.register_panel_rpc("pull.start", _rpc_pull_start)
    api.register_panel_rpc("pull.status", _rpc_pull_status)
    api.register_panel_rpc("pull.cancel", _rpc_pull_cancel)
    api.register_panel_rpc("settings.get", _rpc_settings_get)
    api.register_panel_rpc("settings.set", _rpc_settings_set)
    api.register_panel_rpc("stats.live", _rpc_live)
    api.register_panel_rpc("history.list", _rpc_history)
    api.register_panel_rpc("history.apply", _rpc_apply_history)

    @api.tool(intent=r"\b(ollama|local model|pull model)\b")
    def ollama_search_library(query: str = "", capabilities: str = "vision,tools,thinking", order: str = "popular") -> str:
        """Search the real Ollama library (same filters as ollama.com/search)."""
        caps = [c.strip() for c in str(capabilities or "").split(",") if c.strip()]
        return str(_rpc_search(query=query, capabilities=caps, order=order))

    @api.tool(intent=r"\b(ollama|installed models|local models)\b")
    def ollama_list_local() -> str:
        """List pulled Ollama models, disk storage, and which are loaded."""
        return str(_rpc_local_list())

    @api.tool(intent=r"\b(ollama pull|download model)\b")
    def ollama_pull_model(name: str) -> str:
        """Download an Ollama model (API pull, or listed terminal fallback)."""
        return str(_rpc_pull_start(name=name))

    @api.tool(intent=r"\b(ollama delete|remove model)\b")
    def ollama_delete_model(name: str) -> str:
        """Delete a pulled Ollama model from disk."""
        return str(_rpc_delete(name=name))

    @api.tool(intent=r"\b(ollama|vram|gpu|running model)\b")
    def ollama_live_stats() -> str:
        """Live CPU / RAM / GPU and loaded Ollama models (use while a model is thinking)."""
        return str(_rpc_live())

    @api.tool(intent=r"\b(ollama history|optimize sliders)\b")
    def ollama_call_history() -> str:
        """Last 5 Ollama calls with tokens/s and the slider snapshot used."""
        return str(_rpc_history())

    @api.tool(intent=r"\b(ollama apply|restore sliders)\b")
    def ollama_apply_history(index: int = 0, model: str = "") -> str:
        """Copy sliders from a history call (0 = oldest of the last 5) onto a model."""
        return str(_rpc_apply_history(index=index, model=model))

    @api.tool(intent=r"\b(ollama options|temperature|num_ctx)\b")
    def ollama_get_options(model: str) -> str:
        """Read per-model Ollama optimize sliders (ctx, temperature, GPU layers, …)."""
        return str(_rpc_settings_get(model=model))

    @api.tool(intent=r"\b(ollama tags|library tags)\b")
    def ollama_library_tags(slug: str) -> str:
        """List downloadable tags/sizes for an Ollama library slug."""
        return str(_rpc_tags(slug=slug))

    @api.tool(intent=r"\b(ollama pull status|download progress)\b")
    def ollama_pull_status(job_id: str) -> str:
        """Poll an in-progress Ollama pull (API or terminal)."""
        return str(_rpc_pull_status(job_id=job_id))

    @api.tool(intent=r"\b(ollama cancel pull|stop download)\b")
    def ollama_cancel_pull(job_id: str) -> str:
        """Cancel an in-progress Ollama pull."""
        return str(_rpc_pull_cancel(job_id=job_id))

    @api.tool(intent=r"\b(ollama set|tune model|temperature)\b")
    def ollama_set_options(
        model: str,
        num_ctx: int | None = None,
        keep_alive: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        num_gpu: int | None = None,
        num_thread: int | None = None,
        num_predict: int | None = None,
        min_p: float | None = None,
        repeat_last_n: int | None = None,
        seed: int | None = None,
        num_batch: int | None = None,
        f16_kv: bool | None = None,
        use_mmap: bool | None = None,
        use_mlock: bool | None = None,
    ) -> str:
        """Set per-model Ollama optimize sliders. Omit a field to leave it unchanged."""
        opts = {
            k: v
            for k, v in {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "num_gpu": num_gpu,
                "num_thread": num_thread,
                "num_predict": num_predict,
                "min_p": min_p,
                "repeat_last_n": repeat_last_n,
                "seed": seed,
                "num_batch": num_batch,
                "f16_kv": f16_kv,
                "use_mmap": use_mmap,
                "use_mlock": use_mlock,
            }.items()
            if v is not None
        }
        return str(
            _rpc_settings_set(
                model=model,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                options=opts or None,
            )
        )

    from . import graph_nodes

    graph_nodes.register_nodes(api)
    api.log("Ollama gateway contribution active (Providers)")
