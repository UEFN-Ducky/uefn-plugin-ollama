"""Per-model Ollama prefs (num_ctx / keep_alive) in plugin prefs."""

from __future__ import annotations

from typing import Any

def _tick_label(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def ticks_for_max(max_ctx: int) -> list[int]:
    """Powers of two up to /api/show context_length — never invent a bigger window."""
    cap = max(1, int(max_ctx or 0))
    ticks: list[int] = []
    t = 4096
    while t < cap and len(ticks) < 12:
        ticks.append(t)
        t *= 2
    if not ticks or ticks[-1] != cap:
        ticks.append(cap)
    return ticks


def clamp_num_ctx(value: int, max_ctx: int) -> int:
    """Snap to a tick that exists for this model. Never above /api/show max."""
    cap = max(1, int(max_ctx or 0))
    try:
        raw = int(value)
    except (TypeError, ValueError):
        raw = cap
    ticks = ticks_for_max(cap)
    allowed = [t for t in ticks if t <= cap]
    if not allowed:
        return cap
    # Prefer the greatest tick that is not above the requested/cap value.
    want = min(max(1, raw), cap)
    chosen = allowed[0]
    for t in allowed:
        if t <= want:
            chosen = t
    return chosen


def _prefs() -> dict[str, Any]:
    try:
        from frontend.ui_web.plugin_host_api import prefs_plugin_get

        return dict(prefs_plugin_get("ollama") or {})
    except Exception:
        return {}


def _save(prefs: dict[str, Any]) -> None:
    from frontend.ui_web.plugin_host_api import prefs_plugin_set

    prefs_plugin_set("ollama", prefs)


def saved_num_ctx(model: str) -> int | None:
    name = (model or "").strip()
    if not name:
        return None
    raw = _prefs().get(f"num_ctx:{name}")
    try:
        n = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


OPT_SLIDERS = (
    {"id": "temperature", "label": "Temperature", "lo": 0.0, "hi": 2.0, "step": 0.05, "default": 0.8},
    {"id": "top_p", "label": "Top P", "lo": 0.0, "hi": 1.0, "step": 0.05, "default": 0.9},
    {"id": "top_k", "label": "Top K", "lo": 0, "hi": 100, "step": 1, "default": 40},
    {"id": "repeat_penalty", "label": "Repeat penalty", "lo": 0.8, "hi": 1.5, "step": 0.05, "default": 1.1},
    {"id": "num_gpu", "label": "GPU layers (-1 all)", "lo": -1, "hi": 128, "step": 1, "default": -1},
    {"id": "num_thread", "label": "CPU threads (0 auto)", "lo": 0, "hi": 64, "step": 1, "default": 0},
    {"id": "num_predict", "label": "Max tokens (-1 inf)", "lo": -1, "hi": 8192, "step": 64, "default": -1},
    {"id": "min_p", "label": "Min P", "lo": 0.0, "hi": 1.0, "step": 0.05, "default": 0.0},
    {"id": "repeat_last_n", "label": "Repeat last N", "lo": 0, "hi": 256, "step": 8, "default": 64},
    {"id": "seed", "label": "Seed (-1 random)", "lo": -1, "hi": 99999, "step": 1, "default": -1},
    {"id": "num_batch", "label": "Batch size (0 auto)", "lo": 0, "hi": 2048, "step": 16, "default": 0},
)
OPT_SWITCHES = (
    {"id": "f16_kv", "label": "f16 KV cache", "default": False},
    {"id": "use_mmap", "label": "Memory map weights", "default": True},
    {"id": "use_mlock", "label": "Lock model in RAM", "default": False},
)
_OPT_IDS = {row["id"] for row in OPT_SLIDERS} | {row["id"] for row in OPT_SWITCHES}


def _opt_meta(opt_id: str) -> dict[str, Any] | None:
    for row in OPT_SLIDERS:
        if row["id"] == opt_id:
            return row
    for row in OPT_SWITCHES:
        if row["id"] == opt_id:
            return row
    return None


def _clamp_opt(opt_id: str, value: Any) -> Any:
    meta = _opt_meta(opt_id)
    if meta is None:
        return value
    if opt_id in {r["id"] for r in OPT_SWITCHES}:
        return bool(value)
    lo = meta.get("lo")
    hi = meta.get("hi")
    step = meta.get("step") or 1
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return meta.get("default")
    if lo is not None:
        raw = max(float(lo), raw)
    if hi is not None:
        raw = min(float(hi), raw)
    if isinstance(step, float) or (isinstance(lo, float) or isinstance(hi, float)):
        return round(raw, 3)
    return int(round(raw))


def saved_opt(model: str, opt_id: str) -> Any:
    name = (model or "").strip()
    if not name or opt_id not in _OPT_IDS:
        return None
    raw = _prefs().get(f"opt:{opt_id}:{name}")
    if raw is None:
        return None
    return _clamp_opt(opt_id, raw)


def _api_show(model: str) -> dict[str, Any]:
    try:
        from backend.agent.secrets import get_key

        from .model_fetch import ollama_model_info
        from .ollama_url import normalize_ollama_base

        base = normalize_ollama_base(get_key("ollama") or "")
        return ollama_model_info(base, model)
    except Exception:
        return {}


def saved_options(model: str) -> dict[str, Any]:
    api_defaults = {}
    show = _api_show(model)
    raw = show.get("parameters") if isinstance(show, dict) else None
    if isinstance(raw, dict):
        api_defaults = raw
    out: dict[str, Any] = {}
    for row in OPT_SLIDERS:
        val = saved_opt(model, row["id"])
        if val is None and row["id"] in api_defaults:
            val = _clamp_opt(row["id"], api_defaults[row["id"]])
        out[row["id"]] = row["default"] if val is None else val
    for row in OPT_SWITCHES:
        val = saved_opt(model, row["id"])
        if val is None and row["id"] in api_defaults:
            val = _clamp_opt(row["id"], api_defaults[row["id"]])
        out[row["id"]] = row["default"] if val is None else bool(val)
    return out


def saved_keep_alive(model: str) -> int | None:
    name = (model or "").strip()
    if not name:
        return None
    raw = _prefs().get(f"keep_alive:{name}")
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def get_model_settings(model: str, max_ctx: int = 0) -> dict[str, Any]:
    name = (model or "").strip()
    show = _api_show(name) if name else {}
    api_ctx = 0
    try:
        api_ctx = int(show.get("context_length") or 0)
    except (TypeError, ValueError):
        api_ctx = 0
    cap = max(int(max_ctx or 0), api_ctx)
    ticks = ticks_for_max(cap) if cap else [4096]
    saved = saved_num_ctx(name)
    num_ctx = clamp_num_ctx(saved or (cap or ticks[-1]), cap or ticks[-1]) if cap else saved
    ka = saved_keep_alive(name)
    opts = saved_options(name)
    return {
        "ok": True,
        "model": name,
        "num_ctx": num_ctx,
        "keep_alive": -1 if ka is None else ka,
        "ticks": ticks,
        "tick_labels": [_tick_label(t) for t in ticks],
        "options": opts,
        "sliders": list(OPT_SLIDERS),
        "switches": list(OPT_SWITCHES),
    }


def set_model_settings(
    model: str,
    *,
    num_ctx: int | None = None,
    keep_alive: int | None = None,
    max_ctx: int = 0,
    options: dict[str, Any] | None = None,
    **opt_kw: Any,
) -> dict[str, Any]:
    name = (model or "").strip()
    if not name:
        return {"ok": False, "error": "Model name required"}
    prefs = _prefs()
    if num_ctx is not None:
        prefs[f"num_ctx:{name}"] = clamp_num_ctx(int(num_ctx), max_ctx or int(num_ctx))
    if keep_alive is not None:
        prefs[f"keep_alive:{name}"] = int(keep_alive)
    bag = dict(options or {})
    bag.update(opt_kw)
    for key, val in bag.items():
        if key in _OPT_IDS and val is not None:
            prefs[f"opt:{key}:{name}"] = _clamp_opt(key, val)
    _save(prefs)
    try:
        from .ollama_provider import clear_num_ctx_ratchet

        clear_num_ctx_ratchet()
    except Exception:
        pass
    return get_model_settings(name, max_ctx=max_ctx)


def apply_history(index: int, model: str = "") -> dict[str, Any]:
    """Copy sliders from a saved call onto a model."""
    try:
        from .history import get_call
    except ImportError:
        from history import get_call

    row = get_call(int(index))
    if not row:
        return {"ok": False, "error": "No call at that history index"}
    name = (model or str(row.get("model") or "")).strip()
    opts = row.get("options") if isinstance(row.get("options"), dict) else {}
    return set_model_settings(
        name,
        num_ctx=int(row["num_ctx"]) if row.get("num_ctx") else None,
        keep_alive=int(row["keep_alive"]) if row.get("keep_alive") is not None else None,
        max_ctx=int(row.get("num_ctx") or 0),
        options=opts,
    )
