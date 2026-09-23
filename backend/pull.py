"""Pull a model: /api/pull when the server answers, else listed `ollama pull` terminal."""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

try:
    from .local import api_version_ok
    from .ollama_url import normalize_ollama_base
    from .progress import parse_pull_ndjson_line, parse_pull_tail
except ImportError:
    from local import api_version_ok
    from ollama_url import normalize_ollama_base
    from progress import parse_pull_ndjson_line, parse_pull_tail

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "job_id": job.get("job_id"),
        "model": job.get("model"),
        "mode": job.get("mode"),
        "percent": job.get("percent"),
        "status": job.get("status") or "",
        "error": job.get("error") or "",
        "done": bool(job.get("done")),
        "terminal_session_id": job.get("terminal_session_id") or "",
    }


def _set(job_id: str, **kw: Any) -> None:
    with _LOCK:
        row = _JOBS.get(job_id)
        if row is None:
            return
        row.update(kw)


def _api_pull(job_id: str, base: str, model: str) -> None:
    import httpx

    try:
        with httpx.stream(
            "POST",
            f"{base}/api/pull",
            json={"name": model, "stream": True},
            timeout=httpx.Timeout(connect=15.0, read=None, write=30.0, pool=15.0),
            trust_env=False,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                with _LOCK:
                    if _JOBS.get(job_id, {}).get("cancel"):
                        _set(job_id, done=True, status="cancelled", error="Cancelled")
                        return
                parsed = parse_pull_ndjson_line(line)
                if not parsed:
                    continue
                patch: dict[str, Any] = {}
                if parsed.get("status"):
                    patch["status"] = parsed["status"]
                if parsed.get("percent") is not None:
                    patch["percent"] = parsed["percent"]
                if parsed.get("error"):
                    patch["error"] = parsed["error"]
                    patch["done"] = True
                if parsed.get("done"):
                    patch["done"] = True
                    patch["percent"] = parsed.get("percent", 100.0)
                    patch["status"] = parsed.get("status") or "success"
                if patch:
                    _set(job_id, **patch)
                if patch.get("done"):
                    return
        _set(job_id, done=True, percent=100.0, status="success")
    except Exception as exc:
        _set(job_id, done=True, error=str(exc), status="error")


def _ollama_bin() -> str:
    try:
        from .local import ollama_bin
    except ImportError:
        from local import ollama_bin

    return ollama_bin() or "ollama"


def _ps_quote(path: str) -> str:
    return "'" + path.replace("'", "''") + "'"


def _spawn_pull_terminal(model: str) -> dict[str, Any]:
    from frontend.ui_web.terminal import get_terminal_manager

    mgr = get_terminal_manager()
    workdir = os.getcwd()
    title = f"ollama pull {model}"
    common: dict[str, Any] = {
        "cwd": workdir,
        "title": title,
        "push_open": True,
        "hidden": False,
        "conv_id": "settings:ollama",
    }
    login_shell = "powershell" if os.name == "nt" else "bash"
    binary = _ollama_bin()
    try:
        spawn = mgr.spawn(
            **common,
            command=[binary, "pull", model],
            activate=False,
            shell=login_shell,
        )
        if spawn.get("ok"):
            return spawn
    except TypeError:
        pass
    try:
        spawn = mgr.spawn(shell=login_shell, **common, activate=False)
    except TypeError:
        spawn = mgr.spawn(shell=login_shell, **common)
    session_id = str(spawn.get("session_id") or spawn.get("id") or "").strip()
    session = mgr.get_session(session_id) if spawn.get("ok") and session_id else None
    if session is not None:
        if os.name == "nt":
            session.run_command(f"& {_ps_quote(binary)} pull {_ps_quote(model)}", background=True)
        else:
            session.run_command(f"{binary} pull {model}", background=True)
    return spawn


def _kill_session(session_id: str) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    try:
        from frontend.ui_web.terminal import get_terminal_manager

        get_terminal_manager().kill(sid, push_close=True)
    except Exception:
        pass


def pull_start(base_url: str, name: str) -> dict[str, Any]:
    model = (name or "").strip()
    if not model:
        return {"ok": False, "error": "Model name required"}
    base = normalize_ollama_base(base_url)
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "model": model,
        "mode": "api",
        "percent": 0.0,
        "status": "starting",
        "error": "",
        "done": False,
        "cancel": False,
        "terminal_session_id": "",
        "started": time.time(),
    }
    if api_version_ok(base):
        with _LOCK:
            _JOBS[job_id] = job
        threading.Thread(target=_api_pull, args=(job_id, base, model), daemon=True).start()
        return _job_snapshot(job)

    spawn = _spawn_pull_terminal(model)
    if not spawn.get("ok"):
        return {"ok": False, "error": str(spawn.get("error") or "Could not start ollama pull")}
    session_id = str(spawn.get("session_id") or spawn.get("id") or "").strip()
    job["mode"] = "terminal"
    job["terminal_session_id"] = session_id
    job["status"] = "pulling"
    with _LOCK:
        _JOBS[job_id] = job
    return _job_snapshot(job)


def pull_status(job_id: str) -> dict[str, Any]:
    jid = (job_id or "").strip()
    with _LOCK:
        job = dict(_JOBS.get(jid) or {})
    if not job:
        return {"ok": False, "error": "No pull in progress", "done": True}
    if job.get("mode") == "terminal" and not job.get("done"):
        sid = str(job.get("terminal_session_id") or "")
        session = None
        try:
            from frontend.ui_web.terminal import get_terminal_manager

            session = get_terminal_manager().get_session(sid) if sid else None
        except Exception:
            session = None
        if session is None:
            _set(jid, done=True, error="The pull tab closed.", status="error")
        else:
            tail = ""
            try:
                tail = session.read_output_tail(16000)
            except Exception:
                tail = ""
            parsed = parse_pull_tail(tail)
            alive = True
            try:
                alive = bool(session.is_alive())
            except Exception:
                alive = False
            patch: dict[str, Any] = {}
            if parsed.get("status"):
                patch["status"] = parsed["status"]
            if parsed.get("percent") is not None:
                patch["percent"] = parsed["percent"]
            if parsed.get("error"):
                patch["error"] = parsed["error"]
                patch["done"] = True
            if parsed.get("done") or (not alive and parsed.get("percent") == 100):
                patch["done"] = True
                patch["percent"] = 100.0
                patch["status"] = "success"
            elif not alive and not parsed.get("done"):
                patch["done"] = True
                patch["error"] = patch.get("error") or "Pull finished without success."
                patch["status"] = "error"
            if patch:
                _set(jid, **patch)
        with _LOCK:
            job = dict(_JOBS.get(jid) or job)
    return _job_snapshot(job)


def pull_cancel(job_id: str) -> dict[str, Any]:
    jid = (job_id or "").strip()
    with _LOCK:
        job = dict(_JOBS.get(jid) or {})
        if job:
            _JOBS[jid]["cancel"] = True
            _JOBS[jid]["done"] = True
            _JOBS[jid]["status"] = "cancelled"
            _JOBS[jid]["error"] = "Cancelled"
    _kill_session(str(job.get("terminal_session_id") or ""))
    return {"ok": True, "cancelled": True, "job_id": jid}
