"""Live machine + Ollama /api/ps snapshot (no extra deps)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

try:
    from .local import _ps, format_bytes
    from .ollama_url import normalize_ollama_base
except ImportError:
    from local import _ps, format_bytes
    from ollama_url import normalize_ollama_base

_CPU_PREV: tuple[float, float] | None = None  # (idle, total)
_CPU_PCT = 0.0
_GENERATING = ""
_GPU_CACHE: tuple[float, dict[str, Any]] | None = None
_GPU_MISS = False
_GPU_TTL = 2.0
_PS_CACHE: list[dict[str, Any]] = []
_EMPTY_GPU = {
    "gpu_pct": None,
    "vram_used": 0,
    "vram_total": 0,
    "vram_label": "",
    "gpu_temp": None,
    "gpu_power": None,
    "gpu_power_limit": None,
    "gpu_fan": None,
    "gpu_name": "",
    "gpu_clock": None,
}


def _hidden_popen_kwargs() -> dict[str, Any]:
    """Windows: never flash a console for nvidia-smi."""
    if os.name != "nt":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    kw: dict[str, Any] = {"creationflags": flags}
    try:
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kw["startupinfo"] = info
    except (AttributeError, OSError):
        pass
    return kw


def _mem() -> dict[str, Any]:
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total = int(stat.ullTotalPhys)
            avail = int(stat.ullAvailPhys)
            used = max(0, total - avail)
            return {
                "ram_total": total,
                "ram_used": used,
                "ram_pct": float(stat.dwMemoryLoad),
                "ram_label": f"{format_bytes(used)} / {format_bytes(total)}",
            }
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        avail = page * os.sysconf("SC_AVPHYS_PAGES")
        used = max(0, total - avail)
        pct = (100.0 * used / total) if total else 0.0
        return {
            "ram_total": total,
            "ram_used": used,
            "ram_pct": pct,
            "ram_label": f"{format_bytes(used)} / {format_bytes(total)}",
        }
    except (AttributeError, OSError, ValueError):
        return {"ram_total": 0, "ram_used": 0, "ram_pct": 0.0, "ram_label": "n/a"}


def _file_ints(path: str) -> list[int]:
    try:
        parts = Path_read(path).split()
        return [int(p) for p in parts if p.isdigit() or (p[:1].isdigit())]
    except Exception:
        return []


def Path_read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _cpu_linux() -> float:
    global _CPU_PREV, _CPU_PCT
    try:
        line = Path_read("/proc/stat").splitlines()[0]
        nums = [int(x) for x in line.split()[1:]]
    except (OSError, ValueError, IndexError):
        return _CPU_PCT
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    prev = _CPU_PREV
    _CPU_PREV = (float(idle), float(total))
    if prev is None:
        return _CPU_PCT
    didle = idle - prev[0]
    dtotal = total - prev[1]
    if dtotal <= 0:
        return _CPU_PCT
    _CPU_PCT = max(0.0, min(100.0, 100.0 * (1.0 - didle / dtotal)))
    return _CPU_PCT


def _cpu_windows() -> float:
    global _CPU_PREV, _CPU_PCT
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    def _q(ft: FILETIME) -> int:
        return (int(ft.dwHighDateTime) << 32) + int(ft.dwLowDateTime)

    idle = FILETIME()
    kernel = FILETIME()
    user = FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return _CPU_PCT
    idle_n = _q(idle)
    total_n = _q(kernel) + _q(user)
    prev = _CPU_PREV
    _CPU_PREV = (float(idle_n), float(total_n))
    if prev is None:
        return _CPU_PCT
    didle = idle_n - prev[0]
    dtotal = total_n - prev[1]
    if dtotal <= 0:
        return _CPU_PCT
    _CPU_PCT = max(0.0, min(100.0, 100.0 * (1.0 - didle / dtotal)))
    return _CPU_PCT


def _gpu(*, allow_probe: bool = True) -> dict[str, Any]:
    global _GPU_CACHE, _GPU_MISS
    if _GPU_MISS:
        return dict(_EMPTY_GPU)
    now = time.time()
    if _GPU_CACHE and (not allow_probe or now - _GPU_CACHE[0] < _GPU_TTL):
        return dict(_GPU_CACHE[1])
    if not allow_probe:
        return dict(_GPU_CACHE[1]) if _GPU_CACHE else dict(_EMPTY_GPU)
    exe = shutil.which("nvidia-smi")
    if not exe:
        _GPU_MISS = True
        return dict(_EMPTY_GPU)
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed,name,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
            text=True,
            stderr=subprocess.DEVNULL,
            **_hidden_popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        _GPU_MISS = True
        return dict(_EMPTY_GPU)
    line = (out or "").splitlines()[0] if out else ""
    parts = [p.strip() for p in line.split(",")]
    def _num(idx: int) -> float | None:
        if idx >= len(parts):
            return None
        raw = parts[idx].strip().replace("[N/A]", "").replace("N/A", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    try:
        pct = float(parts[0])
        used = int(float(parts[1]) * 1024 * 1024)
        total = int(float(parts[2]) * 1024 * 1024)
    except (IndexError, ValueError):
        _GPU_MISS = True
        return dict(_EMPTY_GPU)
    temp = _num(3)
    power = _num(4)
    power_lim = _num(5)
    fan = _num(6)
    name = parts[7].strip() if len(parts) > 7 else ""
    clock = _num(8)
    row = {
        "gpu_pct": pct,
        "vram_used": used,
        "vram_total": total,
        "vram_label": f"{format_bytes(used)} / {format_bytes(total)}",
        "gpu_temp": temp,
        "gpu_power": power,
        "gpu_power_limit": power_lim,
        "gpu_fan": fan,
        "gpu_name": name,
        "gpu_clock": clock,
    }
    _GPU_CACHE = (now, row)
    return dict(row)


def _disk() -> dict[str, Any]:
    root = (
        os.environ.get("OLLAMA_MODELS")
        or os.environ.get("OLLAMA_HOME")
        or ("C:\\" if os.name == "nt" else "/")
    )
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        try:
            usage = shutil.disk_usage("C:\\" if os.name == "nt" else "/")
        except OSError:
            return {"disk_total": 0, "disk_used": 0, "disk_pct": 0.0, "disk_label": "n/a"}
    total = int(usage.total)
    used = int(usage.used)
    pct = (100.0 * used / total) if total else 0.0
    return {
        "disk_total": total,
        "disk_used": used,
        "disk_pct": round(pct, 1),
        "disk_label": f"{format_bytes(used)} / {format_bytes(total)}",
    }


def set_generating(model: str = "") -> None:
    global _GENERATING
    _GENERATING = str(model or "").strip()


def _running_from_ps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    running = []
    for row in rows:
        name = str(row.get("name") or row.get("model") or "").strip()
        if not name:
            continue
        vram = int(row.get("size_vram") or 0)
        running.append(
            {
                "name": name,
                "size_vram": vram,
                "size_vram_label": format_bytes(vram),
                "size": int(row.get("size") or 0),
                "expires_at": str(row.get("expires_at") or ""),
                "processor": str((row.get("details") or {}).get("family") or row.get("processor") or ""),
            }
        )
    return running


def live_stats(base_url: str) -> dict[str, Any]:
    """CPU / RAM / GPU + loaded Ollama models. Cheap enough to poll ~1s."""
    global _PS_CACHE
    base = normalize_ollama_base(base_url)
    cpu = _cpu_windows() if os.name == "nt" else _cpu_linux()
    mem = _mem()
    busy = bool(_GENERATING)
    # Prompt-eval owns the GPU + Ollama HTTP. nvidia-smi and /api/ps hitch both.
    gpu = _gpu(allow_probe=not busy)
    disk = _disk()
    if busy:
        running = _running_from_ps(_PS_CACHE)
    else:
        rows = _ps(base)
        _PS_CACHE = list(rows)
        running = _running_from_ps(rows)
    return {
        "ok": True,
        "ts": time.time(),
        "running": bool(running) or bool(_GENERATING),
        "thinking": bool(_GENERATING),
        "generating_model": _GENERATING,
        "cpu_pct": round(cpu, 1),
        "cpu_count": os.cpu_count() or 0,
        **mem,
        **gpu,
        **disk,
        "models": running,
    }
