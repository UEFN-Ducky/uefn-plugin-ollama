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


def _gpu() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"gpu_pct": None, "vram_used": 0, "vram_total": 0, "vram_label": ""}
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return {"gpu_pct": None, "vram_used": 0, "vram_total": 0, "vram_label": ""}
    line = (out or "").splitlines()[0] if out else ""
    parts = [p.strip() for p in line.split(",")]
    try:
        pct = float(parts[0])
        used = int(float(parts[1]) * 1024 * 1024)
        total = int(float(parts[2]) * 1024 * 1024)
    except (IndexError, ValueError):
        return {"gpu_pct": None, "vram_used": 0, "vram_total": 0, "vram_label": ""}
    return {
        "gpu_pct": pct,
        "vram_used": used,
        "vram_total": total,
        "vram_label": f"{format_bytes(used)} / {format_bytes(total)}",
    }


def set_generating(model: str = "") -> None:
    global _GENERATING
    _GENERATING = str(model or "").strip()


def live_stats(base_url: str) -> dict[str, Any]:
    """CPU / RAM / GPU + loaded Ollama models. Cheap enough to poll ~1s."""
    base = normalize_ollama_base(base_url)
    cpu = _cpu_windows() if os.name == "nt" else _cpu_linux()
    mem = _mem()
    gpu = _gpu()
    running = []
    for row in _ps(base):
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
    return {
        "ok": True,
        "ts": time.time(),
        "running": bool(running) or bool(_GENERATING),
        "thinking": bool(_GENERATING),
        "generating_model": _GENERATING,
        "cpu_pct": round(cpu, 1),
        **mem,
        **gpu,
        "models": running,
    }
