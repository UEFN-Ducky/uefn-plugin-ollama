"""Ollama UI must follow the host llm-slot hook — no DOM MutationObserver poll."""

from pathlib import Path

BOOT = Path(__file__).resolve().parent / "ui" / "boot.js"


def test_boot_uses_llm_slot_hook() -> None:
    src = BOOT.read_text(encoding="utf-8")
    assert "ducky:llm-slot" in src
    assert "MutationObserver" not in src
    assert "attachPicker" in src


if __name__ == "__main__":
    test_boot_uses_llm_slot_hook()
    print("ok")
