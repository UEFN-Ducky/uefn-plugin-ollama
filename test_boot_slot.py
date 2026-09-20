"""Ollama UI must follow the host llm-slot hook — no DOM MutationObserver poll."""

from pathlib import Path

BOOT = Path(__file__).resolve().parent / "ui" / "boot.js"


def test_boot_uses_llm_slot_hook() -> None:
    src = BOOT.read_text(encoding="utf-8")
    assert "ducky:llm-slot" in src
    assert "new MutationObserver" not in src
    assert "ollama-board" in src
    assert "ollama-fader" in src
    assert "attachPicker" in src
    assert "ollama-board-modal" not in src
    assert "Open board" not in src
    assert "grid-template-columns:280px minmax(0,1fr)" in src
    assert src.index('aside class="ollama-board-live"') < src.index('div class="ollama-board-main"')


if __name__ == "__main__":
    test_boot_uses_llm_slot_hook()
    print("ok")
