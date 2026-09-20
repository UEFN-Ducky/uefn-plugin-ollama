"""Ollama UI must follow the host llm-slot hook — no DOM MutationObserver poll."""

from pathlib import Path
import json

BOOT = Path(__file__).resolve().parent / "ui" / "boot.js"
MANIFEST = Path(__file__).resolve().parent / "plugin.json"
LIVE = Path(__file__).resolve().parent / "ui" / "live.html"


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


def test_dock_live_panel() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docks = data["contributes"]["dock.panels"]
    assert any(row.get("id") == "ollama-live" and row.get("defaultSide") == "left" for row in docks)
    assert LIVE.is_file()
    html = LIVE.read_text(encoding="utf-8")
    assert "stats.live" in html
    assert "uefn-plugin-ui" in html


if __name__ == "__main__":
    test_boot_uses_llm_slot_hook()
    test_dock_live_panel()
    print("ok")
