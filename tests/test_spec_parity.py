"""The phone app must show the model exactly what it was trained on."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from spec import ACTION_GRAMMAR, SYSTEM_PROMPT, Element, Screen, user_message  # noqa: E402

KOTLIN = (ROOT / "android/app/src/main/java/ai/handsfree/agent/AgentPrompt.kt").read_text(encoding="utf-8")
READER = (ROOT / "android/app/src/main/java/ai/handsfree/agent/ScreenReader.kt").read_text(encoding="utf-8")


def kotlin_raw(name: str) -> str:
    return re.search(rf'val {name} = """(.*?)"""', KOTLIN, re.S).group(1)


def test_system_prompt_matches():
    assert kotlin_raw("SYSTEM_PROMPT") == SYSTEM_PROMPT


def test_grammar_matches():
    assert kotlin_raw("ACTION_GRAMMAR") == ACTION_GRAMMAR


def test_user_message_format_matches():
    # Kotlin: "Goal: $goal\nPrevious actions:\n$hist\nScreen:\n$screen"
    assert '"Goal: $goal\\nPrevious actions:\\n$hist\\nScreen:\\n$screen"' in KOTLIN
    assert user_message("g", [], "S") == "Goal: g\nPrevious actions:\nnone\nScreen:\nS"
    assert user_message("g", ["a", "b"], "S").split("\n")[2:4] == ["1. a", "2. b"]


def test_element_render_matches():
    # Kotlin: "[$id] $role \"$esc\"" + " (on, focused)"
    assert '"[$id] $role \\"$esc\\""' in READER
    assert 'if (checked) "on" else "off"' in READER and 'add("focused")' in READER
    s = Screen("WhatsApp", [Element(1, "switch", "Wi-Fi", checked=True), Element(2, "edit", "Message", focused=True)])
    assert s.render() == 'App: WhatsApp\n[1] switch "Wi-Fi" (on)\n[2] edit "Message" (focused)'
