"""The contract between training data and the phone app.

Everything the model sees at training time must look exactly like what the
Android app shows it at run time. The Kotlin side mirrors this file in
android/app/src/main/java/ai/handsfree/agent/AgentPrompt.kt -- change both
together (tests/test_spec_parity.py checks the system prompt matches).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You control an Android phone for a user who cannot use their hands.
Each turn you get the user's goal, the actions you already took, and the current screen.
Reply with exactly one JSON object: the next action. No other text.
Actions:
{"action":"open_app","app":"<app name>"}
{"action":"tap","id":<element id>}
{"action":"long_press","id":<element id>}
{"action":"type","id":<element id>,"text":"<text>"}
{"action":"enter"}
{"action":"scroll","direction":"up|down|left|right"}
{"action":"back"}
{"action":"home"}
{"action":"wait"}
{"action":"ask_user","question":"<question>"}
{"action":"done","say":"<short reply to the user>"}
Only use ids that appear on the screen. If the goal is ambiguous, ask_user. If the goal is complete, done."""

ACTIONS: dict[str, tuple[str, ...]] = {
    "open_app": ("app",),
    "tap": ("id",),
    "long_press": ("id",),
    "type": ("id", "text"),
    "enter": (),
    "scroll": ("direction",),
    "back": (),
    "home": (),
    "wait": (),
    "ask_user": ("question",),
    "done": ("say",),
}
DIRECTIONS = ("up", "down", "left", "right")

# GBNF grammar used on the phone to force well-formed output from a small model.
ACTION_GRAMMAR = r'''root ::= "{" ws "\"action\"" ws ":" ws body ws "}"
body ::= "\"open_app\"" ws "," ws "\"app\"" ws ":" ws str
       | "\"tap\"" ws "," ws "\"id\"" ws ":" ws int
       | "\"long_press\"" ws "," ws "\"id\"" ws ":" ws int
       | "\"type\"" ws "," ws "\"id\"" ws ":" ws int ws "," ws "\"text\"" ws ":" ws str
       | "\"enter\"" | "\"back\"" | "\"home\"" | "\"wait\""
       | "\"scroll\"" ws "," ws "\"direction\"" ws ":" ws ("\"up\"" | "\"down\"" | "\"left\"" | "\"right\"")
       | "\"ask_user\"" ws "," ws "\"question\"" ws ":" ws str
       | "\"done\"" ws "," ws "\"say\"" ws ":" ws str
str ::= "\"" ([^"\\\n] | "\\" ["\\/bfnrtu])* "\""
int ::= [0-9]+
ws ::= " "?
'''


@dataclass
class Element:
    id: int
    role: str  # button | edit | text | item | switch | tab | image
    label: str
    checked: bool | None = None
    focused: bool = False
    scrollable: bool = False

    def render(self) -> str:
        line = f'[{self.id}] {self.role} "{self.label}"'
        flags = []
        if self.checked is not None:
            flags.append("on" if self.checked else "off")
        if self.focused:
            flags.append("focused")
        if self.scrollable:
            flags.append("scrollable")
        if flags:
            line += " (" + ", ".join(flags) + ")"
        return line


@dataclass
class Screen:
    app: str
    elements: list[Element] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join([f"App: {self.app}"] + [e.render() for e in self.elements])

    def ids(self) -> set[int]:
        return {e.id for e in self.elements}

    def find(self, label: str) -> Element:
        for e in self.elements:
            if e.label == label:
                return e
        raise KeyError(label)


def describe_action(a: dict) -> str:
    """One-line history entry, e.g. 'tap 3 "Ahmed"'. The app writes the same."""
    kind = a["action"]
    if kind == "open_app":
        return f"open_app {a['app']}"
    if kind in ("tap", "long_press"):
        return f"{kind} {a['id']}" + (f' "{a["label"]}"' if a.get("label") else "")
    if kind == "type":
        return f'type {a["id"]} "{a["text"]}"'
    if kind == "scroll":
        return f"scroll {a['direction']}"
    if kind == "ask_user":
        return f'ask_user "{a["question"]}"'
    if kind == "done":
        return "done"
    return kind


def user_message(goal: str, history: list[str], screen: Screen | str) -> str:
    screen_text = screen if isinstance(screen, str) else screen.render()
    hist = "\n".join(f"{i}. {h}" for i, h in enumerate(history, 1)) or "none"
    return f"Goal: {goal}\nPrevious actions:\n{hist}\nScreen:\n{screen_text}"


def to_json(a: dict) -> str:
    """Canonical output string the model is trained to produce."""
    kind = a["action"]
    out = {"action": kind}
    for k in ACTIONS[kind]:
        out[k] = a[k]
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def parse_action(text: str, screen_ids: set[int] | None = None) -> dict | None:
    """Pull the first JSON object out of a model reply and validate it."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("action") not in ACTIONS:
        return None
    for k in ACTIONS[obj["action"]]:
        if k not in obj:
            return None
    if "id" in obj:
        if not isinstance(obj["id"], int) or (screen_ids is not None and obj["id"] not in screen_ids):
            return None
    if obj["action"] == "scroll" and obj["direction"] not in DIRECTIONS:
        return None
    for k in ("app", "text", "question", "say"):
        if k in obj and not isinstance(obj[k], str):
            return None
    return {k: obj[k] for k in ("action",) + ACTIONS[obj["action"]]}


def same_action(a: dict | None, b: dict | None) -> bool:
    """Do two actions do the same thing? Free text only needs to roughly match."""
    if not a or not b or a["action"] != b["action"]:
        return False
    kind = a["action"]
    if kind in ("tap", "long_press"):
        return a["id"] == b["id"]
    if kind == "type":
        return a["id"] == b["id"] and _norm(a["text"]) == _norm(b["text"])
    if kind == "open_app":
        return _norm(a["app"]) == _norm(b["app"])
    if kind == "scroll":
        return a["direction"] == b["direction"]
    return True  # enter/back/home/wait/ask_user/done: wording may differ


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().rstrip(".!").split())
