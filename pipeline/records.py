"""Trajectory records on disk (JSONL) and turning them into training examples."""
from __future__ import annotations

import json
from pathlib import Path

from scenarios import Scenario
from spec import SYSTEM_PROMPT, Element, Screen, describe_action, to_json, user_message


def screen_to_dict(s: Screen) -> dict:
    els = []
    for e in s.elements:
        d = {"id": e.id, "role": e.role, "label": e.label}
        if e.checked is not None:
            d["checked"] = e.checked
        if e.focused:
            d["focused"] = True
        if e.scrollable:
            d["scrollable"] = True
        els.append(d)
    return {"app": s.app, "elements": els}


def screen_from_dict(d: dict) -> Screen:
    return Screen(str(d["app"]), [
        Element(int(e["id"]), str(e["role"]), str(e["label"]), e.get("checked"),
                bool(e.get("focused", False)), bool(e.get("scrollable", False)))
        for e in d["elements"]
    ])


def scenario_to_record(sc: Scenario, source: str = "simulator") -> dict:
    return {"source": source, "family": sc.family, "lang": sc.lang, "goal": sc.goal,
            "steps": [{"screen": screen_to_dict(st.screen), "action": st.action,
                       **({"alts": st.alts} if st.alts else {})} for st in sc.steps]}


def history_entry(action: dict, screen: Screen) -> str:
    """What the app writes into 'Previous actions' after doing this action."""
    a = dict(action)
    if "id" in a:
        try:
            a["label"] = next(e.label for e in screen.elements if e.id == a["id"])
        except StopIteration:
            pass
    return describe_action(a)


def step_prompts(record: dict):
    """Yield (messages, screen, gold_action) for every step, with the true history so far."""
    history: list[str] = []
    for step in record["steps"]:
        screen = screen_from_dict(step["screen"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(record["goal"], history, screen)}]
        yield messages, screen, step["action"]
        history.append(history_entry(step["action"], screen))


def to_sft_examples(record: dict):
    """Prompt/completion pairs: loss is only taken on the action the model writes."""
    for messages, _, action in step_prompts(record):
        yield {"prompt": messages, "completion": [{"role": "assistant", "content": to_json(action)}],
               "family": record["family"], "source": record["source"]}


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | Path, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
