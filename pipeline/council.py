"""The model council: your local LLMs test, teach and check each other.

  leaderboard  Every model drives the simulated phone; scored against the known
               right answers. Tells you which model is the best teacher (and,
               later, how good your trained phone model is).
  paraphrase   The best teacher rewrites task requests the way real people talk;
               another model checks each rewrite still means the same thing.
  invent       A teacher invents new tasks for apps the simulator doesn't cover.
               The other models independently solve each step; when they
               disagree, the author sees their answer and may change its mind.
               Only tasks the council agrees on are kept.

Examples:
  python council.py leaderboard --models qwen2.5:7b llama3.2:3b gemma2:2b
  python council.py paraphrase --models qwen2.5:7b llama3.2:3b --n 300
  python council.py invent --models qwen2.5:7b llama3.2:3b gemma2:2b --n 200
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import KNOWN_SERVERS, LLM, find_servers, parse_model_args
from records import append_jsonl, history_entry, scenario_to_record, screen_from_dict, step_prompts
from scenarios import generate
from spec import SYSTEM_PROMPT, parse_action, same_action, to_json, user_message

DATA = Path(__file__).resolve().parent.parent / "data"
EVAL_SEED = 999  # leaderboard tasks; build_dataset never uses this seed


# ---------------------------------------------------------------- leaderboard

def run_task(model: LLM, record: dict) -> list[bool]:
    """Score each step given the correct history (so one slip doesn't sink the rest)."""
    results = []
    for messages, screen, gold in step_prompts(record):
        try:
            reply = model.chat(messages, max_tokens=120)
        except Exception as e:  # a dead server shouldn't kill the whole run
            print(f"  {model}: request failed: {e}", file=sys.stderr)
            reply = ""
        results.append(same_action(parse_action(reply, screen.ids()), gold))
    return results


def leaderboard(models: list[LLM], n: int, workers: int, out: Path):
    tasks = [scenario_to_record(s) for s in generate(n, seed=EVAL_SEED)]
    board = {}
    for m in models:
        print(f"Testing {m} on {n} tasks...")
        with ThreadPoolExecutor(workers) as ex:
            scores = list(ex.map(lambda r: run_task(m, r), tasks))
        fam = collections.defaultdict(lambda: [0, 0])
        steps_ok = steps = tasks_ok = 0
        for rec, res in zip(tasks, scores):
            steps += len(res)
            steps_ok += sum(res)
            tasks_ok += all(res)
            fam[rec["family"]][0] += sum(res)
            fam[rec["family"]][1] += len(res)
        board[m.model] = {
            "step_accuracy": round(steps_ok / steps, 3),
            "task_success": round(tasks_ok / len(tasks), 3),
            "by_family": {f: round(a / b, 3) for f, (a, b) in sorted(fam.items())},
        }
        print(f"  step accuracy {board[m.model]['step_accuracy']:.0%}, whole tasks right {board[m.model]['task_success']:.0%}")
    ranked = sorted(board, key=lambda k: board[k]["step_accuracy"], reverse=True)
    print("\nLeaderboard:")
    for i, k in enumerate(ranked, 1):
        b = board[k]
        weak = [f for f, v in b["by_family"].items() if v < 0.7]
        print(f"{i}. {k:30s} steps {b['step_accuracy']:.0%}  tasks {b['task_success']:.0%}  weak at: {', '.join(weak) or '-'}")
    out.parent.mkdir(parents=True, exist_ok=True)  # data/ is gitignored, so a fresh checkout lacks it
    out.write_text(json.dumps({"ranking": ranked, "models": board}, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")


# ----------------------------------------------------------------- paraphrase

PARAPHRASE_PROMPT = """Rewrite this phone request {k} different ways, the way a real person would say it out loud to a voice assistant.
Language: {lang}. Keep exactly the same meaning: same person, same app, same place, same time.
Keep these exact words unchanged inside each rewrite: {keep}
Request: {goal}
Reply with only a JSON list of strings."""

SAME_MEANING_PROMPT = """Do these two phone requests ask for exactly the same thing (same person, app, message, place and time)?
A: {a}
B: {b}
Answer only yes or no."""


def must_keep(record: dict) -> list[str]:
    return [s["action"]["text"] for s in record["steps"] if s["action"]["action"] == "type"]


def paraphrase(models: list[LLM], n: int, k: int, out: Path, seed: int):
    writer, checkers = models[0], models[1:] or models[:1]
    lang_name = {"en": "English", "ar": "Egyptian Arabic (colloquial, Arabic script)"}
    kept = tried = 0
    for i, sc in enumerate(generate(n, seed=seed)):
        rec = scenario_to_record(sc, source=f"paraphrase:{writer.model}")
        keep = must_keep(rec)
        prompt = PARAPHRASE_PROMPT.format(k=k, lang=lang_name[sc.lang], goal=sc.goal,
                                          keep=", ".join(f'"{x}"' for x in keep) or "(nothing special)")
        try:
            raw = writer.chat([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=500)
            start = raw.find("[")
            variants, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else ([], 0)
        except Exception:
            continue
        for v in variants[:k]:
            if not isinstance(v, str) or not v.strip() or v.strip() == sc.goal:
                continue
            tried += 1
            if any(x not in v for x in keep):
                continue
            checker = random.Random(i).choice(checkers)
            verdict = checker.chat([{"role": "user", "content": SAME_MEANING_PROMPT.format(a=sc.goal, b=v)}], max_tokens=5)
            if verdict.strip().lower().startswith(("yes", "نعم")):
                append_jsonl(out, {**rec, "goal": v.strip(), "original_goal": sc.goal})
                kept += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n} requests, kept {kept}/{tried} rewrites")
    print(f"Done: kept {kept} of {tried} rewrites -> {out}")


# --------------------------------------------------------------------- invent

INVENT_APPS = ["Gmail", "Calendar", "Spotify", "Calculator", "Uber", "Careem", "Talabat", "InstaPay",
               "Facebook", "Instagram", "Notes", "Contacts", "Files", "Google Photos", "Telegram",
               "Messenger", "Play Store", "Settings (sound)", "Settings (brightness)", "Vodafone Cash",
               "Quran app", "Netflix", "Weather", "Calculator", "Truecaller"]

INVENT_PROMPT = """You are writing training examples for a voice assistant that operates an Android phone.
Invent one realistic task a user would ask by voice in {lang} for the app "{app}", and write the full sequence of screens and the correct action on each screen.

The assistant follows these rules:
---
{rules}
---
Screen elements have: id (1,2,3... in reading order), role (button|edit|text|item|switch|tab|image), label, and optionally checked (true/false), focused, scrollable.
The first screen is whatever the user was looking at (e.g. the home launcher) and its action is open_app. The last action must be done (or ask_user if the request is ambiguous).
Make screens look like the real app. 3 to 7 steps.

Here is an example in the exact format:
{example}

Reply with only the JSON object."""

VOTER_FIX_PROMPT = """Another assistant looked at this same screen and chose a different action:
{other}
You chose:
{mine}
Which is correct? Reply with only the JSON action you now believe is right."""


def validate_invented(obj) -> dict | None:
    try:
        goal = obj["goal"].strip()
        steps = obj["steps"]
        if not goal or not 2 <= len(steps) <= 10:
            return None
        clean = []
        for st in steps:
            screen = screen_from_dict(st["screen"])
            if [e.id for e in screen.elements] != list(range(1, len(screen.elements) + 1)):
                return None
            act = parse_action(json.dumps(st["action"], ensure_ascii=False), screen.ids())
            if act is None:
                return None
            clean.append({"screen": st["screen"], "action": act})
        if clean[-1]["action"]["action"] not in ("done", "ask_user"):
            return None
        return {"goal": goal, "steps": clean}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def council_vote(author: LLM, voters: list[LLM], record: dict) -> tuple[bool, int]:
    """Each voter independently picks the action on every step. A step passes when
    a majority agrees with the author -- or the author is persuaded by a voter's
    answer and switches, which the record then adopts. Returns (kept, switches)."""
    switches = 0
    history: list[str] = []
    for step in record["steps"]:
        screen = screen_from_dict(step["screen"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(record["goal"], history, screen)}]
        votes = [parse_action(v.chat(messages, max_tokens=120), screen.ids()) for v in voters]
        agree = sum(same_action(v, step["action"]) for v in votes)
        if agree * 2 < len(voters):
            alt = next((v for v in votes if v and not same_action(v, step["action"])), None)
            # Small models tend to cave to any objection, so a switch only counts
            # when at least two checkers reached the alternative independently.
            if alt is None or sum(same_action(v, alt) for v in votes) < 2:
                return False, switches
            reply = author.chat(messages + [
                {"role": "assistant", "content": to_json(step["action"])},
                {"role": "user", "content": VOTER_FIX_PROMPT.format(other=to_json(alt), mine=to_json(step["action"]))},
            ], max_tokens=120)
            if not same_action(parse_action(reply, screen.ids()), alt):
                return False, switches
            if sum(same_action(v, alt) for v in votes) * 2 < len(voters):
                return False, switches
            step["action"] = alt
            switches += 1
        history.append(history_entry(step["action"], screen))
    return True, switches


def invent(models: list[LLM], n: int, out: Path, seed: int):
    if len(models) < 2:
        sys.exit("invent needs at least 2 models: one writes, the others check.")
    rng = random.Random(seed)
    example = scenario_to_record(next(generate(1, seed=seed)))
    example = json.dumps({"goal": example["goal"], "steps": example["steps"]}, ensure_ascii=False)
    kept = 0
    for i in range(n):
        author = models[i % len(models)]
        voters = [m for m in models if m is not author]
        lang = rng.choice(["English", "Egyptian Arabic (colloquial, Arabic script)"])
        app = rng.choice(INVENT_APPS)
        try:
            raw = author.chat([{"role": "user", "content": INVENT_PROMPT.format(
                lang=lang, app=app, rules=SYSTEM_PROMPT, example=example)}], temperature=0.8, max_tokens=2500)
            obj, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
        except Exception:
            continue
        rec = validate_invented(obj)
        if rec is None:
            continue
        rec = {"source": f"invent:{author.model}", "family": f"invented:{app}",
               "lang": "en" if lang == "English" else "ar", **rec}
        ok, switched = council_vote(author, voters, rec)
        if ok:
            rec["switches"] = switched
            append_jsonl(out, rec)
            kept += 1
        print(f"{i + 1}/{n} {app}: {'kept' if ok else 'rejected'} (total kept {kept})")
    print(f"Done: {kept} invented tasks -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["servers", "leaderboard", "paraphrase", "invent"])
    ap.add_argument("--models", nargs="*", default=[], help="model names; add @URL for a non-default server")
    ap.add_argument("--server", default=KNOWN_SERVERS["ollama"], help="OpenAI-compatible base URL")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="rewrites per request (paraphrase)")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (keep 1 on a weak laptop)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.command == "servers" or not a.models:
        servers = find_servers()
        if not servers:
            sys.exit("No local model server found. Start Ollama or LM Studio first (see README).")
        for url, ms in servers.items():
            print(url)
            for m in ms:
                print("   ", m)
        if a.command == "servers":
            return
        sys.exit("Pick models with --models (names above).")

    models = parse_model_args(a.models, a.server)
    if a.command == "leaderboard":
        leaderboard(models, a.n, a.workers, a.out or DATA / "leaderboard.json")
    elif a.command == "paraphrase":
        paraphrase(models, a.n, a.k, a.out or DATA / "paraphrased.jsonl", a.seed)
    else:
        invent(models, a.n, a.out or DATA / "invented.jsonl", a.seed)


if __name__ == "__main__":
    main()
