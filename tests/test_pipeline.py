"""End-to-end test of the council and dataset builder against fake model servers.

Run: python -m pytest tests  (no real models needed)
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import council  # noqa: E402
from llm import LLM  # noqa: E402
from records import read_jsonl, scenario_to_record, step_prompts  # noqa: E402
from scenarios import generate  # noqa: E402
from spec import SYSTEM_PROMPT, to_json  # noqa: E402

# Every (prompt -> right answer) the simulator can produce for the leaderboard and a few other seeds.
GOLD = {}
INVENTABLE = [scenario_to_record(sc) for sc in generate(60, seed=3)]
for seed in (council.EVAL_SEED, 1, 2, 3):
    for sc in generate(60, seed=seed):
        for messages, _, action in step_prompts(scenario_to_record(sc)):
            GOLD[messages[1]["content"]] = to_json(action)


def fake_reply(model: str, messages: list[dict]) -> str:
    last = messages[-1]["content"]
    if messages[0]["content"] == SYSTEM_PROMPT and len(messages) == 2:
        if model == "perfect":
            return GOLD.get(last, '{"action":"wait"}')
        ids = re.findall(r"^\[(\d+)\]", last, re.M) or ["1"]
        return json.dumps({"action": "tap", "id": int(random.choice(ids))})
    if last.startswith("Rewrite this phone request"):
        goal = re.search(r"^Request: (.*)$", last, re.M).group(1)
        return json.dumps([goal + " please", "hey " + goal, "unrelated"], ensure_ascii=False)
    if last.startswith("Do these two phone requests"):
        return "yes" if "unrelated" not in last else "no"
    if last.startswith("You are writing training examples"):
        rec = random.choice(INVENTABLE)
        return "Sure:\n" + json.dumps({"goal": rec["goal"], "steps": rec["steps"]}, ensure_ascii=False)
    if last.startswith("Another assistant looked"):
        return last.split("\n")[1]
    return ""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send({"data": [{"id": "perfect"}, {"id": "random"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/chat":  # Ollama's native endpoint
            OLLAMA_CALLS.append(body)
            if "think" in body and body["model"] == "old-ollama":
                return self._send({"error": "invalid field: think"}, 400)
            reply = fake_reply(body["model"].replace("old-ollama", "perfect"), body["messages"])
            return self._send({"message": {"role": "assistant", "content": reply}})
        self._send({"choices": [{"message": {"content": fake_reply(body["model"], body["messages"])}}]})

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


OLLAMA_CALLS: list[dict] = []


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_leaderboard_ranks_the_better_model_first(server, tmp_path):
    out = tmp_path / "data" / "board.json"  # folder doesn't exist yet, like a fresh checkout
    council.leaderboard([LLM("random", server), LLM("perfect", server)], n=30, workers=4, out=out)
    board = json.loads(out.read_text())
    assert board["ranking"][0] == "perfect"
    assert board["models"]["perfect"]["task_success"] == 1.0
    assert board["models"]["random"]["step_accuracy"] < 0.5


def test_paraphrase_keeps_only_checked_rewrites(server, tmp_path):
    out = tmp_path / "para.jsonl"
    council.paraphrase([LLM("perfect", server), LLM("random", server)], n=10, k=3, out=out, seed=2)
    rows = list(read_jsonl(out))
    assert rows and all("unrelated" not in r["goal"] for r in rows)
    for r in rows:  # typed message text survives the rewrite
        for s in r["steps"]:
            if s["action"]["action"] == "type":
                assert s["action"]["text"] in r["goal"]


def test_invent_keeps_council_approved_tasks(server, tmp_path):
    out = tmp_path / "inv.jsonl"
    random.seed(0)
    council.invent([LLM("perfect", server), LLM("perfect", server)], n=4, out=out, seed=3)
    rows = list(read_jsonl(out))
    assert len(rows) == 4 and all(r["family"].startswith("invented:") for r in rows)


def test_invent_rejects_when_voters_disagree(server, tmp_path):
    out = tmp_path / "inv.jsonl"
    random.seed(0)
    council.invent([LLM("perfect", server), LLM("random", server)], n=2, out=out, seed=3)
    # the random voter rarely agrees, and the author never adopts its answer
    assert not out.exists() or len(list(read_jsonl(out))) < 2


def test_build_dataset(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "build_dataset.py"), "--sim", "200", "--out", str(tmp_path)],
                   check=True)
    rows = list(read_jsonl(tmp_path / "train.jsonl"))
    assert len(rows) > 500
    r = rows[0]
    assert r["prompt"][0]["content"] == SYSTEM_PROMPT and r["completion"][0]["role"] == "assistant"
    json.loads(r["completion"][0]["content"])
    assert any("Previous actions:\n1. open_app" in x["prompt"][1]["content"] for x in rows)


def test_ollama_native_turns_thinking_off(server):
    m = LLM("perfect", server)
    m.ollama = True
    rec = scenario_to_record(next(generate(1, seed=1)))
    messages, _, gold = next(step_prompts(rec))
    assert json.loads(m.chat(messages)) == gold
    assert OLLAMA_CALLS[-1]["think"] is False and OLLAMA_CALLS[-1]["options"]["num_predict"] == 400


def test_ollama_without_think_support_falls_back(server):
    m = LLM("old-ollama", server)
    m.ollama = True
    messages, _, gold = next(step_prompts(scenario_to_record(next(generate(1, seed=1)))))
    assert json.loads(m.chat(messages)) == gold
    assert m.send_think is False and "think" not in OLLAMA_CALLS[-1]


def test_equally_valid_alternatives_score_as_correct():
    """Tapping the app icon instead of open_app, or a suggestion instead of Enter, is still right."""
    records = [scenario_to_record(sc) for sc in generate(300, seed=5)]
    with_alts = [r for r in records if any(s.get("alts") for s in r["steps"])]
    assert len(with_alts) > 50
    for r in records:
        for s in r["steps"]:
            ids = {e["id"] for e in s["screen"]["elements"]}
            assert all(a.get("id", 1) in ids for a in s.get("alts", []))

    class PrefersAlternatives:
        def __init__(self, record):
            self.answers = iter(s.get("alts", [s["action"]])[0] for s in record["steps"])

        def chat(self, messages, **kw):
            return json.dumps(next(self.answers), ensure_ascii=False)

    for r in with_alts[:20]:
        assert all(council.run_task(PrefersAlternatives(r), r))
