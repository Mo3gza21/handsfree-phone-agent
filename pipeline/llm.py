"""Talk to local models through an OpenAI-compatible server.

Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1) and
llama.cpp's llama-server (http://localhost:8080/v1) all speak this API, so the
pipeline works with whichever one holds your models.
"""
from __future__ import annotations

import time

import requests

KNOWN_SERVERS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llama.cpp": "http://localhost:8080/v1",
}


def find_servers() -> dict[str, list[str]]:
    """Which local servers are up, and which models each one has."""
    found = {}
    for url in KNOWN_SERVERS.values():
        models = list_models(url)
        if models:
            found[url] = models
    return found


def list_models(base_url: str) -> list[str]:
    try:
        r = requests.get(f"{base_url}/models", timeout=3)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError):
        return []


class LLM:
    def __init__(self, model: str, base_url: str = KNOWN_SERVERS["ollama"], timeout: int = 300):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 400) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": False}
        for attempt in range(3):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (requests.RequestException, KeyError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def __repr__(self):
        return self.model


def parse_model_args(specs: list[str], default_url: str) -> list[LLM]:
    """'qwen2.5:7b' or 'qwen2.5:7b@http://host:port/v1'."""
    out = []
    for s in specs:
        name, _, url = s.partition("@")
        out.append(LLM(name, url or default_url))
    return out
