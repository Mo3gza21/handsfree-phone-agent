"""Talk to local models through an OpenAI-compatible server.

Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1) and
llama.cpp's llama-server (http://localhost:8080/v1) all speak this API, so the
pipeline works with whichever one holds your models.

For Ollama, the native /api/chat endpoint is used instead so "thinking" models
(Qwen 3, Gemma 4) can be told to answer directly: thinking out loud first makes
every answer many times slower on a laptop CPU, and gets cut off by max_tokens.
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
    def __init__(self, model: str, base_url: str = KNOWN_SERVERS["ollama"], timeout: int = 600):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.ollama = ":11434" in self.base_url
        self.send_think = True  # dropped if this Ollama version rejects the field

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 400) -> str:
        for attempt in range(3):
            try:
                return self._ollama_chat(messages, temperature, max_tokens) if self.ollama \
                    else self._openai_chat(messages, temperature, max_tokens)
            except (requests.RequestException, KeyError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _openai_chat(self, messages, temperature, max_tokens) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": False}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def _ollama_chat(self, messages, temperature, max_tokens) -> str:
        root = self.base_url.removesuffix("/v1")
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}}
        if self.send_think:
            body["think"] = False
        r = requests.post(f"{root}/api/chat", json=body, timeout=self.timeout)
        if r.status_code == 400 and self.send_think and "think" in r.text:
            self.send_think = False  # model or Ollama version without thinking support
            body.pop("think")
            r = requests.post(f"{root}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"] or ""

    def __repr__(self):
        return self.model


def parse_model_args(specs: list[str], default_url: str) -> list[LLM]:
    """'qwen3:4b' or 'qwen3:4b@http://host:port/v1'."""
    out = []
    for s in specs:
        name, _, url = s.partition("@")
        out.append(LLM(name, url or default_url))
    return out
