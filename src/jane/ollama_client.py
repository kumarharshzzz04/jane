"""Thin Ollama chat client (stdlib only, no ollama package needed)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .logging_setup import get_logger
from .types import ChatMessage

log = get_logger("jane.ollama")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, timeout: int = 240):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- low level ---------------------------------------------------------

    def _post(self, path: str, payload: dict, stream: bool = False, on_line=None):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if not stream:
                    return json.loads(resp.read().decode("utf-8"))
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        obj = json.loads(line)
                        if on_line:
                            on_line(obj)
                return None
        except urllib.error.HTTPError as e:  # e.g. 404 model missing
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise OllamaError(f"Ollama HTTP {e.code} on {path}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise OllamaError(f"Ollama unreachable at {self.host}: {e}") from e

    # -- public ------------------------------------------------------------

    def tags(self) -> list[dict]:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as r:
                return json.loads(r.read().decode("utf-8")).get("models", [])
        except Exception as e:
            raise OllamaError(f"Ollama unreachable at {self.host}: {e}") from e

    def has_model(self, model: str) -> bool:
        try:
            models = self.tags()
        except OllamaError:
            return False
        wanted = model.split(":")[0]
        return any(m.get("name") == model or m.get("name", "").split(":")[0] == wanted for m in models)

    def chat(self, messages: list[ChatMessage], model: str, temperature: float = 0.1,
             num_ctx: int = 8192) -> str:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "format": "json",  # structured output: model MUST emit a JSON object
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        data = self._post("/api/chat", payload)
        content = (data or {}).get("message", {}).get("content", "")
        return content.strip()

    def generate(self, prompt: str, model: str, system: str | None = None,
                 temperature: float = 0.1, num_ctx: int = 8192) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if system:
            payload["system"] = system
        data = self._post("/api/generate", payload)
        return ((data or {}).get("response") or "").strip()
