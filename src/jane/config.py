"""Central configuration — config.yaml + environment variable overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get(d: dict, path: str, default):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class Settings:
    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"
    primary_model: str = "qwen2.5-coder:32b"
    fallback_model: str = "qwen2.5-coder:7b"
    request_timeout: int = 240

    # Sandbox
    code_timeout: int = 30
    max_memory_mb: int = 4096
    max_output_mb: int = 10

    # Retry loop
    max_attempts: int = 5

    # Limits
    max_upload_mb: int = 200
    max_history_messages: int = 12

    # Paths
    uploads_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    history_db: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "history.sqlite3")

    @classmethod
    def load(cls) -> "Settings":
        cfg: dict = {}
        cfg_path = PROJECT_ROOT / "config.yaml"
        if cfg_path.exists() and yaml is not None:
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}

        def env(name: str, dotted: str, default):
            return os.environ.get(name, _get(cfg, dotted, default))

        s = cls(
            ollama_host=str(env("OLLAMA_HOST", "ollama.host", "http://127.0.0.1:11434")),
            primary_model=str(env("ANALYST_MODEL", "ollama.primary_model", "qwen2.5-coder:32b")),
            fallback_model=str(env("ANALYST_FALLBACK_MODEL", "ollama.fallback_model", "qwen2.5-coder:7b")),
            request_timeout=int(env("ANALYST_REQUEST_TIMEOUT", "ollama.request_timeout", 240)),
            code_timeout=int(env("ANALYST_CODE_TIMEOUT", "sandbox.code_timeout", 30)),
            max_memory_mb=int(env("ANALYST_MAX_MEMORY_MB", "sandbox.max_memory_mb", 4096)),
            max_output_mb=int(env("ANALYST_MAX_OUTPUT_MB", "sandbox.max_output_mb", 10)),
            max_attempts=int(env("ANALYST_MAX_ATTEMPTS", "agent.max_attempts", 5)),
            max_upload_mb=int(env("ANALYST_MAX_UPLOAD_MB", "limits.max_upload_mb", 200)),
            max_history_messages=int(env("ANALYST_MAX_HISTORY", "limits.max_history_messages", 12)),
        )
        return s


settings = Settings.load()
