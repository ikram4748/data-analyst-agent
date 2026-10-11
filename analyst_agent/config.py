"""All settings in one place. Values come from `.env` (see `.env.example`) with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # Every non-empty key is used; each key has its own free-tier quota.
    api_keys: list[str] = field(default_factory=lambda: [
        k for k in (os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API_KEY_2", "")) if k.strip()])

    # Models tried in order. Flash-Lite has the largest free daily quota, so it does the work;
    # the others only answer when it is rate-limited or unavailable.
    models: list[str] = field(default_factory=lambda: _list(
        "MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite"))

    # Free tier allows roughly 10-15 requests/minute per key and model; stay just under it.
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "12"))

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    query_timeout_s: float = float(os.getenv("QUERY_TIMEOUT_S", "30"))
    max_rows_shown: int = int(os.getenv("MAX_ROWS_SHOWN", "200"))

    # default_factory: read the environment when Settings is created, not when this module is imported
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT / "data"))))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", str(ROOT / ".cache"))))

    @property
    def bird_dir(self) -> Path:
        return self.data_dir / "bird"


@lru_cache
def get_settings() -> Settings:
    return Settings()
