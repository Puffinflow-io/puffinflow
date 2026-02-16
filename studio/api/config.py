"""PuffinFlow Studio API configuration."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class StudioSettings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    db_url: str = "sqlite+aiosqlite:///studio.db"
    cors_origins: list[str] = ["http://localhost:3000"]
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "STUDIO_"}


settings = StudioSettings()
