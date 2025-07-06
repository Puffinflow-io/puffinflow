# core/config.py
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PuffinFlow"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Resource limits
    max_cpu_units: float = Field(default=4.0, alias="MAX_CPU_UNITS")
    max_memory_mb: float = Field(default=4096.0, alias="MAX_MEMORY_MB")
    max_io_weight: float = Field(default=100.0, alias="MAX_IO_WEIGHT")
    max_network_weight: float = Field(default=100.0, alias="MAX_NETWORK_WEIGHT")
    max_gpu_units: float = Field(default=0.0, alias="MAX_GPU_UNITS")

    # Worker configuration
    worker_concurrency: int = Field(default=10, alias="WORKER_CONCURRENCY")
    worker_timeout: float = Field(default=300.0, alias="WORKER_TIMEOUT")

    # Observability
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    otlp_endpoint: Optional[str] = Field(default=None, alias="OTLP_ENDPOINT")

    # Core features that are implemented
    enable_scheduling: bool = Field(default=True, alias="ENABLE_SCHEDULING")

    # Storage and checkpointing
    storage_backend: str = Field(default="sqlite", alias="STORAGE_BACKEND")
    checkpoint_interval: int = Field(default=60, alias="CHECKPOINT_INTERVAL")

    class Config:
        env_file = ".env"
        case_sensitive = False


class Features:
    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def scheduling(self) -> bool:
        return self._settings.enable_scheduling

    @property
    def metrics(self) -> bool:
        return self._settings.enable_metrics


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_features() -> Features:
    return Features(get_settings())
