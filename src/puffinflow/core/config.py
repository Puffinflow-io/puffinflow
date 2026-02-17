from functools import lru_cache
from typing import Any, Optional

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    _HAS_PYDANTIC_SETTINGS = False

if _HAS_PYDANTIC_SETTINGS:

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

        # Checkpoint configuration
        checkpoint_format: str = Field(
            default="json", alias="CHECKPOINT_FORMAT"
        )  # json, msgpack, pickle
        checkpoint_backend: str = Field(
            default="file", alias="CHECKPOINT_BACKEND"
        )  # file, memory, redis, postgres, s3
        checkpoint_granularity: str = Field(
            default="per-state", alias="CHECKPOINT_GRANULARITY"
        )  # per-state, on-error

        # Drain configuration
        drain_timeout: float = Field(default=30.0, alias="DRAIN_TIMEOUT")

        # Streaming configuration
        streaming_port: int = Field(default=8080, alias="STREAMING_PORT")

        model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

else:
    # Lightweight fallback when pydantic-settings is not installed
    class Settings:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.app_name: str = kwargs.get("app_name", "PuffinFlow")
            self.environment: str = kwargs.get("environment", "development")
            self.debug: bool = kwargs.get("debug", False)
            self.max_cpu_units: float = kwargs.get("max_cpu_units", 4.0)
            self.max_memory_mb: float = kwargs.get("max_memory_mb", 4096.0)
            self.max_io_weight: float = kwargs.get("max_io_weight", 100.0)
            self.max_network_weight: float = kwargs.get("max_network_weight", 100.0)
            self.max_gpu_units: float = kwargs.get("max_gpu_units", 0.0)
            self.worker_concurrency: int = kwargs.get("worker_concurrency", 10)
            self.worker_timeout: float = kwargs.get("worker_timeout", 300.0)
            self.enable_metrics: bool = kwargs.get("enable_metrics", True)
            self.metrics_port: int = kwargs.get("metrics_port", 9090)
            self.otlp_endpoint: Optional[str] = kwargs.get("otlp_endpoint", None)
            self.enable_scheduling: bool = kwargs.get("enable_scheduling", True)
            self.storage_backend: str = kwargs.get("storage_backend", "sqlite")
            self.checkpoint_interval: int = kwargs.get("checkpoint_interval", 60)
            self.checkpoint_format: str = kwargs.get("checkpoint_format", "json")
            self.checkpoint_backend: str = kwargs.get("checkpoint_backend", "file")
            self.checkpoint_granularity: str = kwargs.get(
                "checkpoint_granularity", "per-state"
            )
            self.drain_timeout: float = kwargs.get("drain_timeout", 30.0)
            self.streaming_port: int = kwargs.get("streaming_port", 8080)


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
