"""Simple tests for observability core module."""

import asyncio

import pytest

from puffinflow.core.observability.config import ObservabilityConfig
from puffinflow.core.observability.core import (
    ObservabilityManager,
    get_observability,
    setup_observability,
)


class TestObservabilityManager:
    """Test ObservabilityManager functionality."""

    def test_manager_initialization_default(self):
        """Test manager initialization with default config."""
        manager = ObservabilityManager()
        assert manager.config is not None
        assert manager._initialized is False

    def test_manager_initialization_custom_config(self):
        """Test manager initialization with custom config."""
        config = ObservabilityConfig()
        manager = ObservabilityManager(config)
        assert manager.config == config
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_manager(self):
        """Test manager initialization."""
        config = ObservabilityConfig()
        manager = ObservabilityManager(config)

        await manager.initialize()
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_manager(self):
        """Test manager shutdown."""
        manager = ObservabilityManager()
        await manager.initialize()

        await manager.shutdown()
        assert manager._initialized is False

    def test_tracing_property(self):
        """Test tracing property."""
        manager = ObservabilityManager()
        # Initially None
        assert manager.tracing is None

    def test_metrics_property(self):
        """Test metrics property."""
        manager = ObservabilityManager()
        # Initially None
        assert manager.metrics is None

    def test_alerting_property(self):
        """Test alerting property."""
        manager = ObservabilityManager()
        # Initially None
        assert manager.alerting is None

    def test_events_property(self):
        """Test events property."""
        manager = ObservabilityManager()
        # Initially None
        assert manager.events is None

    def test_trace_context_manager_no_tracing(self):
        """Test trace context manager when tracing is disabled."""
        manager = ObservabilityManager()

        with manager.trace("test_operation") as span:
            assert span is None

    def test_counter_no_metrics(self):
        """Test counter creation when metrics is disabled."""
        manager = ObservabilityManager()

        counter = manager.counter("test_counter")
        assert counter is None

    def test_gauge_no_metrics(self):
        """Test gauge creation when metrics is disabled."""
        manager = ObservabilityManager()

        gauge = manager.gauge("test_gauge")
        assert gauge is None

    def test_histogram_no_metrics(self):
        """Test histogram creation when metrics is disabled."""
        manager = ObservabilityManager()

        histogram = manager.histogram("test_histogram")
        assert histogram is None

    @pytest.mark.asyncio
    async def test_alert_no_alerting(self):
        """Test alert sending when alerting is disabled."""
        manager = ObservabilityManager()

        # Should not raise error
        await manager.alert("Test alert", "warning")


class TestGlobalObservability:
    """Test global observability functions."""

    def test_get_observability_singleton(self):
        """Test global observability singleton."""
        obs1 = get_observability()
        obs2 = get_observability()

        assert obs1 is obs2
        assert isinstance(obs1, ObservabilityManager)

    @pytest.mark.asyncio
    async def test_setup_observability(self):
        """Test observability setup."""
        config = ObservabilityConfig()
        manager = await setup_observability(config)

        assert isinstance(manager, ObservabilityManager)
        assert manager.config == config
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_setup_observability_default_config(self):
        """Test observability setup with default config."""
        manager = await setup_observability()

        assert isinstance(manager, ObservabilityManager)
        assert manager._initialized is True


class TestObservabilityConfig:
    """Test ObservabilityConfig functionality."""

    def test_config_initialization(self):
        """Test config initialization."""
        config = ObservabilityConfig()

        # Should have default values
        assert hasattr(config, "tracing")
        assert hasattr(config, "metrics")
        assert hasattr(config, "alerting")
        assert hasattr(config, "events")

    def test_config_attributes(self):
        """Test config has expected attributes."""
        config = ObservabilityConfig()

        # Check that config has the expected structure
        assert hasattr(config, "tracing")
        assert hasattr(config, "metrics")
        assert hasattr(config, "alerting")
        assert hasattr(config, "events")


class TestObservabilityIntegration:
    """Test observability integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full observability lifecycle."""
        config = ObservabilityConfig()
        manager = ObservabilityManager(config)

        # Initialize
        await manager.initialize()
        assert manager._initialized is True

        # Use features (should not error even if disabled)
        with manager.trace("test_op"):
            pass

        manager.counter("test_counter")
        manager.gauge("test_gauge")
        manager.histogram("test_histogram")

        await manager.alert("Test alert")

        # Shutdown
        await manager.shutdown()
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_multiple_initialization(self):
        """Test multiple initialization calls."""
        manager = ObservabilityManager()

        await manager.initialize()
        assert manager._initialized is True

        # Second initialization should be safe
        await manager.initialize()
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_before_init(self):
        """Test shutdown before initialization."""
        manager = ObservabilityManager()

        # Should not error
        await manager.shutdown()
        assert manager._initialized is False

    def test_thread_safety(self):
        """Test thread safety of initialization."""
        manager = ObservabilityManager()

        # Multiple threads trying to initialize should be safe
        import threading

        def init_manager():
            asyncio.run(manager.initialize())

        threads = [threading.Thread(target=init_manager) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert manager._initialized is True
