import pytest
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from contextlib import contextmanager

import sys
from pathlib import Path

# Add the src directory to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from puffinflow.core.observability.core import (
        ObservabilityManager,
        get_observability,
        setup_observability
    )
except ImportError:
    # Alternative import path if the structure is different
    sys.path.insert(0, str(project_root))
    from src.puffinflow.core.observability.core import (
        ObservabilityManager,
        get_observability,
        setup_observability
    )


# Global fixtures that can be used across all test classes
@pytest.fixture
def mock_config():
    """Mock observability config"""
    config = Mock()
    config.tracing = Mock()
    config.tracing.enabled = True
    config.metrics = Mock()
    config.metrics.enabled = True
    config.alerting = Mock()
    config.alerting.enabled = True
    config.events = Mock()
    config.events.enabled = True
    return config


@pytest.fixture
def mock_providers():
    """Mock all provider classes"""
    with patch('puffinflow.core.observability.core.OpenTelemetryTracingProvider') as mock_tracing, \
            patch('puffinflow.core.observability.core.PrometheusMetricsProvider') as mock_metrics, \
            patch('puffinflow.core.observability.core.WebhookAlerting') as mock_alerting, \
            patch('puffinflow.core.observability.core.BufferedEventProcessor') as mock_events:
        # Setup return values
        mock_tracing_instance = Mock()
        mock_metrics_instance = Mock()
        mock_alerting_instance = Mock()
        mock_events_instance = AsyncMock()

        # Make send_alert async for alerting
        mock_alerting_instance.send_alert = AsyncMock()

        mock_tracing.return_value = mock_tracing_instance
        mock_metrics.return_value = mock_metrics_instance
        mock_alerting.return_value = mock_alerting_instance
        mock_events.return_value = mock_events_instance

        yield {
            'tracing_class': mock_tracing,
            'metrics_class': mock_metrics,
            'alerting_class': mock_alerting,
            'events_class': mock_events,
            'tracing_instance': mock_tracing_instance,
            'metrics_instance': mock_metrics_instance,
            'alerting_instance': mock_alerting_instance,
            'events_instance': mock_events_instance
        }


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test"""
    # Import the module to access its globals
    try:
        import puffinflow.core.observability.core as core_module
    except ImportError:
        import src.puffinflow.core.observability.core as core_module

    # Store original value
    original_global = getattr(core_module, '_global_observability', None)

    # Reset to None
    core_module._global_observability = None

    yield

    # Restore original value
    core_module._global_observability = original_global


class TestObservabilityManager:
    """Test suite for ObservabilityManager class"""

    @patch('puffinflow.core.observability.core.ObservabilityConfig')
    def test_init_without_config(self, mock_config_class):
        """Test initialization without providing config"""
        mock_config_instance = Mock()
        mock_config_class.return_value = mock_config_instance

        manager = ObservabilityManager()

        assert manager.config == mock_config_instance
        assert not manager._initialized
        assert manager._tracing is None
        assert manager._metrics is None
        assert manager._alerting is None
        assert manager._events is None
        assert isinstance(manager._lock, threading.Lock)

    def test_init_with_config(self, mock_config):
        """Test initialization with provided config"""
        manager = ObservabilityManager(mock_config)

        assert manager.config == mock_config
        assert not manager._initialized

    def test_properties_before_initialization(self):
        """Test that properties return None before initialization"""
        manager = ObservabilityManager()

        assert manager.tracing is None
        assert manager.metrics is None
        assert manager.alerting is None
        assert manager.events is None

    @pytest.mark.asyncio
    async def test_initialize_all_enabled(self, mock_config, mock_providers):
        """Test initialization when all providers are enabled"""
        manager = ObservabilityManager(mock_config)

        await manager.initialize()

        assert manager._initialized
        assert manager._tracing == mock_providers['tracing_instance']
        assert manager._metrics == mock_providers['metrics_instance']
        assert manager._alerting == mock_providers['alerting_instance']
        assert manager._events == mock_providers['events_instance']

        # Verify providers were created with correct config
        mock_providers['tracing_class'].assert_called_once_with(mock_config.tracing)
        mock_providers['metrics_class'].assert_called_once_with(mock_config.metrics)
        mock_providers['alerting_class'].assert_called_once_with(mock_config.alerting)
        mock_providers['events_class'].assert_called_once_with(mock_config.events)

        # Verify events processor was initialized
        mock_providers['events_instance'].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_selective_providers(self, mock_providers):
        """Test initialization with only some providers enabled"""
        config = Mock()
        config.tracing = Mock()
        config.tracing.enabled = True
        config.metrics = Mock()
        config.metrics.enabled = False
        config.alerting = Mock()
        config.alerting.enabled = True
        config.events = Mock()
        config.events.enabled = False

        manager = ObservabilityManager(config)
        await manager.initialize()

        assert manager._tracing == mock_providers['tracing_instance']
        assert manager._metrics is None
        assert manager._alerting == mock_providers['alerting_instance']
        assert manager._events is None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_config, mock_providers):
        """Test that initialize can be called multiple times safely"""
        manager = ObservabilityManager(mock_config)

        await manager.initialize()
        await manager.initialize()  # Second call should do nothing

        # Providers should only be created once
        assert mock_providers['tracing_class'].call_count == 1
        assert mock_providers['metrics_class'].call_count == 1
        assert mock_providers['alerting_class'].call_count == 1
        assert mock_providers['events_class'].call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_config, mock_providers):
        """Test shutdown functionality"""
        manager = ObservabilityManager(mock_config)
        await manager.initialize()

        await manager.shutdown()

        assert not manager._initialized
        mock_providers['events_instance'].shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_events(self):
        """Test shutdown when events processor is None"""
        config = Mock()
        config.events = Mock()
        config.events.enabled = False
        config.tracing = Mock()
        config.tracing.enabled = False
        config.metrics = Mock()
        config.metrics.enabled = False
        config.alerting = Mock()
        config.alerting.enabled = False

        manager = ObservabilityManager(config)
        await manager.initialize()

        # Should not raise an exception
        await manager.shutdown()
        assert not manager._initialized

    def test_trace_context_manager_with_tracing(self, mock_providers):
        """Test trace context manager when tracing is available"""
        manager = ObservabilityManager()
        manager._tracing = mock_providers['tracing_instance']

        # Setup mock span context manager
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_providers['tracing_instance'].span.return_value = mock_context_manager

        with manager.trace("test_operation", key="value") as span:
            assert span == mock_span

        mock_providers['tracing_instance'].span.assert_called_once_with("test_operation", key="value")

    def test_trace_context_manager_without_tracing(self):
        """Test trace context manager when tracing is not available"""
        manager = ObservabilityManager()

        with manager.trace("test_operation") as span:
            assert span is None

    def test_counter_with_metrics(self, mock_providers):
        """Test counter creation when metrics provider is available"""
        manager = ObservabilityManager()
        manager._metrics = mock_providers['metrics_instance']
        mock_counter = Mock()
        mock_providers['metrics_instance'].counter.return_value = mock_counter

        result = manager.counter("test_counter", "Test description", {"label": "value"})

        assert result == mock_counter
        mock_providers['metrics_instance'].counter.assert_called_once_with(
            "test_counter", "Test description", {"label": "value"}
        )

    def test_counter_without_metrics(self):
        """Test counter creation when metrics provider is not available"""
        manager = ObservabilityManager()

        result = manager.counter("test_counter")

        assert result is None

    def test_gauge_with_metrics(self, mock_providers):
        """Test gauge creation when metrics provider is available"""
        manager = ObservabilityManager()
        manager._metrics = mock_providers['metrics_instance']
        mock_gauge = Mock()
        mock_providers['metrics_instance'].gauge.return_value = mock_gauge

        result = manager.gauge("test_gauge", "Test description", {"label": "value"})

        assert result == mock_gauge
        mock_providers['metrics_instance'].gauge.assert_called_once_with(
            "test_gauge", "Test description", {"label": "value"}
        )

    def test_gauge_without_metrics(self):
        """Test gauge creation when metrics provider is not available"""
        manager = ObservabilityManager()

        result = manager.gauge("test_gauge")

        assert result is None

    def test_histogram_with_metrics(self, mock_providers):
        """Test histogram creation when metrics provider is available"""
        manager = ObservabilityManager()
        manager._metrics = mock_providers['metrics_instance']
        mock_histogram = Mock()
        mock_providers['metrics_instance'].histogram.return_value = mock_histogram

        result = manager.histogram("test_histogram", "Test description", {"label": "value"})

        assert result == mock_histogram
        mock_providers['metrics_instance'].histogram.assert_called_once_with(
            "test_histogram", "Test description", {"label": "value"}
        )

    def test_histogram_without_metrics(self):
        """Test histogram creation when metrics provider is not available"""
        manager = ObservabilityManager()

        result = manager.histogram("test_histogram")

        assert result is None

    @pytest.mark.asyncio
    async def test_alert_with_alerting(self, mock_providers):
        """Test alert sending when alerting provider is available"""
        manager = ObservabilityManager()
        manager._alerting = mock_providers['alerting_instance']

        # Mock the AlertSeverity import and creation
        with patch('puffinflow.core.observability.interfaces.AlertSeverity') as mock_alert_severity:
            mock_severity = Mock()
            mock_alert_severity.return_value = mock_severity

            await manager.alert("Test alert", "critical", key="value")

            mock_alert_severity.assert_called_once_with("critical")
            mock_providers['alerting_instance'].send_alert.assert_called_once_with(
                "Test alert", mock_severity, {"key": "value"}
            )

    @pytest.mark.asyncio
    async def test_alert_without_alerting(self):
        """Test alert sending when alerting provider is not available"""
        manager = ObservabilityManager()

        # Should not raise an exception
        await manager.alert("Test alert")

    @pytest.mark.asyncio
    async def test_alert_default_severity(self, mock_providers):
        """Test alert with default severity"""
        manager = ObservabilityManager()
        manager._alerting = mock_providers['alerting_instance']

        with patch('puffinflow.core.observability.interfaces.AlertSeverity') as mock_alert_severity:
            await manager.alert("Test alert")
            mock_alert_severity.assert_called_once_with("warning")


class TestGlobalInstanceManagement:
    """Test suite for global instance management functions"""

    @patch('puffinflow.core.observability.core.ObservabilityConfig')
    def test_get_observability_creates_instance(self, mock_config_class):
        """Test that get_observability creates a global instance"""
        mock_config_instance = Mock()
        mock_config_class.return_value = mock_config_instance

        manager = get_observability()

        assert isinstance(manager, ObservabilityManager)
        assert manager.config == mock_config_instance

    def test_get_observability_returns_same_instance(self):
        """Test that get_observability returns the same instance on multiple calls"""
        manager1 = get_observability()
        manager2 = get_observability()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_setup_observability_with_config(self, mock_config):
        """Test setup_observability with custom config"""
        with patch.object(ObservabilityManager, 'initialize', new_callable=AsyncMock) as mock_init:
            manager = await setup_observability(mock_config)

            assert isinstance(manager, ObservabilityManager)
            assert manager.config == mock_config
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_observability_without_config(self):
        """Test setup_observability without config"""
        with patch.object(ObservabilityManager, 'initialize', new_callable=AsyncMock) as mock_init:
            manager = await setup_observability()

            assert isinstance(manager, ObservabilityManager)
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_observability_replaces_global_instance(self):
        """Test that setup_observability replaces the global instance"""
        # Get initial instance
        initial_manager = get_observability()

        # Setup new instance
        with patch.object(ObservabilityManager, 'initialize', new_callable=AsyncMock):
            new_manager = await setup_observability()

        # Verify it's different and becomes the new global instance
        assert new_manager is not initial_manager
        assert get_observability() is new_manager


class TestThreadSafety:
    """Test suite for thread safety"""

    def test_get_observability_thread_safety(self):
        """Test that get_observability is thread-safe"""
        instances = []
        threads = []

        def get_instance():
            instances.append(get_observability())

        # Create multiple threads that call get_observability
        for _ in range(10):
            thread = threading.Thread(target=get_instance)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(instances) == 10
        assert all(instance is instances[0] for instance in instances)

    @pytest.mark.asyncio
    async def test_initialize_thread_safety(self, mock_config, mock_providers):
        """Test that initialize method is thread-safe"""
        manager = ObservabilityManager(mock_config)
        results = []

        async def init_manager():
            await manager.initialize()
            results.append(manager._initialized)

        # Run multiple initializations concurrently
        tasks = [init_manager() for _ in range(5)]
        await asyncio.gather(*tasks)

        # All should report as initialized
        assert all(results)
        assert manager._initialized

        # Providers should only be created once despite multiple init calls
        assert mock_providers['tracing_class'].call_count == 1


class TestErrorHandling:
    """Test suite for error handling scenarios"""

    @pytest.mark.asyncio
    async def test_initialize_with_events_error(self, mock_config, mock_providers):
        """Test initialization when events processor fails"""
        mock_providers['events_instance'].initialize.side_effect = Exception("Events init failed")

        manager = ObservabilityManager(mock_config)

        with pytest.raises(Exception, match="Events init failed"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_shutdown_with_events_error(self, mock_config, mock_providers):
        """Test shutdown when events processor fails"""
        manager = ObservabilityManager(mock_config)
        await manager.initialize()

        mock_providers['events_instance'].shutdown.side_effect = Exception("Shutdown failed")

        with pytest.raises(Exception, match="Shutdown failed"):
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_alert_with_invalid_severity(self, mock_providers):
        """Test alert with invalid severity"""
        manager = ObservabilityManager()
        manager._alerting = mock_providers['alerting_instance']

        with patch('puffinflow.core.observability.interfaces.AlertSeverity') as mock_alert_severity:
            mock_alert_severity.side_effect = ValueError("Invalid severity")

            with pytest.raises(ValueError, match="Invalid severity"):
                await manager.alert("Test alert", "invalid_severity")


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions"""

    def test_trace_with_no_attributes(self):
        """Test trace context manager with no additional attributes"""
        manager = ObservabilityManager()

        with manager.trace("test_operation") as span:
            assert span is None

    def test_metrics_methods_with_empty_strings(self, mock_providers):
        """Test metrics methods with empty string parameters"""
        manager = ObservabilityManager()
        manager._metrics = mock_providers['metrics_instance']

        manager.counter("", "", None)
        manager.gauge("", "", None)
        manager.histogram("", "", None)

        # Should still call the underlying methods
        assert mock_providers['metrics_instance'].counter.called
        assert mock_providers['metrics_instance'].gauge.called
        assert mock_providers['metrics_instance'].histogram.called

    @pytest.mark.asyncio
    async def test_alert_with_empty_message(self, mock_providers):
        """Test alert with empty message"""
        manager = ObservabilityManager()
        manager._alerting = mock_providers['alerting_instance']

        with patch('puffinflow.core.observability.interfaces.AlertSeverity'):
            await manager.alert("")

            assert mock_providers['alerting_instance'].send_alert.called

    def test_properties_after_partial_initialization(self, mock_config):
        """Test properties when only some providers are initialized"""
        manager = ObservabilityManager(mock_config)

        # Manually set only some providers
        manager._tracing = Mock()
        manager._metrics = None

        assert manager.tracing is not None
        assert manager.metrics is None
        assert manager.alerting is None
        assert manager.events is None


if __name__ == "__main__":
    pytest.main([__file__])