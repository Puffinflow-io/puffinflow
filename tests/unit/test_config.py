"""Tests for core configuration module."""

import os
from unittest.mock import patch

from puffinflow.core.config import Features, Settings, get_features, get_settings


class TestSettings:
    """Test Settings class."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = Settings()

        assert settings.app_name == "PuffinFlow"
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.worker_concurrency == 10
        assert settings.worker_timeout == 300.0
        assert settings.max_cpu_units == 4.0
        assert settings.max_memory_mb == 4096.0
        assert settings.max_io_weight == 100.0
        assert settings.max_network_weight == 100.0
        assert settings.max_gpu_units == 0.0
        assert settings.enable_metrics is True
        assert settings.metrics_port == 9090
        assert settings.otlp_endpoint is None
        assert settings.enable_scheduling is True
        assert settings.storage_backend == "sqlite"
        assert settings.checkpoint_interval == 60

    def test_environment_variables(self):
        """Test configuration from environment variables."""
        env_vars = {
            "ENVIRONMENT": "production",
            "DEBUG": "true",
            "OTLP_ENDPOINT": "http://localhost:4317",
            "WORKER_CONCURRENCY": "20",
            "WORKER_TIMEOUT": "600.0",
            "MAX_CPU_UNITS": "8.0",
            "MAX_MEMORY_MB": "8192.0",
            "MAX_IO_WEIGHT": "200.0",
            "MAX_NETWORK_WEIGHT": "200.0",
            "MAX_GPU_UNITS": "2.0",
            "ENABLE_METRICS": "false",
            "METRICS_PORT": "9091",
            "ENABLE_SCHEDULING": "false",
            "STORAGE_BACKEND": "postgresql",
            "CHECKPOINT_INTERVAL": "120",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.environment == "production"
            assert settings.debug is True
            assert settings.otlp_endpoint == "http://localhost:4317"
            assert settings.worker_concurrency == 20
            assert settings.worker_timeout == 600.0
            assert settings.max_cpu_units == 8.0
            assert settings.max_memory_mb == 8192.0
            assert settings.max_io_weight == 200.0
            assert settings.max_network_weight == 200.0
            assert settings.max_gpu_units == 2.0
            assert settings.enable_metrics is False
            assert settings.metrics_port == 9091
            assert settings.enable_scheduling is False
            assert settings.storage_backend == "postgresql"
            assert settings.checkpoint_interval == 120

    def test_config_class_attributes(self):
        """Test Config class attributes."""
        settings = Settings()
        config = settings.Config

        assert config.env_file == ".env"
        assert config.case_sensitive is False


class TestFeatures:
    """Test Features class."""

    def test_default_features(self):
        """Test default feature flags."""
        settings = Settings()
        features = Features(settings)

        assert features.scheduling is True
        assert features.metrics is True

    def test_enabled_features(self):
        """Test enabled feature flags."""
        env_vars = {
            "ENABLE_SCHEDULING": "false",
            "ENABLE_METRICS": "false",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()
            features = Features(settings)

            assert features.scheduling is False
            assert features.metrics is False


class TestCachedFunctions:
    """Test cached configuration functions."""

    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_features_caching(self):
        """Test that get_features returns cached instance."""
        features1 = get_features()
        features2 = get_features()

        assert features1 is features2

    def test_get_features_uses_cached_settings(self):
        """Test that get_features uses the cached settings."""
        features = get_features()
        settings = get_settings()

        assert features._settings is settings

    @patch("puffinflow.core.config.get_settings")
    def test_get_features_calls_get_settings(self, mock_get_settings):
        """Test that get_features calls get_settings."""
        mock_settings = Settings()
        mock_get_settings.return_value = mock_settings

        # Clear the cache first
        get_features.cache_clear()

        features = get_features()

        mock_get_settings.assert_called_once()
        assert features._settings is mock_settings


class TestIntegration:
    """Integration tests for configuration."""

    def test_full_configuration_flow(self):
        """Test complete configuration flow."""
        env_vars = {
            "ENVIRONMENT": "production",
            "ENABLE_SCHEDULING": "false",
            "STORAGE_BACKEND": "postgresql",
        }

        with patch.dict(os.environ, env_vars):
            # Clear caches to ensure fresh instances
            get_settings.cache_clear()
            get_features.cache_clear()

            settings = get_settings()
            features = get_features()

            # Verify settings
            assert settings.environment == "production"
            assert settings.storage_backend == "postgresql"

            # Verify features
            assert features.scheduling is False

            # Verify caching
            assert get_settings() is settings
            assert get_features() is features

    def teardown_method(self):
        """Clean up after each test."""
        # Clear caches to avoid test interference
        get_settings.cache_clear()
        get_features.cache_clear()
