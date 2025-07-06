"""Tests for alerting functionality"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.observability.alerting import Alert, WebhookAlerting
from puffinflow.core.observability.config import AlertingConfig
from puffinflow.core.observability.interfaces import AlertSeverity


class TestAlert:
    """Test Alert class"""

    def test_alert_creation(self):
        """Test Alert creation with defaults"""
        alert = Alert("Test message", AlertSeverity.WARNING, {"key": "value"})
        assert alert.message == "Test message"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.attributes == {"key": "value"}
        assert alert.timestamp is not None

    def test_alert_creation_with_timestamp(self):
        """Test Alert creation with explicit timestamp"""
        timestamp = datetime.now()
        alert = Alert("Test message", AlertSeverity.ERROR, {}, timestamp)
        assert alert.timestamp == timestamp

    def test_alert_to_dict(self):
        """Test Alert.to_dict() method"""
        timestamp = datetime.now()
        alert = Alert(
            "Test message", AlertSeverity.CRITICAL, {"key": "value"}, timestamp
        )
        result = alert.to_dict()

        expected = {
            "message": "Test message",
            "severity": "critical",
            "attributes": {"key": "value"},
            "timestamp": timestamp.isoformat(),
        }
        assert result == expected


class TestWebhookAlerting:
    """Test WebhookAlerting class"""

    def test_webhook_alerting_creation(self):
        """Test WebhookAlerting creation"""
        config = AlertingConfig()
        alerting = WebhookAlerting(config)
        assert alerting.config == config

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self):
        """Test send_alert when alerting is disabled"""
        config = AlertingConfig(enabled=False)
        alerting = WebhookAlerting(config)

        # Should not raise any exceptions
        await alerting.send_alert("Test", AlertSeverity.WARNING)

    @pytest.mark.asyncio
    async def test_send_alert_no_webhooks(self):
        """Test send_alert with no webhook URLs"""
        config = AlertingConfig(enabled=True, webhook_urls=[])
        alerting = WebhookAlerting(config)

        # Should not raise any exceptions
        await alerting.send_alert("Test", AlertSeverity.WARNING)

    @pytest.mark.asyncio
    async def test_send_alert_with_webhooks(self):
        """Test send_alert with webhook URLs"""
        config = AlertingConfig(
            enabled=True, webhook_urls=["http://webhook1.com", "http://webhook2.com"]
        )
        alerting = WebhookAlerting(config)

        with patch.object(
            alerting, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            await alerting.send_alert(
                "Test message", AlertSeverity.ERROR, {"key": "value"}
            )

            assert mock_send.call_count == 2
            # Check that both webhooks were called
            calls = mock_send.call_args_list
            assert calls[0][0][0] == "http://webhook1.com"
            assert calls[1][0][0] == "http://webhook2.com"

            # Check payload structure
            payload = calls[0][0][1]
            assert "alert" in payload
            assert payload["alert"]["message"] == "Test message"
            assert payload["alert"]["severity"] == "error"
            assert payload["alert"]["attributes"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_send_webhook_success(self):
        """Test _send_webhook success case"""
        config = AlertingConfig()
        alerting = WebhookAlerting(config)

        mock_response = Mock()
        mock_response.status = 200

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            # Create proper async context manager mock
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_response
            mock_context_manager.__aexit__.return_value = None
            mock_session.post.return_value = mock_context_manager

            # Should not raise any exceptions
            await alerting._send_webhook("http://test.com", {"test": "data"})

    @pytest.mark.asyncio
    async def test_send_webhook_http_error(self):
        """Test _send_webhook with HTTP error"""
        config = AlertingConfig()
        alerting = WebhookAlerting(config)

        mock_response = Mock()
        mock_response.status = 500

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            # Create proper async context manager mock
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_response
            mock_context_manager.__aexit__.return_value = None
            mock_session.post.return_value = mock_context_manager

            with patch("builtins.print") as mock_print:
                await alerting._send_webhook("http://test.com", {"test": "data"})
                mock_print.assert_called_once_with("Webhook failed: 500")

    @pytest.mark.asyncio
    async def test_send_webhook_exception(self):
        """Test _send_webhook with exception"""
        config = AlertingConfig()
        alerting = WebhookAlerting(config)

        with patch("aiohttp.ClientSession", side_effect=Exception("Connection error")):
            with patch("builtins.print") as mock_print:
                await alerting._send_webhook("http://test.com", {"test": "data"})
                mock_print.assert_called_once_with(
                    "Failed to send webhook to http://test.com: Connection error"
                )

    @pytest.mark.asyncio
    async def test_send_webhook_timeout(self):
        """Test _send_webhook with timeout"""
        config = AlertingConfig()
        alerting = WebhookAlerting(config)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.side_effect = Exception("Timeout error")

            with patch("builtins.print") as mock_print:
                await alerting._send_webhook("http://test.com", {"test": "data"})
                mock_print.assert_called_once_with(
                    "Failed to send webhook to http://test.com: Timeout error"
                )

    @pytest.mark.asyncio
    async def test_send_alert_with_none_attributes(self):
        """Test send_alert with None attributes"""
        config = AlertingConfig(enabled=True, webhook_urls=["http://webhook.com"])
        alerting = WebhookAlerting(config)

        with patch.object(
            alerting, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            await alerting.send_alert("Test message", AlertSeverity.INFO, None)

            # Should still work with empty attributes
            mock_send.assert_called_once()
            payload = mock_send.call_args[0][1]
            assert payload["alert"]["attributes"] == {}

    @pytest.mark.asyncio
    async def test_send_alert_multiple_severities(self):
        """Test send_alert with different severity levels"""
        config = AlertingConfig(enabled=True, webhook_urls=["http://webhook.com"])
        alerting = WebhookAlerting(config)

        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]

        with patch.object(
            alerting, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            for severity in severities:
                await alerting.send_alert(f"Test {severity.value}", severity)

            assert mock_send.call_count == len(severities)

            # Check that all severity levels are properly serialized
            for i, severity in enumerate(severities):
                payload = mock_send.call_args_list[i][0][1]
                assert payload["alert"]["severity"] == severity.value
