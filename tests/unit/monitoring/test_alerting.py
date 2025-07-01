import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any
from aiohttp import ClientError, ClientTimeout

from src.puffinflow.core.observability.alerting import Alert, WebhookAlerting, AlertSeverity
from src.puffinflow.core.observability.alerting import AlertingConfig


class AsyncContextManagerMock:
    """Custom mock for async context managers"""

    def __init__(self, return_value=None):
        self.return_value = return_value or AsyncMock()

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestAlert:
    """Test cases for Alert class"""

    def test_alert_creation_with_timestamp(self):
        """Test alert creation with explicit timestamp"""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        alert = Alert(
            message="Test alert",
            severity=AlertSeverity.ERROR,
            attributes={"key": "value"},
            timestamp=timestamp
        )

        assert alert.message == "Test alert"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.attributes == {"key": "value"}
        assert alert.timestamp == timestamp

    @patch('src.puffinflow.core.observability.alerting.datetime')
    def test_alert_creation_auto_timestamp(self, mock_datetime):
        """Test alert creation with automatic timestamp"""
        mock_now = datetime(2024, 1, 15, 10, 30, 0)
        mock_datetime.now.return_value = mock_now

        alert = Alert(
            message="Test alert",
            severity=AlertSeverity.WARNING,
            attributes={"source": "test"}
        )

        assert alert.timestamp == mock_now
        mock_datetime.now.assert_called_once()

    def test_alert_creation_empty_attributes(self):
        """Test alert creation with empty attributes"""
        alert = Alert(
            message="Test alert",
            severity=AlertSeverity.INFO,
            attributes={}
        )

        assert alert.attributes == {}

    def test_alert_to_dict(self):
        """Test alert serialization to dictionary"""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        alert = Alert(
            message="Test alert",
            severity=AlertSeverity.CRITICAL,
            attributes={"key": "value", "count": 42},
            timestamp=timestamp
        )

        result = alert.to_dict()

        expected = {
            "message": "Test alert",
            "severity": AlertSeverity.CRITICAL.value,
            "attributes": {"key": "value", "count": 42},
            "timestamp": "2024-01-15T10:30:00"
        }

        assert result == expected

    def test_alert_to_dict_with_complex_attributes(self):
        """Test alert serialization with complex attributes"""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        alert = Alert(
            message="Complex alert",
            severity=AlertSeverity.ERROR,
            attributes={
                "nested": {"inner": "value"},
                "list": [1, 2, 3],
                "boolean": True,
                "null_value": None
            },
            timestamp=timestamp
        )

        result = alert.to_dict()

        assert result["attributes"]["nested"] == {"inner": "value"}
        assert result["attributes"]["list"] == [1, 2, 3]
        assert result["attributes"]["boolean"] is True
        assert result["attributes"]["null_value"] is None


class TestWebhookAlerting:
    """Test cases for WebhookAlerting class"""

    @pytest.fixture
    def mock_config_enabled(self):
        """Fixture for enabled alerting configuration"""
        config = Mock(spec=AlertingConfig)
        config.enabled = True
        config.webhook_urls = ["http://webhook1.com", "http://webhook2.com"]
        return config

    @pytest.fixture
    def mock_config_disabled(self):
        """Fixture for disabled alerting configuration"""
        config = Mock(spec=AlertingConfig)
        config.enabled = False
        config.webhook_urls = ["http://webhook1.com"]
        return config

    @pytest.fixture
    def mock_config_no_urls(self):
        """Fixture for configuration with no webhook URLs"""
        config = Mock(spec=AlertingConfig)
        config.enabled = True
        config.webhook_urls = []
        return config

    def test_webhook_alerting_initialization(self, mock_config_enabled):
        """Test WebhookAlerting initialization"""
        alerting = WebhookAlerting(mock_config_enabled)
        assert alerting.config == mock_config_enabled

    @pytest.mark.asyncio
    async def test_send_alert_disabled_config(self, mock_config_disabled):
        """Test send_alert with disabled configuration"""
        alerting = WebhookAlerting(mock_config_disabled)

        with patch.object(alerting, '_send_webhook') as mock_send:
            await alerting.send_alert("Test message", AlertSeverity.ERROR)
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alert_no_webhook_urls(self, mock_config_no_urls):
        """Test send_alert with no webhook URLs"""
        alerting = WebhookAlerting(mock_config_no_urls)

        with patch.object(alerting, '_send_webhook') as mock_send:
            await alerting.send_alert("Test message", AlertSeverity.ERROR)
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.puffinflow.core.observability.alerting.datetime')
    async def test_send_alert_success_single_webhook(self, mock_datetime, mock_config_enabled):
        """Test successful alert sending to single webhook"""
        mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30, 0)
        mock_config_enabled.webhook_urls = ["http://webhook1.com"]

        alerting = WebhookAlerting(mock_config_enabled)

        with patch.object(alerting, '_send_webhook', new_callable=AsyncMock) as mock_send:
            await alerting.send_alert(
                "Test message",
                AlertSeverity.WARNING,
                {"source": "test"}
            )

            mock_send.assert_called_once()
            call_args = mock_send.call_args[0]

            assert call_args[0] == "http://webhook1.com"
            payload = call_args[1]
            assert payload["alert"]["message"] == "Test message"
            assert payload["alert"]["severity"] == AlertSeverity.WARNING.value
            assert payload["alert"]["attributes"] == {"source": "test"}

    @pytest.mark.asyncio
    @patch('src.puffinflow.core.observability.alerting.datetime')
    async def test_send_alert_success_multiple_webhooks(self, mock_datetime, mock_config_enabled):
        """Test successful alert sending to multiple webhooks"""
        mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        alerting = WebhookAlerting(mock_config_enabled)

        with patch.object(alerting, '_send_webhook', new_callable=AsyncMock) as mock_send:
            await alerting.send_alert("Test message", AlertSeverity.CRITICAL)

            assert mock_send.call_count == 2
            calls = mock_send.call_args_list

            # Verify both webhook URLs were called
            urls_called = [call[0][0] for call in calls]
            assert "http://webhook1.com" in urls_called
            assert "http://webhook2.com" in urls_called

    @pytest.mark.asyncio
    async def test_send_alert_with_none_attributes(self, mock_config_enabled):
        """Test send_alert with None attributes"""
        mock_config_enabled.webhook_urls = ["http://webhook1.com"]
        alerting = WebhookAlerting(mock_config_enabled)

        with patch.object(alerting, '_send_webhook', new_callable=AsyncMock) as mock_send:
            await alerting.send_alert("Test message", AlertSeverity.INFO, None)

            call_args = mock_send.call_args[0]
            payload = call_args[1]
            assert payload["alert"]["attributes"] == {}

    @pytest.mark.asyncio
    async def test_send_webhook_success(self, mock_config_enabled):
        """Test successful webhook sending"""
        alerting = WebhookAlerting(mock_config_enabled)

        # Create mock response
        mock_response = Mock()
        mock_response.status = 200

        # Create response context manager
        mock_response_cm = AsyncContextManagerMock(mock_response)

        # Create mock session
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_response_cm)

        # Create session context manager
        mock_session_cm = AsyncContextManagerMock(mock_session)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await alerting._send_webhook(
                "http://webhook1.com",
                {"alert": {"message": "test"}}
            )

            mock_session.post.assert_called_once_with(
                "http://webhook1.com",
                json={"alert": {"message": "test"}},
                timeout=30
            )

    @pytest.mark.asyncio
    async def test_send_webhook_http_error(self, mock_config_enabled, capsys):
        """Test webhook sending with HTTP error response"""
        alerting = WebhookAlerting(mock_config_enabled)

        # Create mock response with error status
        mock_response = Mock()
        mock_response.status = 500

        # Create response context manager
        mock_response_cm = AsyncContextManagerMock(mock_response)

        # Create mock session
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_response_cm)

        # Create session context manager
        mock_session_cm = AsyncContextManagerMock(mock_session)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await alerting._send_webhook(
                "http://webhook1.com",
                {"alert": {"message": "test"}}
            )

            captured = capsys.readouterr()
            assert "Webhook failed: 500" in captured.out

    @pytest.mark.asyncio
    async def test_send_webhook_network_exception(self, mock_config_enabled, capsys):
        """Test webhook sending with network exception"""
        alerting = WebhookAlerting(mock_config_enabled)

        # Create mock session that raises exception on post
        mock_session = Mock()
        mock_session.post = Mock(side_effect=ClientError("Network error"))

        # Create session context manager
        mock_session_cm = AsyncContextManagerMock(mock_session)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await alerting._send_webhook(
                "http://webhook1.com",
                {"alert": {"message": "test"}}
            )

            captured = capsys.readouterr()
            assert "Failed to send webhook to http://webhook1.com: Network error" in captured.out

    @pytest.mark.asyncio
    async def test_send_webhook_timeout_exception(self, mock_config_enabled, capsys):
        """Test webhook sending with timeout exception"""
        alerting = WebhookAlerting(mock_config_enabled)

        mock_session = Mock()
        mock_session.post = Mock(side_effect=asyncio.TimeoutError("Request timeout"))

        mock_session_cm = AsyncContextManagerMock(mock_session)

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await alerting._send_webhook(
                "http://webhook1.com",
                {"alert": {"message": "test"}}
            )

            captured = capsys.readouterr()
            assert "Failed to send webhook to http://webhook1.com" in captured.out

    @pytest.mark.asyncio
    async def test_send_alert_partial_webhook_failure(self, mock_config_enabled):
        """Test alert sending where some webhooks fail"""
        alerting = WebhookAlerting(mock_config_enabled)

        async def mock_send_webhook(url, payload):
            if "webhook1" in url:
                raise Exception("Webhook 1 failed")
            # webhook2 succeeds silently

        with patch.object(alerting, '_send_webhook', side_effect=mock_send_webhook):
            # Should not raise exception even if some webhooks fail
            await alerting.send_alert("Test message", AlertSeverity.ERROR)

    @pytest.mark.asyncio
    async def test_send_alert_all_severities(self, mock_config_enabled):
        """Test alert sending with all severity levels"""
        mock_config_enabled.webhook_urls = ["http://webhook1.com"]
        alerting = WebhookAlerting(mock_config_enabled)

        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL
        ]

        with patch.object(alerting, '_send_webhook', new_callable=AsyncMock) as mock_send:
            for severity in severities:
                await alerting.send_alert(f"Test {severity.value}", severity)

            assert mock_send.call_count == len(severities)

    @pytest.mark.asyncio
    async def test_concurrent_webhook_calls(self, mock_config_enabled):
        """Test that multiple webhooks are called concurrently"""
        alerting = WebhookAlerting(mock_config_enabled)

        call_order = []

        async def mock_send_webhook(url, payload):
            call_order.append(f"start_{url}")
            await asyncio.sleep(0.1)  # Simulate network delay
            call_order.append(f"end_{url}")

        with patch.object(alerting, '_send_webhook', side_effect=mock_send_webhook):
            await alerting.send_alert("Test message", AlertSeverity.ERROR)

        # Verify that both webhooks started before either finished (concurrent execution)
        start_indices = [i for i, call in enumerate(call_order) if call.startswith("start_")]
        end_indices = [i for i, call in enumerate(call_order) if call.startswith("end_")]

        assert len(start_indices) == 2
        assert len(end_indices) == 2
        assert max(start_indices) < min(end_indices)


# Integration-style tests
class TestWebhookAlertingIntegration:
    """Integration tests for WebhookAlerting"""

    @pytest.mark.asyncio
    async def test_full_alert_flow(self):
        """Test complete alert flow from creation to webhook sending"""
        # Setup
        mock_response = Mock()
        mock_response.status = 200

        mock_response_cm = AsyncContextManagerMock(mock_response)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_response_cm)

        mock_session_cm = AsyncContextManagerMock(mock_session)

        config = Mock(spec=AlertingConfig)
        config.enabled = True
        config.webhook_urls = ["http://test-webhook.com"]

        alerting = WebhookAlerting(config)

        # Execute
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await alerting.send_alert(
                "Database connection failed",
                AlertSeverity.CRITICAL,
                {
                    "database": "user_db",
                    "error_code": 1044,
                    "retry_count": 3
                }
            )

        # Verify
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args

        assert call_args[0][0] == "http://test-webhook.com"
        assert call_args[1]["timeout"] == 30

        payload = call_args[1]["json"]
        alert_data = payload["alert"]

        assert alert_data["message"] == "Database connection failed"
        assert alert_data["severity"] == AlertSeverity.CRITICAL.value
        assert alert_data["attributes"]["database"] == "user_db"
        assert alert_data["attributes"]["error_code"] == 1044
        assert alert_data["attributes"]["retry_count"] == 3
        assert "timestamp" in alert_data


# Fixtures for pytest configuration
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()