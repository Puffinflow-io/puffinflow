import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import aiohttp

from .config import AlertingConfig
from .interfaces import AlertingProvider, AlertSeverity


@dataclass
class Alert:
    """Alert data structure"""

    message: str
    severity: AlertSeverity
    attributes: dict[str, Any]
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()
        else:
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "severity": self.severity.value,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class WebhookAlerting(AlertingProvider):
    """Webhook-based alerting"""

    def __init__(self, config: AlertingConfig):
        self.config = config

    async def send_alert(
        self,
        message: str,
        severity: AlertSeverity,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send alert via webhooks"""
        if not self.config.enabled or not self.config.webhook_urls:
            return

        alert = Alert(message, severity, attributes or {})
        payload = {"alert": alert.to_dict()}

        tasks = []
        for webhook_url in self.config.webhook_urls:
            task = asyncio.create_task(self._send_webhook(webhook_url, payload))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_webhook(self, url: str, payload: dict[str, Any]) -> None:
        """Send single webhook"""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, json=payload, timeout=30) as response,
            ):
                if response.status >= 400:
                    print(f"Webhook failed: {response.status}")
        except Exception as e:
            print(f"Failed to send webhook to {url}: {e}")
