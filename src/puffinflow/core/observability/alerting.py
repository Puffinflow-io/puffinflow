import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import aiohttp

from .interfaces import AlertingProvider, AlertSeverity
from .config import AlertingConfig


@dataclass
class Alert:
    """Alert data structure"""
    message: str
    severity: AlertSeverity
    attributes: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "severity": self.severity.value,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat()
        }


class WebhookAlerting(AlertingProvider):
    """Webhook-based alerting"""

    def __init__(self, config: AlertingConfig):
        self.config = config

    async def send_alert(self, message: str, severity: AlertSeverity,
                         attributes: Dict[str, Any] = None) -> None:
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

    async def _send_webhook(self, url: str, payload: Dict[str, Any]):
        """Send single webhook"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status >= 400:
                        print(f"Webhook failed: {response.status}")
        except Exception as e:
            print(f"Failed to send webhook to {url}: {e}")