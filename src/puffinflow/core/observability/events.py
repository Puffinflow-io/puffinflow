import asyncio
import time
from collections import deque
from typing import Callable, Optional

from .interfaces import EventProcessor, ObservabilityEvent
from .config import EventsConfig


class BufferedEventProcessor(EventProcessor):
    """Buffered event processor"""

    def __init__(self, config: EventsConfig):
        self.config = config
        self.buffer = deque(maxlen=config.buffer_size)
        self.subscribers = []
        self._task = None
        self._shutdown = False

    async def initialize(self):
        """Initialize event processor"""
        if self.config.enabled:
            self._task = asyncio.create_task(self._process_loop())

    async def shutdown(self):
        """Shutdown event processor"""
        self._shutdown = True
        if self._task:
            self._task.cancel()

    async def process_event(self, event: ObservabilityEvent) -> None:
        """Add event to buffer"""
        if self.config.enabled:
            self.buffer.append(event)

    def subscribe(self, callback: Callable[[ObservabilityEvent], None]):
        """Subscribe to events"""
        self.subscribers.append(callback)

    async def _process_loop(self):
        """Process events from buffer"""
        while not self._shutdown:
            try:
                events_to_process = []

                # Collect batch
                for _ in range(min(self.config.batch_size, len(self.buffer))):
                    if self.buffer:
                        events_to_process.append(self.buffer.popleft())

                # Process events
                for event in events_to_process:
                    for subscriber in self.subscribers:
                        try:
                            if asyncio.iscoroutinefunction(subscriber):
                                await subscriber(event)
                            else:
                                subscriber(event)
                        except Exception as e:
                            print(f"Event processing error: {e}")

                # Wait for next batch
                await asyncio.sleep(self.config.flush_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Event processing loop error: {e}")
                await asyncio.sleep(1)