"""Tests for BufferedEventProcessor"""

import asyncio
import contextlib
from collections import deque
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.observability.config import EventsConfig
from puffinflow.core.observability.events import BufferedEventProcessor
from puffinflow.core.observability.interfaces import ObservabilityEvent


class TestBufferedEventProcessor:
    """Test BufferedEventProcessor class"""

    def test_buffered_event_processor_creation(self):
        """Test BufferedEventProcessor creation"""
        config = EventsConfig(
            enabled=True, buffer_size=100, batch_size=10, flush_interval=1.0
        )
        processor = BufferedEventProcessor(config)

        assert processor.config == config
        assert isinstance(processor.buffer, deque)
        assert processor.buffer.maxlen == 100
        assert processor.subscribers == []
        assert processor._task is None
        assert processor._shutdown is False

    def test_buffered_event_processor_disabled(self):
        """Test BufferedEventProcessor when disabled"""
        config = EventsConfig(enabled=False)
        processor = BufferedEventProcessor(config)

        assert processor.config == config
        assert not processor.config.enabled

    @pytest.mark.asyncio
    async def test_initialize_enabled(self):
        """Test initialize when enabled"""
        config = EventsConfig(enabled=True, flush_interval=0.1)
        processor = BufferedEventProcessor(config)

        with patch("asyncio.create_task") as mock_create_task:
            await processor.initialize()
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_disabled(self):
        """Test initialize when disabled"""
        config = EventsConfig(enabled=False)
        processor = BufferedEventProcessor(config)

        with patch("asyncio.create_task") as mock_create_task:
            await processor.initialize()
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutdown"""
        config = EventsConfig(enabled=True)
        processor = BufferedEventProcessor(config)

        # Create an actual task to mock
        async def dummy_task():
            await asyncio.sleep(0.1)

        mock_task = asyncio.create_task(dummy_task())
        processor._task = mock_task

        await processor.shutdown()

        assert processor._shutdown is True
        assert mock_task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_no_task(self):
        """Test shutdown without task"""
        config = EventsConfig(enabled=True)
        processor = BufferedEventProcessor(config)

        await processor.shutdown()

        assert processor._shutdown is True

    @pytest.mark.asyncio
    async def test_process_event_enabled(self):
        """Test process_event when enabled"""
        config = EventsConfig(enabled=True)
        processor = BufferedEventProcessor(config)

        event = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test",
            source="test_source",
            level="info",
            message="Test event",
            attributes={"key": "value"},
        )

        await processor.process_event(event)

        assert len(processor.buffer) == 1
        assert processor.buffer[0] == event

    @pytest.mark.asyncio
    async def test_process_event_disabled(self):
        """Test process_event when disabled"""
        config = EventsConfig(enabled=False)
        processor = BufferedEventProcessor(config)

        event = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test",
            source="test_source",
            level="info",
            message="Test event",
            attributes={"key": "value"},
        )

        await processor.process_event(event)

        assert len(processor.buffer) == 0

    def test_subscribe(self):
        """Test subscribe"""
        config = EventsConfig(enabled=True)
        processor = BufferedEventProcessor(config)

        callback = Mock()
        processor.subscribe(callback)

        assert len(processor.subscribers) == 1
        assert processor.subscribers[0] == callback

    @pytest.mark.asyncio
    async def test_process_loop_sync_callback(self):
        """Test _process_loop with sync callback"""
        config = EventsConfig(enabled=True, batch_size=1, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        callback = Mock()
        processor.subscribe(callback)

        event = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test",
            source="test_source",
            level="info",
            message="Test event",
            attributes={"key": "value"},
        )

        # Add event to buffer
        processor.buffer.append(event)

        # Start the process loop
        task = asyncio.create_task(processor._process_loop())

        # Wait a bit for processing
        await asyncio.sleep(0.05)

        # Stop the loop
        processor._shutdown = True
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that callback was called
        callback.assert_called_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_async_callback(self):
        """Test _process_loop with async callback"""
        config = EventsConfig(enabled=True, batch_size=1, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        callback = AsyncMock()
        processor.subscribe(callback)

        event = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test",
            source="test_source",
            level="info",
            message="Test event",
            attributes={"key": "value"},
        )

        # Add event to buffer
        processor.buffer.append(event)

        # Start the process loop
        task = asyncio.create_task(processor._process_loop())

        # Wait a bit for processing
        await asyncio.sleep(0.05)

        # Stop the loop
        processor._shutdown = True
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that callback was called
        callback.assert_called_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_callback_exception(self):
        """Test _process_loop with callback exception"""
        config = EventsConfig(enabled=True, batch_size=1, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        callback = Mock(side_effect=Exception("Callback error"))
        processor.subscribe(callback)

        event = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test",
            source="test_source",
            level="info",
            message="Test event",
            attributes={"key": "value"},
        )

        # Add event to buffer
        processor.buffer.append(event)

        with patch("builtins.print") as mock_print:
            # Start the process loop
            task = asyncio.create_task(processor._process_loop())

            # Wait a bit for processing
            await asyncio.sleep(0.05)

            # Stop the loop
            processor._shutdown = True
            task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await task

            # Check that error was printed
            mock_print.assert_called_with("Event processing error: Callback error")

    @pytest.mark.asyncio
    async def test_process_loop_multiple_events(self):
        """Test _process_loop with multiple events"""
        config = EventsConfig(enabled=True, batch_size=2, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        callback = Mock()
        processor.subscribe(callback)

        event1 = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test1",
            source="test_source",
            level="info",
            message="Test event 1",
            attributes={"key": "value1"},
        )
        event2 = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567891.0),
            event_type="test2",
            source="test_source",
            level="info",
            message="Test event 2",
            attributes={"key": "value2"},
        )

        # Add events to buffer
        processor.buffer.append(event1)
        processor.buffer.append(event2)

        # Start the process loop
        task = asyncio.create_task(processor._process_loop())

        # Wait a bit for processing
        await asyncio.sleep(0.05)

        # Stop the loop
        processor._shutdown = True
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that callback was called for both events
        assert callback.call_count == 2
        callback.assert_any_call(event1)
        callback.assert_any_call(event2)

    @pytest.mark.asyncio
    async def test_process_loop_batch_processing(self):
        """Test _process_loop batch processing"""
        config = EventsConfig(enabled=True, batch_size=1, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        callback = Mock()
        processor.subscribe(callback)

        event = ObservabilityEvent(
            timestamp=datetime.fromtimestamp(1234567890.0),
            event_type="test",
            source="test_source",
            level="info",
            message="Test event",
            attributes={"key": "value"},
        )

        # Add event to buffer
        processor.buffer.append(event)

        # Start the process loop
        task = asyncio.create_task(processor._process_loop())

        # Wait a bit for processing
        await asyncio.sleep(0.05)

        # Stop the loop
        processor._shutdown = True
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that callback was called
        assert callback.call_count == 1
        callback.assert_called_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_empty_buffer(self):
        """Test _process_loop with empty buffer"""
        config = EventsConfig(enabled=True, batch_size=1, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        callback = Mock()
        processor.subscribe(callback)

        # Start the process loop with empty buffer
        task = asyncio.create_task(processor._process_loop())

        # Wait a bit
        await asyncio.sleep(0.05)

        # Stop the loop
        processor._shutdown = True
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that callback was not called
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_loop_general_exception(self):
        """Test _process_loop with general exception"""
        config = EventsConfig(enabled=True, batch_size=1, flush_interval=0.01)
        processor = BufferedEventProcessor(config)

        # Create a mock that raises exception only for specific delay values
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            # Raise exception only for the flush_interval sleep (0.01)
            # Let the error recovery sleep (1 second) work normally
            if delay == 0.01:
                raise RuntimeError("General error")
            return await original_sleep(delay)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with patch("builtins.print") as mock_print:
                # Start the process loop
                task = asyncio.create_task(processor._process_loop())

                # Wait a bit for the exception to occur and be handled
                await original_sleep(0.05)

                # Stop the loop
                processor._shutdown = True
                task.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await task

                # Check that error was printed
                mock_print.assert_called_with(
                    "Event processing loop error: General error"
                )
