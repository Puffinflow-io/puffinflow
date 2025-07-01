import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque

# Import the class under test
from src.puffinflow.core.observability.events import BufferedEventProcessor
from src.puffinflow.core.observability.interfaces import EventProcessor, ObservabilityEvent
from src.puffinflow.core.observability.config import EventsConfig


class TestBufferedEventProcessor:
    """Comprehensive tests for BufferedEventProcessor"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock EventsConfig"""
        config = Mock(spec=EventsConfig)
        config.enabled = True
        config.buffer_size = 100
        config.batch_size = 10
        config.flush_interval = 0.1
        return config

    @pytest.fixture
    def mock_disabled_config(self):
        """Create a mock EventsConfig with disabled state"""
        config = Mock(spec=EventsConfig)
        config.enabled = False
        config.buffer_size = 100
        config.batch_size = 10
        config.flush_interval = 0.1
        return config

    @pytest.fixture
    def mock_event(self):
        """Create a mock ObservabilityEvent"""
        return Mock(spec=ObservabilityEvent)

    @pytest.fixture
    def processor(self, mock_config):
        """Create BufferedEventProcessor instance"""
        return BufferedEventProcessor(mock_config)

    def test_init_sets_up_instance_correctly(self, mock_config):
        """Test that __init__ properly initializes the instance"""
        processor = BufferedEventProcessor(mock_config)

        assert processor.config == mock_config
        assert isinstance(processor.buffer, deque)
        assert processor.buffer.maxlen == mock_config.buffer_size
        assert processor.subscribers == []
        assert processor._task is None
        assert processor._shutdown is False

    def test_init_with_different_buffer_size(self):
        """Test initialization with different buffer sizes"""
        config = Mock(spec=EventsConfig)
        config.buffer_size = 50

        processor = BufferedEventProcessor(config)
        assert processor.buffer.maxlen == 50

    @pytest.mark.asyncio
    async def test_initialize_when_enabled_creates_task(self, processor):
        """Test that initialize creates a task when enabled"""
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            await processor.initialize()

            mock_create_task.assert_called_once()
            assert processor._task == mock_task

    @pytest.mark.asyncio
    async def test_initialize_when_disabled_no_task(self, mock_disabled_config):
        """Test that initialize doesn't create task when disabled"""
        processor = BufferedEventProcessor(mock_disabled_config)

        with patch('asyncio.create_task') as mock_create_task:
            await processor.initialize()

            mock_create_task.assert_not_called()
            assert processor._task is None

    @pytest.mark.asyncio
    async def test_shutdown_sets_shutdown_flag(self, processor):
        """Test that shutdown sets the _shutdown flag"""
        await processor.shutdown()
        assert processor._shutdown is True

    @pytest.mark.asyncio
    async def test_shutdown_cancels_task_if_exists(self, processor):
        """Test that shutdown cancels the task if it exists"""
        mock_task = Mock()
        processor._task = mock_task

        await processor.shutdown()

        mock_task.cancel.assert_called_once()
        assert processor._shutdown is True

    @pytest.mark.asyncio
    async def test_shutdown_handles_no_task(self, processor):
        """Test that shutdown works when no task exists"""
        processor._task = None

        await processor.shutdown()  # Should not raise exception
        assert processor._shutdown is True

    @pytest.mark.asyncio
    async def test_process_event_adds_to_buffer_when_enabled(self, processor, mock_event):
        """Test that process_event adds event to buffer when enabled"""
        await processor.process_event(mock_event)

        assert len(processor.buffer) == 1
        assert processor.buffer[0] == mock_event

    @pytest.mark.asyncio
    async def test_process_event_ignores_when_disabled(self, mock_disabled_config, mock_event):
        """Test that process_event ignores events when disabled"""
        processor = BufferedEventProcessor(mock_disabled_config)

        await processor.process_event(mock_event)

        assert len(processor.buffer) == 0

    @pytest.mark.asyncio
    async def test_process_event_respects_buffer_max_size(self, processor):
        """Test that buffer respects maxlen and drops old events"""
        # Set a small buffer size
        processor.buffer = deque(maxlen=2)

        event1 = Mock(spec=ObservabilityEvent)
        event2 = Mock(spec=ObservabilityEvent)
        event3 = Mock(spec=ObservabilityEvent)

        await processor.process_event(event1)
        await processor.process_event(event2)
        await processor.process_event(event3)

        assert len(processor.buffer) == 2
        assert processor.buffer[0] == event2  # event1 should be dropped
        assert processor.buffer[1] == event3

    def test_subscribe_adds_callback_to_subscribers(self, processor):
        """Test that subscribe adds callback to subscribers list"""
        callback = Mock()

        processor.subscribe(callback)

        assert len(processor.subscribers) == 1
        assert processor.subscribers[0] == callback

    def test_subscribe_multiple_callbacks(self, processor):
        """Test subscribing multiple callbacks"""
        callback1 = Mock()
        callback2 = Mock()

        processor.subscribe(callback1)
        processor.subscribe(callback2)

        assert len(processor.subscribers) == 2
        assert callback1 in processor.subscribers
        assert callback2 in processor.subscribers

    @pytest.mark.asyncio
    async def test_process_loop_processes_events_in_batches(self, processor):
        """Test that _process_loop processes events in batches"""
        # Setup
        processor.config.batch_size = 2
        processor.config.flush_interval = 0.01

        event1 = Mock(spec=ObservabilityEvent)
        event2 = Mock(spec=ObservabilityEvent)
        event3 = Mock(spec=ObservabilityEvent)

        processor.buffer.extend([event1, event2, event3])

        sync_subscriber = Mock()
        async_subscriber = AsyncMock()
        processor.subscribe(sync_subscriber)
        processor.subscribe(async_subscriber)

        # Use a counter to stop after one iteration
        sleep_call_count = 0

        async def mock_sleep_with_stop(interval):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 1:
                processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # Should process batch_size (2) events
        assert sync_subscriber.call_count == 2
        assert async_subscriber.call_count == 2
        assert len(processor.buffer) == 1  # One event should remain

    @pytest.mark.asyncio
    async def test_process_loop_handles_async_subscribers(self, processor):
        """Test that _process_loop properly handles async subscribers"""
        processor.config.flush_interval = 0.01

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        async_subscriber = AsyncMock()
        processor.subscribe(async_subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        async_subscriber.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_handles_sync_subscribers(self, processor):
        """Test that _process_loop properly handles sync subscribers"""
        processor.config.flush_interval = 0.01

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        sync_subscriber = Mock()
        processor.subscribe(sync_subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        sync_subscriber.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_handles_subscriber_exceptions(self, processor):
        """Test that _process_loop handles exceptions in subscribers"""
        processor.config.flush_interval = 0.01

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        # Subscriber that raises exception
        failing_subscriber = Mock(side_effect=ValueError("Test error"))
        working_subscriber = Mock()

        processor.subscribe(failing_subscriber)
        processor.subscribe(working_subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # Both should be called despite the exception
        failing_subscriber.assert_called_once_with(event)
        working_subscriber.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_handles_async_subscriber_exceptions(self, processor):
        """Test that _process_loop handles exceptions in async subscribers"""
        processor.config.flush_interval = 0.01

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        # Async subscriber that raises exception
        failing_async_subscriber = AsyncMock(side_effect=ValueError("Async test error"))
        working_subscriber = Mock()

        processor.subscribe(failing_async_subscriber)
        processor.subscribe(working_subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # Both should be called despite the exception
        failing_async_subscriber.assert_called_once_with(event)
        working_subscriber.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_handles_empty_buffer(self, processor):
        """Test that _process_loop handles empty buffer gracefully"""
        processor.config.flush_interval = 0.01

        subscriber = Mock()
        processor.subscribe(subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # No events to process, so subscriber shouldn't be called
        subscriber.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_loop_stops_on_shutdown(self, processor):
        """Test that _process_loop stops when _shutdown is True"""
        processor._shutdown = True

        # Should exit immediately without processing
        await processor._process_loop()

        # If we get here without hanging, the test passes

    @pytest.mark.asyncio
    async def test_process_loop_handles_general_exceptions(self, processor):
        """Test that _process_loop handles general exceptions and continues"""
        processor.config.flush_interval = 0.01

        call_count = 0

        async def mock_sleep_with_exception_then_stop(interval):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("General error")
            else:
                processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_exception_then_stop):
            await processor._process_loop()

        # Should have been called twice (once with exception, once to stop)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_process_loop_batch_size_larger_than_buffer(self, processor):
        """Test _process_loop when batch_size is larger than buffer contents"""
        processor.config.batch_size = 10
        processor.config.flush_interval = 0.01

        # Only add 3 events but batch size is 10
        events = [Mock(spec=ObservabilityEvent) for _ in range(3)]
        processor.buffer.extend(events)

        subscriber = Mock()
        processor.subscribe(subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # Should process all 3 events
        assert subscriber.call_count == 3
        assert len(processor.buffer) == 0

    @pytest.mark.asyncio
    async def test_process_loop_respects_flush_interval(self, processor):
        """Test that _process_loop respects the flush interval"""
        processor.config.flush_interval = 0.5

        async def mock_sleep_with_stop(interval):
            assert interval == 0.5
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop) as mock_sleep:
            await processor._process_loop()

        # Should call sleep with the flush interval
        mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_process_loop_handles_cancelled_error(self, processor):
        """Test that _process_loop properly handles CancelledError"""
        processor.config.flush_interval = 0.01

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        subscriber = Mock()
        processor.subscribe(subscriber)

        async def mock_sleep_with_cancellation(interval):
            raise asyncio.CancelledError()

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_cancellation):
            await processor._process_loop()

        # Should process the event before cancellation
        subscriber.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, mock_config):
        """Integration test of the full workflow"""
        processor = BufferedEventProcessor(mock_config)

        # Initialize
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            await processor.initialize()

        # Add events
        events = [Mock(spec=ObservabilityEvent) for _ in range(5)]
        for event in events:
            await processor.process_event(event)

        # Add subscribers
        sync_subscriber = Mock()
        async_subscriber = AsyncMock()
        processor.subscribe(sync_subscriber)
        processor.subscribe(async_subscriber)

        # Verify state
        assert len(processor.buffer) == 5
        assert len(processor.subscribers) == 2
        assert processor._task == mock_task

        # Shutdown
        await processor.shutdown()
        assert processor._shutdown is True
        mock_task.cancel.assert_called_once()

    def test_implements_event_processor_interface(self, processor):
        """Test that BufferedEventProcessor implements EventProcessor interface"""
        assert isinstance(processor, EventProcessor)

    @pytest.mark.asyncio
    async def test_multiple_initialize_calls(self, processor):
        """Test behavior when initialize is called multiple times"""
        with patch('asyncio.create_task') as mock_create_task:
            mock_task1 = Mock()
            mock_task2 = Mock()
            mock_create_task.side_effect = [mock_task1, mock_task2]

            await processor.initialize()
            await processor.initialize()

            # Should create task twice (no protection against multiple calls)
            assert mock_create_task.call_count == 2
            assert processor._task == mock_task2

    @pytest.mark.asyncio
    async def test_process_event_after_shutdown(self, processor, mock_event):
        """Test processing events after shutdown"""
        processor._shutdown = True

        # Should still add to buffer (shutdown only affects processing loop)
        await processor.process_event(mock_event)
        assert len(processor.buffer) == 1

    def test_buffer_is_thread_safe_deque(self, processor):
        """Test that buffer uses thread-safe deque"""
        assert isinstance(processor.buffer, deque)
        # deque is thread-safe for append/popleft operations


# Additional test class for edge cases and error conditions
class TestBufferedEventProcessorEdgeCases:
    """Edge cases and error condition tests"""

    @pytest.mark.asyncio
    async def test_process_loop_with_mixed_subscriber_types(self):
        """Test _process_loop with mixture of sync, async, and callable objects"""
        config = Mock(spec=EventsConfig)
        config.enabled = True
        config.buffer_size = 100
        config.batch_size = 10
        config.flush_interval = 0.01

        processor = BufferedEventProcessor(config)

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        # Different types of subscribers
        sync_function = Mock()
        async_function = AsyncMock()
        callable_object = Mock()
        callable_object.__call__ = Mock()

        processor.subscribe(sync_function)
        processor.subscribe(async_function)
        processor.subscribe(callable_object)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        sync_function.assert_called_once_with(event)
        async_function.assert_called_once_with(event)
        callable_object.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_very_large_batch_processing(self):
        """Test processing very large batches"""
        config = Mock(spec=EventsConfig)
        config.enabled = True
        config.buffer_size = 1000
        config.batch_size = 500
        config.flush_interval = 0.01

        processor = BufferedEventProcessor(config)

        # Add many events
        events = [Mock(spec=ObservabilityEvent) for _ in range(300)]
        processor.buffer.extend(events)

        subscriber = Mock()
        processor.subscribe(subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # Should process all 300 events (limited by buffer contents, not batch size)
        assert subscriber.call_count == 300
        assert len(processor.buffer) == 0

    @pytest.mark.asyncio
    async def test_process_loop_with_zero_flush_interval(self):
        """Test _process_loop with zero flush interval"""
        config = Mock(spec=EventsConfig)
        config.enabled = True
        config.buffer_size = 100
        config.batch_size = 10
        config.flush_interval = 0

        processor = BufferedEventProcessor(config)

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        subscriber = Mock()
        processor.subscribe(subscriber)

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            assert interval == 0
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        subscriber.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_process_loop_with_no_subscribers(self):
        """Test _process_loop when there are no subscribers"""
        config = Mock(spec=EventsConfig)
        config.enabled = True
        config.buffer_size = 100
        config.batch_size = 10
        config.flush_interval = 0.01

        processor = BufferedEventProcessor(config)

        event = Mock(spec=ObservabilityEvent)
        processor.buffer.append(event)

        # Don't add any subscribers

        # Stop after one iteration
        async def mock_sleep_with_stop(interval):
            processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_stop):
            await processor._process_loop()

        # Should process events even with no subscribers (removes them from buffer)
        assert len(processor.buffer) == 0

    @pytest.mark.asyncio
    async def test_concurrent_event_processing_and_adding(self):
        """Test that events can be added while processing is happening"""
        config = Mock(spec=EventsConfig)
        config.enabled = True
        config.buffer_size = 100
        config.batch_size = 2
        config.flush_interval = 0.01

        processor = BufferedEventProcessor(config)

        # Add initial events
        initial_events = [Mock(spec=ObservabilityEvent) for _ in range(3)]
        processor.buffer.extend(initial_events)

        subscriber = Mock()
        processor.subscribe(subscriber)

        processed_count = 0

        async def mock_sleep_with_event_addition(interval):
            nonlocal processed_count
            processed_count += 1

            if processed_count == 1:
                # Add more events during processing
                new_event = Mock(spec=ObservabilityEvent)
                processor.buffer.append(new_event)
            elif processed_count >= 2:
                processor._shutdown = True

        with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep_with_event_addition):
            await processor._process_loop()

        # Should have processed some events (batch size limits per iteration)
        assert subscriber.call_count >= 2