"""Tests for schedule builder fluent API."""

from unittest.mock import Mock

from puffinflow.core.agent.scheduling.builder import (
    ScheduleBuilder,
    create_schedule_builder,
)
from puffinflow.core.agent.scheduling.inputs import InputType


class TestScheduleBuilder:
    """Test ScheduleBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_agent.name = "test_agent"
        self.mock_agent.schedule = Mock()
        self.schedule_string = "daily"
        self.builder = ScheduleBuilder(self.mock_agent, self.schedule_string)

    def test_schedule_builder_creation(self):
        """Test ScheduleBuilder can be created."""
        assert self.builder._agent == self.mock_agent
        assert self.builder._schedule_string == self.schedule_string
        assert self.builder._inputs == {}

    def test_with_inputs_single(self):
        """Test with_inputs with single input."""
        result = self.builder.with_inputs(key1="value1")

        assert result is self.builder  # Returns self for chaining
        assert len(self.builder._inputs) == 1
        assert "key1" in self.builder._inputs
        assert self.builder._inputs["key1"].key == "key1"
        assert self.builder._inputs["key1"].value == "value1"
        assert self.builder._inputs["key1"].input_type == InputType.VARIABLE

    def test_with_inputs_multiple(self):
        """Test with_inputs with multiple inputs."""
        result = self.builder.with_inputs(key1="value1", key2="value2", key3=123)

        assert result is self.builder
        assert len(self.builder._inputs) == 3

        for key in ["key1", "key2", "key3"]:
            assert key in self.builder._inputs
            assert self.builder._inputs[key].input_type == InputType.VARIABLE

    def test_with_inputs_chaining(self):
        """Test with_inputs can be chained."""
        result = self.builder.with_inputs(key1="value1").with_inputs(key2="value2")

        assert result is self.builder
        assert len(self.builder._inputs) == 2
        assert "key1" in self.builder._inputs
        assert "key2" in self.builder._inputs

    def test_with_secrets_single(self):
        """Test with_secrets with single secret."""
        result = self.builder.with_secrets(api_key="secret123")

        assert result is self.builder
        assert len(self.builder._inputs) == 1
        assert "api_key" in self.builder._inputs
        assert self.builder._inputs["api_key"].input_type == InputType.SECRET
        assert self.builder._inputs["api_key"].value == "secret123"

    def test_with_secrets_multiple(self):
        """Test with_secrets with multiple secrets."""
        result = self.builder.with_secrets(
            api_key="secret123", db_password="password456"
        )

        assert result is self.builder
        assert len(self.builder._inputs) == 2

        for key in ["api_key", "db_password"]:
            assert key in self.builder._inputs
            assert self.builder._inputs[key].input_type == InputType.SECRET

    def test_with_constants_single(self):
        """Test with_constants with single constant."""
        result = self.builder.with_constants(max_retries=3)

        assert result is self.builder
        assert len(self.builder._inputs) == 1
        assert "max_retries" in self.builder._inputs
        assert self.builder._inputs["max_retries"].input_type == InputType.CONSTANT
        assert self.builder._inputs["max_retries"].value == "3"

    def test_with_constants_multiple(self):
        """Test with_constants with multiple constants."""
        result = self.builder.with_constants(
            timeout=30, base_url="https://api.example.com"
        )

        assert result is self.builder
        assert len(self.builder._inputs) == 2

        for key in ["timeout", "base_url"]:
            assert key in self.builder._inputs
            assert self.builder._inputs[key].input_type == InputType.CONSTANT

    def test_with_cache_single(self):
        """Test with_cache with single cached input."""
        result = self.builder.with_cache(300, user_data="cached_value")

        assert result is self.builder
        assert len(self.builder._inputs) == 1
        assert "user_data" in self.builder._inputs
        assert self.builder._inputs["user_data"].input_type == InputType.CACHED
        assert self.builder._inputs["user_data"].value == "cached_value"
        assert self.builder._inputs["user_data"].ttl == 300

    def test_with_cache_multiple(self):
        """Test with_cache with multiple cached inputs."""
        result = self.builder.with_cache(
            600, config="config_data", settings="settings_data"
        )

        assert result is self.builder
        assert len(self.builder._inputs) == 2

        for key in ["config", "settings"]:
            assert key in self.builder._inputs
            assert self.builder._inputs[key].input_type == InputType.CACHED
            assert self.builder._inputs[key].ttl == 600

    def test_with_cache_dict_value(self):
        """Test with_cache with dictionary value."""
        dict_value = {"key": "value", "number": 42}
        result = self.builder.with_cache(300, data=dict_value)

        assert result is self.builder
        assert "data" in self.builder._inputs
        assert self.builder._inputs["data"].input_type == InputType.CACHED
        assert self.builder._inputs["data"].value == dict_value
        assert self.builder._inputs["data"].ttl == 300

    def test_with_cache_list_value(self):
        """Test with_cache with list value."""
        list_value = [1, 2, 3, "test"]
        result = self.builder.with_cache(300, items=list_value)

        assert result is self.builder
        assert "items" in self.builder._inputs
        assert self.builder._inputs["items"].input_type == InputType.CACHED
        assert self.builder._inputs["items"].value == list_value
        assert self.builder._inputs["items"].ttl == 300

    def test_with_typed_single(self):
        """Test with_typed with single typed input."""
        result = self.builder.with_typed(count=42)

        assert result is self.builder
        assert len(self.builder._inputs) == 1
        assert "count" in self.builder._inputs
        assert self.builder._inputs["count"].input_type == InputType.TYPED
        assert self.builder._inputs["count"].value == 42

    def test_with_typed_multiple(self):
        """Test with_typed with multiple typed inputs."""
        result = self.builder.with_typed(count=42, rate=3.14)

        assert result is self.builder
        assert len(self.builder._inputs) == 2

        for key in ["count", "rate"]:
            assert key in self.builder._inputs
            assert self.builder._inputs[key].input_type == InputType.TYPED

    def test_with_typed_dict_value(self):
        """Test with_typed with dictionary value."""
        dict_value = {"type": "object", "value": 123}
        result = self.builder.with_typed(config=dict_value)

        assert result is self.builder
        assert "config" in self.builder._inputs
        assert self.builder._inputs["config"].input_type == InputType.TYPED
        assert self.builder._inputs["config"].value == dict_value

    def test_with_typed_list_value(self):
        """Test with_typed with list value."""
        list_value = ["item1", "item2", "item3"]
        result = self.builder.with_typed(items=list_value)

        assert result is self.builder
        assert "items" in self.builder._inputs
        assert self.builder._inputs["items"].input_type == InputType.TYPED
        assert self.builder._inputs["items"].value == list_value

    def test_with_outputs_single(self):
        """Test with_outputs with single output."""
        result = self.builder.with_outputs(result="success")

        assert result is self.builder
        assert len(self.builder._inputs) == 1
        assert "result" in self.builder._inputs
        assert self.builder._inputs["result"].input_type == InputType.OUTPUT
        assert self.builder._inputs["result"].value == "success"

    def test_with_outputs_multiple(self):
        """Test with_outputs with multiple outputs."""
        result = self.builder.with_outputs(status="completed", message="Task finished")

        assert result is self.builder
        assert len(self.builder._inputs) == 2

        for key in ["status", "message"]:
            assert key in self.builder._inputs
            assert self.builder._inputs[key].input_type == InputType.OUTPUT

    def test_method_chaining_all_types(self):
        """Test chaining all input methods together."""
        result = (
            self.builder.with_inputs(var="value")
            .with_secrets(secret="hidden")
            .with_constants(const="fixed")
            .with_cache(300, cached="data")
            .with_typed(typed=42)
            .with_outputs(output="result")
        )

        assert result is self.builder
        assert len(self.builder._inputs) == 6

        # Check all input types are present
        input_types = {inp.input_type for inp in self.builder._inputs.values()}
        expected_types = {
            InputType.VARIABLE,
            InputType.SECRET,
            InputType.CONSTANT,
            InputType.CACHED,
            InputType.TYPED,
            InputType.OUTPUT,
        }
        assert input_types == expected_types

    def test_run_method_calls_agent_schedule(self):
        """Test run method calls agent.schedule with correct parameters."""
        mock_scheduled_agent = Mock()
        self.mock_agent.schedule.return_value = mock_scheduled_agent

        # Set up some inputs
        self.builder.with_inputs(var="value").with_secrets(secret="hidden")

        result = self.builder.run()

        # Verify agent.schedule was called
        self.mock_agent.schedule.assert_called_once()
        args, kwargs = self.mock_agent.schedule.call_args

        # Check schedule string
        assert args[0] == self.schedule_string

        # Check inputs were converted back to string format
        assert "var" in kwargs
        assert kwargs["var"] == "value"
        assert "secret" in kwargs
        assert kwargs["secret"] == "secret:hidden"

        assert result == mock_scheduled_agent

    def test_run_method_with_all_input_types(self):
        """Test run method with all input types."""
        mock_scheduled_agent = Mock()
        self.mock_agent.schedule.return_value = mock_scheduled_agent

        # Set up all input types
        (
            self.builder.with_inputs(var="value")
            .with_secrets(secret="hidden")
            .with_constants(const="fixed")
            .with_cache(300, cached="data")
            .with_typed(typed=42)
            .with_outputs(output="result")
        )

        result = self.builder.run()

        # Verify agent.schedule was called
        self.mock_agent.schedule.assert_called_once()
        args, kwargs = self.mock_agent.schedule.call_args

        # Check all input types were converted correctly
        assert kwargs["var"] == "value"
        assert kwargs["secret"] == "secret:hidden"
        assert kwargs["const"] == "const:fixed"
        assert kwargs["cached"] == "cache:300:data"
        assert kwargs["typed"] == "typed:42"
        assert kwargs["output"] == "output:result"

        assert result == mock_scheduled_agent

    def test_run_method_with_complex_cached_values(self):
        """Test run method with complex cached values."""
        mock_scheduled_agent = Mock()
        self.mock_agent.schedule.return_value = mock_scheduled_agent

        dict_value = {"key": "value", "number": 42}
        list_value = [1, 2, 3]

        self.builder.with_cache(600, dict_data=dict_value, list_data=list_value)

        self.builder.run()

        args, kwargs = self.mock_agent.schedule.call_args

        # Check complex values were serialized correctly
        assert "dict_data" in kwargs
        assert "list_data" in kwargs

        # The values should contain the original data, not JSON strings
        # because parse_inputs handles the JSON parsing
        assert "cache:600:" in kwargs["dict_data"]
        assert "cache:600:" in kwargs["list_data"]

    def test_run_method_empty_inputs(self):
        """Test run method with no inputs."""
        mock_scheduled_agent = Mock()
        self.mock_agent.schedule.return_value = mock_scheduled_agent

        result = self.builder.run()

        self.mock_agent.schedule.assert_called_once_with(self.schedule_string)
        assert result == mock_scheduled_agent

    def test_input_overwriting(self):
        """Test that inputs with same key overwrite previous ones."""
        self.builder.with_inputs(key="value1").with_inputs(key="value2")

        assert len(self.builder._inputs) == 1
        assert self.builder._inputs["key"].value == "value2"

    def test_mixed_input_types_same_key(self):
        """Test mixing different input types with same key."""
        (self.builder.with_inputs(key="variable").with_secrets(key="secret"))

        # Last one should win
        assert len(self.builder._inputs) == 1
        assert self.builder._inputs["key"].input_type == InputType.SECRET
        assert self.builder._inputs["key"].value == "secret"


class TestCreateScheduleBuilder:
    """Test create_schedule_builder function."""

    def test_create_schedule_builder(self):
        """Test create_schedule_builder function."""
        mock_agent = Mock()
        schedule_string = "hourly"

        builder = create_schedule_builder(mock_agent, schedule_string)

        assert isinstance(builder, ScheduleBuilder)
        assert builder._agent == mock_agent
        assert builder._schedule_string == schedule_string
        assert builder._inputs == {}

    def test_create_schedule_builder_returns_working_builder(self):
        """Test that created builder works correctly."""
        mock_agent = Mock()
        mock_agent.schedule = Mock()
        mock_scheduled_agent = Mock()
        mock_agent.schedule.return_value = mock_scheduled_agent

        builder = create_schedule_builder(mock_agent, "daily")
        result = builder.with_inputs(test="value").run()

        mock_agent.schedule.assert_called_once()
        assert result == mock_scheduled_agent


class TestScheduleBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_agent.schedule = Mock()
        self.builder = ScheduleBuilder(self.mock_agent, "daily")

    def test_empty_inputs_methods(self):
        """Test calling input methods with no arguments."""
        # These should not fail and should return self
        result1 = self.builder.with_inputs()
        result2 = self.builder.with_secrets()
        result3 = self.builder.with_constants()
        result4 = self.builder.with_cache(300)
        result5 = self.builder.with_typed()
        result6 = self.builder.with_outputs()

        assert all(
            r is self.builder
            for r in [result1, result2, result3, result4, result5, result6]
        )
        assert len(self.builder._inputs) == 0

    def test_none_values(self):
        """Test handling None values in inputs."""
        result = (
            self.builder.with_inputs(none_var=None)
            .with_secrets(none_secret=None)
            .with_constants(none_const=None)
            .with_cache(300, none_cached=None)
            .with_typed(none_typed=None)
            .with_outputs(none_output=None)
        )

        assert result is self.builder
        assert len(self.builder._inputs) == 6

        # All should be processed normally
        for key, _scheduled_input in self.builder._inputs.items():
            assert key.startswith("none_")

    def test_zero_ttl_cache(self):
        """Test cache with zero TTL."""
        result = self.builder.with_cache(0, data="test")

        assert result is self.builder
        assert self.builder._inputs["data"].ttl == 0

    def test_negative_ttl_cache(self):
        """Test cache with negative TTL."""
        result = self.builder.with_cache(-1, data="test")

        assert result is self.builder
        assert self.builder._inputs["data"].ttl == -1

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        result = self.builder.with_inputs(
            **{
                "key-with-dashes": "value1",
                "key_with_underscores": "value2",
                "key.with.dots": "value3",
                "key with spaces": "value4",
            }
        )

        assert result is self.builder
        assert len(self.builder._inputs) == 4

        for key in [
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key with spaces",
        ]:
            assert key in self.builder._inputs

    def test_unicode_values(self):
        """Test handling unicode values."""
        unicode_value = "测试值 🚀 émojis"
        result = self.builder.with_inputs(unicode_key=unicode_value)

        assert result is self.builder
        assert self.builder._inputs["unicode_key"].value == unicode_value
