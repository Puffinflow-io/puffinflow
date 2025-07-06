"""Tests for scheduling inputs and magic prefix parsing."""

from unittest.mock import Mock

import pytest

from puffinflow.core.agent.scheduling.exceptions import InvalidInputTypeError
from puffinflow.core.agent.scheduling.inputs import (
    InputType,
    ScheduledInput,
    parse_inputs,
    parse_magic_prefix,
)


class TestInputType:
    """Test InputType enum."""

    def test_input_type_values(self):
        """Test InputType enum has expected values."""
        assert InputType.VARIABLE.value == "variable"
        assert InputType.SECRET.value == "secret"
        assert InputType.CONSTANT.value == "const"
        assert InputType.CACHED.value == "cache"
        assert InputType.TYPED.value == "typed"
        assert InputType.OUTPUT.value == "output"

    def test_input_type_members(self):
        """Test InputType enum has all expected members."""
        expected_members = {
            "VARIABLE",
            "SECRET",
            "CONSTANT",
            "CACHED",
            "TYPED",
            "OUTPUT",
        }
        actual_members = {member.name for member in InputType}
        assert actual_members == expected_members


class TestScheduledInput:
    """Test ScheduledInput dataclass."""

    def test_scheduled_input_creation(self):
        """Test ScheduledInput can be created with required fields."""
        input_obj = ScheduledInput("key", "value", InputType.VARIABLE)

        assert input_obj.key == "key"
        assert input_obj.value == "value"
        assert input_obj.input_type == InputType.VARIABLE
        assert input_obj.ttl is None

    def test_scheduled_input_with_ttl(self):
        """Test ScheduledInput can be created with TTL."""
        input_obj = ScheduledInput("key", "value", InputType.CACHED, ttl=300)

        assert input_obj.key == "key"
        assert input_obj.value == "value"
        assert input_obj.input_type == InputType.CACHED
        assert input_obj.ttl == 300

    def test_apply_to_context_variable(self):
        """Test apply_to_context for VARIABLE type."""
        context = Mock()
        input_obj = ScheduledInput("test_key", "test_value", InputType.VARIABLE)

        input_obj.apply_to_context(context)

        context.set_variable.assert_called_once_with("test_key", "test_value")

    def test_apply_to_context_secret(self):
        """Test apply_to_context for SECRET type."""
        context = Mock()
        input_obj = ScheduledInput("secret_key", "secret_value", InputType.SECRET)

        input_obj.apply_to_context(context)

        context.set_secret.assert_called_once_with("secret_key", "secret_value")

    def test_apply_to_context_constant(self):
        """Test apply_to_context for CONSTANT type."""
        context = Mock()
        input_obj = ScheduledInput("const_key", "const_value", InputType.CONSTANT)

        input_obj.apply_to_context(context)

        context.set_constant.assert_called_once_with("const_key", "const_value")

    def test_apply_to_context_cached(self):
        """Test apply_to_context for CACHED type."""
        context = Mock()
        input_obj = ScheduledInput(
            "cache_key", "cache_value", InputType.CACHED, ttl=600
        )

        input_obj.apply_to_context(context)

        context.set_cached.assert_called_once_with("cache_key", "cache_value", 600)

    def test_apply_to_context_typed(self):
        """Test apply_to_context for TYPED type."""
        context = Mock()
        input_obj = ScheduledInput("typed_key", {"data": "value"}, InputType.TYPED)

        input_obj.apply_to_context(context)

        context.set_typed_variable.assert_called_once_with(
            "typed_key", {"data": "value"}
        )

    def test_apply_to_context_output(self):
        """Test apply_to_context for OUTPUT type."""
        context = Mock()
        input_obj = ScheduledInput("output_key", "output_value", InputType.OUTPUT)

        input_obj.apply_to_context(context)

        context.set_output.assert_called_once_with("output_key", "output_value")


class TestParseMagicPrefix:
    """Test parse_magic_prefix function."""

    def test_parse_non_string_value(self):
        """Test parsing non-string values returns VARIABLE type."""
        test_cases = [
            123,
            45.67,
            True,
            False,
            None,
            ["list", "value"],
            {"dict": "value"},
        ]

        for value in test_cases:
            result = parse_magic_prefix("key", value)
            assert result.key == "key"
            assert result.value == value
            assert result.input_type == InputType.VARIABLE
            assert result.ttl is None

    def test_parse_string_without_colon(self):
        """Test parsing string without colon returns VARIABLE type."""
        result = parse_magic_prefix("key", "simple_value")

        assert result.key == "key"
        assert result.value == "simple_value"
        assert result.input_type == InputType.VARIABLE
        assert result.ttl is None

    def test_parse_secret_prefix(self):
        """Test parsing secret: prefix."""
        result = parse_magic_prefix("key", "secret:my_secret_value")

        assert result.key == "key"
        assert result.value == "my_secret_value"
        assert result.input_type == InputType.SECRET
        assert result.ttl is None

    def test_parse_secret_prefix_invalid(self):
        """Test parsing invalid secret: prefix raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "secret:")

        assert exc_info.value.prefix == "secret"
        assert "Secret format: secret:value" in str(exc_info.value)

    def test_parse_const_prefix(self):
        """Test parsing const: prefix."""
        result = parse_magic_prefix("key", "const:constant_value")

        assert result.key == "key"
        assert result.value == "constant_value"
        assert result.input_type == InputType.CONSTANT
        assert result.ttl is None

    def test_parse_const_prefix_invalid(self):
        """Test parsing invalid const: prefix raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "const:")

        assert exc_info.value.prefix == "const"
        assert "Constant format: const:value" in str(exc_info.value)

    def test_parse_cache_prefix_valid(self):
        """Test parsing valid cache: prefix."""
        result = parse_magic_prefix("key", "cache:300:cached_value")

        assert result.key == "key"
        assert result.value == "cached_value"
        assert result.input_type == InputType.CACHED
        assert result.ttl == 300

    def test_parse_cache_prefix_json_value(self):
        """Test parsing cache: prefix with JSON value."""
        json_value = '{"data": "test", "number": 42}'
        result = parse_magic_prefix("key", f"cache:600:{json_value}")

        assert result.key == "key"
        assert result.value == {"data": "test", "number": 42}
        assert result.input_type == InputType.CACHED
        assert result.ttl == 600

    def test_parse_cache_prefix_invalid_format(self):
        """Test parsing invalid cache: prefix format raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "cache:300")

        assert exc_info.value.prefix == "cache"
        assert "Cache format: cache:TTL:value" in str(exc_info.value)

    def test_parse_cache_prefix_invalid_ttl(self):
        """Test parsing cache: prefix with invalid TTL raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "cache:invalid:value")

        assert exc_info.value.prefix == "cache"
        assert "Cache TTL must be an integer" in str(exc_info.value)

    def test_parse_typed_prefix_string(self):
        """Test parsing typed: prefix with string value."""
        result = parse_magic_prefix("key", "typed:string_value")

        assert result.key == "key"
        assert result.value == "string_value"
        assert result.input_type == InputType.TYPED
        assert result.ttl is None

    def test_parse_typed_prefix_json(self):
        """Test parsing typed: prefix with JSON value."""
        json_value = '{"type": "object", "value": 123}'
        result = parse_magic_prefix("key", f"typed:{json_value}")

        assert result.key == "key"
        assert result.value == {"type": "object", "value": 123}
        assert result.input_type == InputType.TYPED
        assert result.ttl is None

    def test_parse_typed_prefix_invalid_format(self):
        """Test parsing invalid typed: prefix format raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "typed:")

        assert exc_info.value.prefix == "typed"
        assert "Typed format: typed:value" in str(exc_info.value)

    def test_parse_invalid_prefix(self):
        """Test parsing with unknown prefix returns VARIABLE type."""
        # Unknown prefixes are treated as regular variables, not errors
        result = parse_magic_prefix("key", "unknown:value")

        assert result.key == "key"
        assert result.value == "unknown:value"
        assert result.input_type == InputType.VARIABLE

    def test_parse_secret_prefix_empty_value(self):
        """Test parsing secret: prefix with empty value raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "secret:")

        assert exc_info.value.prefix == "secret"
        assert "Secret format: secret:value" in str(exc_info.value)

    def test_parse_const_prefix_empty_value(self):
        """Test parsing const: prefix with empty value raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "const:")

        assert exc_info.value.prefix == "const"
        assert "Constant format: const:value" in str(exc_info.value)

    def test_parse_non_string_input(self):
        """Test parsing non-string input returns VARIABLE type."""
        result = parse_magic_prefix("key", 123)

        assert result.key == "key"
        assert result.value == 123
        assert result.input_type == InputType.VARIABLE

    def test_parse_no_colon_input(self):
        """Test parsing string without colon returns VARIABLE type."""
        result = parse_magic_prefix("key", "simple_value")

        assert result.key == "key"
        assert result.value == "simple_value"
        assert result.input_type == InputType.VARIABLE

    def test_parse_empty_prefix(self):
        """Test parsing string with empty prefix before colon."""
        # Empty prefix is not recognized, so it should be treated as variable
        result = parse_magic_prefix("key", ":value")

        assert result.key == "key"
        assert result.value == ":value"
        assert result.input_type == InputType.VARIABLE

    def test_parse_case_insensitive_prefix(self):
        """Test that prefixes are case insensitive."""
        result1 = parse_magic_prefix("key", "SECRET:value")
        result2 = parse_magic_prefix("key", "Secret:value")
        result3 = parse_magic_prefix("key", "secret:value")

        assert result1.input_type == InputType.SECRET
        assert result2.input_type == InputType.SECRET
        assert result3.input_type == InputType.SECRET
        assert result1.value == result2.value == result3.value == "value"

    def test_parse_output_prefix(self):
        """Test parsing output: prefix."""
        result = parse_magic_prefix("key", "output:output_value")

        assert result.key == "key"
        assert result.value == "output_value"
        assert result.input_type == InputType.OUTPUT
        assert result.ttl is None

    def test_parse_output_prefix_invalid(self):
        """Test parsing invalid output: prefix format raises error."""
        with pytest.raises(InvalidInputTypeError) as exc_info:
            parse_magic_prefix("key", "output:")

        assert exc_info.value.prefix == "output"
        assert "Output format: output:value" in str(exc_info.value)

    def test_parse_unknown_prefix(self):
        """Test parsing unknown prefix returns VARIABLE type."""
        result = parse_magic_prefix("key", "unknown:value")

        assert result.key == "key"
        assert result.value == "unknown:value"
        assert result.input_type == InputType.VARIABLE
        assert result.ttl is None

    def test_parse_case_insensitive_prefixes(self):
        """Test parsing prefixes is case insensitive."""
        test_cases = [
            ("SECRET:value", InputType.SECRET, "value"),
            ("Const:value", InputType.CONSTANT, "value"),
            ("CACHE:300:value", InputType.CACHED, "value"),
            ("Typed:value", InputType.TYPED, "value"),
            ("OUTPUT:value", InputType.OUTPUT, "value"),
        ]

        for input_str, expected_type, expected_value in test_cases:
            result = parse_magic_prefix("key", input_str)
            assert result.input_type == expected_type
            assert result.value == expected_value


class TestParseInputs:
    """Test parse_inputs function."""

    def test_parse_inputs_empty(self):
        """Test parsing empty inputs."""
        result = parse_inputs()
        assert result == {}

    def test_parse_inputs_single(self):
        """Test parsing single input."""
        result = parse_inputs(key1="value1")

        assert len(result) == 1
        assert "key1" in result
        assert result["key1"].key == "key1"
        assert result["key1"].value == "value1"
        assert result["key1"].input_type == InputType.VARIABLE

    def test_parse_inputs_multiple(self):
        """Test parsing multiple inputs with different types."""
        result = parse_inputs(
            var="simple_value",
            secret="secret:my_secret",
            const="const:my_constant",
            cache="cache:300:cached_data",
            typed="typed:typed_data",
            output="output:output_data",
        )

        assert len(result) == 6

        # Check variable
        assert result["var"].input_type == InputType.VARIABLE
        assert result["var"].value == "simple_value"

        # Check secret
        assert result["secret"].input_type == InputType.SECRET
        assert result["secret"].value == "my_secret"

        # Check constant
        assert result["const"].input_type == InputType.CONSTANT
        assert result["const"].value == "my_constant"

        # Check cached
        assert result["cache"].input_type == InputType.CACHED
        assert result["cache"].value == "cached_data"
        assert result["cache"].ttl == 300

        # Check typed
        assert result["typed"].input_type == InputType.TYPED
        assert result["typed"].value == "typed_data"

        # Check output
        assert result["output"].input_type == InputType.OUTPUT
        assert result["output"].value == "output_data"

    def test_parse_inputs_mixed_types(self):
        """Test parsing inputs with mixed value types."""
        result = parse_inputs(
            string_val="test",
            int_val=42,
            float_val=3.14,
            bool_val=True,
            list_val=[1, 2, 3],
            dict_val={"key": "value"},
            none_val=None,
            secret_val="secret:hidden",
        )

        assert len(result) == 8

        # Non-string values should be VARIABLE type
        for key in [
            "int_val",
            "float_val",
            "bool_val",
            "list_val",
            "dict_val",
            "none_val",
        ]:
            assert result[key].input_type == InputType.VARIABLE

        # String values without prefix should be VARIABLE
        assert result["string_val"].input_type == InputType.VARIABLE

        # String with prefix should be parsed
        assert result["secret_val"].input_type == InputType.SECRET
        assert result["secret_val"].value == "hidden"

    def test_parse_inputs_preserves_key_names(self):
        """Test that parse_inputs preserves original key names."""
        result = parse_inputs(
            CamelCase="value1",
            snake_case="value2",
            kebab_case="value3",
            number123="value4",
        )

        expected_keys = {"CamelCase", "snake_case", "kebab_case", "number123"}
        actual_keys = set(result.keys())
        assert actual_keys == expected_keys

        for key in expected_keys:
            assert result[key].key == key


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_value(self):
        """Test parsing empty string value."""
        result = parse_magic_prefix("key", "")

        assert result.key == "key"
        assert result.value == ""
        assert result.input_type == InputType.VARIABLE

    def test_colon_only_value(self):
        """Test parsing value that is just a colon."""
        result = parse_magic_prefix("key", ":")

        assert result.key == "key"
        assert result.value == ":"
        assert result.input_type == InputType.VARIABLE

    def test_multiple_colons_unknown_prefix(self):
        """Test parsing value with multiple colons and unknown prefix."""
        result = parse_magic_prefix("key", "unknown:part1:part2:part3")

        assert result.key == "key"
        assert result.value == "unknown:part1:part2:part3"
        assert result.input_type == InputType.VARIABLE

    def test_whitespace_in_values(self):
        """Test parsing values with whitespace."""
        result = parse_magic_prefix("key", "secret: value with spaces ")

        assert result.key == "key"
        assert result.value == " value with spaces "
        assert result.input_type == InputType.SECRET

    def test_special_characters_in_values(self):
        """Test parsing values with special characters."""
        special_value = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = parse_magic_prefix("key", f"const:{special_value}")

        assert result.key == "key"
        assert result.value == special_value
        assert result.input_type == InputType.CONSTANT
