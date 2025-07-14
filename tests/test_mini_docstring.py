from typing import Any
import pytest
from fifo_dev_common.introspection.mini_docstring import (
    MiniDocString,
    MiniDocStringArg,
    MiniDocStringType,
    _match_closing_quote,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
    _parse_str_to_scalar_supported_type,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
    _parse_str_to_supported_type,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
    _split_by_coma  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
)

def test_parses_short_and_detailed_description():
    doc = """
    Short description.

    This function does something useful.
    2nd description line.

    Args:
        x (int):
            A number to use
    """
    mds = MiniDocString(doc)
    assert mds.description_short == "Short description."
    assert mds.description_detailed == "This function does something useful.\n2nd description line."


def test_parses_args_section_without_description():
    # Negative test: missing description
    doc_missing_desc = """
    Args:
        x (int):
            First number

        y (Optional[str]):
            Second one, optional
    """
    with pytest.raises(ValueError, match="description.*before.*Args"):
        MiniDocString(doc_missing_desc)


def test_parses_args_section_with_description():
    # Positive test: valid description included
    doc_with_desc = """
    Parses two values.

    Args:
        x (int):
            First number
        y (Optional[str]):
            Second one, optional
    """
    mds = MiniDocString(doc_with_desc)
    assert len(mds.args) == 2
    assert mds.args[0].name == "x"
    assert mds.args[0].pytype.to_string() == "int"
    assert mds.args[1].pytype.to_string() == "Optional[str]"


def test_parses_returns_section_without_description():
    # Negative test: missing description before Returns section
    doc_missing_desc = """
    Returns:
        str:
            A message
    """
    with pytest.raises(ValueError, match="description.*before.*Returns"):
        MiniDocString(doc_missing_desc)


def test_parses_returns_section_with_description():
    # Positive test: parses return type and description correctly
    doc_with_desc = """
    Returns a message.

    Returns:
        str:
            A message
    """
    mds = MiniDocString(doc_with_desc)
    assert mds.return_type is not None
    assert mds.return_type.to_string() == "str"
    assert mds.return_desc is not None and mds.return_desc == "A message"


def test_parses_raises_section_without_description():
    # Negative test: missing description before Raises section
    doc_missing_desc = """
    Raises:
        ValueError: if the input is invalid
    """
    with pytest.raises(ValueError, match="description.*before.*Raises"):
        MiniDocString(doc_missing_desc)


def test_parses_raises_section_with_description():
    # Positive test: parses raises section when description is present
    doc_with_desc = """
    Validates the input and raises if invalid.

    Raises:
        ValueError: if the input is invalid
    """
    mds = MiniDocString(doc_with_desc)
    assert mds.raises is not None and mds.raises == "ValueError: if the input is invalid"


def test_get_arg_by_name():
    doc = """
    Short description.

    Args:
        foo (int):
            an argument
    """
    mds = MiniDocString(doc)
    arg = mds.get_arg_by_name("foo")
    assert isinstance(arg, MiniDocStringArg)
    assert arg.name == "foo"
    assert arg.pytype.to_string() == "int"


def test_validate_runtime_args_valid():
    doc = """
    Short description.

    Args:
        a (int):
            required
        b (Optional[list[str]]):
            optional list
    """
    mds = MiniDocString(doc)
    mds.validate_runtime_args({"a": 123, "b": ["x", "y"]})


def test_validate_runtime_args_missing():
    doc = """
    Short description.

    Args:
        a (int):
            required
        b (str):
            required
    """
    mds = MiniDocString(doc)
    with pytest.raises(ValueError, match="Missing required arguments"):
        mds.validate_runtime_args({"a": 1})


def test_validate_runtime_args_extra():
    doc = """
    Short description.

    Args:
        a (int):
            required
    """
    mds = MiniDocString(doc)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        mds.validate_runtime_args({"a": 1, "extra": 42})


def test_validate_runtime_args_type_mismatch():
    doc = """
    Short description.

    Args:
        a (int):
            required
    """
    mds = MiniDocString(doc)
    with pytest.raises(ValueError, match="expected .*int.* got str"):
        mds.validate_runtime_args({"a": "not an int"})


def test_to_schema_yaml_structure():
    doc = """
    Short description.

    Does something. Line1
    Does something. Line2

    Args:
        id (int):
            ID value
        name (Optional[str]):
            optional label

    Returns:
        str:
            confirmation
    """
    mds = MiniDocString(doc)
    yaml = mds.to_schema_yaml("function", "my_func")

    assert yaml == \
"""- function: my_func
  description: Short description. Does something. Line1 Does something. Line2
  parameters:
    - name: id
      type: int
      description: ID value
      optional: False
    - name: name
      type: str
      description: optional label
      optional: True
  return:
    type: str
    description: confirmation"""


@pytest.mark.parametrize(
    "input_value,expected_output,should_raise",
    [
        ("['1', '2', '3']", [1, 2, 3], False),
        (['1', '2', '3'], [1, 2, 3], False),
        (['1', 2, '3'], [1, 2, 3], False),
        (['1', ['2'], '3'], None, True),
    ]
)
def test_cast_list_of_strings_to_list_of_ints_parametrized(
    input_value: list[Any],
    expected_output: list[Any] | None,
    should_raise: bool
):
    type_parser = MiniDocStringType("list[int]")

    if should_raise:
        with pytest.raises(ValueError) as exc_info:
            type_parser.cast(input_value)
        assert "Failed to cast elements" in str(exc_info.value)
    else:
        result = type_parser.cast(input_value)
        assert result == expected_output
        assert all(isinstance(x, int) for x in result)

@pytest.mark.parametrize(
    "start_idx,input_value,expected_output,exception_message",
    [
        (0, "test", None, "String value does not start by a valid quote."),
        (0, "[test", None, "String value does not start by a valid quote."),
        (0, "]test", None, "String value does not start by a valid quote."),
        (0, "", None, "String value is empty."),

        (0, "'", None, "String value does not end by a valid quote."),
        (0, '"', None, "String value does not end by a valid quote."),
        (0, "'\"", None, "String value does not end by a valid quote."),
        (0, '"\'', None, "String value does not end by a valid quote."),
        (0, "'\\'", None, "String value does not end by a valid quote."),
        (0, '"\\"', None, "String value does not end by a valid quote."),

        (1, "''", None, "String value does not end by a valid quote."),
        (1, '""', None, "String value does not end by a valid quote."),
        (2, "''", None, "String value is empty."),
        (2, '""', None, "String value is empty."),
        (6, '"test"', None, "String value is empty."),
        (7, '"test"', None, "String value is empty."),

        (0, "''", 1, None),
        (0, '""', 1, None),
        (3, "...''", 4, None),
        (3, '...""', 4, None),

        (0, "''...", 1, None),
        (0, '""...', 1, None),

        (0, "'\\''", 3, None),
        (0, '"\\""', 3, None),

        (0, '"test"...', 5, None),

        (0, '"test" more', 5, None),
        (0, "'a\\'b'", 5, None),
    ]
)
def test_match_closing_quote(
    start_idx: int,
    input_value: str,
    expected_output: int | None,
    exception_message: str | None
):
    if exception_message is not None:
        assert expected_output is None
        with pytest.raises(ValueError) as exc_info:
            _match_closing_quote(input_value, start_idx)
        assert exception_message == str(exc_info.value)
    else:
        assert expected_output is not None
        assert expected_output == _match_closing_quote(input_value, start_idx)

@pytest.mark.parametrize(
    "input_value,expected_output,exception_message",
    [
        ("test", None, "Unsupported type."),
        ("[test", None, "Unsupported type."),
        ("-test", None, "Unsupported type."),
        ("42+51", None, "Unsupported type."),
        ("'test'", "test", None),
        ('"test"', "test", None),
        ('"\\"test"', '\\"test', None),

        (42, None, "Value is not a string and cannot be converted."),
        ('"test...', None, "String is not terminated properly."),
        ("'test...", None, "String is not terminated properly."),

        ('"test', None, "String is not terminated properly."),
        ('42', 42, None),
        ('+42', 42, None),
        ('-42', -42, None),
        ('42.0', 42.0, None),
        ('-42.0', -42.0, None),
        ('', None, "Empty value."),
    ]
)
def test_parse_str_to_scalar_supported_type(
    input_value: str,
    expected_output: int | float | str | None,
    exception_message: str | None
):
    if exception_message is not None:
        assert expected_output is None
        with pytest.raises(ValueError) as exc_info:
            _parse_str_to_scalar_supported_type(input_value)
        assert exception_message == str(exc_info.value)
    else:
        assert expected_output is not None
        actual_output = _parse_str_to_scalar_supported_type(input_value)
        assert expected_output == actual_output and type(expected_output) == type(actual_output)

@pytest.mark.parametrize(
    "input_value,expected_output,exception_message",
    [
        ('', [], None),
        ('0,1,2', ["0", "1", "2"], None),
        (',0,1,2', ["", "0", "1", "2"], None),
        ('0,1,,2', ["0", "1", "", "2"], None),
        ('0,1,2,', ["0", "1", "2", ""], None),
        ('0.5,1.42, -2.5,', ["0.5", "1.42", "-2.5", ""], None),

        ('"a,b", 42, "c"', ['"a,b"', '42', '"c"'], None),
        ('1, 2, 3', ['1', '2', '3'], None),
        ('"test"', ['"test"'], None),
        ('', [], None),

        ('"a,b",3', ['"a,b"', "3"], None),
        ('   "a,b"  ,  3   ', ['"a,b"', "3"], None),
        ('   " a , b "  ,  3   ', ['" a , b "', "3"], None),

        ('"a" "b", "c"', None, "Missing delimiter."),
        ('4 4, 5', None, "Missing delimiter."),
        ('4 4', None, "Missing delimiter."),
        ('4, 4 5', None, "Missing delimiter."),
        ('4, 4, 5 6', None, "Missing delimiter."),
        ('4, 4, 5 "6"', None, "Missing delimiter."),
        ('4, 4, "5" 6', None, "Missing delimiter."),
        ('4, 4, "5" "6"', None, "Missing delimiter."),

        ('\\"test"', None, "Invalid character in numerical value."),
        ('0++00 , +42', ["0++00", "+42"], None),

        ('   " a , b \\"  ,  c"   ', ['" a , b \\"  ,  c"'], None),
        ('   " a , b \\"  ,  c   ', None, "String value does not end by a valid quote."),
    ]
)
def test_split_by_coma(
    input_value: str,
    expected_output: int | float | str | None,
    exception_message: str | None
):
    if exception_message is not None:
        assert expected_output is None
        with pytest.raises(ValueError) as exc_info:
            _split_by_coma(input_value)
        assert exception_message == str(exc_info.value)
    else:
        assert expected_output is not None
        actual_output = _split_by_coma(input_value)
        assert expected_output == actual_output and type(expected_output) == type(actual_output)

@pytest.mark.parametrize(
    "input_value,expected_output,exception_message",
    [
        ('42', 42, None),
        ('42.51', 42.51, None),
        ('"42.51"', "42.51", None),
        ('[]', [], None),
        ('[42]', [42], None),
        ('[42, 51]', [42, 51], None),
        ('    [42,   51   ]    ', [42, 51], None),
        ('    [42,   51  , -54 ]    ', [42, 51, -54], None),
        ('[42.51]', [42.51], None),
        ('["42.51"]', ["42.51"], None),
        ('"[42.51]"', "[42.51]", None),
        ('[42, 51, 3.14]', None, "Type of values mismatch."),
        ('[42, 51, ""]', None, "Type of values mismatch."),
        ('[42.51, ""]', None, "Type of values mismatch."),
        ('[1, 2, 3', None, "List of values not closed by ]."),
        ('[1, 2, "]"', None, "List of values not closed by ]."),
        ('[', None, "List of values not closed by ]."),
        ('[   ', None, "List of values not closed by ]."),
        ('1, 2', None, "List of values not enclosed in []."),
        ('', None, "Empty value."),
        ('   ', None, "Empty value."),
        ('"foo,bar', None, "String value does not end by a valid quote."),
        ('"mixed quote\'', None, "String value does not end by a valid quote."),

        # valid quoted strings with escaped quotes
        ('"abc"', "abc", None),
        ("'abc'", "abc", None),
        ('"a\\"b"', 'a\\"b', None),
        ("'a\\'b'", "a\\'b", None),

        # commas inside quoted strings
        ('"a,b"', "a,b", None),
        ('["a,b", "c"]', ["a,b", "c"], None),

        # extra whitespace handling
        ('[  "x"  ,   "y" , " z "  ]', ["x", "y", " z "], None),
        ('[1 ,2   ,3]', [1, 2, 3], None),
        ('  123  ', 123, None),

        # trailing comma
        ('[1, 2,]', None, "Empty value."),

        # malformed input
        ('[a, b]', None, "Invalid character in numerical value."),
        ('a', None, "Invalid character in numerical value."),

        # disallowed nested structure
        ('[[1], [2]]', None, "Invalid character in numerical value."),
        ('[ [ 1 ], [ 2 ] ]', None, "Invalid character in numerical value."),
        ('[[ 1 ], [ 2 ]]', None, "Invalid character in numerical value."),
        ('[1, [2] ]', None, "Invalid character in numerical value."),
        ('[1, [ 2] ]', None, "Invalid character in numerical value."),
        ('[1, [ 2 ] ]', None, "Invalid character in numerical value."),

        # negative numbers and floats
        ('-42', -42, None),
        ('-3.14', -3.14, None),
        ('[-3.0, -4.0]', [-3.0, -4.0], None),

        # no unescaping
        ('"\\"test\\""', '\\"test\\"', None),
    ]
)
def test_parse_str_to_supported_type(
    input_value: str,
    expected_output: int | float | str | None,
    exception_message: str | None
):
    if exception_message is not None:
        assert expected_output is None
        with pytest.raises(ValueError) as exc_info:
            _parse_str_to_supported_type(input_value)
        assert exception_message == str(exc_info.value)
    else:
        assert expected_output is not None
        actual_output = _parse_str_to_supported_type(input_value)
        assert expected_output == actual_output and type(expected_output) == type(actual_output)
