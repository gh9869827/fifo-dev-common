from typing import Any, Type, get_origin
import pytest
from fifo_dev_common.introspection.mini_docstring import MiniDocStringType

# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

@pytest.mark.parametrize(
    "type_input, expected_type, is_optional",
    [
        ("int", int, False),
        ("Optional[int]", int, True),
        ("None | int", int, True),
        ("int | None", int, True),
        ("list[float]", list, False),
        (int, int, False),
        (int | None, int, True),
        (None | str, str, True),
    ]
)
def test_init_from_valid_inputs(
    type_input: str | type,
    expected_type: type,
    is_optional: bool
) -> None:
    t = MiniDocStringType(type_input)
    base = t._type
    assert (base if not hasattr(base, "__origin__") else base.__origin__) == expected_type
    assert t.is_optional() is is_optional


@pytest.mark.parametrize(
    "invalid_input",
    [
        "dict",           # not in SUPPORTED_TYPES
        "Optional[dict]", # not in SUPPORTED_TYPES
        "None | dict",    # same
        "something random",
        123,              # not a type or string
        ["int"],          # wrong type entirely
    ]
)
def test_init_unsupported_type_raises(invalid_input: str | object) -> None:
    with pytest.raises(ValueError, match="Unsupported type string|is not supported|Unsupported type input"):
        MiniDocStringType(invalid_input)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "type_str, expected_type, is_optional, is_list, inner_type",
    [
        ("int", int, False, False, None),
        ("Optional[int]", int, True, False, None),
        ("None | int", int, True, False, None),
        ("int | None", int, True, False, None),
        ("list[float]", list, False, True, float),
        ("Optional[list[int]]", list, True, True, int),
    ]
)
def test_type_parsing_and_structure(
    type_str: str,
    expected_type: type,
    is_optional: bool,
    is_list: bool,
    inner_type: type | None
) -> None:
    t = MiniDocStringType(type_str)
    if get_origin(t._type):
        assert get_origin(t._type) == expected_type
    else:
        assert t._type == expected_type
    assert t.is_optional() == is_optional
    assert t._is_list() == is_list
    if is_list:
        assert t._get_inner_type() == inner_type
        assert t.is_list() is not None
    else:
        assert t.is_list() is None


@pytest.mark.parametrize(
    "type_str, value, expected",
    [
        ("int", 42, True),
        ("int", "not-an-int", False),
        ("Optional[int]", None, True),
        ("Optional[int]", 0, True),
        ("list[str]", ["a", "b"], True),
        ("list[str]", ["a", 2], False),
        ("list[float]", 1.0, False),
    ]
)
def test_matches_by_value(type_str: str, value: Any, expected: bool):
    assert MiniDocStringType(type_str).matches_by_value(value) is expected


@pytest.mark.parametrize(
    "type_str, input_type, expected",
    [
        ("int", int, True),
        ("int", float, False),
        ("Optional[int]", int, True),
        ("Optional[int]", type(None), True),
        ("list[int]", list, True),
        ("Optional[list[str]]", list, True),
        ("Optional[int]", list, False),
    ]
)
def test_matches_by_type(type_str: str, input_type: Type[Any], expected: bool):
    assert MiniDocStringType(type_str).matches_by_type(input_type) is expected


@pytest.mark.parametrize(
    "type_str, strip_optional, expected_str",
    [
        ("int", False, "int"),
        ("Optional[int]", False, "Optional[int]"),
        ("int | None", False, "Optional[int]"),
        ("Optional[int]", True, "int"),
        ("list[str]", False, "list[str]"),
        ("Optional[list[int]]", False, "Optional[list[int]]"),
        ("Optional[list[int]]", True, "list[int]"),
    ]
)
def test_to_string(type_str: str, strip_optional: bool, expected_str: str):
    t = MiniDocStringType(type_str)
    assert t.to_string(strip_optional=strip_optional) == expected_str


@pytest.mark.parametrize(
    "type_str, input_str, expected",
    [
        ("int", "42", 42),
        ("float", "3.14", 3.14),
        ("str", "hello", "hello"),
        ("list[int]", "[1, 2, 3]", [1, 2, 3]),
        ("list[int]", 42, [42]),
        ("Optional[int]", None, None),
        ("list[str]", "['x', 'y']", ["x", "y"]),
        ("list[float]", "2.5", [2.5]),
        ("list[float]", "[2.5, 4.2]", [2.5, 4.2]),
        ("list[float]", 2.5, [2.5]),
        ("list[float]", [2.5, 4.2], [2.5, 4.2]),
        ("list[str]", "['a', 'b', 'c']", ["a", "b", "c"]),
        ("str", "['a', 'b', 'c']", "['a', 'b', 'c']"),
    ]
)
def test_cast_valid(type_str: str, input_str: str, expected: Any):
    t = MiniDocStringType(type_str)
    result = t.cast(input_str, allow_scalar_to_list=True)
    assert result == expected


@pytest.mark.parametrize(
    "type_str, input_str, allow_scalar_to_list, error_fragment",
    [
        ("int", "not-an-int", False, "Failed to cast value"),
        ("float", "oops", False, "Failed to cast value"),
        ("list[int]", "not-a-list", False, "Failed to cast list value"),
        ("list[str]", "42", False, "Expected a list"),
        ("list[str]", "abc", False, "Failed to cast list value"),
    ]
)
def test_cast_invalid(
    type_str: str,
    input_str: str,
    allow_scalar_to_list:
    bool, error_fragment: str
):
    t = MiniDocStringType(type_str)
    with pytest.raises(ValueError, match=error_fragment):
        t.cast(input_str, allow_scalar_to_list=allow_scalar_to_list)


@pytest.mark.parametrize(
    "type_str, input_str, expected",
    [
        ("Optional[int]", None, None),
        ("Optional[list[int]]", None, None),
    ]
)
def test_cast_none_allowed(type_str: str, input_str: Any | None, expected: Any | None):
    t = MiniDocStringType(type_str)
    assert t.cast(input_str) == expected


@pytest.mark.parametrize(
    "type_str, input_str",
    [
        ("int", None),
        ("list[int]", None),
    ]
)
def test_cast_none_not_allowed(type_str: str, input_str: Any | None):
    t = MiniDocStringType(type_str)
    with pytest.raises(ValueError, match="Cannot cast None"):
        t.cast(input_str)


@pytest.mark.parametrize(
    "type_input, expected_repr",
    [
        ("int", "ArgType(int)"),
        ("Optional[float]", "ArgType(Optional[float])"),
        ("list[str]", "ArgType(list[str])"),
        ("Optional[list[int]]", "ArgType(Optional[list[int]])"),
    ]
)
def test_repr_outputs(type_input: str, expected_repr: str) -> None:
    t = MiniDocStringType(type_input)
    assert repr(t) == expected_repr
