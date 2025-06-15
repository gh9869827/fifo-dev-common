import pytest

# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

from fifo_dev_common.typeutils.strict_cast import strict_cast, _format_type_name


def test_format_type_name_single() -> None:
    assert _format_type_name(str) == "str"
    assert _format_type_name(int) == "int"


def test_format_type_name_tuple() -> None:
    assert _format_type_name((str, int, float)) == "str, int, float"
    assert _format_type_name((bool,)) == "bool"


def test_strict_cast_single_type_success() -> None:
    assert strict_cast(str, "hello") == "hello"
    assert strict_cast(int, 123) == 123
    assert strict_cast(list, [1, 2, 3]) == [1, 2, 3]
    assert strict_cast(list, [1, "2", 3]) == [1, "2", 3]


def test_strict_cast_tuple_type_success() -> None:
    assert strict_cast((str, bytes), b"bytes") == b"bytes"
    assert strict_cast((list, tuple), [1, 2]) == [1, 2]
    assert strict_cast((list, tuple), (1, 2)) == (1, 2)


def test_strict_cast_failure_single_type() -> None:
    with pytest.raises(TypeError, match="expected int, got str"):
        strict_cast(int, "wrong type")

    with pytest.raises(TypeError, match="expected str, got int"):
        strict_cast(str, 123)


def test_strict_cast_failure_tuple_type() -> None:
    with pytest.raises(TypeError, match="expected int, float, got list"):
        strict_cast((int, float), [1.0])
