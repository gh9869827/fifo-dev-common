from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from types import SimpleNamespace
from typing import Any, Type, cast
import pytest
import math
import struct
import numpy as np
from numpy.typing import NDArray
from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable, compile_field


def floats_equal_list(list1: list[float],
                      list2: list[float],
                      rel_tol: float=1e-6,
                      abs_tol: float=1e-8):
    assert len(list1) == len(list2)
    for a, b in zip(list1, list2):
        assert math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol), f"{a} != {b}"


class TestEnum(IntEnum):
    A = 1
    B = 2

@pytest.mark.parametrize("fmt", ["E<b>", "E<B>", "E<h>", "E<H>", "E<i>", "E<I>"])
def test_serialize_enum(fmt: str):
    @serializable
    @dataclass
    class _TestSerializeEnum(FifoSerializable):
        a: TestEnum = field(metadata={"format": fmt, "ptype": TestEnum})
        b: TestEnum = field(metadata={"format": fmt, "ptype": TestEnum})

    test = _TestSerializeEnum(TestEnum.A, TestEnum.B)

    byte_size = test.serialized_byte_size()
    expected_size = 2 * struct.calcsize('<' + fmt[2])
    assert byte_size == expected_size

    buffer = bytearray(byte_size)

    test.serialize_to_bytes(buffer, 0)

    deserialized_test, _ = _TestSerializeEnum.deserialize_from_bytes(buffer, 0)

    assert type(deserialized_test.a) == TestEnum
    assert type(deserialized_test.b) == TestEnum

    assert deserialized_test.a == TestEnum.A
    assert deserialized_test.b == TestEnum.B



@serializable
@dataclass
class TestBasic(FifoSerializable):
    a: int = field(metadata={"format": "I"})
    b: int = field(metadata={"format": "I"})

def test_basic():
    test = TestBasic(42, 51)

    byte_size = test.serialized_byte_size()

    assert byte_size == 8

    buffer = bytearray(byte_size)

    test.serialize_to_bytes(buffer, 0)

    deserialized_test, _ = TestBasic.deserialize_from_bytes(buffer, 0)

    assert deserialized_test.a == 42
    assert deserialized_test.b == 51


@serializable
@dataclass
class TestOptional(FifoSerializable):
    a: int | None = field(metadata={"format": "?I"})
    b: int | None = field(metadata={"format": "?I"})

def test_optional():
    test = TestOptional(None, 51)

    byte_size = test.serialized_byte_size()

    assert byte_size == 6

    buffer = bytearray(byte_size)

    test.serialize_to_bytes(buffer, 0)

    deserialized_test, _ = TestOptional.deserialize_from_bytes(buffer, 0)

    assert deserialized_test.a is None
    assert deserialized_test.b == 51


@serializable
@dataclass
class TestArray(FifoSerializable):
    lst1: list[float] = field(metadata={"format": "[f]"})
    lst2: list[int]   = field(metadata={"format": "[I]"})

def test_array():
    l1 = [1.2, 3.4]
    l2 = [1, 2, 3, 4]
    test = TestArray(l1, l2)

    byte_size = test.serialized_byte_size()

    assert byte_size == (len(l1) + len(l2)) * 4 + 4 + 4

    buffer = bytearray(byte_size)

    test.serialize_to_bytes(buffer, 0)

    deserialized_test, _ = TestArray.deserialize_from_bytes(buffer, 0)

    floats_equal_list(deserialized_test.lst1, l1)
    assert deserialized_test.lst2 == l2


@serializable
@dataclass
class TestTupleInt(FifoSerializable):
    pair: tuple[int, int] = field(metadata={"format": "T<II>"})


@serializable
@dataclass
class TestTupleMixed(FifoSerializable):
    pair: tuple[int, float] = field(metadata={"format": "T<If>"})


def test_tuple_basic():
    obj = TestTupleInt((7, 9))

    size = obj.serialized_byte_size()
    assert size == 8

    buf = bytearray(size)
    obj.serialize_to_bytes(buf, 0)

    restored, _ = TestTupleInt.deserialize_from_bytes(buf, 0)
    assert restored.pair == (7, 9)


def test_tuple_mixed():
    obj = TestTupleMixed((42, 3.5))

    size = obj.serialized_byte_size()
    assert size == 8

    buf = bytearray(size)
    obj.serialize_to_bytes(buf, 0)

    restored, _ = TestTupleMixed.deserialize_from_bytes(buf, 0)
    assert restored.pair[0] == 42
    assert math.isclose(restored.pair[1], 3.5, rel_tol=1e-6, abs_tol=1e-8)


@serializable
@dataclass
class TestOptionalArray(FifoSerializable):
    items: list[TestBasic | None] = field(metadata={"format": "[?_]", "ptype": TestBasic})


def test_optional_array():
    arr: list[TestBasic | None] = [TestBasic(1, 2), None, TestBasic(3, 4)]
    obj = TestOptionalArray(arr)

    size = obj.serialized_byte_size()
    expected = 4 + ((len(arr) + 7) // 8) + 2 * TestBasic(0, 0).serialized_byte_size()
    assert size == expected

    buf = bytearray(size)
    obj.serialize_to_bytes(buf, 0)

    restored, _ = TestOptionalArray.deserialize_from_bytes(buf, 0)
    assert restored.items[1] is None
    assert restored.items[0] is not None and restored.items[2] is not None
    assert restored.items[0].a == 1 and restored.items[2].b == 4


@serializable
@dataclass
class TestSuperCombo(FifoSerializable):
    arr: TestArray                = field(metadata={ "ptype": TestArray })
    opt: TestOptional             = field(metadata={ "ptype": TestOptional })
    basic: TestBasic              = field(metadata={ "ptype": TestBasic })
    opt_arr: TestArray | None     = field(metadata={ "format": "?_", "ptype": TestArray })
    opt_opt: TestOptional | None  = field(metadata={ "format": "?_", "ptype": TestOptional })
    opt_basic: TestBasic | None   = field(metadata={ "format": "?_", "ptype": TestBasic })
    lst_arr: list[TestArray]      = field(metadata={ "format": "[_]", "ptype": TestArray })
    lst_opt: list[TestOptional]   = field(metadata={ "format": "[_]", "ptype": TestOptional })
    lst_basic: list[TestBasic]    = field(metadata={ "format": "[_]", "ptype": TestBasic })

def test_super_combo():
    arr = TestArray([1.2, 3.4], [1, 2, 3])
    opt = TestOptional(10, None)
    basic = TestBasic(42, 99)

    opt_arr = TestArray([7.8], [9])
    opt_opt = None
    opt_basic = TestBasic(123, 456)

    lst_arr = [TestArray([5.5], [6]), TestArray([7.7, 8.8], [9, 10])]
    lst_opt = [TestOptional(None, 2), TestOptional(5, 6)]
    lst_basic = [TestBasic(1, 2), TestBasic(3, 4), TestBasic(5, 6)]

    combo = TestSuperCombo(
        arr=arr,
        opt=opt,
        basic=basic,
        opt_arr=opt_arr,
        opt_opt=opt_opt,
        opt_basic=opt_basic,
        lst_arr=lst_arr,
        lst_opt=lst_opt,
        lst_basic=lst_basic,
    )

    buf = bytearray(combo.serialized_byte_size())
    combo.serialize_to_bytes(buf, 0)
    deserialized, _ = TestSuperCombo.deserialize_from_bytes(buf, 0)

    # Check that all fields match (robust to float precision)
    floats_equal_list(deserialized.arr.lst1, arr.lst1)
    assert deserialized.arr.lst2 == arr.lst2

    assert deserialized.opt.a == opt.a
    assert deserialized.opt.b == opt.b
    assert deserialized.basic.a == basic.a
    assert deserialized.basic.b == basic.b

    assert deserialized.opt_arr is not None and deserialized.opt_arr.lst2 == opt_arr.lst2
    floats_equal_list(deserialized.opt_arr.lst1, opt_arr.lst1)

    assert deserialized.opt_opt is None
    assert deserialized.opt_basic is not None
    assert deserialized.opt_basic.a == opt_basic.a
    assert deserialized.opt_basic.b == opt_basic.b

    assert len(deserialized.lst_arr) == len(lst_arr)
    for d, o in zip(deserialized.lst_arr, lst_arr):
        floats_equal_list(d.lst1, o.lst1)
        assert d.lst2 == o.lst2

    assert len(deserialized.lst_opt) == len(lst_opt)
    for d, o in zip(deserialized.lst_opt, lst_opt):
        assert d.a == o.a and d.b == o.b

    assert len(deserialized.lst_basic) == len(lst_basic)
    for d, o in zip(deserialized.lst_basic, lst_basic):
        assert d.a == o.a and d.b == o.b


@pytest.mark.parametrize("format_string,ptype,err_msg", [
    ("[f", None, "Struct format for array type invalid"),
    ("[__", None, "Struct format for array type invalid"),
    ("[",   None, "Struct format for array type invalid"),
    ("[_]", None, "Type must be provided for generic array"),
    ("[?_]", None, "Type must be provided for optional array"),

    ("?",   None, "Struct format for optional type invalid"),
    ("?__", None, "Struct format for optional type invalid"),
    ("?_",  None, "Type must be provided for generic optional"),

    ("E<",  None, "Struct format for enum type invalid"),
    ("E<>",  None, "Struct format for enum type invalid"),
    ("E<_>",  None, "Struct only supports integer formats bBhHiI"),
    ("E<d>",  None, "Struct only supports integer formats bBhHiI"),
    ("E<I>",  None, "Type must be provided for Struct"),

    ("T<",   None, "Struct format for tuple type invalid"),
    ("T<>",  None, "Struct format for tuple type invalid"),
    ("T<__>", None, "Struct format for tuple type invalid"),

    ("S[", None, "Struct format for string type invalid"),
    ("S[]", None, "Struct format for string type invalid"),
    ("S[a]", None, "Struct format for string type invalid"),

    ("[np:bad]", None, "Unsupported numpy dtype"),

    (None,  None, "Type or Struct format must be provided"),
])
def test_compile_field_value_errors(format_string: str | None, ptype: Type[Any] | None, err_msg: str):
    metadata = {}
    if format_string is not None:
        metadata["format"] = format_string
    # Simulate a dataclass Field with .name and .metadata
    mock_field = SimpleNamespace(name="x", metadata=metadata, ptype=ptype)
    with pytest.raises(ValueError, match=err_msg):
        compile_field(mock_field)  # type: ignore[arg-type]


@serializable
@dataclass(kw_only=True)
class TestOtherNonSerializableFields(FifoSerializable):
    d: int = 0
    a: int = field(metadata={"format": "I"})
    b: int = field(metadata={"format": "I"})
    c: int = 0

def test_other_non_serializable_fields():
    test = TestOtherNonSerializableFields(d=4, a=42, b=51, c=2)

    byte_size = test.serialized_byte_size()

    assert byte_size == 8

    buffer = bytearray(byte_size)

    test.serialize_to_bytes(buffer, 0)

    deserialized_test, _ = TestOtherNonSerializableFields.deserialize_from_bytes(buffer, 0)

    assert deserialized_test.d == 0
    assert deserialized_test.a == 42
    assert deserialized_test.b == 51
    assert deserialized_test.c == 0


@serializable
@dataclass
class TestNumpy1D(FifoSerializable):
    arr: NDArray[np.uint8] = field(metadata={"format": "[np:u8]"})


@serializable
@dataclass
class TestNumpy2D(FifoSerializable):
    arr: NDArray[np.float32] = field(metadata={"format": "[np:f32]"})


@serializable
@dataclass
class TestNumpy3D(FifoSerializable):
    arr: NDArray[np.int16] = field(metadata={"format": "[np:i16]"})


def test_numpy_arrays_roundtrip() -> None:
    a1 = np.arange(10, dtype=np.uint8)
    a2 = np.arange(6, dtype=np.float32).reshape(2, 3)
    a3 = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    o1 = TestNumpy1D(a1)
    o2 = TestNumpy2D(a2)
    o3 = TestNumpy3D(a3)

    lst: list[tuple[FifoSerializable, NDArray[Any], Type[FifoSerializable]]] = [
        (o1, a1, TestNumpy1D),
        (o2, a2, TestNumpy2D),
        (o3, a3, TestNumpy3D),
    ]

    for obj, arr, cls in lst:
        buf = bytearray(obj.serialized_byte_size())
        obj.serialize_to_bytes(buf, 0)
        restored, _ = cls.deserialize_from_bytes(buf, 0)
        restored_arr = cast(NDArray[Any], restored.arr) # type: ignore
        assert np.array_equal(restored_arr, arr)
        assert restored_arr.dtype == arr.dtype


@serializable
@dataclass
class TestStringVar(FifoSerializable):
    name: str = field(metadata={"format": "S"})


@serializable
@dataclass
class TestStringFixed(FifoSerializable):
    tag: str = field(metadata={"format": "S[8]"})


def test_string_variable_roundtrip() -> None:
    obj = TestStringVar("hello\u03c0")
    buf = bytearray(obj.serialized_byte_size())
    obj.serialize_to_bytes(buf, 0)
    restored, _ = TestStringVar.deserialize_from_bytes(buf, 0)
    assert restored.name == "hello\u03c0"


def test_string_fixed_padding_and_truncation() -> None:
    short = TestStringFixed("hi")
    buf = bytearray(short.serialized_byte_size())
    short.serialize_to_bytes(buf, 0)
    assert buf.decode("utf-8") == "hi" + " " * 6
    restored, _ = TestStringFixed.deserialize_from_bytes(buf, 0)
    assert restored.tag == "hi"

    long = TestStringFixed("\u00e9" * 5)
    buf2 = bytearray(long.serialized_byte_size())
    long.serialize_to_bytes(buf2, 0)
    restored2, _ = TestStringFixed.deserialize_from_bytes(buf2, 0)
    assert restored2.tag == "\u00e9" * 4

def test_string_fixed_strip_trailing_spaces_only() -> None:
    # Leading and trailing spaces
    s = "  abc  "  # Two spaces before and after
    obj = TestStringFixed(s)
    buf = bytearray(obj.serialized_byte_size())
    obj.serialize_to_bytes(buf, 0)
    # Buffer should be b'  abc   ' (8 bytes: 2+3+3)
    assert buf.decode("utf-8") == "  abc   "
    restored, _ = TestStringFixed.deserialize_from_bytes(buf, 0)
    # Trailing spaces stripped, leading spaces preserved
    assert restored.tag == "  abc"

    # Trailing spaces only
    obj2 = TestStringFixed("abc   ")
    buf2 = bytearray(obj2.serialized_byte_size())
    obj2.serialize_to_bytes(buf2, 0)
    assert buf2.decode("utf-8") == "abc     "
    restored2, _ = TestStringFixed.deserialize_from_bytes(buf2, 0)
    assert restored2.tag == "abc"

    # Leading spaces only
    obj3 = TestStringFixed("   abc")
    buf3 = bytearray(obj3.serialized_byte_size())
    obj3.serialize_to_bytes(buf3, 0)
    assert buf3.decode("utf-8") == "   abc  "
    restored3, _ = TestStringFixed.deserialize_from_bytes(buf3, 0)
    assert restored3.tag == "   abc"

def test_string_fixed_emoji_truncation_no_length_hack() -> None:
    s = "..🚀🚀"
    # ".." is 2 bytes (1 each), each rocket is 4 bytes => total 10 bytes
    # If field is fixed-length 8, only "..🚀" (2+4=6 bytes) plus part of the next, 
    # but since we only fit full codepoints, we should get "..🚀" (6 bytes) + 2 spaces.
    obj = TestStringFixed(s)
    # Fixed length 8, as defined in field metadata
    buf = bytearray(obj.serialized_byte_size())
    obj.serialize_to_bytes(buf, 0)
    # After serialization: b'..🚀   '
    assert buf == ("..🚀" + "  ").encode("utf-8")
    restored, _ = TestStringFixed.deserialize_from_bytes(buf, 0)
    # After deserialization, only "..🚀" should remain
    assert restored.tag == "..🚀"
