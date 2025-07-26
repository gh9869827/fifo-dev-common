from __future__ import annotations
from abc import ABC
import abc
from dataclasses import Field, fields
from enum import Enum
import struct
from typing import Any, Self, Tuple, Type, TypeVar
import numpy as np
from numpy.typing import NDArray
from fifo_dev_common.socket.socket_utils import SupportsRecvInto, SupportsSendAll, recv_all

_NUMPY_DTYPES: dict[str, np.dtype[Any]] = {
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}

# Supported struct format characters for all primitive field types
_ALLOWED_STRUCT_FORMAT_CHARS = {
    "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", "y",
}

def compile_field(field: Field[Any]) -> FieldSpecCompiled:
    """
    Compile a dataclass Field into a corresponding FieldSpecCompiled instance.

    This function reads the serialization metadata from the dataclass field
    and returns a concrete compiled field handler that knows how to serialize
    and deserialize that field according to the specified format or nested type.

    Supported formats in `field.metadata["format"]`:

      - Basic types (struct format strings), e.g.:
          - 'I' — unsigned 4-byte integer
          - 'f' — 4-byte float
          - 'B' — unsigned 1-byte integer

      - Enum types serialized as unsigned int:
          - 'E<I>' — Enum stored as a 4-byte unsigned int.
            Requires 'ptype' metadata specifying the Enum class.
            For now, support only type 'I'

      - Fixed-size tuple of basic types:
          - 'T<xy...>' where x, y, ... are struct format characters.
            Example: 'T<II>' means a tuple of two ints, 'T<If>' means a tuple of (int, float).
            The tuple is serialized as a tightly packed binary sequence of those types.
            Only basic types are supported inside the tuple. Nested tuples, arrays, or optional
            values are not supported in the current version.

      - Fixed-length array of basic types:
          - '[x]' where x is a single struct format character.
            Example: '[f]' means a variable-length array of floats.

      - NumPy arrays of fixed dtype:
          - '[np:x]' where x is one of 'u8', 'u16', 'u32', 'i8', 'i16', 'i32',
            'f32', 'f64'. The dtype must be fixed for the field.
            Supports arrays of arbitrary dimension.

      - Optional values (nullable) of basic types:
          - '?x' where x is a single struct format character.
            Example: '?I' means an optional unsigned int (presence flag + value).

      - Generic optional nested serializable object:
          - '?_' indicates an optional nested object.
            Requires 'ptype' metadata specifying the nested class type.

      - Generic array of nested serializable objects:
          - '[_]' indicates a variable-length array of nested objects.
            Requires 'ptype' metadata specifying the element class type.

      - Array of optional nested serializable objects:
          - '[?_]' indicates a variable-length array where each element may be
            `None`.
            Requires 'ptype' metadata specifying the element class type.

      - No format specified:
          - Assumes a nested serializable object.
            Requires 'ptype' metadata specifying the class type.

    Args:
        field (Field[Any]):
            A dataclasses.Field object representing a field in a dataclass.
            Must have metadata specifying either:
              - 'format': a struct-like format string defining the binary layout.
              - 'ptype': a nested serializable class type for custom serialization.

    Returns:
        FieldSpecCompiled:
            An instance of a FieldSpecCompiled subclass corresponding to the field's
            serialization format or nested type.

    Raises:
        ValueError:
            - If the 'format' string is invalid for array, optional, or enum types.
            - If a generic array ('[_]'), optional ('?_'), or optional array ('[?_]') format is specified without a 'ptype'.
            - If an enum format ('E<I>') is used without a 'ptype' Enum class.
            - If neither 'format' nor 'ptype' is provided in the field metadata.
    """
    struct_format = field.metadata.get("format")
    ptype = field.metadata.get("ptype")
    name = field.name

    if struct_format is not None:
        if struct_format[0] == "[":
            if struct_format.startswith("[np:"):
                if not struct_format.endswith("]"):
                    raise ValueError("Struct format for numpy array type invalid")
                dtype_key = struct_format[4:-1]
                dtype = _NUMPY_DTYPES.get(dtype_key)
                if dtype is None:
                    raise ValueError("Unsupported numpy dtype")
                return FieldSpecCompiledNumpyArray(name, dtype)

            if struct_format == "[?_]":
                if ptype is None:
                    raise ValueError("Type must be provided for optional array")
                return FieldSpecCompiledGenericOptionalArray(name, ptype)

            if (    struct_format.startswith("[?")
                and struct_format.endswith("]")
                and len(struct_format) == 4
            ):
                element_type = struct_format[2]
                if element_type not in _ALLOWED_STRUCT_FORMAT_CHARS:
                    raise ValueError(
                        "Struct format for optional type in arrays only supports format characters "
                        + "".join(sorted(_ALLOWED_STRUCT_FORMAT_CHARS))
                    )
                return FieldSpecCompiledOptionalPrimitiveArray(name, element_type)

            if not (len(struct_format) == 3 and struct_format[2] == "]"):
                raise ValueError("Struct format for array type invalid")

            if struct_format == "[_]":
                if ptype is None:
                    raise ValueError("Type must be provided for generic array")
                return FieldSpecCompiledGenericArray(name, ptype)

            element_type = struct_format[1]
            if element_type not in _ALLOWED_STRUCT_FORMAT_CHARS:
                raise ValueError(
                    "Struct only supports format characters "
                    + "".join(sorted(_ALLOWED_STRUCT_FORMAT_CHARS))
                )
            return FieldSpecCompiledArray(name, element_type)

        if struct_format[0] == "?":
            if not len(struct_format) == 2:
                raise ValueError("Struct format for optional type invalid")

            if struct_format == "?_":
                if ptype is None:
                    raise ValueError("Type must be provided for generic optional")
                return FieldSpecCompiledGenericOptional(name, ptype)

            inner_type = struct_format[1]
            if inner_type not in _ALLOWED_STRUCT_FORMAT_CHARS:
                raise ValueError(
                    "Struct only supports format characters "
                    + "".join(sorted(_ALLOWED_STRUCT_FORMAT_CHARS))
                )
            return FieldSpecCompiledOptional(name, inner_type)

        if struct_format.startswith("E<"):
            if not (len(struct_format) == 4 and struct_format[3] == ">"):
                raise ValueError("Struct format for enum type invalid")

            inner_type = struct_format[2]

            supported_enum_ints = {"b", "B", "h", "H", "i", "I"}
            if inner_type not in supported_enum_ints:
                raise ValueError("Struct only supports integer formats bBhHiI")

            if ptype is None:
                raise ValueError("Type must be provided for Struct")

            return FieldSpecCompiledEnum(name, inner_type, ptype)

        if struct_format.startswith("T<"):
            if not (len(struct_format) >= 4 and struct_format[-1] == ">"):
                raise ValueError("Struct format for tuple type invalid")
            inner_types = struct_format[2:-1]
            try:
                struct.calcsize('<' + inner_types.replace('y', 'B'))
            except struct.error as exc:
                raise ValueError("Struct format for tuple type invalid") from exc
            if any(c not in _ALLOWED_STRUCT_FORMAT_CHARS for c in inner_types):
                raise ValueError(
                    "Struct only supports format characters "
                    + "".join(sorted(_ALLOWED_STRUCT_FORMAT_CHARS))
                )
            return FieldSpecCompiledTuple(name, inner_types)

        if struct_format == "S":
            return FieldSpecCompiledString(name)

        if struct_format.startswith("S["):
            if not struct_format.endswith("]"):
                raise ValueError("Struct format for string type invalid")
            length_txt = struct_format[2:-1]
            if not length_txt.isdigit():
                raise ValueError("Struct format for string type invalid")
            return FieldSpecCompiledFixedString(name, int(length_txt))

        if len(struct_format) != 1 or struct_format not in _ALLOWED_STRUCT_FORMAT_CHARS:
            raise ValueError(
                "Struct only supports format characters "
                + "".join(sorted(_ALLOWED_STRUCT_FORMAT_CHARS))
            )
        return FieldSpecCompiledBasic(name, struct_format)

    if ptype is not None:
        return FieldSpecCompiledGeneric(name, ptype)

    raise ValueError("Type or Struct format must be provided")


class FieldSpecCompiled(ABC):
    """
    Abstract base class representing a compiled field specification
    responsible for serializing and deserializing a single field
    in a serializable class.

    Each subclass implements the specific logic to convert between
    Python objects and their binary representation according to
    the field's format or type.

    Attributes:
        name (str):
            The name of the field this instance handles.
    """

    name: str

    def __init__(self, name: str):
        """
        Initialize the compiled field specifier.

        Args:
            name (str):
                The name of the field.
        """
        self.name = name

    @abc.abstractmethod
    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the field's value from `class_obj` into the provided
        `buffer` starting at index `idx`.

        Args:
            class_obj (Any):
                The instance containing the field value to serialize.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the field data.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """

    @abc.abstractmethod
    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the field's value from the `buffer` starting at index `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized field value.
                - The updated buffer index after reading the field data.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """

    @abc.abstractmethod
    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the field's value
        from `class_obj`.

        Args:
            class_obj (Any):
                The instance containing the field value.

        Returns:
            int:
                The byte size needed for serialization of this field.
        """


class FieldSpecCompiledBasic(FieldSpecCompiled):
    """
    Concrete implementation of FieldSpecCompiled for fixed-size fields
    defined by a struct format string.

    Uses Python's `struct` module to serialize and deserialize
    primitive fixed-length types with little-endian byte order.

    Attributes:
        struct_format (str):
            The struct format string describing the binary layout
            (e.g., 'I', 'f', 'B').

        _struct_format_byte_length (int):
            The fixed size in bytes computed from `struct_format`,
            representing the number of bytes required to serialize
            the field value.

        _is_bool (bool):
            True if the original `struct_format` was `'y'` (boolean).
            Used to distinguish native booleans even though they are encoded as `'B'`.
    """

    struct_format: str
    _struct_format_byte_length: int
    _is_bool: bool

    def __init__(self, name: str, struct_format: str, *args: Any, **kwargs: Any) -> None:
        """
        Initialize with field name and struct format string.

        Args:
            name (str):
                The name of the field.

            struct_format (str):
                The struct format string defining the binary layout.
        """
        super().__init__(name, *args, **kwargs)
        self._is_bool = struct_format == "y"
        self.struct_format = "B" if self._is_bool else struct_format
        self._struct_format_byte_length = struct.calcsize('<' + self.struct_format)

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the field's value from `class_obj` into the provided
        `buffer` starting at index `idx`.

        Args:
            class_obj (Any):
                The instance containing the field value to serialize.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the field data.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        value = getattr(class_obj, self.name)
        struct.pack_into('<' + self.struct_format, buffer, idx, value)
        return idx + self._struct_format_byte_length

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the field's value from the `buffer` starting at index `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized field value.
                - The updated buffer index after reading the field data.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        obj, = struct.unpack_from('<' + self.struct_format, buffer, idx)
        if self._is_bool:
            obj = bool(obj)
        return obj, idx + self._struct_format_byte_length

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the field's value
        from `class_obj`.

        Args:
            class_obj (Any):
                The instance containing the field value.

        Returns:
            int:
                The byte size needed for serialization of this field.
        """
        return self._struct_format_byte_length


class FieldSpecCompiledEnum(FieldSpecCompiledBasic):
    """
    Handles serialization and deserialization of Enum fields
    stored as fixed-size primitive types using struct formats.

    This class extends FieldSpecCompiledBasic and adds support
    for converting between the raw integer representation and
    the Enum instance using the specified `ptype`.

    Attributes:
        ptype (Type[Enum]):
            The Enum type used for deserialization and validation.
    """

    ptype: Type[Enum]

    def __init__(self, name: str, struct_format: str, ptype: Type[Enum]):
        """
        Initialize the Enum field with its name, struct format, and Enum type.

        Args:
            name (str):
                The name of the field.

            struct_format (str):
                The struct format string defining the binary layout.

            ptype (Type[Enum]):
                The Enum class used to convert the integer value
                to the corresponding Enum instance.
        """
        super().__init__(name, struct_format)
        self.ptype = ptype

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the Enum field's value from the buffer starting at index `idx`.

        Reads the raw value using the struct format, then
        converts it to an Enum instance using the provided `ptype`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized Enum instance.
                - The updated buffer index after reading the data.

        Raises:
            Implementation-specific exceptions if deserialization fails,
            such as if the raw value does not correspond to a valid Enum member.
        """
        obj = struct.unpack_from('<' + self.struct_format, buffer, idx)
        if len(obj) == 1:
            obj = obj[0]
        return self.ptype(obj), idx + self._struct_format_byte_length


class FieldSpecCompiledOptional(FieldSpecCompiledBasic):
    """
    FieldSpecCompiled subclass for optional (nullable) fields with a presence flag.

    This class serializes an optional field as a single byte presence flag (`0` or `1`),
    followed by the field's value if present, using the specified struct format.

    In deserialization, the presence flag is checked first to determine if the
    value follows or if the field is `None`.
    """

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the optional field's value from `class_obj` into the buffer,
        prefixing with a presence byte.

        Args:
            class_obj (Any):
                The instance containing the field value to serialize.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the presence flag and
                optionally the field data.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        obj = getattr(class_obj, self.name)
        if obj is None:
            struct.pack_into("<b", buffer, idx, 0)
            return idx + 1

        struct.pack_into('<b' + self.struct_format, buffer, idx, 1, obj)
        return idx + 1 + self._struct_format_byte_length

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the optional field's value from the buffer starting at index `idx`,
        first reading the presence flag.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized field value or None if not present.
                - The updated buffer index after reading the presence flag and value.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        present = struct.unpack_from("<b", buffer, idx)[0]
        idx += 1

        if present == 0:
            return None, idx

        obj, = struct.unpack_from('<' + self.struct_format, buffer, idx)
        if self._is_bool:
            obj = bool(obj)
        return obj, idx + self._struct_format_byte_length

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the optional field,
        including one byte for the presence flag.

        Args:
            class_obj (Any):
                The instance containing the field value.

        Returns:
            int:
                1 if the value is None (presence flag only),
                otherwise 1 plus the fixed size of the serialized value.
        """
        if getattr(class_obj, self.name) is None:
            return 1
        return 1 + self._struct_format_byte_length


class FieldSpecCompiledArray(FieldSpecCompiledBasic):
    """
    FieldSpecCompiled subclass for variable-length arrays of basic fixed-size elements.

    Serializes the array by first writing its length as a 4-byte unsigned integer,
    followed by the concatenated serialized elements according to the struct format.
    """

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the array field from `class_obj` into the buffer starting at `idx`.

        Writes the length of the array (4 bytes, little-endian unsigned int),
        followed by each element serialized using the struct format and little-endian
        encoding.

        Args:
            class_obj (Any):
                The instance containing the array field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the array length and elements.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        obj = getattr(class_obj, self.name)
        length = len(obj)
        struct.pack_into("<I", buffer, idx, length)
        idx += 4
        struct.pack_into('<' + self.struct_format * length, buffer, idx, *obj)
        return idx + self._struct_format_byte_length * length

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the array field from the buffer starting at `idx`.

        Reads the array length (4 bytes), then unpacks that many elements
        using the struct format, returning them as a list.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized list of elements.
                - The updated buffer index after reading the array.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        length, = struct.unpack_from("<I", buffer, idx)
        idx += 4
        items = list(struct.unpack_from('<' + self.struct_format * length, buffer, idx))
        if self._is_bool:
            items = [bool(v) for v in items]
        return items, idx + self._struct_format_byte_length * length

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the array field,
        including 4 bytes for the length prefix plus the serialized size of all elements.

        Args:
            class_obj (Any):
                The instance containing the array field.

        Returns:
            int:
                The total byte size needed to serialize the array field.
        """
        return 4 + len(getattr(class_obj, self.name)) * self._struct_format_byte_length


class _FieldSpecCompiledOptionalArrayBase(FieldSpecCompiled):
    """
    Base helper for array-of-optional fields (`list[Optional[...]]`), serialized using
    a presence bitmap.

    This refers to arrays where each individual element may be present or `None`,
    as opposed to the entire array being optional.

    Arrays using this layout encode whether each element is present with a
    bit-packed bitmap placed right after the length prefix. Only the values for
    present elements are stored in the payload. Subclasses are responsible for
    the element-level (de)serialization via the `_serialize_element` and
    `_deserialize_element` hooks and for reporting the size of an element with
    `_element_size`.

    Features:
        - Serializes the array length as a 4-byte unsigned int.
        - Packs a presence bitmap (1 bit per element) immediately after the
          length.
        - Writes only the serialized form of present elements in sequence.
        - Computes the serialized size accounting for the bitmap and present
          elements only.
    """

    def __init__(self, *args: Any, **kwargs: Any):  # pylint: disable=useless-parent-delegation
        """
        Cooperative constructor for MRO-based initialization.

        Accepts and forwards all arguments to the next class in the MRO chain.
        """
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def _serialize_element(self, elem: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize a single element into `buffer` starting at `idx`.

        Args:
            elem (Any):
                The element value to serialize.

            buffer (bytearray):
                The buffer into which to serialize the element.

            idx (int):
                The starting index in `buffer` at which to write data.

        Returns:
            int:
                The updated buffer index after writing the element.
        """

    @abc.abstractmethod
    def _deserialize_element(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize a single element from `buffer` starting at `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in `buffer` at which to read data.

        Returns:
            tuple[Any, int]:
                - The deserialized element value.
                - The updated buffer index after reading the element.
        """

    @abc.abstractmethod
    def _element_size(self, elem: Any) -> int:
        """
        Return the serialized size in bytes of `elem`.

        Args:
            elem (Any):
                The element to measure.

        Returns:
            int:
                The number of bytes required to serialize `elem`.
        """

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the array of optional values into `buffer`, starting at index `idx`.

        Writes the following sequence:
            - 4 bytes: array length (little-endian unsigned int)
            - N bits: presence bitmap for each element (rounded up to full bytes)
            - Serialized payloads of present elements in order

        Args:
            class_obj (Any):
                The instance containing the array field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in `buffer` at which to write data.

        Returns:
            int:
                The updated buffer index after writing the entire array.
        """
        array = getattr(class_obj, self.name)
        length = len(array)
        struct.pack_into("<I", buffer, idx, length)
        idx += 4
        bitmap_len = (length + 7) // 8
        bitmap = bytearray(bitmap_len)
        for i, elem in enumerate(array):
            if elem is not None:
                bitmap[i // 8] |= 1 << (i % 8)
        buffer[idx:idx + bitmap_len] = bitmap
        idx += bitmap_len
        for elem in array:
            if elem is not None:
                idx = self._serialize_element(elem, buffer, idx)
        return idx

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize an array of optional values from `buffer`, starting at index `idx`.

        Reads the sequence produced by `serialize_to_bytes` and reconstructs
        a list containing either element values or `None` for missing items.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in `buffer` from which to read.

        Returns:
            tuple[list[Any | None], int]:
                - The list of deserialized elements.
                - The updated buffer index after reading the entire array.
        """
        length, = struct.unpack_from("<I", buffer, idx)
        idx += 4
        bitmap_len = (length + 7) // 8
        bitmap = buffer[idx:idx + bitmap_len]
        idx += bitmap_len
        items: list[Any | None] = []
        for i in range(length):
            if bitmap[i // 8] & (1 << (i % 8)):
                item, idx = self._deserialize_element(buffer, idx)
                items.append(item)
            else:
                items.append(None)
        return items, idx

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the optional array.

        The calculation accounts for:
            - 4 bytes for the length prefix
            - N bits (rounded up to bytes) for the presence bitmap
            - Only the serialized sizes of present elements

        Args:
            class_obj (Any):
                The instance containing the array field.

        Returns:
            int:
                The total byte size needed to serialize the array field.
        """
        array = getattr(class_obj, self.name)
        total = 4 + (len(array) + 7) // 8
        for elem in array:
            if elem is not None:
                total += self._element_size(elem)
        return total


class FieldSpecCompiledOptionalPrimitiveArray(_FieldSpecCompiledOptionalArrayBase,
                                              FieldSpecCompiledBasic):
    """
    FieldSpecCompiled subclass for arrays of optional primitive values.

    Supports efficient serialization and deserialization of arrays where each element
    is either a primitive value or `None`. Presence is encoded using a compact bitmask.

    Features:
        - Serializes the array length as a 4-byte unsigned int.
        - Encodes presence with a packed bitmap (1 bit per element) immediately after the length.
        - Serializes only the present elements in sequence (no data for `None` entries).
        - Supports arbitrary array lengths and efficient deserialization.
    """

    def __init__(self, name: str, struct_format: str):
        """
        Initialize with field name and struct format of the primitive element.

        Args:
            name (str):
                The name of the array field.

            struct_format (str):
                Struct format character describing each element's binary layout.
        """
        super().__init__(name, struct_format)

    def _serialize_element(self, elem: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize a primitive element into `buffer` at `idx`.

        Args:
            elem (Any):
                The element value to serialize.

            buffer (bytearray):
                The buffer into which to serialize the element.

            idx (int):
                The starting index in `buffer` for the element bytes.

        Returns:
            int:
                The updated buffer index after writing the element.
        """
        struct.pack_into('<' + self.struct_format, buffer, idx, elem)
        return idx + self._struct_format_byte_length

    def _deserialize_element(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize a primitive element from `buffer` at `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in `buffer` from which to read.

        Returns:
            tuple[Any, int]:
                - The deserialized element value.
                - The updated buffer index after reading the element.
        """
        val = struct.unpack_from('<' + self.struct_format, buffer, idx)[0]
        if self._is_bool:
            val = bool(val)
        return val, idx + self._struct_format_byte_length

    def _element_size(self, elem: Any) -> int:
        """
        Return the fixed serialized size of a primitive element.

        Args:
            elem (Any):
                Unused; size is constant for all elements.

        Returns:
            int:
                The number of bytes used to serialize one element.
        """
        return self._struct_format_byte_length


class FieldSpecCompiledNumpyArray(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for variable-size NumPy arrays of a fixed dtype.

    Serializes the array as:
        - 1 byte: number of dimensions (ndim, unsigned)
        - 4 * ndim bytes: each dimension size as a 4-byte unsigned int (little-endian)
        - N bytes: array data as a contiguous bytes buffer (row-major/C order)

    Only arrays with the exact dtype as specified at construction are supported.
    Raises ValueError on dtype mismatch.

    Attributes:
        dtype (np.dtype[Any]):
            The required NumPy dtype of the array (e.g., np.uint8, np.float32).
    """

    dtype: np.dtype[Any]

    def __init__(self, name: str, dtype: np.dtype[Any]):
        """
        Initialize the field for a variable-size NumPy array.

        Args:
            name (str):
                The name of the field.

            dtype (np.dtype[Any]):
                The NumPy dtype of the array.
        """
        super().__init__(name)
        self.dtype = np.dtype(dtype)

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the NumPy array field from `class_obj` into the buffer at `idx`.

        Args:
            class_obj (Any):
                The instance containing the array field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing.

        Raises:
            ValueError: If the array's dtype does not match the required dtype.
        """
        arr: NDArray[Any] = getattr(class_obj, self.name)
        if arr.dtype != self.dtype:
            raise ValueError("NDArray dtype mismatch")
        ndim = arr.ndim
        struct.pack_into("<B", buffer, idx, ndim)
        idx += 1
        struct.pack_into("<" + "I" * ndim, buffer, idx, *arr.shape)
        idx += 4 * ndim
        data = arr.tobytes(order="C")
        buffer[idx:idx + len(data)] = data
        return idx + len(data)

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize a NumPy array field from the buffer at `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[NDArray, int]:
                - The deserialized NumPy array.
                - The updated buffer index after reading.
        """
        ndim = struct.unpack_from("<B", buffer, idx)[0]
        idx += 1
        shape = struct.unpack_from("<" + "I" * ndim, buffer, idx)
        idx += 4 * ndim
        count = 1
        for dim in shape:
            count *= dim
        byte_len = count * self.dtype.itemsize
        arr = np.frombuffer(buffer, dtype=self.dtype, count=count, offset=idx).reshape(shape)
        arr = arr.copy()  # Defensive copy (frombuffer is always read-only)
        idx += byte_len
        return arr, idx

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the total number of bytes required to serialize the NumPy array field.

        Args:
            class_obj (Any):
                The instance containing the array field.

        Returns:
            int:
                The total byte size for serialization.
        """
        arr: NDArray[Any] = getattr(class_obj, self.name)
        return 1 + 4 * arr.ndim + arr.nbytes


class FieldSpecCompiledString(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for variable-length UTF-8 strings.

    Serializes the string by first writing its length as a 4-byte unsigned integer (little-endian),
    followed by the UTF-8 encoded bytes.

    On deserialization, reads the length prefix and decodes the following bytes as a UTF-8 string.

    Raises UnicodeDecodeError on invalid UTF-8 sequences.
    """

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the UTF-8 string field from `class_obj` into the buffer at `idx`.

        Writes the string length (4 bytes, little-endian unsigned int),
        followed by the UTF-8 encoded bytes.

        Args:
            class_obj (Any):
                The instance containing the string field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the length and string bytes.

        Raises:
            UnicodeEncodeError: If the string cannot be encoded as UTF-8.
        """
        data = getattr(class_obj, self.name).encode("utf-8")
        struct.pack_into("<I", buffer, idx, len(data))
        idx += 4
        buffer[idx:idx + len(data)] = data
        return idx + len(data)

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the UTF-8 string field from the buffer at `idx`.

        Reads the string length (4 bytes, little-endian unsigned int),
        then reads and decodes that many bytes as UTF-8.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[str, int]:
                - The deserialized string.
                - The updated buffer index after reading the string.

        Raises:
            UnicodeDecodeError: If the bytes cannot be decoded as UTF-8.
        """
        length = struct.unpack_from("<I", buffer, idx)[0]
        idx += 4
        data = bytes(buffer[idx:idx + length])
        return data.decode("utf-8"), idx + length

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the UTF-8 string field,
        including 4 bytes for the length prefix plus the encoded string bytes.

        Args:
            class_obj (Any):
                The instance containing the string field.

        Returns:
            int:
                The total byte size needed to serialize the string field.
        """
        return 4 + len(getattr(class_obj, self.name).encode("utf-8"))


class FieldSpecCompiledFixedString(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for fixed-length UTF-8 strings.

    Serializes the string to exactly `length` bytes:
      - If the UTF-8 encoded string is shorter than `length`, pads with ASCII spaces (' ').
      - If the encoded string is longer than `length`, truncates so the UTF-8 sequence fits,
        never splitting a multi-byte Unicode codepoint.

    On deserialization, reads exactly `length` bytes, strips trailing ASCII spaces,
    and decodes as UTF-8. Raises UnicodeDecodeError on invalid UTF-8 sequences.

    Note:
        Any trailing ASCII spaces in the original string will **not** be preserved;
        all trailing spaces (whether from padding or the original string) are removed
        during deserialization.

    Attributes:
        length (int):
            The fixed number of bytes used to store the string (after encoding).
    """

    length: int

    def __init__(self, name: str, length: int):
        """
        Initialize the field for a fixed-length UTF-8 string.

        Args:
            name (str):
                The name of the field.

            length (int):
                The fixed number of bytes to allocate for the UTF-8 encoded string.
        """
        super().__init__(name)
        self.length = length

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the fixed-length UTF-8 string field from `class_obj` into the buffer at `idx`.

        Encodes the string as UTF-8, truncating or padding with spaces as needed
        to ensure the output is exactly `self.length` bytes.
        Truncation never splits multi-byte codepoints.

        Args:
            class_obj (Any):
                The instance containing the string field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the fixed-length string.

        Raises:
            UnicodeEncodeError: If the string cannot be encoded as UTF-8.
        """
        data = getattr(class_obj, self.name).encode("utf-8")
        if len(data) > self.length:
            trunc = data[: self.length]
            while True:
                try:
                    trunc.decode("utf-8")
                    data = trunc
                    break
                except UnicodeDecodeError as exc:
                    trunc = trunc[: exc.start]
        buffer[idx:idx + len(data)] = data
        if len(data) < self.length:
            buffer[idx + len(data):idx + self.length] = b" " * (self.length - len(data))
        return idx + self.length

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the fixed-length UTF-8 string field from the buffer at `idx`.

        Reads exactly `self.length` bytes, strips trailing ASCII spaces,
        and decodes the result as a UTF-8 string.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[str, int]:
                - The deserialized string (with trailing spaces removed).
                - The updated buffer index after reading.

        Raises:
            UnicodeDecodeError: If the bytes cannot be decoded as UTF-8.
        """
        data = bytes(buffer[idx:idx + self.length])
        s = data.rstrip(b" ").decode("utf-8")
        return s, idx + self.length

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Return the fixed number of bytes used to serialize this string field.

        Args:
            class_obj (Any):
                The instance containing the string field.

        Returns:
            int:
                The fixed byte size for serialization (`self.length`).
        """
        return self.length


class FieldSpecCompiledTuple(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for fixed-size tuples of basic struct types.

    Serializes and deserializes a tuple field as a tightly packed sequence
    of fixed-size elements, according to the specified struct format string.

    The number and types of elements in the tuple are determined by the struct format,
    e.g. 'II' for (int, int) or 'If' for (int, float).

    Attributes:

        struct_format (str):
            The struct format string describing the binary layout
            (e.g., 'Ii', 'ffff', 'BB').

        _struct_format_byte_length (int):
            The fixed size in bytes computed from `struct_format`,
            representing the number of bytes required to serialize
            the tuple value.

        _bool_flags (list[bool]):
            A list indicating which positions in the original `struct_format` were
            `'y'` (booleans).
            Each element is `True` if the corresponding format character was `'y'`,
            otherwise `False`.

        _has_bool (bool):
            True if at least one `'y'` (boolean) was present in the original `struct_format`.
    """

    _bool_flags: list[bool]
    _has_bool: bool
    struct_format: str
    _struct_format_byte_length: int

    def __init__(self, name: str, struct_format: str):
        super().__init__(name)
        self._bool_flags = [c == "y" for c in struct_format]
        self._has_bool = any(self._bool_flags)
        self.struct_format = struct_format.replace("y", "B")
        self._struct_format_byte_length = struct.calcsize('<' + self.struct_format)

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the tuple field from `class_obj` into the buffer starting at `idx`.

        Writes the tuple elements in order using the struct format and little-endian encoding.
        No length prefix is included, as the size and structure are fixed by the format.

        Args:
            class_obj (Any):
                The instance containing the tuple field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the tuple data.

        Raises:
            Implementation-specific exceptions if serialization fails,
            such as a mismatch in tuple length or type.
        """
        values = getattr(class_obj, self.name)
        struct.pack_into('<' + self.struct_format, buffer, idx, *values)
        return idx + self._struct_format_byte_length

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the tuple field from the buffer starting at `idx`.

        Reads a fixed number of elements according to the struct format,
        returning them as a tuple.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized tuple of values.
                - The updated buffer index after reading the tuple.

        Raises:
            Implementation-specific exceptions if deserialization fails,
            such as if not enough data is available in the buffer.
        """
        obj = struct.unpack_from('<' + self.struct_format, buffer, idx)
        if self._has_bool:
            obj = tuple(
                bool(v) if flag else v
                for v, flag in zip(obj, self._bool_flags)
            )
        return obj, idx + self._struct_format_byte_length

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the tuple field.

        The size is fixed and determined entirely by the struct format string.

        Args:
            class_obj (Any):
                The instance containing the tuple field.

        Returns:
            int:
                The total byte size needed to serialize the tuple.
        """
        return self._struct_format_byte_length


class FieldSpecCompiledGeneric(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for nested serializable objects.

    Handles serialization and deserialization by delegating to the nested object's
    own serialization methods (`serialize_to_bytes`, `deserialize_from_bytes`,
    `serialized_byte_size`).

    Attributes:
        ptype (FifoSerializable):
            The type of the nested serializable object.
    """

    ptype: FifoSerializable

    def __init__(self, name: str, ptype: FifoSerializable):
        """
        Initialize with the field name and the nested serializable type.

        Args:
            name (str):
                The name of the field.

            ptype (FifoSerializable):
                The nested serializable class type.
        """
        super().__init__(name)
        self.ptype = ptype

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the nested serializable field from `class_obj` into the buffer.

        Delegates serialization to the nested object's `serialize_to_bytes` method.

        Args:
            class_obj (Any):
                The instance containing the nested serializable field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing the nested serialized data.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        return getattr(class_obj, self.name).serialize_to_bytes(buffer, idx)

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the nested serializable field from the buffer starting at `idx`.

        Delegates deserialization to the nested type's `deserialize_from_bytes` class method.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized nested object.
                - The updated buffer index after reading the nested data.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        return self.ptype.deserialize_from_bytes(buffer, idx)

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the byte size required to serialize the nested serializable field.

        Delegates size computation to the nested object's `serialized_byte_size` method.

        Args:
            class_obj (Any):
                The instance containing the nested serializable field.

        Returns:
            int:
                The number of bytes required for the nested serialized field.
        """
        return getattr(class_obj, self.name).serialized_byte_size()


class FieldSpecCompiledGenericArray(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for variable-length arrays of nested serializable objects.

    Serializes an array by writing a 4-byte length prefix followed by the serialization
    of each element using the element's own serialization methods.

    Attributes:
        ptype (FifoSerializable):
            The type of the nested serializable elements in the array.
    """

    ptype: FifoSerializable

    def __init__(self, name: str, ptype: FifoSerializable):
        """
        Initialize with field name and nested serializable element type.

        Args:
            name (str):
                The name of the field.

            ptype (FifoSerializable):
                The class type of elements contained in the array.
        """
        super().__init__(name)
        self.ptype = ptype

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the array field from `class_obj` into the buffer starting at `idx`.

        Writes the length of the array (4 bytes, little-endian unsigned int),
        followed by serializing each element in order.

        Args:
            class_obj (Any):
                The instance containing the array field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing all elements.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        array = getattr(class_obj, self.name)
        length = len(array)
        struct.pack_into("<I", buffer, idx, length)
        idx += 4
        for elem in array:
            idx = elem.serialize_to_bytes(buffer, idx)
        return idx

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize the array field from the buffer starting at `idx`.

        Reads the length prefix, then deserializes that many elements
        from the buffer using the element's deserialization method.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            tuple[Any, int]:
                - The list of deserialized elements.
                - The updated buffer index after reading the array.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        length, = struct.unpack_from("<I", buffer, idx)
        idx += 4
        items: list[Any] = []
        for _ in range(length):
            item, idx = self.ptype.deserialize_from_bytes(buffer, idx)
            items.append(item)
        return items, idx

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the number of bytes required to serialize the array field,
        including 4 bytes for the length prefix plus the sum of all serialized element sizes.

        Args:
            class_obj (Any):
                The instance containing the array field.

        Returns:
            int:
                The total byte size needed to serialize the entire array field.
        """
        array = getattr(class_obj, self.name)
        total = 4
        for elem in array:
            total += elem.serialized_byte_size()
        return total


class FieldSpecCompiledGenericOptional(FieldSpecCompiled):
    """
    FieldSpecCompiled subclass for optional nested serializable objects.

    Serializes the field with a 1-byte presence flag followed by the serialized
    nested object if present. If the object is None, only the presence byte is stored.
    
    Attributes:
        ptype (FifoSerializable):
            The type of the nested serializable object.
    """

    ptype: FifoSerializable

    def __init__(self, name: str, ptype: FifoSerializable):
        """
        Initialize with field name and nested serializable type.

        Args:
            name (str):
                The name of the field.

            ptype (FifoSerializable):
                The class type of the nested serializable object.
        """
        super().__init__(name)
        self.ptype = ptype

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the optional nested object field from `class_obj` into the buffer.

        Writes a 1-byte presence flag (0 if None, 1 if present) followed by the
        serialized nested object if present.

        Args:
            class_obj (Any):
                The instance containing the optional nested object.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after writing presence flag and optional data.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        obj = getattr(class_obj, self.name)
        if obj is None:
            struct.pack_into("<b", buffer, idx, 0)
            return idx + 1
        struct.pack_into("<b", buffer, idx, 1)
        idx += 1
        idx = obj.serialize_to_bytes(buffer, idx)
        return idx

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize the optional nested object field from the buffer starting at `idx`.

        Reads a 1-byte presence flag; if zero, returns None. Otherwise,
        deserializes the nested object using the nested type's method.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            tuple[Any, int]:
                - The deserialized nested object or None if not present.
                - The updated buffer index after reading the data.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        present = struct.unpack_from("<b", buffer, idx)[0]
        idx += 1
        if not present:
            return None, idx
        obj, idx = self.ptype.deserialize_from_bytes(buffer, idx)
        return obj, idx

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the byte size required to serialize the optional nested object field.

        Returns 1 if the field is None (presence flag only), otherwise
        1 plus the size of the nested object's serialized form.

        Args:
            class_obj (Any):
                The instance containing the optional nested object.

        Returns:
            int:
                The number of bytes needed to serialize this field.
        """
        obj = getattr(class_obj, self.name)
        if obj is None:
            return 1
        return 1 + obj.serialized_byte_size()


class FieldSpecCompiledGenericOptionalArray(_FieldSpecCompiledOptionalArrayBase):
    """
    FieldSpecCompiled subclass for arrays of optional nested serializable objects.

    Supports serializing and deserializing fields where each element is either an instance of a
    serializable class or None. Serialization is efficient and compact, encoding presence with a
    bitmask.

    Features:
        - Serializes array length as a 4-byte unsigned int.
        - Serializes a packed presence bitmap (1 bit per element) directly after the length.
        - Serializes only present elements in order (no data for None elements).
        - Handles arbitrary array lengths and supports efficient deserialization.

    Args:
        ptype (FifoSerializable):
            The serializable element type for array elements.
    """

    ptype: FifoSerializable

    def __init__(self, name: str, ptype: FifoSerializable):
        """
        Initialize with field name and nested serializable type for the array.

        Args:
            name (str):
                The name of the array field.

            ptype (FifoSerializable):
                The serializable element type for array elements.
        """
        super().__init__(name)
        self.ptype = ptype

    def _serialize_element(self, elem: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize a nested element into `buffer` at `idx`.

        Args:
            elem (Any):
                The element value to serialize.

            buffer (bytearray):
                The buffer into which to serialize the element.

            idx (int):
                The starting index in `buffer` for the element bytes.

        Returns:
            int:
                The updated buffer index after writing the element.
        """
        return elem.serialize_to_bytes(buffer, idx)

    def _deserialize_element(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize a nested element from `buffer` at `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in `buffer` from which to read.

        Returns:
            tuple[Any, int]:
                - The deserialized nested element.
                - The updated buffer index after reading the element.
        """
        return self.ptype.deserialize_from_bytes(buffer, idx)

    def _element_size(self, elem: Any) -> int:
        """
        Return the fixed serialized size of a nested element.

        Args:
            elem (Any):
                The element whose serialized size should be computed.

        Returns:
            int:
                The number of bytes used to serialize one nested element.
        """
        return elem.serialized_byte_size()


C = TypeVar('C', bound=type)


class FifoSerializable:
    """
    Base interface class for serializable objects using the FIFO serialization protocol
    defined by this module.

    The actual implementations of `serialize_to_bytes`, `deserialize_from_bytes`,
    and `serialized_byte_size` are injected by the `@serializable` decorator,
    so subclasses do not need to implement them manually.

    This class primarily exists for typing, interface enforcement, and documentation.
    """

    def serialize_to_bytes(self, buffer: bytearray, idx: int) -> int:
        """
        Serialize the object into the given buffer starting at index `idx`.

        Args:
            buffer (bytearray):
                The buffer into which to serialize the data.

            idx (int):
                The starting index in the buffer at which to write data.

        Returns:
            int:
                The updated buffer index after serialization.
        """
        raise RuntimeError("not implemented yet") # pragma: no cover

    @classmethod
    def deserialize_from_bytes(cls, buffer: bytearray, idx: int) -> Tuple[Self, int]:
        """
        Deserialize an instance from the given buffer starting at index `idx`.

        Args:
            buffer (bytearray):
                The buffer containing the serialized data.

            idx (int):
                The starting index in the buffer at which to read data.

        Returns:
            Tuple[Self, int]:
                - The deserialized instance of the class.
                - The updated buffer index after deserialization.
        """
        raise RuntimeError("not implemented yet") # pragma: no cover

    def serialized_byte_size(self) -> int:
        """
        Return the number of bytes required to serialize this object.

        Returns:
            int:
                The byte size needed for serialization.
        """
        raise RuntimeError("not implemented yet") # pragma: no cover

    def serialize_to_socket(self, sock: SupportsSendAll):
        """
        Serialize the object and send it over the given socket with a 4-byte length prefix.

        This method computes the serialized byte size, allocates a buffer,
        writes the length as a 4-byte little-endian unsigned integer, serializes
        the object into the buffer, and sends the entire buffer using `sock.sendall()`.

        Args:
            sock (SupportsSendAll):
                A socket-like object that supports `sendall()` for writing bytes.
        """
        length = self.serialized_byte_size()
        buffer = bytearray(4 + length)
        struct.pack_into("<I", buffer, 0, length)
        self.serialize_to_bytes(buffer, 4)
        sock.sendall(buffer)

    @classmethod
    def deserialize_from_socket(cls, sock: SupportsRecvInto) -> Self:
        """
        Receive and deserialize an object from the given socket.

        Reads a 4-byte length prefix to determine the size of the incoming message,
        then reads the specified number of bytes from the socket and deserializes
        the object using the class's `deserialize_from_bytes()` method.

        Args:
            sock (SupportsRecvInto):
                A socket-like object that supports `recv_into()` for reading bytes.

        Returns:
            Self:
                The deserialized instance of the class.

        Raises:
            ConnectionError:
                If the socket is closed or incomplete data is received.
        """
        length_bytes = recv_all(sock, 4)

        if len(length_bytes) != 4:
            raise ConnectionError(
                f"Incomplete message header: expected 4 bytes, got {len(length_bytes)}"
            )

        length, = struct.unpack("<I", length_bytes)
        payload = recv_all(sock, length)

        obj, _ = cls.deserialize_from_bytes(payload, 0)
        return obj


def serializable(cls: C) -> C:
    """
    Class decorator that adds FIFO serialization methods to a dataclass.

    This decorator inspects the dataclass fields' metadata to compile
    serialization logic, then injects three methods into the class:

        - serialize_to_bytes(self, buffer: bytearray, idx: int) -> int
        - deserialize_from_bytes(cls, buffer: bytearray, idx: int) -> Tuple[cls, int]
        - serialized_byte_size(self) -> int

    The injected methods handle binary serialization and deserialization
    of the class fields according to their specified formats and types.

    If the class defines a classmethod named `from_deserialized_fields(**kwargs)`,
    it will be used instead of the regular constructor during deserialization.
    This allows support for classes that require custom initialization logic
    or that use private fields not accepted by `__init__`.

    Args:
        cls (Type[C]):
            The dataclass type to be enhanced with serialization and deserialization methods.

    Returns:
        Type[C]:
            The same class type with added serialization and deserialization capabilities.
    """
    compiled_fields: list[FieldSpecCompiled] = []
    for f in fields(cls):
        # Only serialize fields with 'format' or 'ptype' in metadata
        meta = getattr(f, "metadata", None)
        if meta and ("format" in meta or "ptype" in meta):
            compiled_fields.append(compile_field(f))

    setattr(cls, "_fifo_compiled_fields", compiled_fields)

    def serialize_to_bytes(self: C, buffer: bytearray, idx: int) -> int:
        for field in compiled_fields:
            idx = field.serialize_to_bytes(self, buffer, idx)
        return idx

    def deserialize_from_bytes(cls: Type[C], buffer: bytearray, idx: int) -> Tuple[C, int]:
        args = {}
        for field in compiled_fields:
            obj, idx = field.deserialize_from_bytes(buffer, idx)
            args[field.name] = obj

        # If the class defines a custom deserialization constructor, use it first
        factory = getattr(cls, "from_deserialized_fields", None)
        if callable(factory):
            return factory(**args), idx  # type: ignore

        # if not present then call the regular constructor
        return cls(**args), idx  # type: ignore

    def serialized_byte_size(self: C) -> int:
        return sum(field.serialized_byte_size(self) for field in compiled_fields)

    setattr(cls, "serialize_to_bytes", serialize_to_bytes)
    setattr(cls, "deserialize_from_bytes", classmethod(deserialize_from_bytes))
    setattr(cls, "serialized_byte_size", serialized_byte_size)

    return cls
