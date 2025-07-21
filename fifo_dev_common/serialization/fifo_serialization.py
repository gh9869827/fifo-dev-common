"""
fifo_event.py

Serializable base class for events used in `fifo-*` projects, suitable for cross-platform use
(including microcontrollers).

- Event ID is always serialized as a 4-byte unsigned int ('I'), little-endian.
- Event priority is serialized as a 4-byte signed int ('i'), immediately after the event ID.
- Subclasses should override `event_id` and declare serializable fields using the
  @serializable class decorator and `FieldSpec` descriptors.
- Field serialization supports composite fields and custom packing/unpacking logic via
  optional `from_fields` and `to_fields` callables.
"""

from __future__ import annotations
from abc import ABC
import abc
from dataclasses import Field, fields
from enum import Enum
import struct
from typing import Any, Self, Tuple, Type, TypeVar

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
            if struct_format == "[?_]":
                if ptype is None:
                    raise ValueError("Type must be provided for optional array")
                return FieldSpecCompiledGenericOptionalArray(name, ptype)

            if not (len(struct_format) == 3 and struct_format[2] == "]"):
                raise ValueError("Struct format for array type invalid")

            if struct_format == "[_]":
                if ptype is None:
                    raise ValueError("Type must be provided for generic array")
                return FieldSpecCompiledGenericArray(name, ptype)

            return FieldSpecCompiledArray(name, struct_format[1])

        if struct_format[0] == "?":
            if not len(struct_format) == 2:
                raise ValueError("Struct format for optional type invalid")

            if struct_format == "?_":
                if ptype is None:
                    raise ValueError("Type must be provided for generic optional")
                return FieldSpecCompiledGenericOptional(name, ptype)

            return FieldSpecCompiledOptional(name, struct_format[1])

        if struct_format.startswith("E<"):
            if not (len(struct_format) == 4 and struct_format[3] == ">"):
                raise ValueError("Struct format for enum type invalid")

            inner_type = struct_format[2]

            if inner_type != "I":
                raise ValueError("Struct only support I format for now")

            if ptype is None:
                raise ValueError("Type must be provided for Struct")

            return FieldSpecCompiledEnum(name, inner_type, ptype)

        if struct_format.startswith("T<"):
            if not (len(struct_format) >= 4 and struct_format[-1] == ">"):
                raise ValueError("Struct format for tuple type invalid")
            inner_types = struct_format[2:-1]
            if len(inner_types) == 0:
                raise ValueError("Struct format for tuple type invalid")
            try:
                struct.calcsize('<' + inner_types)
            except struct.error as exc:
                raise ValueError("Struct format for tuple type invalid") from exc
            return FieldSpecCompiledTuple(name, inner_types)

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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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
    """

    struct_format: str
    _struct_format_byte_length: int

    def __init__(self, name: str, struct_format: str):
        """
        Initialize with field name and struct format string.

        Args:
            name (str):
                The name of the field.

            struct_format (str):
                The struct format string defining the binary layout.
        """
        super().__init__(name)
        self.struct_format = struct_format
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
                The starting index in the buffer to write data.

        Returns:
            int:
                The updated buffer index after writing the field data.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        struct.pack_into('<' + self.struct_format, buffer, idx, getattr(class_obj, self.name))
        return idx + self._struct_format_byte_length

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> Tuple[Any, int]:
        """
        Deserialize the field's value from the `buffer` starting at index `idx`.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized field value.
                - The updated buffer index after reading the field data.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        obj = struct.unpack_from('<' + self.struct_format, buffer, idx)
        if len(obj) == 1:
            obj = obj[0]
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
                The starting index in the buffer to read data.

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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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

        obj = struct.unpack_from(self.struct_format, buffer, idx)
        if len(obj) == 1:
            obj = obj[0]
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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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


class FieldSpecCompiledTuple(FieldSpecCompiledBasic):
    """
    FieldSpecCompiled subclass for fixed-size tuples of basic struct types.

    Serializes and deserializes a tuple field as a tightly packed sequence
    of fixed-size elements, according to the specified struct format string.

    The number and types of elements in the tuple are determined by the struct format,
    e.g. 'II' for (int, int) or 'If' for (int, float).
    """

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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

        Returns:
            Tuple[Any, int]:
                - The deserialized tuple of values.
                - The updated buffer index after reading the tuple.

        Raises:
            Implementation-specific exceptions if deserialization fails,
            such as if not enough data is available in the buffer.
        """
        obj = struct.unpack_from('<' + self.struct_format, buffer, idx)
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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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


class FieldSpecCompiledGenericOptionalArray(FieldSpecCompiled):
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

    def serialize_to_bytes(self, class_obj: Any, buffer: bytearray, idx: int) -> int:
        """
        Serialize the array of optional nested serializable objects.

        Writes the following to the buffer, starting at the given index:
            - 4 bytes: array length (unsigned int, little-endian)
            - N bits (packed into bytes): presence bitmap (1 if element is present, 0 if None)
            - Serialized payloads of present elements, in order.

        Args:
            class_obj (Any):
                The instance containing the array field.

            buffer (bytearray):
                The buffer into which to serialize data.

            idx (int):
                The starting index in the buffer to write data.

        Returns:
            int:
                The updated buffer index after writing the entire array.

        Raises:
            Implementation-specific exceptions if serialization fails.
        """
        array: list[Any | None] = getattr(class_obj, self.name)
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
                idx = elem.serialize_to_bytes(buffer, idx)

        return idx

    def deserialize_from_bytes(self, buffer: bytearray, idx: int) -> tuple[Any, int]:
        """
        Deserialize an array of optional nested serializable objects from a buffer.

        Reads:
            - 4 bytes: array length (unsigned int, little-endian)
            - N bits (packed into bytes): presence bitmap (1 if element is present, 0 if None)
            - Serialized payloads of present elements, in order.

        Args:
            buffer (bytearray):
                The buffer containing serialized data.

            idx (int):
                The starting index in the buffer to read data.

        Returns:
            tuple[list[Any | None], int]:
                - The deserialized list of elements (either instances or None).
                - The updated buffer index after reading the entire array.

        Raises:
            Implementation-specific exceptions if deserialization fails.
        """
        length, = struct.unpack_from("<I", buffer, idx)
        idx += 4

        bitmap_len = (length + 7) // 8
        bitmap = buffer[idx:idx + bitmap_len]
        idx += bitmap_len

        items: list[Any | None] = []
        for i in range(length):
            if bitmap[i // 8] & (1 << (i % 8)):
                item, idx = self.ptype.deserialize_from_bytes(buffer, idx)
                items.append(item)
            else:
                items.append(None)
        return items, idx

    def serialized_byte_size(self, class_obj: Any) -> int:
        """
        Compute the total number of bytes required to serialize the array field.

        Includes:
            - 4 bytes for the array length prefix
            - N bits (rounded up to bytes) for the presence bitmap
            - Only the size of present (non-None) elements

        Args:
            class_obj (Any):
                The instance containing the array field.

        Returns:
            int:
                The total number of bytes needed to serialize the field.
        """
        array: list[Any | None] = getattr(class_obj, self.name)
        total = 4 + (len(array) + 7) // 8
        for elem in array:
            if elem is not None:
                total += elem.serialized_byte_size()
        return total


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
                The starting index in the buffer to write data.

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
                The starting index in the buffer to read data.

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

    Args:
        cls (Type[C]):
            The dataclass type to be enhanced with serialization methods.

    Returns:
        Type[C]:
            The same class type with added serialization and deserialization methods.
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
        return cls(**args), idx  # type: ignore

    def serialized_byte_size(self: C) -> int:
        return sum(field.serialized_byte_size(self) for field in compiled_fields)

    setattr(cls, "serialize_to_bytes", serialize_to_bytes)
    setattr(cls, "deserialize_from_bytes", classmethod(deserialize_from_bytes))
    setattr(cls, "serialized_byte_size", serialized_byte_size)

    return cls
