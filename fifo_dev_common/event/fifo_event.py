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
from dataclasses import dataclass, field
import struct
from typing import Any, Tuple, Type, TypeVar, Callable, ClassVar, cast
import threading


@dataclass
class FieldSpec:
    """
    Describes a serializable field in a FifoEvent.

    Attributes:
        name (str):
            Name of the attribute on the event instance.

        struct_format (str):
            Struct format string (e.g., 'i', 'f', 'B', 'II', '10B') for this field.

        from_fields (Callable[[Any], Any] | None):
            Optional function to convert the unpacked value(s) after deserialization
            (e.g., reconstruct a dataclass or cast an int to an Enum type).

            This function receives the field's raw value as a scalar (for single-value formats)
            or a tuple (for composite fields, e.g., 'II'). It should return the decoded object.

        to_fields (Callable[[Any], Any | Tuple[Any, ...] | list[Any]] | None):
            Optional function to convert the attribute value before serialization.
            Should return either a single value, a tuple, or a list matching the struct format.
            For example, an Enum's .value, or a tuple of integers from a dataclass.

        field_count (int):
            The number of struct fields this FieldSpec serializes/deserializes.
            Automatically computed.
    """
    name: str
    struct_format: str
    from_fields: Callable[[Any], Any] | None = None
    to_fields: Callable[[Any], Any | Tuple[Any, ...] | list[Any]] | None = None
    field_count: int = field(init=False)

    def __post_init__(self):
        size = struct.calcsize('<' + self.struct_format)
        # The number of fields is the length of the tuple unpacked from a dummy buffer
        self.field_count = len(struct.unpack('<' + self.struct_format, b'\x00' * size))


T = TypeVar('T', bound="FifoEvent")

def serializable(fields: list[FieldSpec]) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator to declare which fields are serialized, and their struct formats.

    Args:
        fields (list[FieldSpec]):
            List of (field_name, struct_format) tuples.
            `struct_format` must be a format character supported by `struct.pack`
            (e.g., 'i' for int32, 'f' for float32, 'h' for int16, etc.).
    """
    def wrapper(cls: Type[T]) -> Type[T]:
        # Accessing protected class variable to store the fields that can be serialized.
        cls._fifo_fields = fields  # pylint: disable=protected-access  # pyright: ignore[reportPrivateUsage]
        return cls
    return wrapper


class FifoEvent:
    """
    Base class for all serializable events in the `fifo-*` projects.

    Subclasses should override the `event_id` class attribute with a unique integer,
    and declare serializable fields using the @serializable class decorator.

    Provides methods for binary serialization and deserialization, suitable for use
    across different platforms (including microcontrollers).

    Attributes:
        event_id (int):
            Class Attribute. Unique event type identifier.
            Must be overridden by subclasses.

        default_priority (int):
            Class Attribute. Default priority for sorting and queueing.
            Can be overridden per instance.

        priority (int):
            Instance attribute. Actual priority used for queueing/comparison.

        _registry_lock (ClassVar[threading.Lock]):
            Class-level threading lock used to synchronize access to the event class registry
            (_registry).
            Ensures thread-safe registration and lookup of event subclasses during serialization and
            deserialization.

        _registry (dict[int, Type[FifoEvent]]):
            Class-level registry mapping event_id to event subclass for deserialization.

        _fifo_fields (ClassVar[list[FieldSpec]]):
            Class attribute. List of (field_name, struct_format) tuples specifying serializable
            fields for the event.
    """
    event_id: int = -1  # will be overridden in subclasses

    default_priority: int = 0  # can be overridden in subclasses

    _registry_lock: ClassVar[threading.Lock] = threading.Lock()
    _registry: dict[int, Type["FifoEvent"]] = {}

    _fifo_fields: ClassVar[list[FieldSpec]] = []

    def __init__(self, priority: int | None = None):
        if priority is not None:
            self.priority = priority
        else:
            self.priority = self.default_priority

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, FifoEvent):
            return NotImplemented
        return self.priority < other.priority

    @classmethod
    def register(cls, event_cls: Type[T]) -> Type[T]:
        """
        Decorator to register an event subclass with its event_id for the factory.

        Args:
            event_cls (Type[FifoEvent]):
                The event subclass to register. Must define a unique `event_id` class attribute.
        
        Usage:
            @FifoEvent.register
            @serializable([...])
            class MyEvent(FifoEvent):
                event_id = ...
                ...
        """
        event_id = getattr(event_cls, 'event_id', -1)
        if event_id == -1:
            raise AttributeError(f"Class {event_cls.__name__} must define an 'event_id' attribute.")

        with cls._registry_lock:
            if event_id in cls._registry:
                raise ValueError(f"Duplicate event_id {event_id} "
                                 f"for class {event_cls.__name__}.")
            cls._registry[event_id] = event_cls

        return event_cls

    @classmethod
    def get_fields(cls) -> list[FieldSpec]:
        """
        Returns the list of FieldSpec objects for this event.

        Returns:
            list[FieldSpec]:
                FieldSpec entries describing the serializable fields for the event.
        """
        return cast(list[FieldSpec], getattr(cls, '_fifo_fields', []))

    def to_bytes(self) -> bytes:
        """
        Serializes the event instance (including event_id and priority) to bytes.

        Returns:
            bytes:
                The serialized event as a byte string, including the 4-byte event_id,
                4-byte priority (int32), and all declared fields.
        """
        fmt = '<Ii' + ''.join(field.struct_format for field in self.get_fields())
        values = [
            self.__class__.event_id,
            self.priority
        ]
        # Apply per-field to_fields if set, else take the attribute as-is
        for fifo_field in self.get_fields():
            v = getattr(self, fifo_field.name)
            if fifo_field.to_fields:
                v = fifo_field.to_fields(v)
                if isinstance(v, list) or isinstance(v, tuple):
                    values += v
                else:
                    values.append(v)
            else:
                values.append(v)
        return struct.pack(fmt, *values)

    @classmethod
    def from_bytes(cls, data: bytes) -> FifoEvent:
        """
        Factory: Deserializes a FifoEvent subclass instance from bytes,
        dispatching based on the 4-byte event_id at the start.

        Args:
            data (bytes):
                Serialized event packet (event_id + priority + fields).

        Returns:
            FifoEvent:
                An instance of the correct subclass.

        Raises:
            ValueError: If the event_id is unknown or payload size is incorrect.
        """
        if len(data) < 8:
            raise ValueError("Data too short to contain event_id and priority.")

        # Extract event_id and priority
        event_id, priority = struct.unpack('<Ii', data[:8])

        # Retrieve the corresponding subclass from the registry
        with cls._registry_lock:
            subclass = cls._registry.get(event_id)
        if subclass is None:
            raise ValueError(f"Unknown event_id {event_id} in packet.")

        fields = subclass.get_fields()

        # Compute expected payload size and validate
        fmt = '<' + ''.join(field.struct_format for field in fields)
        expected_size = 8 + struct.calcsize(fmt)
        if len(data) != expected_size:
            raise ValueError(
                f"Incorrect packet size for event_id {event_id}: "
                f"expected {expected_size}, got {len(data)}"
            )

        # Unpack fields (after event_id and priority) efficiently, no copy
        all_values = struct.unpack_from(fmt, data, offset=8)
        obj = subclass.__new__(subclass)
        obj.priority = priority
        i = 0
        while i < len(all_values):
            # for field, value in zip(fields, all_values):
            fifo_field = fields[i]
            if fifo_field.from_fields:
                assert fifo_field.field_count > 0
                if fifo_field.field_count == 1:
                    value = fifo_field.from_fields(all_values[i])
                else:
                    value = fifo_field.from_fields(all_values[i:i + fifo_field.field_count])
                setattr(obj, fifo_field.name, value)
                i += fifo_field.field_count
            else:
                setattr(obj, fifo_field.name, all_values[i])
                i += 1

        return obj
