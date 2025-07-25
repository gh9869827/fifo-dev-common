from __future__ import annotations
from dataclasses import dataclass, field
import struct
from typing import Any, Type, TypeVar, ClassVar
import threading

from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable
from fifo_dev_common.socket.socket_utils import SupportsRecvInto, SupportsSendAll, recv_all

T = TypeVar('T', bound='FifoEvent')

@serializable
@dataclass(kw_only=True)
class FifoEvent(FifoSerializable):
    """
    Base class for all binary-serializable, priority-aware events in the `fifo-*` projects.

    Subclasses must define a unique `event_id` (class attribute)
    and should be decorated with both @serializable and @dataclass (kw_only recommended).

    Features:
        - Compact binary serialization for event exchange across embedded, desktop,
          or distributed systems.
        - Event ID-based factory registration for deserialization (see `@FifoEvent.register`).
        - Priority field and comparison support for event queueing.
        - Thread-safe registry for event types.

    Attributes:
        priority (int):
            Instance-level event priority.
            Defaults to -1, which is replaced by `default_priority` at instantiation
            if not specified.
            Serialized as a standard field.

        event_id (ClassVar[int]):
            Unique integer identifier for this event type.

            - **Must be overridden** by each subclass.
            - Serialized as the first 4 bytes in every event packet.
            - Used by the event registry to map event IDs to Python classes during deserialization.

        default_priority (ClassVar[int]):
            Class-level default priority, used if no instance-level value is provided to set a
            value to the `priority` field.

        _registry_lock (ClassVar[threading.Lock]):
            Internal lock for thread-safe access to the class registry.

        _registry (ClassVar[dict[int, Type[FifoEvent]]]):
            Maps event_id to event subclasses for deserialization.
    """
    priority: int = field(default=-1, metadata={"format": "i"})

    # Class-level (not serialized) attributes
    event_id: ClassVar[int] = -1
    default_priority: ClassVar[int] = 0
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()
    _registry: ClassVar[dict[int, Type[FifoEvent]]] = {}

    def __post_init__(self):
        """
        After dataclass initialization, assigns `priority` from the class default
        if it was left unset (-1).
        """
        if self.priority == -1:
            self.priority = type(self).default_priority

    def __lt__(self, other: Any) -> bool:
        """
        Compare events for priority-based sorting.

        Args:
            other (Any):
                Another event instance.

        Returns:
            bool:
                True if this event's priority is less than `other`'s.
        """
        if not isinstance(other, FifoEvent):
            return NotImplemented
        return self.priority < other.priority

    @classmethod
    def register(cls, event_cls: Type[T]) -> Type[T]:
        """
        Registers the event class for deserialization by event ID.

        - Associates the class's `event_id` with the class in a global registry.
        - Enables reconstructing event objects from raw bytes based on their ID.

        Args:
            event_cls (Type[T]):
                The class to register. Must define `event_id`.

        Returns:
            Type[T]:
                The registered class.

        Raises:
            AttributeError: If the class does not define `event_id`.
            ValueError: If the event_id is already registered.
        """
        event_id = getattr(event_cls, 'event_id', -1)
        if event_id == -1:
            raise AttributeError(f"Class {event_cls.__name__} must define an 'event_id' attribute.")

        with cls._registry_lock:
            if event_id in cls._registry:
                raise ValueError(f"Duplicate event_id {event_id} for class {event_cls.__name__}.")
            cls._registry[event_id] = event_cls

        return event_cls

    def to_bytes(self) -> bytearray:
        """
        Serialize the event instance to a binary buffer (little-endian encoding).

        The serialized format is:
            [event_id (4 bytes)] + [payload (includes priority and all fields)].

        Returns:
            bytearray:
                The complete serialized event packet.
        """
        payload_size = self.serialized_byte_size()

        buffer = bytearray(4 + payload_size)

        # Write header: the event id
        struct.pack_into("<I", buffer, 0, self.event_id)

        # Write payload after the header
        self.serialize_to_bytes(buffer, 4)

        return buffer

    def serialize_to_socket(self, sock: SupportsSendAll):
        """
        Serialize the object and send it over the given socket with a 4-byte length prefix.

        The serialized format is:
            [length (4 bytes)] + [event_id (4 bytes)] + [payload]

        This method computes the serialized byte size, allocates a single buffer,
        writes the total message length and event ID, serializes the payload, and
        sends the entire buffer using `sock.sendall()`.

        Args:
            sock (SupportsSendAll):
                A socket-like object that supports `sendall()` for writing bytes.

        Note:
            This method intentionally duplicates part of the logic from `to_bytes()`
            to avoid multiple memory allocations. Calling `to_bytes()` would result
            in one allocation for the payload and another for prefixing with the length.
            This version constructs the full framed message in a single buffer for efficiency.
        """
        payload_size = self.serialized_byte_size()
        length = payload_size + 4  # +4 for the event ID

        buffer = bytearray(4 + length)

        # Write the length and the event id
        struct.pack_into("<II", buffer, 0, length, self.event_id)

        # Write payload
        self.serialize_to_bytes(buffer, 8)  # +8 for the length and the event ID

        sock.sendall(buffer)

    @classmethod
    def from_bytes(cls, data: bytearray) -> FifoEvent:
        """
        Deserialize a binary buffer to an event instance.

        Args:
            data (bytearray):
                The binary buffer starting with 4-byte event_id.

        Returns:
            FifoEvent:
                The deserialized event instance.

        Raises:
            ValueError:
                If the event_id is unknown or data is too short.
        """
        if len(data) < 4:
            raise ValueError("Data too short to contain event_id.")
        # Deserialize header
        event_id = struct.unpack_from("<I", data, 0)[0]
        with cls._registry_lock:
            subclass = cls._registry.get(event_id)
        if subclass is None:
            raise ValueError(f"Unknown event_id {event_id} in packet.")
        # Parse the payload
        obj, _ = subclass.deserialize_from_bytes(data, 4)

        return obj

    @classmethod
    def deserialize_from_socket(cls, sock: SupportsRecvInto) -> FifoEvent:
        """
        Receive and deserialize a FIFO event from the given socket.

        This method reads a 4-byte length prefix to determine the size of the
        incoming message, then reads the specified number of bytes from the socket.
        It then delegates deserialization to `from_bytes()`, which performs
        event ID dispatch and constructs the appropriate subclass instance.

        Args:
            sock (SupportsRecvInto):
                A socket-like object that supports `recv_into()` for reading bytes.

        Returns:
            FifoEvent:
                The deserialized event instance.

        Raises:
            ConnectionError:
                If the socket is closed or incomplete data is received.

            ValueError:
                If the event ID is unknown or the buffer is too short to decode.
        """
        length_bytes = recv_all(sock, 4)

        if len(length_bytes) != 4:
            raise ConnectionError(
                f"Incomplete message header: expected 4 bytes, got {len(length_bytes)}"
            )

        length, = struct.unpack("<I", length_bytes)
        payload = recv_all(sock, length)

        return cls.from_bytes(payload)

    @classmethod
    def clear_registry(cls):
        """
        Clear the event class registry.

        Useful for test isolation.
        """
        with cls._registry_lock:
            cls._registry.clear()


@FifoEvent.register
@serializable
class FifoEventPoison(FifoEvent):
    """
    Sentinel event used as a "poison pill" for queues or pipelines.

    In concurrent or asynchronous systems, inserting a FifoEventPoison into a queue
    signals consumers to terminate gracefully. This event carries no data payload, 
    only an event ID and optional priority.

    Usage:
        queue.put(FifoEventPoison())
        # Consumer:
        event = queue.get()
        if isinstance(event, FifoEventPoison):
            # Clean shutdown
            break
    """
    event_id = 0
    default_priority = 0
