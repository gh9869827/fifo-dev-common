from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
import struct
import time
from typing import Any, Awaitable, Callable, Type, TypeVar, ClassVar
import threading
from uuid import UUID, uuid4
from fifo_dev_common.serialization.fifo_serialization import field_meta_serialize_handler_uuid
from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable
from fifo_dev_common.socket.socket_utils import (
    SupportsRecvInto,
    SupportsSendAll,
    SupportsRead,
    SupportsWrite,
    recv_all
)

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

    def _get_serialized_buffer(self) -> bytearray:
        """
        Serialize the event with a 4-byte length prefix.

        The serialized format is:
            [length (4 bytes)] + [event_id (4 bytes)] + [payload]

        This method computes the serialized byte size, allocates a single buffer,
        writes the total message length and event ID and serializes the payload.

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

        return buffer

    def serialize_to_socket(self, sock: SupportsSendAll):
        """
        Serialize the event and send it over the given socket with a 4-byte length prefix.

        The socket is expected to be a raw, unbuffered socket. Therefore, this function does not
        automatically call `flush()`. If you wrap the socket in a buffered stream (e.g.,
        using `makefile()`), you must manually flush the stream after calling this function.

        The serialized format is:
            [length (4 bytes)] + [event_id (4 bytes)] + [payload]

        This method computes the serialized byte size, allocates a single buffer,
        writes the total message length and event ID, serializes the payload, and
        sends the entire buffer using `sock.sendall()`.

        Args:
            sock (SupportsSendAll):
                A socket-like object that supports `sendall()` for writing bytes.
        """
        sock.sendall(self._get_serialized_buffer())

    def serialize_to_serial(self, serial: SupportsWrite):
        """
        Serialize the event and send it over the given serial connection with a 4-byte length
        prefix. Automatically flushes the serial connection.

        The serialized format is:
            [length (4 bytes)] + [event_id (4 bytes)] + [payload]

        This method computes the serialized byte size, allocates a single buffer,
        writes the total message length and event ID, serializes the payload, and
        sends the entire buffer using `serial.write()`. If `write` succeeds, the
        serial connection is then flushed using `serial.flush()`.

        Args:
            serial (SupportsWrite):
                A serial-like object that supports `write()` for writing bytes and `flush()`.
        """
        buffer = self._get_serialized_buffer()
        if serial.write(buffer) != len(buffer):
            raise RuntimeError("Invalid number of bytes written to serial connection")
        serial.flush()

    async def serialize_to_stream_async(self, stream: asyncio.StreamWriter):
        """
        Serialize the event and send it over the given asyncio stream writer with a 4-byte length
        prefix. Automatically flushes the stream.

        The serialized format is:
            [length (4 bytes)] + [event_id (4 bytes)] + [payload]

        This method computes the serialized byte size, allocates a single buffer,
        writes the total message length and event ID, serializes the payload, and
        sends the entire buffer using `stream.write()` followed by `await stream.drain()`.

        Args:
            stream (asyncio.StreamWriter):
                An asyncio stream writer (e.g. network socket or serial connection) that supports
                `write()` for writing bytes and `drain()`.
        """
        stream.write(self._get_serialized_buffer())
        await stream.drain()

    @classmethod
    def from_bytes(cls, data: bytes) -> FifoEvent:
        """
        Deserialize a binary buffer to an event instance.

        Args:
            data (bytes):
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
    def _deserialize_from_stream(cls, read_nb_bytes: Callable[[int], bytes]) -> FifoEvent:
        """
        Receive and deserialize a FIFO event from a stream using the `read_nb_bytes` function.

        This method reads a 4-byte length prefix to determine the size of the
        incoming message, then reads the specified number of bytes from the socket.
        It then delegates deserialization to `from_bytes()`, which performs
        event ID dispatch and constructs the appropriate subclass instance.

        Args:
            read_nb_bytes (Callable[[int], bytes]):
                A function that reads exactly `nb_bytes` from a stream, like a socket or a serial
                connection.

        Returns:
            FifoEvent:
                The deserialized event instance.

        Raises:
            ConnectionError:
                If the stream is closed or incomplete data is received.

            ValueError:
                If the event ID is unknown or the buffer is too short to decode.
        """
        length_bytes = read_nb_bytes(4)

        if len(length_bytes) != 4:
            raise ConnectionError(
                f"Incomplete message header: expected 4 bytes, got {len(length_bytes)}"
            )

        length, = struct.unpack("<I", length_bytes)
        payload = read_nb_bytes(length)

        return cls.from_bytes(payload)

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
        return cls._deserialize_from_stream(lambda nb_bytes: recv_all(sock, nb_bytes))

    @classmethod
    def deserialize_from_serial(cls, serial: SupportsRead) -> FifoEvent:
        """
        Receive and deserialize a FIFO event from the given serial connection.

        This method reads a 4-byte length prefix to determine the size of the
        incoming message, then reads the specified number of bytes from the socket.
        It then delegates deserialization to `from_bytes()`, which performs
        event ID dispatch and constructs the appropriate subclass instance.

        Args:
            serial (SupportsRead):
                A serial-like object that supports `read()` for reading exactly a given number of
                bytes.

        Returns:
            FifoEvent:
                The deserialized event instance.

        Raises:
            ConnectionError:
                If the socket is closed or incomplete data is received.

            ValueError:
                If the event ID is unknown or the buffer is too short to decode.
        """
        return cls._deserialize_from_stream(serial.read)

    @classmethod
    async def _deserialize_from_stream_async(
        cls,
        read_nb_bytes: Callable[[int], Awaitable[bytes]]
    ) -> FifoEvent:
        """
        Receive and deserialize a FifoEvent from a stream using the async `read_nb_bytes`
        function.

        This method reads a 4-byte length prefix to determine the size of the
        incoming message, then reads the specified number of bytes from the stream.
        It then delegates deserialization to `from_bytes()`, which performs
        event ID dispatch and constructs the appropriate subclass instance.

        Args:
            read_nb_bytes (Callable[[int], Awaitable[bytes]]):
                An async function that reads exactly `nb_bytes` from a stream, like an
                asyncio StreamReader.

        Returns:
            FifoEvent:
                The deserialized event instance.

        Raises:
            ConnectionError:
                If the stream is closed or incomplete data is received.

            ValueError:
                If the event ID is unknown or the buffer is too short to decode.
        """
        length_bytes = await read_nb_bytes(4)

        if len(length_bytes) != 4:
            raise ConnectionError(
                f"Incomplete message header: expected 4 bytes, got {len(length_bytes)}"
            )

        length, = struct.unpack("<I", length_bytes)
        payload = await read_nb_bytes(length)

        return cls.from_bytes(payload)

    @classmethod
    async def deserialize_from_stream_async(cls, stream: asyncio.StreamReader) -> FifoEvent:
        """
        Receive and deserialize a FifoEvent from the given asyncio stream reader.

        This method reads a 4-byte length prefix to determine the size of the
        incoming message, then reads the specified number of bytes from the stream.
        It then delegates deserialization to `from_bytes()`, which performs
        event ID dispatch and constructs the appropriate subclass instance.

        Args:
            stream (asyncio.StreamReader):
                An asyncio StreamReader (e.g. network socket or serial connection) that supports
                `readexactly()` for reading exact byte counts.

        Returns:
            FifoEvent:
                The deserialized event instance.

        Raises:
            ConnectionError:
                If the stream is closed or incomplete data is received.

            ValueError:
                If the event ID is unknown or the buffer is too short to decode.
        """
        return await cls._deserialize_from_stream_async(stream.readexactly)

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
class FifoEventShutdown(FifoEvent):
    """
    Sentinel event used to signal termination in queues or pipelines.

    In concurrent or asynchronous systems, inserting a FifoEventShutdown into a queue
    signals consumers to terminate gracefully. This event carries no data payload, 
    only an event ID and optional priority.

    Usage:
        queue.put(FifoEventShutdown())
        # Consumer:
        event = queue.get()
        if isinstance(event, FifoEventShutdown):
            # Clean shutdown
            break
    """
    event_id = 0
    default_priority = 0


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class FifoEventException(FifoEvent):
    """
    Serializable event for transporting exception information across threads, processes, or systems.

    This event is used to report exceptions in a form that can be serialized and sent across
    thread, process, or network boundaries.

    Attributes:
        class_name (str):
            [Serializable] Name of the exception class, such as "RuntimeError".

        message (str):
            [Serializable] Exception message.

        source (str | None):
            [Serializable] Optional source identifier (e.g., thread or worker name).

    Usage:
        # Construct from class name and message
        try:
            raise RuntimeError("Test message")
        except RuntimeError as e:
            event = FifoEventException(
                class_name=e.__class__.__name__,
                message=str(e),
                source="worker-1"
            )

        # Construct directly from an exception object
        try:
            raise RuntimeError("Test message")
        except RuntimeError as e:
            event = FifoEventException(
                exception=e,
                source=threading.current_thread().name
            )

    Raises:
        ValueError:
            Raised if the constructor arguments are invalid. You must provide either:
                - both `class_name` and `message`, or
                - only `exception`.
    """
    event_id = 1
    default_priority = 0

    class_name: str = field(metadata={"format": "S"})
    message: str = field(metadata={"format": "S"})
    source: str | None = field(default=None, metadata={"format": "?S"})

    def __init__(self, class_name: str | None = None,
                 message: str | None = None,
                 priority: int = -1,
                 exception: Exception | None = None,
                 source: str | None = None):
        """
        Initialize a FifoEventException for serializing exception details.

        Args:
            class_name (str, optional):
                Name of the exception class, such as "RuntimeError".
                Must be provided together with `message` if `exception` is not given.

            message (str, optional):
                Exception message.
                Must be provided together with `class_name` if `exception` is not given.

            priority (int, optional):
                Event priority. If set to -1 (default), the class's default_priority is used.

            exception (Exception, optional):
                Exception object. If provided, `class_name` and `message` must not be set.

            source (str, optional):
                Optional identifier of the thread, worker, or system component that raised
                the exception.

        Raises:
            ValueError:
                If both `exception` and either `class_name` or `message` are provided.
            ValueError:
                If neither `exception` nor both `class_name` and `message` are provided.
        """
        super().__init__(priority=priority)
        self.source = source

        if exception is not None:
            if class_name is not None or message is not None:
                raise ValueError("Cannot set class_name or message when exception is provided")
            self.class_name = exception.__class__.__name__
            self.message = str(exception)
        else:
            if class_name is None or message is None:
                raise ValueError("Both class_name and message must be provided when exception "
                                 "is not set")
            self.class_name = class_name
            self.message = message


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class FifoEventKeepAlive(FifoEvent):
    """
    Keep-alive event carrying the current epoch timestamp.

    This standard event can be periodically sent to indicate that a connection
    or component is still alive. It contains the epoch time at which it was
    created.

    Attributes:
        epoch (float):
            [Serializable] Epoch timestamp of when the event was created.
            If `None` is passed to the constructor, the current time from
            `time.time` is used.
    """
    event_id = 2
    default_priority = 0

    epoch: float = field(metadata={"format": "d"})

    def __init__(self, epoch: float | None = None, priority: int = -1):
        """
        Initialize a `FifoEventKeepAlive` with the epoch timestamp at creation time.

        Args:
            epoch (float | None, optional):
                Explicit epoch timestamp. If `None` (default), the current from `time.time` is used.

            priority (int, optional):
                Event priority. If set to -1 (default), the class's default_priority is used.
        """
        super().__init__(priority=priority)
        if epoch is None:
            epoch = time.time()
        self.epoch = epoch


class EErrorCode(IntEnum):
    """
    Enumeration of standard error codes for FifoEventResultBase and its subclasses.

    This enum provides a standardized way to categorize operation results across
    different event types. Additional error codes can be added as needed for
    specific use cases.

    Values:
        OK (0):
            Operation completed successfully with no errors.

        ERROR (1):
            Generic error occurred. Use the message field for specific details.

        EXCEPTION (2):
            An unhandled exception occurred. See the message for details. Note that
            a corresponding FifoEventException may not always be sent separately.
    """
    OK = 0
    ERROR = 1
    EXCEPTION = 2



@serializable
@dataclass(kw_only=True)
class FifoEventResultSimpleBase(FifoEvent):
    """
    Abstract base class for simple result events that report operation outcomes using only an
    error code (no messages).

    This class provides a standardized structure for events that need to communicate
    success/failure status without descriptive messages. It is designed
    to be subclassed, with each subclass defining its own event_id and default_priority.

    Note: This class is not registered with @FifoEvent.register because it lacks
    an event_id and is intended only as a base class.

    Attributes:
        code (EErrorCode):
            [Serializable] The result status of the operation.
    """

    code: EErrorCode = field(metadata={"format": "E<B>", "ptype": EErrorCode})

    def __init__(self, code: EErrorCode, priority: int = -1):
        """
        Initialize a FifoEventResultSimpleBase with result status.

        Args:
            code (EErrorCode):
                The result status of the operation.

            priority (int, optional):
                Event priority. If set to -1 (default), the class's default_priority is used.
        """
        super().__init__(priority=priority)
        self.code = code


@serializable
@dataclass(kw_only=True)
class FifoEventResultBase(FifoEventResultSimpleBase):
    """
    Abstract base class for result events that report operation outcomes with error codes.

    This class provides a standardized structure for events that need to communicate
    success/failure status along with optional descriptive messages. It is designed
    to be subclassed, with each subclass defining its own event_id and default_priority.

    Note: This class is not registered with @FifoEvent.register because it lacks
    an event_id and is intended only as a base class.

    Attributes:
        message (str | None):
            [Serializable] Optional descriptive message providing additional context
            about the result, especially useful for error cases.

    Usage:
        @FifoEvent.register
        class FifoEventMyResult(FifoEventResultBase):
            event_id = 100
            default_priority = 10

        # Success case
        result = FifoEventMyResult(EErrorCode.OK, "Operation completed successfully")

        # Error case
        result = FifoEventMyResult(EErrorCode.ERROR, "Database connection failed")
    """

    message: str | None = field(default=None, metadata={"format": "?S"})

    def __init__(self, code: EErrorCode, message: str | None = None, priority: int = -1):
        """
        Initialize a FifoEventResultBase with result status and optional message.

        Args:
            code (EErrorCode):
                The result status of the operation.

            message (str, optional):
                Optional descriptive message providing additional context.

            priority (int, optional):
                Event priority. If set to -1 (default), the class's default_priority is used.
        """
        super().__init__(code=code, priority=priority)
        self.message = message


@serializable
@dataclass(kw_only=True)
class FifoEventWithCID(FifoEvent):
    """
    Base class for events that include a correlation ID.

    The correlation ID uniquely identifies this event and is intended to be copied into
    corresponding response events, such as reception acknowledgements or completion
    acknowledgements. This enables tracking and matching of requests and responses
    across asynchronous or distributed systems.

    Attributes:
        correlation_id (UUID):
            [Serializable] Unique correlation ID for this event, used to match requests and
            responses.
    """
    correlation_id: UUID = field(metadata=field_meta_serialize_handler_uuid())

    def __init__(self,
                 correlation_id: UUID | None = None,
                 priority: int = -1):
        """
        Initialize a FifoEventWithCID with a correlation ID.

        The correlation ID uniquely identifies this event and is intended to be copied into
        corresponding response events, such as reception acknowledgements or completion
        acknowledgements. This enables tracking and matching of requests and responses
        across asynchronous or distributed systems.

        Args:
            correlation_id (UUID, optional):
                Unique correlation ID for this event, used to match requests and responses.
                If not provided, a new UUID is generated.

            priority (int, optional):
                Event priority. If set to -1 (default), the class's default_priority is used.
        """
        super().__init__(priority=priority)
        self.correlation_id = correlation_id if correlation_id is not None else uuid4()


@serializable
@dataclass(kw_only=True)
class FifoEventResultWithCID(FifoEventResultBase):
    """
    Base class for result (response) events that include a correlation ID.

    The correlation ID is copied from the corresponding request event, allowing the
    recipient to match this result event to its original request. This is essential for
    tracking asynchronous operations and ensuring correct pairing of requests and responses.

    Attributes:
        correlation_id (UUID):
            [Serializable] Correlation ID copied from the corresponding request event, used for
            matching.
    """
    correlation_id: UUID = field(metadata=field_meta_serialize_handler_uuid())

    def __init__(self,
                 code: EErrorCode,
                 correlation_id: UUID,
                 message: str | None = None,
                 priority: int = -1):
        """
        Initialize a FifoEventResultWithCID with result status, an optional message, and a
        correlation ID copied from a corresponding request event.

        The correlation ID is used to match this result event to its request, which is
        essential for tracking asynchronous operations and ensuring correct pairing of
        requests and responses.

        Args:
            code (EErrorCode):
                The result status of the operation.

            correlation_id (UUID):
                Correlation ID copied from the corresponding request event, used for matching.

            message (str, optional):
                Optional descriptive message providing additional context.

            priority (int, optional):
                Event priority. If set to -1 (default), the class's default_priority is used.
        """
        super().__init__(code, message, priority)
        self.correlation_id = correlation_id
