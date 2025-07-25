import struct
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from fifo_dev_common.event.fifo_event import FifoEvent
from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable


class DummySendSocket:
    """Collects bytes sent via sendall."""
    def __init__(self):
        self.data = bytearray()

    def sendall(self, data: bytes | bytearray | memoryview, flags: int = 0):
        _ = flags
        self.data.extend(data)


class DummyRecvSocket:
    """Provides bytes via recv_into in preset chunks."""
    def __init__(self, chunks: list[bytes]):
        self._chunks = [memoryview(c) for c in chunks]

    def recv_into(self, buffer: memoryview, nbytes: int = 0):
        if not self._chunks:
            return 0
        chunk = self._chunks.pop(0)
        to_copy = min(len(chunk), nbytes, len(buffer))
        buffer[:to_copy] = chunk[:to_copy]
        if to_copy < len(chunk):
            self._chunks.insert(0, chunk[to_copy:])
        return to_copy


def test_fifo_serializable_socket_roundtrip():
    @serializable
    @dataclass
    class SockObj(FifoSerializable):
        a: int = field(metadata={"format": "I"})
        b: int = field(metadata={"format": "I"})

    obj = SockObj(1, 2)
    send_sock = DummySendSocket()
    obj.serialize_to_socket(send_sock)

    payload_size = obj.serialized_byte_size()
    payload = bytearray(payload_size)
    obj.serialize_to_bytes(payload, 0)
    expected = struct.pack("<I", payload_size) + payload
    assert send_sock.data == expected

    recv_sock = DummyRecvSocket([expected[:3], expected[3:]])
    restored = SockObj.deserialize_from_socket(recv_sock)
    assert isinstance(restored, SockObj)
    assert restored.a == obj.a and restored.b == obj.b


def test_fifo_event_socket_roundtrip():
    @FifoEvent.register
    @serializable
    @dataclass(kw_only=True)
    class SockEvent(FifoEvent):
        event_id: ClassVar[int] = 6543
        value: int = field(metadata={"format": "i"})

        def __init__(self, value: int, priority: int = -1):
            super().__init__(priority=priority)
            self.value = value

    event = SockEvent(value=42, priority=7)
    send_sock = DummySendSocket()
    event.serialize_to_socket(send_sock)

    event_bytes = event.to_bytes()
    expected = struct.pack("<I", len(event_bytes)) + event_bytes
    assert send_sock.data == expected

    recv_sock = DummyRecvSocket([expected[:2], expected[2:-1], expected[-1:]])
    restored = FifoEvent.deserialize_from_socket(recv_sock)
    assert isinstance(restored, SockEvent)
    assert restored.value == event.value
    assert restored.priority == event.priority


def test_fifo_serializable_socket_incomplete_header():
    """Simulate socket closing before 4-byte length is read."""

    @serializable
    @dataclass
    class SockObj(FifoSerializable):
        a: int = field(metadata={"format": "I"})
        b: int = field(metadata={"format": "I"})

    recv_sock = DummyRecvSocket([b"\x08\x00"])  # only 2 header bytes
    with pytest.raises(ConnectionError):
        SockObj.deserialize_from_socket(recv_sock)


def test_fifo_serializable_socket_incomplete_payload():
    """Header ok but payload shorter than advertised."""

    @serializable
    @dataclass
    class SockObj(FifoSerializable):
        a: int = field(metadata={"format": "I"})
        b: int = field(metadata={"format": "I"})

    obj = SockObj(5, 6)
    payload_size = obj.serialized_byte_size()
    payload = bytearray(payload_size)
    obj.serialize_to_bytes(payload, 0)
    full = struct.pack("<I", payload_size) + payload
    truncated = full[:-1]  # remove one byte of payload

    recv_sock = DummyRecvSocket([truncated])
    with pytest.raises(ConnectionError):
        SockObj.deserialize_from_socket(recv_sock)


def test_fifo_event_socket_incomplete_header():
    """FifoEvent fails if length prefix is truncated."""

    recv_sock = DummyRecvSocket([b"\x01\x00\x00"])  # only 3 bytes
    with pytest.raises(ConnectionError):
        FifoEvent.deserialize_from_socket(recv_sock)


def test_fifo_event_socket_incomplete_payload():
    """Event payload shorter than specified length triggers error."""

    @FifoEvent.register
    @serializable
    @dataclass(kw_only=True)
    class SockEvent(FifoEvent):
        event_id: ClassVar[int] = 6544
        value: int = field(metadata={"format": "i"})

        def __init__(self, value: int, priority: int = -1):
            super().__init__(priority=priority)
            self.value = value

    event = SockEvent(value=99, priority=3)
    event_bytes = event.to_bytes()
    packet = struct.pack("<I", len(event_bytes)) + event_bytes
    truncated = packet[:-2]

    recv_sock = DummyRecvSocket([truncated])
    with pytest.raises(ConnectionError):
        FifoEvent.deserialize_from_socket(recv_sock)
