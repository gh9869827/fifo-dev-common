from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import struct
import threading
from typing import ClassVar
import pytest
from fifo_dev_common.event.fifo_event import FifoEvent
from fifo_dev_common.serialization.fifo_serialization import serializable

# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

@serializable
@dataclass
class MyData:
    a: int = field(metadata={"format": "I"})
    b: int = field(metadata={"format": "I"})

@serializable
@dataclass
class MyDynamicDataFormat:
    lst: list[int] = field(metadata={"format": "[I]"})

@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DataEvent(FifoEvent):
    event_id: ClassVar[int] = 300
    default_priority: ClassVar[int] = 2

    payload1: MyData = field(metadata={"ptype": MyData})
    payload2: MyDynamicDataFormat = field(metadata={"ptype": MyDynamicDataFormat})

    def __init__(self, payload1: MyData, payload2: MyDynamicDataFormat, priority: int = -1):
        super().__init__(priority=priority)
        self.payload1 = payload1
        self.payload2 = payload2


def test_data_event_serialization_roundtrip():
    # Create example payload instances
    my_data = MyData(a=123, b=456)
    my_dyn_data = MyDynamicDataFormat(lst=[10, 20, 30])

    # Create the DataEvent instance
    event = DataEvent(payload1=my_data, payload2=my_dyn_data, priority=5)

    # Serialize to bytes
    serialized = event.to_bytes()

    # Deserialize from bytes
    deserialized_event = FifoEvent.from_bytes(serialized)

    # Verify deserialized event is the right type
    assert isinstance(deserialized_event, DataEvent)

    # Verify priority and event_id
    assert deserialized_event.priority == 5
    assert deserialized_event.event_id == 300

    # Verify nested payloads match original data
    assert deserialized_event.payload1.a == my_data.a
    assert deserialized_event.payload1.b == my_data.b
    assert deserialized_event.payload2.lst == my_dyn_data.lst




@FifoEvent.register
@serializable
@dataclass
class TestEvent(FifoEvent):
    value: int = field(metadata={"format": "i"})
    flag: int = field(metadata={"format": "B"})
    event_id: ClassVar[int] = 42

    def __init__(self, value: int, flag: int, priority: int = -1):
        super().__init__(priority=priority)
        self.value = value
        self.flag = flag

def test_serialization_and_deserialization():
    event = TestEvent(value=12345, flag=1, priority=7)
    data = event.to_bytes()

    # Check that the event_id is correctly serialized
    assert data[:4] == (42).to_bytes(4, "little")

    # Check that the priority is correctly serialized (int32 LE)
    assert data[4:8] == (7).to_bytes(4, "little", signed=True)

    # Deserialize and check fields
    event2 = FifoEvent.from_bytes(data)
    assert isinstance(event2, TestEvent)
    assert event2.value == 12345
    assert event2.flag == 1
    assert event2.priority == 7




@FifoEvent.register
@serializable
@dataclass
class TestEventDefaultPriorityChanged(FifoEvent):
    value: int = field(metadata={"format": "i"})
    flag: int = field(metadata={"format": "B"})
    event_id: ClassVar[int] = 421
    default_priority: ClassVar[int] = 100

    def __init__(self, value: int, flag: int, priority: int = -1):
        super().__init__(priority=priority)
        self.value = value
        self.flag = flag

def test_serialization_and_deserialization_default_priority_changed():
    event = TestEventDefaultPriorityChanged(value=54321, flag=8)
    data = event.to_bytes()

    # Check that the event_id is correctly serialized
    assert data[:4] == (421).to_bytes(4, "little")

    # Check that the priority is correctly serialized (int32 LE)
    assert data[4:8] == (100).to_bytes(4, "little", signed=True)

    # Deserialize and check fields
    event2 = TestEventDefaultPriorityChanged.from_bytes(data)
    assert isinstance(event2, TestEventDefaultPriorityChanged)
    assert event2.value == 54321
    assert event2.flag == 8
    assert event2.priority == 100


def test_serialization_and_deserialization_default_priority_changed_overriden():
    event = TestEventDefaultPriorityChanged(value=54321, flag=8, priority=123)
    data = event.to_bytes()

    # Check that the event_id is correctly serialized
    assert data[:4] == (421).to_bytes(4, "little")

    # Check that the priority is correctly serialized (int32 LE)
    assert data[4:8] == (123).to_bytes(4, "little", signed=True)

    # Deserialize and check fields
    event2 = TestEventDefaultPriorityChanged.from_bytes(data)
    assert isinstance(event2, TestEventDefaultPriorityChanged)
    assert event2.value == 54321
    assert event2.flag == 8
    assert event2.priority == 123

def test_registry_duplicate_event_id():
    # Register one class with a specific event_id
    @FifoEvent.register
    @serializable
    @dataclass
    class _UniqueEvent(FifoEvent): # type: ignore
        other: int = field(metadata={"format": "i"})
        event_id: ClassVar[int] = 3333

    # Attempt to register another class with the same event_id should raise ValueError
    with pytest.raises(ValueError, match="Duplicate event_id 3333"):
        @FifoEvent.register
        @serializable
        @dataclass
        class _DuplicateEvent(FifoEvent):  # type: ignore
            another: int = field(metadata={"format": "i"})
            event_id: ClassVar[int] = 3333


def test_incorrect_packet_size():
    event = TestEvent(value=1, flag=2, priority=0)
    data = event.to_bytes()
    # Remove one byte to make the packet size incorrect (truncate payload)
    bad_data = data[:-1]
    with pytest.raises(Exception):
        FifoEvent.from_bytes(bad_data)

def test_unknown_event_id():
    # Create a packet with an unknown event_id
    unknown_event_id = 9999
    # Build minimal packet: event_id + dummy payload bytes (empty payload here)
    unknown_packet = bytearray(struct.pack("<I", unknown_event_id))
    # Payload empty or minimal to trigger parse
    with pytest.raises(ValueError, match=f"Unknown event_id {unknown_event_id}"):
        FifoEvent.from_bytes(unknown_packet)

def test_from_bytes_too_short():
    # Now header is only 4 bytes (event_id)
    # Provide fewer than 4 bytes, should raise ValueError about data being too short
    with pytest.raises(ValueError, match="Data too short to contain event_id."):
        FifoEvent.from_bytes(bytearray(b"\x01\x02\x03"))  # only 3 bytes, less than required 4

def test_register_missing_event_id():
    # Should raise AttributeError if event_cls does not define event_id or event_id == -1
    class NoEventId(FifoEvent):
        pass

    with pytest.raises(AttributeError, match="must define an 'event_id' attribute"):
        FifoEvent.register(NoEventId)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class PriorityEvent(FifoEvent):
    event_id: ClassVar[int] = 101
    default_priority: ClassVar[int] = 5

    value: int = field(metadata={"format": "i"})

    def __init__(self, value: int, priority: int = -1):
        super().__init__(priority=priority)
        self.value = value

def test_priority_default_and_override():
    e1 = PriorityEvent(10)
    e2 = PriorityEvent(20, priority=2)
    assert e1.priority == 5  # uses default_priority
    assert e2.priority == 2  # overridden

def test_priority_serialization():
    e1 = PriorityEvent(10, priority=9)
    data = e1.to_bytes()
    # Check that event_id and priority are serialized correctly
    assert data[:4] == (101).to_bytes(4, "little")
    assert data[4:8] == (9).to_bytes(4, "little", signed=True)
    restored = FifoEvent.from_bytes(data)
    assert isinstance(restored, PriorityEvent)
    assert restored.priority == 9
    assert restored.value == 10

def test_lt_comparison():
    e1 = PriorityEvent(10, priority=1)
    e2 = PriorityEvent(20, priority=3)
    assert e1 < e2
    assert not (e2 < e1)

def test_lt_with_non_fifo_event():
    event = FifoEvent(priority=1)
    class NotAnEvent:
        pass
    other = NotAnEvent()
    # Should return NotImplemented when comparing with non-FifoEvent
    assert event.__lt__(other) is NotImplemented


class TestEnum(IntEnum):
    A = 1
    B = 2


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class PriorityEvent1(FifoEvent):
    event_id: ClassVar[int] = 102
    default_priority: ClassVar[int] = 5

    value: int = field(metadata={"format": "i"})
    int_enum: TestEnum = field(default=TestEnum.A, metadata={"format": "E<I>", "ptype": TestEnum})

    def __init__(self, value: int, int_enum: TestEnum = TestEnum.A, priority: int = -1):
        super().__init__(priority=priority)
        self.value = value
        self.int_enum = int_enum

def test_enum_serialization_roundtrip():
    # Serialize with B
    event = PriorityEvent1(42, int_enum=TestEnum.B, priority=99)
    blob = event.to_bytes()
    # Deserialize
    restored = FifoEvent.from_bytes(blob)
    assert isinstance(restored, PriorityEvent1)
    assert restored.value == 42
    assert restored.int_enum == TestEnum.B
    assert restored.priority == 99

    # Serialize with A
    event2 = PriorityEvent1(100, int_enum=TestEnum.A, priority=1)
    blob2 = event2.to_bytes()
    restored2 = FifoEvent.from_bytes(blob2)
    assert isinstance(restored2, PriorityEvent1)
    assert type(restored2.int_enum) == TestEnum
    assert restored2.value == 100
    assert restored2.int_enum == TestEnum.A
    assert restored2.priority == 1


def test_registry_thread_safety() -> None:
    results: list[bool] = []
    def register_event():
        class ThreadEvent(FifoEvent):
            event_id = 999
        try:
            FifoEvent.register(ThreadEvent)
            results.append(True)
        except ValueError:
            results.append(False)
    threads = [threading.Thread(target=register_event) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    # Only one should succeed, rest should fail due to duplicate event_id
    assert results.count(True) == 1
    assert results.count(False) == 9


def test_clear_registry_removes_all_registered_events():
    # Register a dummy event
    @FifoEvent.register
    @serializable
    @dataclass
    class DummyEvent(FifoEvent):  # type: ignore  # pylint: disable=unused-variable
        event_id: ClassVar[int] = 12345
        value: int = field(metadata={"format": "i"})

    # Ensure the event is registered
    assert 12345 in FifoEvent._registry

    # Clear the registry
    FifoEvent.clear_registry()

    # Registry should be empty
    assert FifoEvent._registry == {}

def test_clear_registry_allows_re_registration():
    # Register an event
    @FifoEvent.register
    @serializable
    @dataclass
    class DummyEvent2(FifoEvent):  # type: ignore  # pylint: disable=unused-variable
        event_id: ClassVar[int] = 54321
        value: int = field(metadata={"format": "i"})

    assert 54321 in FifoEvent._registry

    # Clear the registry
    FifoEvent.clear_registry()
    assert FifoEvent._registry == {}

    # Should be able to register again without ValueError
    @FifoEvent.register
    @serializable
    @dataclass
    class DummyEvent3(FifoEvent):  # type: ignore  # pylint: disable=unused-variable
        event_id: ClassVar[int] = 54321
        value: int = field(metadata={"format": "i"})

    assert 54321 in FifoEvent._registry
