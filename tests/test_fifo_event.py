from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Tuple
import pytest
from fifo_dev_common.event.fifo_event import FifoEvent, serializable, FieldSpec
import threading

# Example subclass for testing
@FifoEvent.register
@serializable([
    FieldSpec("value", "i"),
    FieldSpec("flag", "B")
])
class TestEvent(FifoEvent):
    event_id = 42

    def __init__(self, value: int, flag: int, priority: int | None = None):
        super().__init__(priority)
        self.value = value
        self.flag = flag

def test_serialization_and_deserialization():
    event = TestEvent(12345, 1, priority=7)
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

def test_registry_duplicate_event_id():
    # Attempt to register another class with the same event_id should raise ValueError
    with pytest.raises(ValueError):
        @FifoEvent.register
        @serializable([
            FieldSpec("other", "i")
        ])
        class _DuplicateEvent(FifoEvent): # type: ignore
            event_id = 42

def test_incorrect_packet_size():
    event = TestEvent(1, 2, priority=0)
    data = event.to_bytes()
    # Remove one byte to make the packet size incorrect
    bad_data = data[:-1]
    with pytest.raises(ValueError):
        FifoEvent.from_bytes(bad_data)

def test_unknown_event_id():
    # Create a packet with an unknown event_id and priority=0
    unknown_id = (9999).to_bytes(4, "little") + (0).to_bytes(4, "little", signed=True) + b"\x00"
    with pytest.raises(ValueError):
        FifoEvent.from_bytes(unknown_id)

def test_from_bytes_too_short():
    # Should raise ValueError if data is less than 8 bytes (4 for id, 4 for priority)
    with pytest.raises(ValueError, match="Data too short to contain event_id and priority."):
        FifoEvent.from_bytes(b"\x01\x02\x03\x04\x05\x06\x07")

def test_register_missing_event_id():
    # Should raise AttributeError if event_cls does not define event_id or event_id == -1
    class NoEventId(FifoEvent):
        pass

    with pytest.raises(AttributeError, match="must define an 'event_id' attribute"):
        FifoEvent.register(NoEventId)

@FifoEvent.register
@serializable([
    FieldSpec("value", "i")
])
class PriorityEvent(FifoEvent):
    event_id = 101
    default_priority = 5

    def __init__(self, value: int, priority: int | None = None):
        super().__init__(priority)
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

    @staticmethod
    def from_tuple(value: Any) -> Any:
        return TestEnum(value)

    def to_tuple(self) -> int:
        return self.value

@FifoEvent.register
@serializable([
    FieldSpec("value", "i"),
    FieldSpec("int_enum", "i", from_fields=TestEnum.from_tuple, to_fields=TestEnum.to_tuple)
])
class PriorityEvent1(FifoEvent):
    event_id = 102
    default_priority = 5

    def __init__(self, value: int, int_enum: TestEnum = TestEnum.A, priority: int | None = None):
        super().__init__(priority)
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
    assert restored2.value == 100
    assert restored2.int_enum == TestEnum.A
    assert restored2.priority == 1



@dataclass
class MyData:
    a: int
    b: int

    @staticmethod
    def from_tuple(values: Tuple[Any, ...]) -> Any:
        return MyData(values[0], values[1])

    def to_tuple(self):
        return (self.a, self.b)

@FifoEvent.register
@serializable([
    FieldSpec("payload", "II", from_fields=MyData.from_tuple, to_fields=MyData.to_tuple)
])
class DataEvent(FifoEvent):
    event_id = 300
    default_priority = 2

    def __init__(self, payload: MyData, priority: int | None = None):
        super().__init__(priority)
        self.payload = payload

def test_dataclass_field_serialization_roundtrip():
    d = MyData(7, 77)
    event = DataEvent(d, priority=42)
    blob = event.to_bytes()
    restored = FifoEvent.from_bytes(blob)
    assert isinstance(restored, DataEvent)
    assert restored.payload == MyData(7, 77)
    assert restored.priority == 42

    event2 = DataEvent(MyData(123, 456))
    blob2 = event2.to_bytes()
    restored2 = FifoEvent.from_bytes(blob2)
    assert isinstance(restored2, DataEvent)
    assert restored2.payload == MyData(123, 456)
    assert restored2.priority == 2  # default

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
