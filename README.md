[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Test Status](https://github.com/gh9869827/fifo-dev-common/actions/workflows/test.yml/badge.svg)

# ⚠️ Experimental Branch: `experimental/event` ⚠️

This branch contains experimental code for binary-serializable event with factory deserialization and class registration, designed for cross-system use.

Features, APIs, and behavior are subject to change or removal at any time.  
**Use at your own risk.**

# `fifo-dev-common`

Shared core utilities for all `fifo-dev` repositories, under the `fifo_dev_common` namespace.

This package is designed to support the `fifo-dev` ecosystem with minimal dependencies.  
It provides the following for runtime type checks and casting, docstring parsing, LLM tool support, serialization, and socket utilities:

- `strict_cast()`: Runtime-enforced type casting.  
- `class MiniDocStringFunction`: Lightweight parser for Google-style function docstrings. Extracts minimal structured information for runtime type checking of arguments and return values—useful for LLM-based function calling and agent execution without third-party dependencies.
- `class MiniDocStringClass`: Lightweight parser for Google-style class docstrings. Extracts the short and detailed description along with a list of declared `Attributes:` as raw name/type/description triples.
- `class ReadOnlyList`: Immutable wrapper for list-like data.  
- `@tool_handler` / `@tool_query_source`: Decorators for defining tools and query sources in LLM-based agents.
- `@serializable` decorator and `FifoSerializable` base class: Efficient binary serialization and deserialization for dataclasses, supporting primitives, enums, optional fields, arrays, and nested objects.
- `recv_all(sock, n)`: Efficiently receive exactly `n` bytes from a socket-like object supporting `recv_into()`.
- `class FifoEvent`: Base class for binary-serializable events, with factory deserialization and class registration for cross-system use.
- `class FifoEventQueueForwarderMpToAsync`: Bridges multiprocessing and asyncio event queues via a background thread.
- `class FifoEventQueueConnectorAsyncClient`: Shared asyncio client helpers for stream-based transports.
- `class FifoEventQueueNetworkAsyncClient` / `class FifoEventQueueNetworkAsyncServer`: Asyncio-based network communication for FifoEvent objects over TCP with optional TLS 1.3 encryption.
- `class FifoEventQueueSerialAsyncClient`: Asyncio serial communication for environments exposing serial connections via `serial_asyncio`.
- `class FifoEventQueueConnectorAsyncHandlerCID`: Correlation-ID request/response helpers via `register_cid_template`, `send_and_wait`, and `send_in_background`; outcomes use `FifoEventCIDOutcome`.
- `class FifoProcessManager`: Manager for running workers in separate OS processes with interprocess communication. Abstracts process creation, startup, and shutdown while bridging multiprocessing queues with asyncio. Supports both async and sync worker callbacks, with correlation ID tracking for request/response workflows.
- `get_logger()`: Returns a logger instance with `.trace()` support for fine-grained debugging. Registers a custom TRACE level and logger class.
- `class FifoRefreshableValue`: Lock-free cache for asynchronously refreshed values with explicit state transitions and immutable snapshots.

## 📚 Table of Contents

- [🎯 Project Status & Audience](#-project-status--audience)
- [🧩 Python Modules](#-python-modules)
  - [Install](#-install)
  - [strict_cast](#fifo_dev_commontypeutilsstrict_cast)
  - [mini_docstring](#fifo_dev_commonintrospectionmini_docstring)
  - [read_only_list](#fifo_dev_commoncontainersread_onlyread_only_list)
  - [tool_decorator](#fifo_dev_commonintrospectiontool_decorator)
  - [socket_utils](#fifo_dev_commonsocketsocket_utils)
  - [fifo_serialization](#fifo_dev_commonserializationfifo_serialization)
  - [fifo_event](#fifo_dev_commoneventfifo_event)
  - [fifo_event_queue_forwarder](#fifo_dev_commoneventfifo_event_queue_forwarder)
  - [fifo_event_queue_connector](#fifo_dev_commoneventfifo_event_queue_connector)
  - [fifo_event_queue_network](#fifo_dev_commoneventfifo_event_queue_network)
  - [fifo_event_queue_serial](#fifo_dev_commoneventfifo_event_queue_serial)
  - [fifo_process_manager](#fifo_dev_commonprocessutilsfifo_process_manager)
  - [logger](#fifo_dev_commonlogginglogger)
  - [fifo_refreshable_value](#fifo_dev_commonstatefifo_refreshable_value)
- [📦 C++ Library](#-c-library)
  - [Key Features](#key-features)
  - [Requirements](#requirements)
  - [Core API](#core-api)
  - [Integration with Arduino](#integration-with-arduino)
  - [Integration with Raspberry Pi (or Linux)](#integration-with-raspberry-pi-or-linux)
  - [Cross-Language Example](#cross-language-example)
  - [Documentation](#documentation)
- [🧪 Tests](#-tests)
- [📄 License](#-license)

---

## 🎯 Project Status & Audience

🚧 **Work in Progress** — This project is in **early development**. 🚧

This is a personal project developed and maintained by a solo developer.  
Contributions, ideas, and feedback are welcome, but development is driven by personal time and priorities.

`fifo-dev-common` provides **shared core utilities** for other `fifo-dev-*` projects developed by the author.  
It is **primarily developed to support those projects**, but **individual developers experimenting with the ecosystem are welcome to explore and use it.**

No official release or pre-release has been published yet. The code is provided for **preview and experimentation**.  
**Use at your own risk.**

---

## 🧩 Python Modules

### Install

This repo is meant for local development. Install in editable mode:

```bash
python3 -m pip install -e .
```

Python 3.10+ is required.

---

### `fifo_dev_common.typeutils.strict_cast`

Defines `strict_cast(tp, value)` — a runtime-enforced version of `typing.cast()`.  
Raises `TypeError` if the value does not match the expected type(s).  
**Shallow check**: only verifies the outermost type (e.g., `list`, not `list[int]`).

**Examples:**

```python
from fifo_dev_common.typeutils.strict_cast import strict_cast

# Valid cast: the value matches the expected type (int)
value = strict_cast(int, 42)

# Attempting to cast a value of the wrong type (str instead of int)
# This demonstrates how strict_cast raises a TypeError on mismatch
try:
    value = strict_cast(int, "42")
except TypeError as e:
    print(e)
    # Output:
    # TypeError: strict_cast failed: expected int, got str
```

---

### `fifo_dev_common.introspection.mini_docstring`

Provides the `MiniDocStringFunction` and `MiniDocStringClass` classes for parsing Google-style docstrings into structured form.  
Includes:

- `MiniDocStringFunction`:
  - Description parsing (short + detailed)
  - Argument type extraction from `Args:` sections (`MiniDocStringArg`)
  - Return and raise parsing
  - Runtime type validation
  - Export to YAML schema for structured function calls

- `MiniDocStringClass`:
  - Description parsing (short + detailed)
  - Attribute extraction from `Attributes:` sections (`MiniDocStringAttribute`)

**Examples:**

```python
from fifo_dev_common.introspection.mini_docstring import MiniDocStringFunction

def move_to(x: int, y: int) -> str:
    """
    Move the robot to an (x, y) position in millimeters.

    This function generates a movement command that instructs the robot to move 
    to a specific location on a 2D surface. Coordinates are given in millimeters 
    relative to the robot's current workspace origin.

    Args:
        x (int):
            The target X position in millimeters.
        y (int):
            The target Y position in millimeters.

    Returns:
        str:
            A confirmation string like "Moving to (100, 200)".
    """
    return f"Moving to ({x}, {y})"


# Create a `MiniDocStringFunction` object to parse the docstring
parsed = MiniDocStringFunction(move_to.__doc__)

# Access the parsed types
print(parsed.get_arg_by_name("x").pytype.to_string())   # Output: int
print(parsed.get_arg_by_name("y").pytype.to_string())   # Output: int
assert parsed.return_type is not None
print(parsed.return_type.to_string())                   # Output: str

# Access descriptions
print(parsed.description_short)    # Output: Move the robot to an (x, y) position in millimeters.
print(parsed.description_detailed) # Output: This function generates a movement command ...

# Access return description
print(parsed.return_desc)          # Output: A confirmation string like "Moving to (100, 200)".

# ✅ Runtime validation: correct types
parsed.validate_runtime_args({
    "x": 100,
    "y": 200
})
# Should succeed silently

# ❌ Type mismatch: 'x' is a string, not an int
try:
    parsed.validate_runtime_args({
        "x": "100",
        "y": 200
    })
except ValueError as e:
    print(e)
    # Output: Argument 'x' expected ArgType(int), but got str

# ❌ Unexpected extra argument
try:
    parsed.validate_runtime_args({
        "x": 100,
        "y": 200,
        "speed": 50
    })
except ValueError as e:
    print(e)
    # Output: Unexpected arguments: speed
```

---

### `fifo_dev_common.containers.read_only.read_only_list`

Implements `ReadOnlyList`, a lightweight wrapper that disables mutation of a list.  
Supports indexing, iteration, equality, and containment.

> ⚠️ Inner objects (like nested lists/dicts) are not automatically frozen.  
> For example, `ReadOnlyList([{"x": 1}])[0]["x"] = 2` is still allowed.

**Examples:**

```python
from fifo_dev_common.containers.read_only.read_only_list import ReadOnlyList

# Create a simple read-only list
items = ReadOnlyList([1, 2, 3])

# Read access works like a regular list
print(items[0])  # Output: 1

# Attempting to modify the list raises an error
try:
    # This will raise TypeError because ReadOnlyList is immutable
    # Linters also report a warning in the editor:
    #   - Pylance warning: "__setitem__" method not defined on type "ReadOnlyList[int]"
    #   - Pylint warning: 'items' does not support item assignment
    items[0] = 2
except TypeError as e:
    print("Top-level modification error:", e)

# Wrap inner lists with ReadOnlyList to enforce nested immutability
nested = ReadOnlyList([
    ReadOnlyList([1, 2]),
    ReadOnlyList([3, 4])
])

# Attempting to modify the top-level nested list raises an error
try:
    # This will raise TypeError because ReadOnlyList is immutable
    # Linters also report a warning in the editor
    nested[0] = [9, 9]
except TypeError as e:
    print("Nested top-level modification error:", e)

# Attempting to modify the inner list also raises an error
try:
    # This will raise TypeError because the inner ReadOnlyList is also immutable
    # Linters also report a warning in the editor
    nested[0][0] = 99
except TypeError as e:
    print("Nested inner modification error:", e)

# Create a ReadOnlyList containing a mutable inner list (not wrapped)
shallow = ReadOnlyList([
    [1, 2],  # This is a regular list, still mutable
    [3, 4]
])

# Modifying the outer list will raise a TypeError
try:
    shallow[0] = [9, 9]
except TypeError as e:
    print("Shallow top-level modification error:", e)

# But modifying the inner list itself works — ReadOnlyList perform a shallow check
shallow[0][0] = 99
print("Modified shallow inner list:", shallow[0])  # Output: [99, 2]
```

---

### `fifo_dev_common.introspection.tool_decorator`

Decorators to define tools and runtime query sources callable by large language models:

- `@tool_handler(name)`: Declare a function as an executable tool with schema support.  
- `@tool_query_source(name)`: Define a no-arg runtime data source that provides context for LLM execution planning.

These attach structured metadata derived from docstrings—enabling parsing, validation, and schema generation for transparent agent planning and execution.

**Examples:**

```python
from fifo_dev_common.introspection.tool_decorator import tool_handler

@tool_handler("describe_task")
def describe_task(task_id: int) -> str:
    """
    Describe the task based on its ID.

    Args:
        task_id (int): 
            ID to fetch

    Returns:
        str:
            Description
    """
    return f"Task #{task_id}"

# Convert the function metadata into structured schema (YAML format)
print(describe_task.to_schema_yaml())

# Output:
# - intent: describe_task
#   description: Describe the task based on its ID.
#   parameters:
#     - name: task_id
#       type: int
#       description: ID to fetch
#       optional: False
#   return:
#     type: str
#     description: Description
```

---

### `fifo_dev_common.socket.socket_utils`

Defines `recv_all(sock, n)`, a helper that receives exactly `n` bytes from a socket-like object using `recv_into()` and `memoryview` for efficient reading.

Also provides two runtime-checkable protocols for socket abstraction:

- `SupportsRecvInto`: defines `recv_into(buffer, nbytes)`
- `SupportsSendAll`: defines `sendall(data, flags=...)`

These allow code to accept real sockets or compatible mock objects without relying on concrete types.

**Examples:**

```bash
import socket
from fifo_dev_common.socket.socket_utils import recv_all, SupportsRecvInto

# Create a pair of connected sockets for local communication
sock_server, sock_client = socket.socketpair()

# Send 6 bytes from one end
sock_client.sendall(b"ABCDEF")

# Receive exactly 6 bytes from the other end using recv_all()
data = recv_all(sock_server, 6)
print(data)
# Output: b'ABCDEF'

# Define a function that accepts any object supporting recv_into()
def read_exact(sock: SupportsRecvInto, size: int) -> bytes:
    """
    Receive exactly `size` bytes from any object implementing recv_into().
    Useful for both real sockets and mocks during testing.
    """
    return recv_all(sock, size)

# Demonstrate using it with a real socket:
sock_client.sendall(b"GHIJKL")
result = read_exact(sock_server, 6)
print(result)
# Output: b'GHIJKL'

# Cleanup
sock_server.close()
sock_client.close()
```

---

### `fifo_dev_common.serialization.fifo_serialization`

Provides a lightweight, efficient binary serialization framework for Python dataclasses.

- `@serializable`: Decorator to enable serialization/deserialization on dataclasses.
- Supports scalar types, enums and optional enums, optional fields, arrays, and fixed-length tuples.
- Uses dataclass `field` metadata (e.g., `format`, `ptype`) for flexible, extensible field definitions.
- Custom per-field (de)serialization via `field` metadata callables (`serialize`, `deserialize`, `bytelength`).
- All serialized data is written and read in little-endian byte order, independent of host architecture.
- Designed for preallocated buffers to maximize performance and minimize allocations.
- Works well with microcontroller and embedded system data formats as it is a compact binary format prioritizing direct raw serialization with very little overhead.

#### Supported Format Strings

| Format     | Description                          | Serialization Details                                                  | Notes                                                                 |
|------------|--------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `x`        | **Primitive value**                  | Serialized directly using `struct`                                     | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `[x]`      | **Array of primitives**              | 4-byte length prefix + consecutive `x` values                          | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `[?x]`     | **Array of optional primitives**     | 4-byte length + presence bitmap + serialized values for present items  | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `?x`       | **Optional primitive**               | 1-byte presence flag + `x` if present                                  | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `T<xx...>` | **Fixed-length tuple of primitives** | Each element serialized consecutively (no prefix)                    | Each `x` must be one of: `b B h H i I l L q Q e f d y`                  |
| `?T<xx...>` | **Optional fixed-length tuple of primitives** | 1-byte presence flag + serialized elements if present | Each `x` must be one of: `b B h H i I l L q Q e f d y` |
| `[np:x]`   | **NumPy array**                      | 1-byte ndim + N×4-byte shape + raw data buffer                         | `x` must be NumPy dtype: `u8`, `u16`, `u32`, `i8`, `i16`, `i32`, `f32`, `f64` |
| `[np:x:shape]` | **Fixed-shape NumPy array**       | Raw data buffer only; shape is provided in format string                | `shape` is comma-separated dims, e.g. `[np:f32:2,3]` |
| `S`        | **Variable-length UTF-8 string**     | 4-byte length prefix + UTF-8 encoded bytes                             | -                                                                     |
| `S[x]`     | **Fixed-length UTF-8 string**        | UTF-8 encoded, space-padded or truncated to `x` bytes (UTF-8 safe)     | -                                                                     |
| `?S`       | **Optional variable-length UTF-8 string** | 1-byte presence flag + 4-byte length + UTF-8 encoded bytes            | -
| `?S[x]`    | **Optional fixed-length UTF-8 string**    | 1-byte presence flag + UTF-8 encoded, space-padded or truncated to `x` bytes | -
| `E<x>`     | **Enum stored as integer**           | Stored as `x` (e.g., `B`, `H`, `I`)                                    | `x` must be one of: `b B h H i I`                                     |
| `?E<x>`    | **Optional enum stored as integer**  | 1-byte presence flag + enum stored as `x`                              | `x` must be one of: `b B h H i I`                                     |
| `_`        | **Nested object**                    | Nested serialization using `ptype`                                     | Requires `ptype`; equivalent to omitting `format`                     |
| `?_`       | **Optional nested object**           | 1-byte presence flag + nested serialization if present                 | -                                                                     |
| `[_]`      | **Array of nested objects**          | 4-byte length prefix + consecutive nested serializations               | -                                                                     |
| `[?_]`     | **Array of optional nested objects** | 4-byte length + presence bitmap + serialized present objects           | -                                                                     |

**Note:** Primitive format codes `b B h H i I l L q Q e f d` follow the [Python `struct` module](https://docs.python.org/3/library/struct.html).  
`y` is a special format for booleans, serialized as a single byte (`0` for `False`, `1` for `True`).

#### Custom per-field serialization

Fields can provide `serialize`, `deserialize`, and `bytelength` callables in their
`field` metadata. When present, these functions are used instead of a `format` or
`ptype`, enabling support for custom types without any global registry or wrapper. The helper
`field_meta_serialize_handler_uuid()` demonstrates how to serialize a `uuid.UUID`
using this mechanism, showing a reusable approach: create the three handlers in a
helper function and return them as a `FieldMetaDataSerializeHandlers` `TypedDict`
for type-checking.

#### Examples

```python
from dataclasses import dataclass, field
import socket
from typing import List
from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable

# Define a serializable dataclass for a single sensor reading.
# The temperature and humidity fields are both floats, which means we use the format string 'f'.
@serializable
@dataclass
class SensorReadings(FifoSerializable):
    temperature: float = field(metadata={"format": "f"})
    humidity: float = field(metadata={"format": "f"})

# Define a serializable dataclass for an array of sensor readings.
# `readings` is a list of `SensorReadings` objects. Since it's a list, we use the format string 
# `[]`. The `_` inside the brackets indicates that each item is a serializable object.
# We also set the `ptype` metadata attribute to `SensorReadings` so that each item can be properly
# instantiated during deserialization.
@serializable
@dataclass
class SensorArray(FifoSerializable):
    readings: List[SensorReadings] = field(metadata={"format": "[_]", "ptype": SensorReadings})

# Create some example sensor readings.
s1 = SensorReadings(temperature=22.5, humidity=40.0)
s2 = SensorReadings(temperature=23.0, humidity=38.5)
sensor_data = SensorArray(readings=[s1, s2])

# Allocate a buffer of the correct size and serialize the data into it.
buffer = bytearray(sensor_data.serialized_byte_size())
sensor_data.serialize_to_bytes(buffer, 0)

# Deserialize the data back from the buffer.
deserialized, _ = SensorArray.deserialize_from_bytes(buffer, 0)


# Display the deserialized data to confirm it matches the original
print(f"Number of readings: {len(deserialized.readings)}")                   # Output=2
print(f"First reading temperature: {deserialized.readings[0].temperature}")  # Output=22.5
print(f"Second reading humidity: {deserialized.readings[1].humidity}")       # Output=38.5

# This object can also be sent over a socket using the built-in serialization methods.

# Create a pair of connected sockets for local communication
sock_server, sock_client = socket.socketpair()

# Serialize the object and send it over the socket
sensor_data.serialize_to_socket(sock_client)

# Deserialize the object from the receiving socket
deserialized_socket = SensorArray.deserialize_from_socket(sock_server)

# Display the deserialized data to confirm it matches the original
print(f"Number of readings: {len(deserialized_socket.readings)}")                   # Output=2
print(f"First reading temperature: {deserialized_socket.readings[0].temperature}")  # Output=22.5
print(f"Second reading humidity: {deserialized_socket.readings[1].humidity}")       # Output=38.5

# Clean up the sockets
sock_server.close()
sock_client.close()
```

---

```python
# Example with optional (nullable) elements in the array.
# `readings` is similar to the previous example, but each element in the list may be None.
# We use `[?_]` as the format string:
# - `[]` indicates a list,
# - `_` indicates the element type is a serializable object,
# - `?` means each element is optional (can be None).
@serializable
@dataclass
class MaybeSensorArray(FifoSerializable):
    readings: List[SensorReadings | None] = field(metadata={"format": "[?_]", "ptype": SensorReadings})

# Create an array with one reading and one None
data = MaybeSensorArray(readings=[s1, None])
buf = bytearray(data.serialized_byte_size())
data.serialize_to_bytes(buf, 0)

# Deserialize and check that the None is preserved
restored, _ = MaybeSensorArray.deserialize_from_bytes(buf, 0)

# Display the result
print(f"Restored[0] is None? {restored.readings[0] is None}")  # Output=False
print(f"Restored[1] is None? {restored.readings[1] is None}")  # Output=True
```

---

```python
import uuid
from dataclasses import dataclass
from fifo_dev_common.serialization.fifo_serialization import (
    FifoSerializable, serializable, field_meta_serialize_handler_uuid,
)

@serializable
@dataclass
class MyEvent(FifoSerializable):
    # Per-field serialization
    uid: uuid.UUID = field(metadata=field_meta_serialize_handler_uuid())
```

---

### `fifo_dev_common.event.fifo_event`

Defines the `FifoEvent` base class for priority-aware, factory-registered event types.

- Supports event ID defined as a class-level attribute.
- Supports priority with a class-level default that can be overridden at instantiation.
- Uses `@FifoEvent.register` to register event subclasses for factory-based deserialization.
- Supports serialization/deserialization by:
  - Inheriting from `FifoSerializable`.
  - Using the `@serializable` decorator.
  - Automatically serializing the event header (event ID) alongside the payload (which includes priority).
- Designed for efficient, compact event exchange in embedded and distributed systems.

**Examples:**

```python
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

from fifo_dev_common.event.fifo_event import FifoEvent, serializable

# Define an enum for event states.
class State(IntEnum):
    INIT = 1
    RUN = 2
    DONE = 3

# Define a serializable dataclass representing a 2D point.
@serializable
@dataclass
class Point:
    x: int = field(metadata={"format": "i"})
    y: int = field(metadata={"format": "i"})

# Define a serializable event class and register it with FifoEvent.
@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DemoEvent(FifoEvent):
    # These are class-level constants and not serialized.
    event_id: ClassVar[int] = 99
    default_priority: ClassVar[int] = 3

    # Instance fields that are serialized.
    score: int = field(metadata={"format": "i"})
    state: State = field(metadata={"format": "E<B>", "ptype": State})
    position: Point = field(metadata={"ptype": Point})

    def __init__(self, score: int, state: State, position: Point, priority: int = -1):
        # Initialize the base event (sets priority).
        super().__init__(priority=priority)
        self.score = score
        self.state = state
        self.position = position

# Example usage:

# Create an event instance.
evt = DemoEvent(score=42, state=State.RUN, position=Point(7, 8), priority=77)

# Serialize the event to a byte array.
blob = evt.to_bytes()

# Deserialize the event from bytes using the event registry.
restored = FifoEvent.from_bytes(blob)

# Verify instance and cast for the linters
assert isinstance(restored, DemoEvent)

# Display the deserialized event to confirm it matches the original
print(f"Score: {restored.score}")                                         # Output=42
print(f"State: {restored.state.name}")                                    # Output=RUN
print(f"Position: ({restored.position.x}, {restored.position.y})")        # Output=(7, 8)
print(f"Priority: {restored.priority}")                                   # Output=77
```

In addition to the `FifoEvent` base class, which can be subclassed to create custom events, the `fifo_dev_common` library provides two event classes that already inherit from `FifoEvent` and can be used directly:

#### `FifoEventShutdown`
- **Description:** A sentinel event used to signal queue consumers to terminate gracefully.
- **Use Case:** Insert this event into a queue to notify consumers (e.g., threads or processes) to shut down cleanly.
- **Fields:** Inherits `priority` from `FifoEvent`. No additional fields.

#### `FifoEventException`
- **Description:** A serializable event for transporting exception details across threads, processes, or network boundaries.
- **Use Case:** Use this event to report exceptions in distributed or concurrent systems.
- **Fields:**
  - `class_name`: Name of the exception class (e.g., `"RuntimeError"`).
  - `message`: Exception message.
  - `source`: *(Optional)* Identifier of the source (such as thread or worker name).

#### `FifoEventKeepAlive`
- **Description:** Lightweight keep-alive message containing the epoch timestamp at creation.
- **Use Case:** Send periodically to indicate that a connection or component is still active.
- **Fields:**
  - `epoch`: Timestamp in seconds from `time.time()` when the event was created. Automatically set if not provided.

The `fifo_dev_common` library also includes other classes that inherit from `FifoEvent`, but are intended to be used as a base class for creating standardized events:

#### `FifoEventResultBase`
- **Description:** Base class for reporting operation outcomes with error codes and optional messages.
- **Use Case:** Subclass this to define application-specific result events.
- **Fields:**
  - `code`: An `EErrorCode` value (e.g., `OK`, `ERROR`).
  - `message`: *(Optional)* Descriptive message providing additional context.

You can subclass `FifoEventResultBase` to create strongly-typed result events tailored to the application's needs.  
Each subclass must define a unique `event_id` and `default_priority`, and may include additional fields if needed.

#### `FifoEventWithCID`
- **Description:** Base class for events that include a correlation ID (`correlation_id: UUID`).
- **Use Case:** Use as a base class when you need to track and match requests and responses across asynchronous or distributed systems.
- **How it works:**  
  - `FifoEventWithCID` assigns a unique correlation ID to each event, which is then copied into the corresponding result event.
- **Fields:**
  - `correlation_id`: A UUID used to match requests and responses. Automatically generated if not provided.

#### `FifoEventResultWithCID`
- **Description:** Base class for result (response) events that include a correlation ID (`correlation_id: UUID`).
- **Use Case:** Use as a base class for response/result events that need to be matched to their original request.
- **How it works:**  
  - `FifoEventResultWithCID` carries the correlation ID from the original request, enabling reliable matching of responses to requests.
- **Fields:**
  - `correlation_id`: A UUID copied from the corresponding request event.

**Example:**

```python
from fifo_dev_common.event.fifo_event import (
    FifoEventResultBase, EErrorCode, FifoEvent
)

@FifoEvent.register
class FifoEventMyResult(FifoEventResultBase):
    event_id = 100
    default_priority = 10

# Usage
# Success
result = FifoEventMyResult(code=EErrorCode.OK, message="Operation completed successfully")

# Error
result = FifoEventMyResult(code=EErrorCode.ERROR, message="Database connection failed")
```

Result events can be extended with custom fields.  
Be sure to use the `@dataclass(kw_only=True)` and `@serializable` decorators when adding new serializable fields.
Even if a constructor is not strictly required, it is recommended to provide one to avoid `pylint` warnings.

```python
from dataclasses import dataclass, field
from fifo_dev_common.event.fifo_event import EErrorCode, FifoEvent, FifoEventResultBase
from fifo_dev_common.serialization.fifo_serialization import serializable

@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class FifoEventMyCustomResult(FifoEventResultBase):
    event_id = 101
    default_priority = 5

    details: str = field(default="", metadata={"format": "S"})

    def __init__(self,
                 details: str,
                 code: EErrorCode,
                 message: str | None = None,
                 priority: int = -1):
        super().__init__(code=code, message=message, priority=priority)
        self.details = details

result = FifoEventMyCustomResult(details="...", code=EErrorCode.OK)
```

The example below illustrates how to create a custom request with a correlation ID and copy this correlation ID into a custom answer.

```python
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID
from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventResultWithCID, EErrorCode, FifoEventWithCID
from fifo_dev_common.serialization.fifo_serialization import serializable

@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class MyRequest(FifoEventWithCID):
    event_id: ClassVar[int] = 42011
    default_priority: ClassVar[int] = 10

    data: int = field(metadata={"format": "I"})

    def __init__(self,
                 data: int,
                 correlation_id: UUID | None = None,
                 priority: int = -1):
        super().__init__(correlation_id=correlation_id, priority=priority)
        self.data = data

req = MyRequest(data=123)
# A new correlation ID is automatically generated and assigned to the request
print(req.correlation_id)

@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class MyResult(FifoEventResultWithCID):
    event_id: ClassVar[int] = 42012
    default_priority: ClassVar[int] = 10

    result: int = field(metadata={"format": "I"})

    def __init__(self,
                 result: int,
                 code: EErrorCode,
                 correlation_id: UUID,
                 message: str | None = None,
                 priority: int = -1):
        super().__init__(code=code,
                         correlation_id=correlation_id,
                         priority=priority,
                         message=message)
        self.result = result

# Copy the request correlation ID into the result event
res = MyResult(code=EErrorCode.OK, correlation_id=req.correlation_id, result=456)
```

---

### `fifo_dev_common.event.fifo_event_queue_forwarder`

Bridges a blocking `multiprocessing.Queue` of `FifoEvent` instances to an `asyncio.PriorityQueue` using a background thread.

- Forwards events from `multiprocessing.Queue` → `asyncio.PriorityQueue`.
- Can stop cleanly via `.stop()` **or** by pushing a `FifoEventShutdown` event into the source queue.
- Optionally forwards the shutdown event to the asyncio queue.

**Example:**

```python
from __future__ import annotations
import asyncio
import multiprocessing
from typing import TYPE_CHECKING

from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_queue_forwarder import FifoEventQueueForwarderMpToAsync

# Proper type hints for the mp queue
if TYPE_CHECKING:
    from multiprocessing.queues import Queue as MpQueue
else:
    MpQueue = multiprocessing.Queue  # type: ignore[misc]


# Minimal concrete event for the demo
class DummyEvent(FifoEvent):
    event_id = 42
    default_priority = 0


async def main() -> None:
    # Create queues
    mp_q: MpQueue[FifoEvent] = MpQueue()
    async_q: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()

    # Bridge mp_q -> async_q with a background thread
    forwarder = FifoEventQueueForwarderMpToAsync(
        mp_q, async_q, asyncio.get_running_loop(), forward_shutdown_event=True
    )

    # Start the background thread
    forwarder.start()

    # Send an event into the multiprocessing queue
    mp_q.put(DummyEvent())

    # Receive it from the asyncio priority queue
    evt = await async_q.get()
    print("Got:", type(evt).__name__, "priority:", evt.priority)

    # Request shutdown (also forwards FifoEventShutdown into async_q)
    forwarder.stop()
    # Alternatively, instead of calling `stop()` you can directly push a FifoEventShutdown event:
    # mp_q.put(FifoEventShutdown())

    # Optionally consume the forwarded shutdown sentinel
    shutdown = await async_q.get()
    print("Got shutdown:", type(shutdown).__name__)
    assert isinstance(shutdown, FifoEventShutdown)

    # Wait for the background thread to stop
    forwarder.join()

if __name__ == "__main__":
    asyncio.run(main())
```

---

### `fifo_dev_common.event.fifo_event_queue_connector`

Reusable asyncio components shared by the transport-specific clients.

- `FifoEventQueueConnectorAsyncClient`: manages the background reader task,
  output queue, and graceful shutdown for any ``StreamReader``/``StreamWriter`` pair.
- `FifoEventQueueConnectorAsyncHandlerCID`: correlation-ID helper powering
  the higher-level network and serial helpers.

---

### `fifo_dev_common.event.fifo_event_queue_network`

Asyncio-based TCP transport for `FifoEvent` objects, built on top of
`asyncio.open_connection()` / `asyncio.start_server()` with optional TLS 1.3
encryption.

- `FifoEventQueueNetworkAsyncClient`: sends events immediately; receives events in a background task and enqueues them in a local `asyncio.PriorityQueue`.
- `FifoEventQueueNetworkAsyncServer`: accepts **exactly one** client at a time (additional clients are rejected until the connection closes); sends events immediately and receives events in a background task, enqueuing them in a local `asyncio.PriorityQueue`.
- `make_server_tls_context()` / `make_client_tls_context()`: helpers to build **TLS 1.3–only** SSL contexts.
- Optional pre-queue async handlers to process and optionally consume incoming events; includes a built-in correlation-ID handler for acknowledgements.
- Optional mutual TLS (mTLS): peer authentication.
- Scope: transport layer only (no application-layer authentication/authorization).

> **Security Note**
> - Certificate revocation (OCSP/CRL) is out of scope and **not** implemented.
> - Not audited or penetration-tested; not for safety-critical use.
> - Intended for personal use on an internal network with **non-sensitive data** (e.g., hobby robotics / experimentation).
> - TLS 1.3 and optional mTLS are supported, but there is **no application-layer authentication**.
> - Do **not** expose this service to the public internet or untrusted networks.

**Example:**

```python
import asyncio
from fifo_dev_common.event.fifo_event_queue_network import (
    FifoEventQueueNetworkAsyncServer, 
    FifoEventQueueNetworkAsyncClient,
    make_server_tls_context,
    make_client_tls_context
)
from fifo_dev_common.event.fifo_event import FifoEvent

# Define a custom event
@FifoEvent.register
class MyEvent(FifoEvent):
    event_id = 42
    default_priority = 5

async def run_server():
    # Create TLS context for server (mutual TLS)
    server_ctx = make_server_tls_context(
        certfile="path/to/server.crt",
        keyfile="path/to/server.key",
        cafile="path/to/ca.crt",
        require_client_cert=True  # Enable mTLS
    )

    # Create a priority queue for events received by the server
    queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()

    # Start server (accepts one client, then stops listening)
    server = await FifoEventQueueNetworkAsyncServer.accept(
        "127.0.0.1", 8800, ssl_ctx=server_ctx, out_queue=queue
    )
    
    # Send an event
    await server.send(MyEvent())
    
    # Receive events from client
    event = await queue.get()
    print(f"Server received: {type(event).__name__}")
    
    # Graceful shutdown
    await server.stop()
    await server.join()

async def run_client():
    # wait for the server to start
    await asyncio.sleep(1)

    # Create TLS context for client (mutual TLS)
    client_ctx = make_client_tls_context(
        certfile="path/to/client.crt",
        keyfile="path/to/client.key", 
        cafile="path/to/ca.crt",
        check_hostname=True
    )
    
    # Connect to server
    client = await FifoEventQueueNetworkAsyncClient.connect(
        "127.0.0.1", 8800, 
        ssl_ctx=client_ctx,
        server_hostname="localhost"
    )
    
    # Receive events from server
    event = await client._out_queue.get()
    print(f"Client received: {type(event).__name__}")
    
    # Send response
    await client.send(MyEvent())
    
    # Graceful shutdown  
    await client.stop()
    await client.join()

# Run server and client (in practice, these would be separate processes)
async def main():
    await asyncio.gather(run_server(), run_client())

if __name__ == "__main__":
    asyncio.run(main())
```

---

### `fifo_dev_common.event.fifo_event_queue_serial`

Asyncio serial transport built on top of `serial_asyncio.open_serial_connection()`.

- `FifoEventQueueSerialAsyncClient`: leverages the shared connector base to reuse
  the same API as the TCP client while targeting serial links.

---

### `fifo_dev_common.process.utils.FifoProcessManager`

Manager class for running workers in separate OS processes and handling interprocess communication.

This class abstracts the creation, startup, and shutdown of worker processes (either async or sync), and manages the threads that transfer events between the main process and worker processes. It provides async methods for sending events and ensures clean shutdown by propagating shutdown events and joining all threads and processes.

Key features:
- Supports both `FifoAsyncProcessWorkerCallback` and `FifoSyncProcessWorkerCallback`
- Automatic correlation ID tracking for request/response patterns via `send_and_wait_response()`
- Thread-safe event bridging between multiprocessing queues and asyncio priority queues
- Clean shutdown handling with proper thread and process joining

**Example:**

```python
import asyncio
from fifo_dev_common.process.utils import FifoProcessManager, FifoAsyncProcessWorkerCallback
from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventKeepAlive

class Echo(FifoAsyncProcessWorkerCallback):
    def initialize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        pass

    def finalize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        pass

    async def loop(self,
                   incoming_event: FifoEvent | None,
                   incoming_queue_size: int,
                   outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        if incoming_event is not None:
            await outgoing_queue.put(incoming_event)

    def get_timeout(self) -> float:
        return -1

async def main():
    loop = asyncio.get_running_loop()
    out_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    process_manager = FifoProcessManager(loop, Echo(), out_queue)
    process_manager.start()
    await process_manager.send(FifoEventKeepAlive())
    event = await out_queue.get()
    print(event)
    await process_manager.stop()
    process_manager.join()

if __name__ == "__main__":
    asyncio.run(main())
```

---

### `fifo_dev_common.logging.logger`

Defines `get_logger(name=None)` — returns a logger instance with `.trace()` support.  
Registers a custom TRACE log level (`level=5`) and uses a subclassed logger to enable fine-grained tracing.  
Compatible with the standard Python `logging` module.

**Examples:**

```python
from fifo_dev_common.logging.logger import get_logger

logger = get_logger(__name__)

logger.trace("This is a low-level trace message for debugging internals")
logger.debug("Standard debug message")
logger.info("Informational message")
```

> `trace()` messages are only shown if the logging level is set to `TRACE` (i.e. `level=5`).  
> You can enable them via:

```python
import logging
logging.basicConfig(level=5)
```

---

### `fifo_dev_common.state.fifo_refreshable_value`

Defines `FifoRefreshableValue[T]` — a lock-free, one-writer/many-readers cache
for asynchronously refreshed values.

The cache maintains an immutable `Snapshot` `(state, value, timestamp)` and
supports explicit transitions via `mark_refreshing()`, `set_success()`,
and `set_failure()`. States are managed through a simple state machine:
`STALE → REFRESHING → FRESH/ERROR`.

**Examples:**

```python
from fifo_dev_common.state.fifo_refreshable_value import (
    FifoRefreshableValue,
    CacheState,
)

# create a cache for float values (e.g., distances)
cache = FifoRefreshableValue[float]()

# writer side (e.g., from an async callback)
cache.mark_refreshing()
cache.set_success(1.23)    # publish a new value
# cache.set_failure()      # publish an error state

# reader side (control loop or other tasks)
snap = cache.snapshot()
if snap.state is CacheState.FRESH:
    print(f"Latest value: {snap.value} at {snap.ts}")
elif snap.state is CacheState.ERROR:
    print("Value source reported an error")
```

> Readers always see either the old snapshot or the new one and never a torn or partially updated state.
> For multiple writers across threads, funnel updates through a single owner (e.g., `loop.call_soon_threadsafe`).  

---

## 📦 C++ Library

`fifo-dev-common\fifo_buffer.h` is a header-only C++ library for binary serialization and deserialization, fully compatible with Python's `FifoSerializable` format.

### Key Features

- **Cross-language compatibility**: Binary format matches Python `FifoSerializable` exactly
- **Header-only**: No separate compilation required (see `include/fifo_dev_common/fifo_buffer.h`)
- **Type-safe**: Template-based API with compile-time checks
- **Memory-aware design**: Buffers grow dynamically and can be preallocated via constructor parameters to reduce allocation overhead and enable reuse
- **Contiguous memory**: Direct access to the underlying buffer for efficient I/O operations
- **Versatile type support**: Handles primitives, strings, arrays, optional values, nested objects, and basic read-only UUIDs
- **I/O extensions**: Includes optional helpers for sockets (Boost.Asio) and serial communication (Arduino)

### Requirements

- **C++20** or later (requires `std::endian`)
- **Optional**: Boost.Asio (for socket I/O helpers)
- **Optional**: Arduino (for Serial communication helpers)

### Core API

The `FifoBuffer` class provides the main serialization interface:

```cpp
#include <fifo_buffer.h>

// Writing
FifoBuffer buf;
buf.write_int32_t(42);
buf.write_string("hello");
buf.write_float(3.14f);

// Reading
int32_t value;
std::string text;
float pi;
if (buf.read_int32_t(value) &&
    buf.read_string(text) &&
    buf.read_float(pi)) {
    // Success
}

// Custom objects implement serialize() and a constructor taking FifoBuffer&
struct MyData {
    int32_t x;
    std::string name;

    MyData(int32_t _x, const char* lpsz_name) :
        x(_x),
        name(lpsz_name) {}
    
    void serialize(FifoBuffer& buf) const {
        buf.write_int32_t(x);
        buf.write_string(name);
    }
    
    explicit MyData(FifoBuffer& buf) {
        if (!buf.read_int32_t(x) || !buf.read_string(name)) {
            throw std::runtime_error("Deserialization failed");
        }
    }
};

// Serialize nested object
MyData data{42, "test"};
buf.write_nested_object(data);

// Deserialize nested object
MyData restored = buf.read_nested_object<MyData>();
```

### Integration with Arduino

To use this library with Arduino:

1. **Supported platforms**:
   - **ARM Cortex-M0/M4** (e.g., Adafruit Feather M0/M4)
   - **ESP32**
   - **Not supported**: AVR-based boards (Arduino Uno, Nano, Mega, etc.) lack C++20 support and sufficient RAM

2. **Enable C++20 in `platform.txt`**:
   ```
   compiler.cpp.flags=-std=gnu++2a -fexceptions
   compiler.c.flags=-std=gnu2x
   ```

3. **Install the library**:
   
   **On Linux**:
   ```bash
   cd ~/Arduino/libraries/
   ln -s /path/to/fifo-dev-common/include/fifo_dev_common fifo_dev_common
   ```
   
   **On Windows**:
   ```cmd
   cd %USERPROFILE%\Documents\Arduino\libraries\
   mklink /D fifo_dev_common C:\path\to\fifo-dev-common\include\fifo_dev_common
   ```

4. **Include in your sketch**:
   ```cpp
   #include <fifo_buffer.h>
   
   void setup() {
       Serial.begin(115200);
       
       FifoBuffer buf;
       buf.write_int32_t(42);
       buf.write_string("Arduino");
       
       // Send to Python via Serial
       Serial.write(buf.data(), buf.size());
   }
   ```

### Integration with Raspberry Pi (or Linux)

To use the C++ header-only library in your project, reference the `include` folder in your build system.

**For Makefile-based projects:**

1. Define the root directory of this repository:
    ```bash
    export FIFO_DEV_COMMON_ROOT=/path/to/fifo-dev-common
    ```

2. Add the include path to your `CXXFLAGS` or `CPPFLAGS`:
    ```makefile
    CXXFLAGS += -I${FIFO_DEV_COMMON_ROOT}/include/fifo_dev_common
    ```

**Example Makefile snippet:**
```makefile
CXXFLAGS += -I$(FIFO_DEV_COMMON_ROOT)/include/fifo_dev_common

main: main.cpp
    $(CXX) $(CXXFLAGS) -o main main.cpp
```

### Cross-Language Example

**Python side:**
```python
from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable
from dataclasses import dataclass, field

@serializable
@dataclass
class SensorData(FifoSerializable):
    temperature: float = field(metadata={"format": "f"})
    humidity: float = field(metadata={"format": "f"})
    
data = SensorData(temperature=22.5, humidity=45.0)
binary = data.to_bytes()
# Send to C++ via socket/serial...
```

**C++ side:**
```cpp
struct SensorData {
    float temperature;
    float humidity;

    // Default constructor for initialization
    SensorData() : temperature(0.0f), humidity(0.0f) {}
    
    // Constructor with values
    SensorData(float temp, float hum) : temperature(temp), humidity(hum) {}
    
    void serialize(FifoBuffer& buf) const {
        buf.write_float(temperature);
        buf.write_float(humidity);
    }
    
    explicit SensorData(FifoBuffer& buf) {
        if (!buf.read_float(temperature) || !buf.read_float(humidity)) {
            throw std::runtime_error("Failed to deserialize SensorData");
        }
    }
};

// Receive binary data from Python...
FifoBuffer buf;
// ... populate buffer with received bytes ...
SensorData data(buf);  // Deserialize
```

### Documentation

See the header file `include/fifo_dev_common/fifo_buffer.h` for complete API documentation, including:
- All supported data types and formats
- Socket I/O helpers (with Boost.Asio)
- Serial I/O helpers (on Arduino platforms)
- Basic read-only UUID support (`FifoUuid`)
- Endianness handling
- Memory management and capacity control

---

## 🧪 Tests

Run the test suite using `pytest`:

```bash
pytest tests/
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
