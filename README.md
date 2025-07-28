[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Test Status](https://github.com/gh9869827/fifo-dev-common/actions/workflows/test.yml/badge.svg)

# ⚠️ Experimental Branch: `experimental/event` ⚠️

This branch contains experimental code for binary-serializable event with factory deserialization and class registration, designed for cross-system use.

Features, APIs, and behavior are subject to change or removal at any time.  
**Use at your own risk.**

# `fifo-dev-common`

Shared core utilities for all `fifo-dev` repositories, under the `fifo_dev_common` namespace.

This package is designed to support the `fifo-dev` ecosystem with minimal dependencies. It provides the following for runtime type checks and casting, docstring parsing, and LLM tool support:

- `strict_cast()`: Runtime-enforced type casting.  
- `class MiniDocStringFunction`: Lightweight parser for Google-style function docstrings. Extracts minimal structured information for runtime type checking of arguments and return values—useful for LLM-based function calling and agent execution without third-party dependencies.
- `class MiniDocStringClass`: Lightweight parser for Google-style class docstrings. Extracts the short and detailed description along with a list of declared `Attributes:` as raw name/type/description triples.
- `class ReadOnlyList`: Immutable wrapper for list-like data.  
- `@tool_handler` / `@tool_query_source`: Decorators for defining tools and query sources in LLM-based agents.
- `class FifoEvent`: Base class for binary-serializable events, with factory deserialization and class registration for cross-system use.

See the [Example Usage](#-example-usage) section below for how these functions, classes, and decorators can be used.

---

## 📚 Table of Contents

- [🎯 Project Status & Audience](#-project-status--audience)
- [📦 Install](#-install)
- [🧩 Modules](#-modules)
  - [strict_cast](#fifo_dev_commontypeutilsstrict_cast)
  - [mini_docstring](#fifo_dev_commonintrospectionmini_docstring)
  - [read_only_list](#fifo_dev_commoncontainersread_onlyread_only_list)
  - [tool_decorator](#fifo_dev_commonintrospectiontool_decorator)
  - [socket_utils](#fifo_dev_commonsocketsocket_utils)
  - [fifo_serialization](#fifo_dev_commonserializationfifo_serialization)
  - [fifo_event](#fifo_dev_commoneventfifo_event)
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

## 📦 Install

This repo is meant for local development. Install in editable mode:

```bash
python3 -m pip install -e .
```

Python 3.10+ is required.

---

## 🧩 Modules

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
- Supports scalar types, enums, optional fields, arrays, and fixed-length tuples.
- Uses dataclass `field` metadata (e.g., `format`, `ptype`) for flexible, extensible field definitions.  
- Designed for preallocated buffers to maximize performance and minimize allocations.  
- Works well with microcontroller and embedded system data formats as it is a compact binary format prioritizing direct raw serialization with very little overhead.

**Supported Format Strings:**

| Format     | Description                          | Serialization Details                                                  | Notes                                                                 |
|------------|--------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `x`        | **Primitive value**                  | Serialized directly using `struct`                                     | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `[x]`      | **Array of primitives**              | 4-byte length prefix + consecutive `x` values                          | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `[?x]`     | **Array of optional primitives**     | 4-byte length + presence bitmap + serialized values for present items  | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `?x`       | **Optional primitive**               | 1-byte presence flag + `x` if present                                  | `x` must be one of: `b B h H i I l L q Q e f d y`                     |
| `T[xx...]` | **Fixed-length tuple of primitives** | Each element serialized consecutively (no prefix)                    | Each `x` must be one of: `b B h H i I l L q Q e f d y`                  |
| `[np:x]`   | **NumPy array**                      | 1-byte ndim + N×4-byte shape + raw data buffer                         | `x` must be NumPy dtype: `u8`, `u16`, `u32`, `i8`, `i16`, `i32`, `f32`, `f64` |
| `S`        | **Variable-length UTF-8 string**     | 4-byte length prefix + UTF-8 encoded bytes                             | -                                                                     |
| `S[x]`     | **Fixed-length UTF-8 string**        | UTF-8 encoded, space-padded or truncated to `x` bytes (UTF-8 safe)     | -                                                                     |
| `E<x>`     | **Enum stored as integer**           | Stored as `x` (e.g., `B`, `H`, `I`)                                    | `x` must be one of: `b B h H i I`                                     |
| `?_`       | **Optional nested object**           | 1-byte presence flag + nested serialization if present                 | -                                                                     |
| `[_]`      | **Array of nested objects**          | 4-byte length prefix + consecutive nested serializations               | -                                                                     |
| `[?_]`     | **Array of optional nested objects** | 4-byte length + presence bitmap + serialized present objects           | -                                                                     |

**Note:** Primitive format codes `b B h H i I l L q Q e f d` follow the [Python `struct` module](https://docs.python.org/3/library/struct.html).  
`y` is a special format for booleans, serialized as a single byte (`0` for `False`, `1` for `True`).

**Examples:**

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

---

## 🧪 Tests

Run the test suite using `pytest`:

```bash
pytest tests/
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
