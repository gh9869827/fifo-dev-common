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
- `class MiniDocString`: Lightweight implementation designed specifically to parse Google-style docstrings
   and extract minimal, structured information for runtime type checking of inputs and outputs—ideal
   for LLM-based function calling and agent execution, without requiring any third-party dependencies.
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
- [✅ Example Usage](#-example-usage)
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

---

### `fifo_dev_common.introspection.mini_docstring`

Provides the `MiniDocString` class for parsing Google-style docstrings into structured form.  
Includes:

- Argument type extraction (`MiniDocStringArg`)
- Return and raise parsing
- Runtime type validation
- Export to YAML schema for structured function calls

---

### `fifo_dev_common.containers.read_only.read_only_list`

Implements `ReadOnlyList`, a lightweight wrapper that disables mutation of a list.  
Supports indexing, iteration, equality, and containment.

> ⚠️ Inner objects (like nested lists/dicts) are not automatically frozen.  
> For example, `ReadOnlyList([{"x": 1}])[0]["x"] = 2` is still allowed.

---

### `fifo_dev_common.introspection.tool_decorator`

Decorators to define tools and runtime query sources callable by large language models:

- `@tool_handler(name)`: Declare a function as an executable tool with schema support.  
- `@tool_query_source(name)`: Define a no-arg runtime data source that provides context for LLM execution planning.

These attach structured metadata derived from docstrings—enabling parsing, validation, and schema generation for transparent agent planning and execution.

### `fifo_dev_common.serialization.fifo_serialization`

Provides a lightweight, efficient binary serialization framework for Python dataclasses.

- `@serializable`: Decorator to enable serialization/deserialization on dataclasses.  
- Supports scalar types, enums, optional fields, arrays, and fixed-length tuples.
- Uses dataclass `field` metadata (e.g., `format`, `ptype`) for flexible, extensible field definitions.  
- Designed for preallocated buffers to maximize performance and minimize allocations.  
- Works well with microcontroller and embedded system data formats as it is a compact binary format prioritizing direct raw serialization with very little overhead.

**Supported format strings examples:**

| Format String | Description                                  | Serialization Details                            | Example                                  |
|---------------|----------------------------------------------|-------------------------------------------------|------------------------------------------|
| `I`           | Unsigned 4-byte integer                      | 4 bytes little-endian integer                    | `field(metadata={"format": "I"})`        |
| `f`           | 4-byte float                                 | 4 bytes little-endian float                      | `field(metadata={"format": "f"})`        |
| `[f]`         | Array of floats                              | 4-byte length prefix + consecutive float values | `field(metadata={"format": "[f]"})`      |
| `[I]`         | Array of unsigned ints                       | 4-byte length prefix + consecutive uint32 values| `field(metadata={"format": "[I]"})`      |
| `[np:u8]`     | NumPy array of uint8                        | 1 byte ndim + N x 4 byte shape + raw data       | `field(metadata={"format": "[np:u8]"})` |
| `[np:f32]`    | NumPy array of float32                      | 1 byte ndim + N x 4 byte shape + raw data       | `field(metadata={"format": "[np:f32]"})` |
| `?I`          | Optional unsigned int (presence flag + value) | 1 byte presence flag (0/1) + 4-byte uint if present | `field(metadata={"format": "?I"})`        |
| `?_`          | Optional nested serializable object (`_` is literal) | 1 byte presence flag + serialized nested object if present | `field(metadata={"format": "?_", "ptype": MyClass})` |
| `[_]`         | Array of nested serializable objects (`_` is literal) | 4-byte length prefix + serialized nested objects in sequence | `field(metadata={"format": "[_]", "ptype": MyClass})` |
| `[?_]`        | Array of optional nested objects (`_` is literal) | 4-byte length + presence bitmap + serialized present objects | `field(metadata={"format": "[?_]", "ptype": MyClass})` |
| `E<x>`        | Enum stored as integer type `x` (`b`, `B`, `h`, `H`, `i`, `I`) | Integer representing the Enum value using chosen size | `field(metadata={"format": "E<B>", "ptype": MyEnum})` |
| `T<x>`        | Fixed-length tuple of basic types            | Raw binary data for each tuple element           | `field(metadata={"format": "T<If>"})` |

`x` in `[np:x]` can be one of `u8`, `u16`, `u32`, `i8`, `i16`, `i32`, `f32`, or `f64`.

**Note:** Standard [Python `struct` format characters](https://docs.python.org/3/library/struct.html#format-characters) are supported for scalar types, such as `B`, `b`, `H`, `h`, `I`, `i`, `Q`, `q`, `f`, and `d`.

Tuples encode elements consecutively with no length prefix or additional metadata.

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

---

## ✅ Example Usage

### `fifo_dev_common.typeutils.strict_cast` example

```python
from fifo_dev_common.typeutils.strict_cast import strict_cast

value = strict_cast(int, 42)

try:
    value = strict_cast(int, "42")
except TypeError as e:
    print(e)
    # Output:
    # TypeError: strict_cast failed: expected int, got str
```

### `fifo_dev_common.introspection.mini_docstring` example

```python
from fifo_dev_common.introspection.mini_docstring import MiniDocString

doc = """
Brief summary.

Args:
    task_id (int):
        Unique identifier for the task.
    tags (list[str]):
        List of tags. Can be empty.

Returns:
    str:
        The task description in serialized format.
"""

parsed = MiniDocString(doc)
assert parsed.get_arg_by_name("task_id").pytype.to_string() == "int"
assert parsed.return_desc == "The task description in serialized format."

parsed.validate_runtime_args({
    "task_id": 42,
    "tags": ["tag1", "tag2", "tag3"]
})
# Validation completes successfully

try:
    parsed.validate_runtime_args({
        "task_id": 42,
        "tags": [1, 2, 3]
    })
except ValueError as e:
    print(e)
    # Output:
    # ValueError: Argument 'tags' expected ArgType(list[str]), but got list

try:
    parsed.validate_runtime_args({
        "task_id": 42,
        "tags": ["tag1", "tag2", "tag3"],
        "extra_args": "extra_value"
    })
except ValueError as e:
    print(e)
    # Output:
    # Unexpected arguments: extra_args
```

### `fifo_dev_common.containers.read_only.read_only_list` example

```python
from fifo_dev_common.containers.read_only.read_only_list import ReadOnlyList


items = ReadOnlyList([1, 2, 3])
print(items[0])
# Output:
# 1

try:
    # Pylance warning: "__setitem__" method not defined on type "ReadOnlyList[int]"
    # Pylint warning: 'items' does not support item assignment
    items[0] = 2
except TypeError as e:
    print(e)
    # Output:
    # TypeError: 'ReadOnlyList' object does not support item assignment
```
### `fifo_dev_common.introspection.tool_decorator` example

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

### `fifo_dev_common.serialization.fifo_serialization` example

```python
from dataclasses import dataclass, field
from typing import List
from fifo_dev_common.serialization.fifo_serialization import FifoSerializable, serializable

@serializable
@dataclass
class SensorReadings(FifoSerializable):
    temperature: float = field(metadata={"format": "f"})
    humidity: float = field(metadata={"format": "f"})

@serializable
@dataclass
class SensorArray(FifoSerializable):
    readings: List[SensorReadings] = field(metadata={"format": "[_]", "ptype": SensorReadings})

# Usage
s1 = SensorReadings(temperature=22.5, humidity=40.0)
s2 = SensorReadings(temperature=23.0, humidity=38.5)
sensor_data = SensorArray(readings=[s1, s2])

buffer = bytearray(sensor_data.serialized_byte_size())
sensor_data.serialize_to_bytes(buffer, 0)

deserialized, _ = SensorArray.deserialize_from_bytes(buffer, 0)
assert len(deserialized.readings) == 2
assert abs(deserialized.readings[0].temperature - 22.5) < 1e-6
assert abs(deserialized.readings[1].humidity - 38.5) < 1e-6
```

```python
@serializable
@dataclass
class MaybeSensorArray(FifoSerializable):
    readings: List[SensorReadings | None] = field(metadata={"format": "[?_]", "ptype": SensorReadings})

data = MaybeSensorArray(readings=[s1, None])
buf = bytearray(data.serialized_byte_size())
data.serialize_to_bytes(buf, 0)
restored, _ = MaybeSensorArray.deserialize_from_bytes(buf, 0)
assert restored.readings[0] is not None
assert restored.readings[1] is None
```


### `fifo_dev_common.event.fifo_event` example

```python
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

from fifo_dev_common.event.fifo_event import FifoEvent, serializable

class State(IntEnum):
    INIT = 1
    RUN = 2
    DONE = 3

@serializable
@dataclass
class Point:
    x: int = field(metadata={"format": "i"})
    y: int = field(metadata={"format": "i"})

@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DemoEvent(FifoEvent):
    event_id: ClassVar[int] = 99
    default_priority: ClassVar[int] = 3

    score: int = field(metadata={"format": "i"})
    state: State = field(metadata={"format": "E<B>", "ptype": State})
    position: Point = field(metadata={"ptype": Point})

    def __init__(self, score: int, state: State, position: Point, priority: int = -1):
        super().__init__(priority=priority)
        self.score = score
        self.state = state
        self.position = position

# Usage example:
evt = DemoEvent(score=42, state=State.RUN, position=Point(7, 8), priority=77)

blob = evt.to_bytes()

restored = FifoEvent.from_bytes(blob)
assert isinstance(restored, DemoEvent)
assert restored.score == 42
assert restored.state == State.RUN
assert restored.position == Point(7, 8)
assert restored.priority == 77
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
