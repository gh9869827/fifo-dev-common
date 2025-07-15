[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Test Status](https://github.com/gh9869827/fifo-dev-common/actions/workflows/test.yml/badge.svg)

# `fifo-dev-common`

Shared core utilities for all `fifo-dev` repositories, under the `fifo_dev_common` namespace.

This package is designed to support the `fifo-dev` ecosystem with minimal dependencies. It provides the following for runtime type checks and casting, docstring parsing, and LLM tool support:

- `strict_cast()`: Runtime-enforced type casting.  
- `class MiniDocString`: Lightweight implementation designed specifically to parse Google-style docstrings
   and extract minimal, structured information for runtime type checking of inputs and outputs—ideal
   for LLM-based function calling and agent execution, without requiring any third-party dependencies.
- `class ReadOnlyList`: Immutable wrapper for list-like data.  
- `@tool_handler` / `@tool_query_source`: Decorators for defining tools and query sources in LLM-based agents.

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

---

## 🧪 Tests

Run the test suite using `pytest`:

```bash
pytest tests/
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
