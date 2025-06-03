[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

# `fifo-dev-common`

Shared core utilities for all `fifo-dev` repositories, under the `fifo_dev_common` namespace.

This package provides foundational tools for runtime safety, docstring parsing, and LLM tool support:

- `strict_cast`: Runtime-enforced type casting.  
- `MiniDocString`: Lightweight implementation designed specifically to parse Google-style docstrings
   and extract minimal, structured information for runtime type checking of inputs and outputs—ideal
   for LLM-based function calling and agent execution, without requiring any third-party dependencies.
- `ReadOnlyList`: Immutable wrapper for list-like data.  
- `@tool_handler` / `@tool_query_source`: Decorators for defining tools and query sources in LLM-based agents.

It is designed to support the `fifo-dev` ecosystem with minimal dependencies and strong runtime guarantees.  
Primarily intended for internal use.

---

## 📚 Table of Contents

- [📦 Install](#-install)
- [🧩 Modules](#-modules)
  - [strict_cast](#fifo_dev_commonstrict_cast)
  - [mini_docstring](#fifo_dev_commonmini_docstring)
  - [read_only_list](#fifo_dev_commonread_only_list)
  - [tool_decorator](#fifo_dev_commontool_decorator)
- [✅ Example Usage](#-example-usage)
- [🧪 Tests](#-tests)
- [📄 License](#-license)

---

## 📦 Install

This repo is meant for local development. Install in editable mode:

```bash
python3 -m pip install -e .
```

Python 3.10+ is required.

---

## 🧩 Modules

### `fifo_dev_common.strict_cast`

Defines `strict_cast(tp, value)` — a runtime-enforced version of `typing.cast()`.  
Raises `TypeError` if the value does not match the expected type(s).  
**Shallow check**: only verifies the outermost type (e.g., `list`, not `list[int]`).

---

### `fifo_dev_common.mini_docstring`

Provides the `MiniDocString` class for parsing Google-style docstrings into structured form.  
Includes:

- Argument type extraction (`MiniDocStringArg`)
- Return and raise parsing
- Runtime type validation
- Export to YAML schema for structured function calls

---

### `fifo_dev_common.read_only_list`

Implements `ReadOnlyList`, a lightweight wrapper that disables mutation of a list.  
Supports indexing, iteration, equality, and containment.

> ⚠️ Inner objects (like nested lists/dicts) are not automatically frozen.  
> For example, `ReadOnlyList([{"x": 1}])[0]["x"] = 2` is still allowed.

---

### `fifo_dev_common.tool_decorator`

Decorators to define tools and runtime query sources callable by large language models:

- `@tool_handler(name)`: Declare a function as an executable tool with schema support.  
- `@tool_query_source(name)`: Define a no-arg runtime data source that provides context for LLM execution planning.

These attach structured metadata derived from docstrings and function signatures—enabling parsing, validation, and schema generation for transparent agent planning and execution.

---

## ✅ Example Usage

```python
from fifo_dev_common.strict_cast import strict_cast
from fifo_dev_common.mini_docstring import MiniDocString
from fifo_dev_common.read_only_list import ReadOnlyList
from fifo_dev_common.tool_decorator import tool_handler

value = strict_cast(int, 42)

doc = """
Brief summary.

Args:
    task_id (int):
        Unique identifier for the task.
    tags (list[str]):
        Optional list of tags.

Returns:
    str:
        The task description in serialized format.
"""

parsed = MiniDocString(doc)
assert parsed.get_arg_by_name("task_id").pytype.to_string() == "int"

items = ReadOnlyList([1, 2, 3])
print(items[0])  # → 1

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
