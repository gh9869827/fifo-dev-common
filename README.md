[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

# fifo-dev-common

Core Python modules shared across all `fifo-dev` repositories, available under the `fifo_dev_common`
namespace.

This package provides:

- `strict_cast`: runtime-enforced type casting  
- `MiniDocString`: Google-style docstring parser for schema extraction and validation  
- `ReadOnlyList`: immutable wrapper for list-like data  

It is designed for the `fifo-dev` ecosystem, to enforce type safety and explicit runtime behavior.  
Primarily intended for internal use.

## Install

This repo is meant for local development use. Install via editable mode:

```bash
pip install -e .
```

Python 3.10+ is required.

## Modules

### `fifo_dev_common.cast`

Provides `strict_cast(tp, value)`, which raises a `TypeError` unless the value matches the expected
type(s). Useful for defensive programming where `typing.cast()` is not strict enough.

### `fifo_dev_common.docstring`

Defines `MiniDocString`, a parser that converts a Google-style docstring into a structured object.  
Includes runtime type checking via `MiniDocStringType` and support for YAML schema serialization.

### `fifo_dev_common.read_only_list`

Implements `ReadOnlyList`, a lightweight, immutable wrapper around a Python list.  
Supports indexing, iteration, and comparison.

Note: Nested contents are not recursively frozen, i.e. inner lists or dicts must themselves be 
      wrapped or treated immutably by convention.

## Example Usage

```python
from fifo_dev_common.typeutils.cast import strict_cast
from fifo_dev_common.introspection.docstring import MiniDocString
from fifo_dev_common.containers.read_only.read_only_list import ReadOnlyList

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
print(items[0]) # 1
```

## License

MIT — see [LICENSE](LICENSE) for details.