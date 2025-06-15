"""
Decorators for annotating functions as LLM-executable tools or query sources.

These decorators attach metadata used for planning, execution, and schema generation:
  - `@tool_handler(name)`: marks a callable as a tool with structured schema metadata
  - `@tool_query_source(name)`: marks a function returning context info (e.g., current date)

Decorated functions expose `tool_name`, `tool_docstring`, and structured format methods like
`to_schema_yaml()` or `get_description()`, depending on the type.
"""

import re
from textwrap import indent
from typing import Protocol, runtime_checkable, Callable, cast, Any
from fifo_dev_common.introspection.mini_docstring import MiniDocString

@runtime_checkable
class ToolHandler(Protocol):
    tool_docstring: MiniDocString
    tool_name: str
    def to_schema_yaml(self) -> str: ...
    def __call__(self, **kwargs: Any) -> Any: ...

def tool_handler(name: str) -> Callable[[Callable[..., Any]], ToolHandler]:
    """
    Decorator to annotate a callable tool with a `MiniDocString` and a logical tool name, used as a
    unique identifier in schemas or execution plans.

    This attaches three attributes to the target function:
      - `tool_name`: a string identifier used for naming or routing
      - `tool_docstring`: a parsed representation of the function's docstring
      - `to_schema_yaml()`: a method that returns the tool's schema in YAML format

    Args:
        name (str):
            The logical name of the tool, typically used as an identifier in schemas.

    Returns:
        Callable: The original function, enriched with `.tool_name` and `.tool_docstring`
        attributes, and recognized as conforming to the ToolHandler protocol.
    """
    def decorator(fn: Callable[..., Any]) -> ToolHandler:
        tool = cast(ToolHandler, fn)

        # Attach metadata
        setattr(tool, "tool_name", name)
        setattr(tool, "tool_docstring", MiniDocString(fn.__doc__))

        def to_schema_yaml() -> str:
            return tool.tool_docstring.to_schema_yaml("intent", tool.tool_name)

        setattr(tool, "to_schema_yaml", to_schema_yaml)

        return tool

    return decorator


@runtime_checkable
class ToolQuerySource(Protocol):
    source_docstring: MiniDocString
    source_name: str
    def get_description(self) -> str: ...
    def __call__(self, **kwargs: Any) -> Any: ...

_ALLOWED_PREFIXES = ("Returns", "Provides", "Gets", "Fetches", "Supplies")

def tool_query_source(name: str) -> Callable[[Callable[[Any], str]], ToolQuerySource]:
    """
    Decorator to annotate a callable tool query source with a MiniDocString and a logical tool name.

    This attaches three attributes to the target function:
      - `source_name`: a string identifier used for naming or routing
      - `source_docstring`: a parsed representation of the function's docstring
      - `get_description()`: a method that returns the tool's human-readable description

    The decorated function must:
      - Have no parameters
      - Return a `str`

    Args:
        name (str):
            The logical name of the tool query source, typically used to identify queryable runtime
            context.

    Returns:
        Callable: The original function, enriched with `.source_name` and `.source_docstring`
        attributes, and recognized as conforming to the ToolQuerySource protocol.
    """
    def decorator(fn: Callable[[Any], str]) -> ToolQuerySource:
        tool = cast(ToolQuerySource, fn)

        # Attach metadata
        setattr(tool, "source_name", name)
        setattr(tool, "source_docstring", MiniDocString(fn.__doc__))

        def _clean_summary_for_prompt(summary: str) -> str:
            match = re.match(
                r"^(Returns|Provides|Gets|Fetches|Supplies)\s+(.*)", summary, flags=re.IGNORECASE
            )
            if not match:
                raise ValueError(
                    f"Source '{name}' summary must start with one of: "
                    f"{', '.join(_ALLOWED_PREFIXES)}.\n"
                    f"Got: {summary!r}"
                )
            rest = match.group(2)
            return rest[0].upper() + rest[1:]

        short = _clean_summary_for_prompt(tool.source_docstring.description_short)

        if tool.source_docstring.description_detailed == "":
            cleaned_desc = short
        else:
            cleaned_desc = short + "\n" + tool.source_docstring.description_detailed

        cleaned_desc = f"- {name}:\n  description: >\n" + indent(cleaned_desc, "    ")

        def get_description() -> str:
            return cleaned_desc

        setattr(tool, "get_description", get_description)

        if tool.source_docstring.args:
            raise RuntimeError(f"Source {name} has unexpected arguments.")

        if tool.source_docstring.return_type is None:
            raise RuntimeError(f"Source {name} must return a string (str), not None.")

        if tool.source_docstring.return_type.matches_by_type(str) is False:
            raise RuntimeError(f"Source {name} must return a string (str),"
                               f" not a '{tool.source_docstring.return_type}'.")

        if tool.source_docstring.return_desc == "":
            raise RuntimeError(f"Source {name} does not document its return value.")

        return tool

    return decorator

__all__ = [
    "tool_handler",
    "tool_query_source",
    "ToolHandler",
    "ToolQuerySource"
]
