import pytest
from typing import cast, Callable
from fifo_dev_common.introspection.tool_decorator import (
    tool_handler,
    tool_query_source,
    ToolHandler,
    ToolQuerySource,
)


def test_summary_truncation_raises_value_error():

    # The decorator runs at import time, so the exception is raised during decoration.
    with pytest.raises(ValueError) as exc_info:

        # Define a function with an invalid docstring summary that is longer than 10 chars
        @tool_query_source("bad_desc")
        def bad_func() -> str: # type: ignore
            """
            InvalidSummaryThatDoesNotStartWithAllowedPrefix
            Returns: something
            """
            return "test"

    # Verify that the message contains only the first 10 characters of the summary plus
    # an ellipsis
    msg = str(exc_info.value)
    assert "Got: 'InvalidSum…'" in msg or "Got: \"InvalidSum…\"" in msg
