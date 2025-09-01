import pytest
from typing import cast, Callable
from fifo_dev_common.introspection.tool_decorator import (
    tool_handler,
    tool_query_source,
    ToolHandler,
    ToolQuerySource,
)


# --------------------------------------------------------------------------- #
# Positive tests - tool_handler
# --------------------------------------------------------------------------- #

ADD_YAML_DESCRIPTION = \
"""- intent: add
  description: Add two numbers.
  parameters:
    - name: x
      type: int
      description: first operand
      optional: False
    - name: y
      type: int
      description: second operand
      optional: False
  return:
    type: int
    description: sum of the operands"""

DESCRIBE_TASK_DESCRIPTION = \
"""- describe_task:
  description: >
    A description of the task."""


def test_tool_handler_basic_success():
    @tool_handler("add")
    def add(x: int, y: int) -> int:
        """
        Add two numbers.

        Args:
            x (int): first operand
            y (int): second operand

        Returns:
            int: sum of the operands
        """
        return x + y

    # The decorator should have returned a callable that conforms to ToolHandler
    assert isinstance(add, ToolHandler)

    assert hasattr(add, "tool_name") and add.tool_name == "add"
    assert hasattr(add, "tool_docstring")
    assert hasattr(add, "to_schema_yaml")

    # The generated schema must contain the tool name and a description
    assert add.to_schema_yaml() == ADD_YAML_DESCRIPTION


def test_tool_handler_method_success():
    """
    The `tool_handler` decorator should also work on instance methods.

    The generated schema must be identical to the one produced when decorating a plain function.
    """

    class Calculator:

        @tool_handler("add")
        def add(self, x: int, y: int) -> int:
            """Add two numbers.

            Args:
                x (int): first operand
                y (int): second operand

            Returns:
                int: sum of the operands
            """
            return x + y

    calc = Calculator()
    # The wrapper lives on the class, not on each instance.
    assert isinstance(Calculator.add, ToolHandler)

    assert hasattr(calc.add, "tool_name") and calc.add.tool_name == "add"
    assert hasattr(calc.add, "tool_docstring")
    assert hasattr(calc.add, "to_schema_yaml")

    # The generated schema must contain the tool name and a description
    assert calc.add.to_schema_yaml() == ADD_YAML_DESCRIPTION


def test_tool_handler_callable_behavior():

    @tool_handler("echo")
    def echo(msg: str) -> str:
        """
        Echo a message.

        Args:
            msg (str): the message to return

        Returns:
            str: same as ``msg``
        """
        return msg

    echo(msg="hello")

    # Direct call should work exactly like the original function
    assert echo(msg="hello") == "hello"

    # The wrapper retains the original __name__ and __doc__
    echo_func = cast(Callable[..., str], echo)
    assert "echo" == echo_func.__name__
    assert echo_func.__doc__ is not None and "Echo a message" in echo_func.__doc__


def test_tool_handler_class_method_callable_behavior():

    class Greeter:
        @tool_handler("greet")
        def greet(self, msg: str) -> str:
            """
            Greet a message.

            Args:
                msg (str): the message to return

            Returns:
                str: same as ``msg``
            """
            return msg

    greeter = Greeter()

    # The wrapper lives on the class, not per‑instance.
    assert isinstance(Greeter.greet, ToolHandler)

    # Calling through the instance works exactly like the original function.
    assert greeter.greet(msg="hello") == "hello"

    # The wrapper retains the original __name__ and __doc__
    greet_func = cast(Callable[..., str], greeter.greet)
    assert greet_func.__name__ == "greet"
    assert greet_func.__doc__ is not None and "Greet a message" in greet_func.__doc__


# --------------------------------------------------------------------------- #
# Positive tests - tool_query_source
# --------------------------------------------------------------------------- #

def test_tool_query_source_function_no_args():
    @tool_query_source("describe_task")
    def describe_task() -> str:
        """Returns a description of the task.

        Returns:
            str: description of the current task
        """
        return "Task #42"

    assert isinstance(describe_task, ToolQuerySource)
    assert hasattr(describe_task, "source_name") and describe_task.source_name == "describe_task"
    assert hasattr(describe_task, "source_docstring")
    assert hasattr(describe_task, "get_description")

    # The callable must still be invocable with no arguments
    assert describe_task() == "Task #42"
    assert describe_task.get_description() == DESCRIBE_TASK_DESCRIPTION


def test_tool_query_source_instance_method_self_only():
    class Test:
        @tool_query_source("describe_task")
        def describe_task(self) -> str:
            """Returns a description of the task.

            Returns:
                str: description of the current task
            """
            return "Task #99"

    t = Test()

    assert isinstance(Test.describe_task, ToolQuerySource)
    assert hasattr(t.describe_task, "source_name") and t.describe_task.source_name == "describe_task"
    assert hasattr(t.describe_task, "source_docstring")
    assert hasattr(t.describe_task, "get_description")

    # The callable must still be invocable with no arguments
    assert t.describe_task() == "Task #99"
    assert t.describe_task.get_description() == DESCRIBE_TASK_DESCRIPTION


# --------------------------------------------------------------------------- #
# Negative tests - tool_query_source
# --------------------------------------------------------------------------- #

def test_tool_query_source_extra_arg_raises():
    with pytest.raises(ValueError) as excinfo:

        @tool_query_source("invalid")
        def invalid(_a: int) -> str: # type: ignore
            """Invalid source.

            Args:
                a (int): stray argument

            Returns:
                str: something
            """
            return "nope"

    assert "Source 'invalid' summary must start with one of: Returns, " in str(excinfo.value)


def test_tool_query_source_missing_return_doc_raises():
    with pytest.raises(RuntimeError) as excinfo:

        @tool_query_source("missing_ret")
        def missing_ret() -> str: # type: ignore
            """Returns a value.

            # No Returns section!
            """
            return "oops"

    assert "Source missing_ret must return a string (str), not None." in str(excinfo.value)


def test_tool_query_source_non_str_return_type_raises():
    with pytest.raises(RuntimeError) as excinfo:

        @tool_query_source("non_str") # type: ignore
        def non_str() -> int: # type: ignore
            """Returns an int, which is illegal for a query source.

            Returns:
                int: some number
            """
            return 123

    assert "must return a string (str)" in str(excinfo.value)


def test_tool_query_source_invalid_args_raises():
    """The decorator must reject query sources that have arguments other than a single `self`."""
    # The function has two parameters, which is not allowed for a query source.
    with pytest.raises(RuntimeError) as excinfo:

        @tool_query_source("bad_source")
        def bad_source(_var_a: int, _var_b: str) -> str: # type: ignore
            """
            Returns a description.
            
            Args:
                _var_a (int):
                    First argument
                
                _var_b (int):
                    Second argument
            
            Returns:
                str:
                    description
            """
            return "oops"

    # The error message should not list the unexpected argument names.
    assert "unexpected arguments" in str(excinfo.value)
    assert "_var_a" not in str(excinfo.value)
    assert "_var_b" not in str(excinfo.value)


def test_tool_query_source_invalid_args_raises_1_arg():
    """The decorator must reject query sources that have arguments other than a single ``self``."""
    # The function has one parameter (which is not self), which is not allowed for a query source.
    with pytest.raises(RuntimeError) as excinfo:

        @tool_query_source("bad_source")
        def bad_source(_var_a: int) -> str: # type: ignore
            """
            Returns a description.
            
            Args:
                _var_a (int):
                    First argument

            Returns:
                str:
                    description
            """
            return "oops"

    # The error message should not list the unexpected argument names.
    assert "unexpected arguments" in str(excinfo.value)
    assert "_var_a" not in str(excinfo.value)


def test_tool_query_source_invalid_method_args_raises():
    """The decorator must reject query‑source *methods* that have arguments other than ``self``."""
    # The method incorrectly declares two additional positional args.
    with pytest.raises(RuntimeError) as excinfo:

        class BadSource:
            @tool_query_source("bad_method")
            def bad_method(self, _var_a: int, _var_b: str) -> str:  # type: ignore
                """
                Returns a description.

                Args:
                    _var_a (int):
                        First argument

                    _var_b (str):
                        Second argument

                Returns:
                    str:
                        description
                """
                return "oops"

        # Instantiation is not needed; the decorator runs at class definition time.
        BadSource()

    # The error message should mention the unexpected arguments.
    assert "unexpected arguments" in str(excinfo.value)
    assert "_var_a" not in str(excinfo.value)
    assert "_var_b" not in str(excinfo.value)


def test_tool_query_source_invalid_no_return_desc():
    """The decorator must reject query sources that return a value but does not provide a description."""

    with pytest.raises(RuntimeError) as excinfo:

        @tool_query_source("bad_source")
        def bad_source() -> str: # type: ignore
            """
            Returns a str but no description.
            
            Returns:
                str:
            """
            return "oops"

    # The error message should list the unexpected argument names.
    assert "Source bad_source does not document its return value" in str(excinfo.value)


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
