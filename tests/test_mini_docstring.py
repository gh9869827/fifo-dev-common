import pytest
from fifo_dev_common.introspection.mini_docstring import MiniDocString, MiniDocStringArg


def test_parses_short_and_detailed_description():
    doc = """
    Short description.

    This function does something useful.
    2nd description line.

    Args:
        x (int):
            A number to use
    """
    mds = MiniDocString(doc)
    assert mds.description_short == "Short description."
    assert mds.description_detailed == "This function does something useful.\n2nd description line."


def test_parses_args_section_without_description():
    # Negative test: missing description
    doc_missing_desc = """
    Args:
        x (int):
            First number

        y (Optional[str]):
            Second one, optional
    """
    with pytest.raises(ValueError, match="description.*before.*Args"):
        MiniDocString(doc_missing_desc)


def test_parses_args_section_with_description():
    # Positive test: valid description included
    doc_with_desc = """
    Parses two values.

    Args:
        x (int):
            First number
        y (Optional[str]):
            Second one, optional
    """
    mds = MiniDocString(doc_with_desc)
    assert len(mds.args) == 2
    assert mds.args[0].name == "x"
    assert mds.args[0].pytype.to_string() == "int"
    assert mds.args[1].pytype.to_string() == "Optional[str]"


def test_parses_returns_section_without_description():
    # Negative test: missing description before Returns section
    doc_missing_desc = """
    Returns:
        str:
            A message
    """
    with pytest.raises(ValueError, match="description.*before.*Returns"):
        MiniDocString(doc_missing_desc)


def test_parses_returns_section_with_description():
    # Positive test: parses return type and description correctly
    doc_with_desc = """
    Returns a message.

    Returns:
        str:
            A message
    """
    mds = MiniDocString(doc_with_desc)
    assert mds.return_type is not None
    assert mds.return_type.to_string() == "str"
    assert mds.return_desc is not None and mds.return_desc == "A message"


def test_parses_raises_section_without_description():
    # Negative test: missing description before Raises section
    doc_missing_desc = """
    Raises:
        ValueError: if the input is invalid
    """
    with pytest.raises(ValueError, match="description.*before.*Raises"):
        MiniDocString(doc_missing_desc)


def test_parses_raises_section_with_description():
    # Positive test: parses raises section when description is present
    doc_with_desc = """
    Validates the input and raises if invalid.

    Raises:
        ValueError: if the input is invalid
    """
    mds = MiniDocString(doc_with_desc)
    assert mds.raises is not None and mds.raises == "ValueError: if the input is invalid"


def test_get_arg_by_name():
    doc = """
    Short description.

    Args:
        foo (int):
            an argument
    """
    mds = MiniDocString(doc)
    arg = mds.get_arg_by_name("foo")
    assert isinstance(arg, MiniDocStringArg)
    assert arg.name == "foo"
    assert arg.pytype.to_string() == "int"


def test_validate_runtime_args_valid():
    doc = """
    Short description.

    Args:
        a (int):
            required
        b (Optional[list[str]]):
            optional list
    """
    mds = MiniDocString(doc)
    mds.validate_runtime_args({"a": 123, "b": ["x", "y"]})


def test_validate_runtime_args_missing():
    doc = """
    Short description.

    Args:
        a (int):
            required
        b (str):
            required
    """
    mds = MiniDocString(doc)
    with pytest.raises(ValueError, match="Missing required arguments"):
        mds.validate_runtime_args({"a": 1})


def test_validate_runtime_args_extra():
    doc = """
    Short description.

    Args:
        a (int):
            required
    """
    mds = MiniDocString(doc)
    with pytest.raises(ValueError, match="Unexpected arguments"):
        mds.validate_runtime_args({"a": 1, "extra": 42})


def test_validate_runtime_args_type_mismatch():
    doc = """
    Short description.

    Args:
        a (int):
            required
    """
    mds = MiniDocString(doc)
    with pytest.raises(ValueError, match="expected .*int.* got str"):
        mds.validate_runtime_args({"a": "not an int"})


def test_to_schema_yaml_structure():
    doc = """
    Short description.

    Does something. Line1
    Does something. Line2

    Args:
        id (int):
            ID value
        name (Optional[str]):
            optional label

    Returns:
        str:
            confirmation
    """
    mds = MiniDocString(doc)
    yaml = mds.to_schema_yaml("function", "my_func")

    assert yaml == \
"""- function: my_func
  description: Short description. Does something. Line1 Does something. Line2
  parameters:
    - name: id
      type: int
      description: ID value
      optional: False
    - name: name
      type: str
      description: optional label
      optional: True
  return:
    type: str
    description: confirmation"""
