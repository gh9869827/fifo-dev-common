import pytest
from fifo_dev_common.introspection.mini_docstring import MiniDocStringClass, MiniDocStringAttribute


def test_empty_docstring_initializes_defaults():
    mdc = MiniDocStringClass(None)
    assert mdc.description_short == ""
    assert mdc.description_detailed == ""
    assert mdc.attributes == []
    with pytest.raises(AttributeError):
        mdc.get_attribute_by_name("foo")


def test_parses_description_and_attributes():
    doc = """
    Short summary.

    Detailed description line1.
    Detailed description line2.

    Attributes:
        foo (int):
            first attribute
        bar (str): second attribute
    """
    mdc = MiniDocStringClass(doc)
    assert mdc.description_short == "Short summary."
    assert mdc.description_detailed == "Detailed description line1.\nDetailed description line2."
    assert len(mdc.attributes) == 2
    foo = mdc.get_attribute_by_name("foo")
    assert foo == MiniDocStringAttribute("foo", "int", "first attribute")
    bar = mdc.get_attribute_by_name("bar")
    assert bar == MiniDocStringAttribute("bar", "str", "second attribute")


def test_missing_description_raises_error():
    doc = """
    Attributes:
        foo (int): something
    """
    with pytest.raises(ValueError, match="description.*before.*structured"):
        MiniDocStringClass(doc)


def test_multiline_attribute_description():
    doc = """
    Summary.

    Attributes:
        foo (int):
            first line
            second line
    """
    mdc = MiniDocStringClass(doc)
    foo = mdc.get_attribute_by_name("foo")
    assert foo.desc == "first line\nsecond line"


def test_docstring_without_attributes_section():
    doc = """
    Only description here.

    More details on the class.
    """
    mdc = MiniDocStringClass(doc)
    assert mdc.description_short == "Only description here."
    assert mdc.description_detailed == "More details on the class."
    assert mdc.attributes == []

