from typing import Type, TypeVar, Union, Tuple, cast

T = TypeVar('T')

def _format_type_name(tp: Union[Type, Tuple[Type, ...]]) -> str:
    """
    Helper to format type names nicely for error messages.
    """
    if isinstance(tp, tuple):
        return ", ".join(t.__name__ for t in tp)
    return tp.__name__

def strict_cast(tp: Union[Type[T], Tuple[Type, ...]], value: object) -> T:
    """
    Perform a runtime type check before casting a value.

    Unlike `typing.cast`, this function enforces that the value is actually
    an instance of the specified type(s) at runtime. If the value does not match,
    a `TypeError` is raised immediately.

    Args:
        tp (Type[T] or Tuple[Type, ...]):
            The expected type or tuple of types to cast to.
        value (object):
            The value to check and cast.

    Returns:
        T:
            The value casted to the specified type if it matches.

    Raises:
        TypeError:
            If the value is not an instance of the given type(s).

    Example:
        >>> strict_cast(str, "hello")
        'hello'

        >>> strict_cast((str, bytes), b"hello")
        b'hello'

        >>> strict_cast(int, "not an int")
        TypeError: strict_cast failed: expected int, got str
    """
    if not isinstance(value, tp):
        raise TypeError(
            f"strict_cast failed: expected {_format_type_name(tp)}, got {type(value).__name__}"
        )
    return cast(T, value)
