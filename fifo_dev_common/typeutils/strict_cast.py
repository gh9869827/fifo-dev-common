from typing import Type, TypeVar, Tuple, cast

T = TypeVar('T')

def _format_type_name(tp: Type[T] | Tuple[Type[T], ...]) -> str:
    """
    Helper to format type names nicely for error messages.
    """
    if isinstance(tp, tuple):
        return ", ".join(t.__name__ for t in tp)
    return tp.__name__

def strict_cast(tp: Type[T] | Tuple[Type[T], ...], value: object) -> T:
    """
    Perform a shallow runtime type check before casting a value.

    Unlike `typing.cast`, this function enforces that the value is actually
    an instance of the specified type(s) at runtime. If the value does not match,
    a `TypeError` is raised immediately.

    This check is **shallow**, verifying only the outermost type.
    Generic types like `list[int]` are supported for static type checkers (e.g., MyPy, Pylance),
    but only the outer type (`list`) is enforced at runtime. Element types are not checked.
    For example, `strict_cast(list, [1, "2"])` will pass, as the contents are not inspected.

    Args:
        tp (Type[T] or Tuple[Type[T], ...]):
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

        >>> strict_cast(list, [1, 2, 3])
        [1, 2, 3]

        >>> strict_cast(int, "not an int")
        TypeError: strict_cast failed: expected int, got str
    """
    # avoid casting within a narrowed scope; this prevents Pylance from flagging the cast
    # as unnecessary
    value_ = value

    # validating type
    if not isinstance(value, tp):
        raise TypeError(
            f"strict_cast failed: expected {_format_type_name(tp)}, got {type(value).__name__}"
        )

    # value/value_ is guaranteed to be type T after isinstance; cast for type checker
    return cast(T, value_)
