from collections.abc import Sequence
from typing import TypeVar, Generic, Iterator, cast, overload

T = TypeVar("T")

class ReadOnlyList(Sequence[T], Generic[T]):
    """
    A read-only wrapper around a list. Prevents mutation while allowing full read access.

    This container is intended for internal use when exposing list-like data that should not be
    accidentally modified. Nested contents are not recursively frozen, which means any contained
    lists or dicts must themselves be wrapped or treated immutably by convention.

    Attributes:
        _data (list[T]):
            The underlying list that this ReadOnlyList wraps.
    """

    _data: list[T]

    def __init__(self, data: list[T]):
        self._data = data

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def count(self, value: T) -> int:
        """
        Returns the number of times the specified value appears in the list.

        Args:
            value (T):
                The value to count.

        Returns:
            int:
                Number of occurrences of the value.
        """
        return self._data.count(value)

    def index(self, value: T, start: int = 0, stop: int | None = None) -> int:
        """
        Returns the index of the first occurrence of the value.

        Args:
            value (T):
                The value to locate.
            start (int):
                Optional start index.
            stop (int | None):
                Optional stop index.

        Returns:
            int:
                Index of the value.

        Raises:
            ValueError: If the value is not present.
        """
        if stop is None:
            return self._data.index(value, start)
        return self._data.index(value, start, stop)

    def __repr__(self) -> str:
        return f"ReadOnlyList({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReadOnlyList):
            # Cast is for the type checker only; T is not enforced at runtime, but attribute access
            # is safe for comparison
            return self._data == cast(ReadOnlyList[T], other)._data
        if isinstance(other, Sequence):
            return self._data == other
        return False

    def __contains__(self, item: object) -> bool:
        return item in self._data
