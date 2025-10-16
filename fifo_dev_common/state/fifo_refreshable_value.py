from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar, Callable

T = TypeVar("T")


class CacheState(Enum):
    """
    Lifecycle of a refreshable value.
    """
    STALE = auto()
    REFRESHING = auto()
    FRESH = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Snapshot(Generic[T]):
    """
    Immutable, atomic view of the cache.

    Attributes:
        state (CacheState):
            Freshness / lifecycle state.

        value (T | None):
            Latest value (if any).

        ts (float):
            Publish timestamp (seconds since epoch).
    """
    state: CacheState
    value: T | None
    ts: float

    def get_if_fresh(self, ttl: float | None=None) -> T | None:
        """
        Return the cached value if state is valid and within time-to-live.

        Returns the value when the state is FRESH or REFRESHING (with data),
        and the elapsed time since timestamp is within ttl. If ttl is None,
        skips the timestamp check entirely.

        Args:
            ttl (float | None):
                Time-to-live in seconds, or None to ignore timestamp.

        Returns:
            T | None:
                The cached value if constraints met, otherwise None.
        """
        if self.state in [CacheState.FRESH, CacheState.REFRESHING] and self.value is not None:
            if ttl is None or time.monotonic() - self.ts < ttl:
                return self.value
        return None


class FifoRefreshableValue(Generic[T]):
    """
    Lock-free, one-writer/many-readers cache for asynchronously refreshed values with
    explicit state transitions (STALE → REFRESHING → FRESH/ERROR) and immutable snapshots.

    Concurrency (CPython):
        - Safe with exactly one writer thread/task and any number of readers.
          Publishing is a single assignment of a new `Snapshot`, which is atomic
          under the GIL; readers observe either the old or new snapshot.
        - For multiple writers across threads, marshal updates to a single owner.

    Typical flow:
        - Writers (e.g., event interceptors handling ACK/DONE) call mark_refreshing(),
          then publish results with set_success(value) or set_failure().
        - Readers (e.g., control loops) call snapshot() to check the latest state/value
          without blocking the event queue.

    Attributes:
        _now (Callable[[], float]):
            Function returning current time in seconds. Defaults to time.time.
        
        _snap (Snapshot[T]):
            Current immutable snapshot containing the cached value, state, and timestamp.
    """

    _now: Callable[[], float]
    _snap: Snapshot[T]

    def __init__(self, time_fn: Callable[[], float] = time.time) -> None:
        """
        Initialize a refreshable value cache.

        Args:
            time_fn (Callable[[], float], optional):
                Function to get current time in seconds since epoch. Defaults to time.time.
        """
        self._now = time_fn
        self._snap = Snapshot(CacheState.STALE, None, 0.0)

    # -------- Reads --------
    def snapshot(self) -> Snapshot[T]:
        """
        Return the current immutable snapshot (no copying).
        
        Returns:
            Snapshot[T]:
                Current immutable snapshot containing state, value, and timestamp (no copying).
        """
        return self._snap

    # -------- Writes (single-assignment publish) --------
    def mark_refreshing(self) -> None:
        """
        Publish REFRESHING while preserving the last known value.

        Sets the state to REFRESHING with the current timestamp and keeps the
        previous value available to readers. Callers may treat that value as
        a best-effort placeholder until a new FRESH snapshot is published.
        """
        s = self._snap
        self._snap = Snapshot(CacheState.REFRESHING, s.value, self._now())

    def set_success(self, val: T) -> None:
        """
        Publish a fresh value and update state to FRESH.
        
        Args:
            val (T):
                The new value to cache with current timestamp.
        """
        self._snap = Snapshot(CacheState.FRESH, val, self._now())

    def set_failure(self) -> None:
        """
        Publish ERROR, retaining last value for readers that can tolerate stale data.
        """
        s = self._snap
        self._snap = Snapshot(CacheState.ERROR, s.value, self._now())
