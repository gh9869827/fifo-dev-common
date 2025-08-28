import pytest
from collections.abc import Callable
from dataclasses import FrozenInstanceError

from fifo_dev_common.state.fifo_refreshable_value import (
    FifoRefreshableValue,
    Snapshot,
    CacheState,
)

# ---- Type aliases ----

Clock = tuple[Callable[[], float], Callable[[float], None]]

# ---- Fixtures ----

@pytest.fixture(name="fake_clock")
def fake_clock_impl() -> Clock:
    """A simple controllable clock."""
    t = {"now": 1000.0}

    def now() -> float:
        return t["now"]

    def advance(dt: float) -> None:
        t["now"] += dt

    return now, advance


@pytest.fixture(name="fifo_cache")
def cache_impl(fake_clock: Clock) -> FifoRefreshableValue[float]:
    now, _ = fake_clock
    return FifoRefreshableValue[float](time_fn=now)

# ---- Tests ----

def test_initial_snapshot_is_stale_and_empty(
    fifo_cache: FifoRefreshableValue[float],
) -> None:
    snap = fifo_cache.snapshot()
    assert isinstance(snap, Snapshot)
    assert snap.state is CacheState.STALE
    assert snap.value is None
    assert snap.ts == 0.0


def test_set_success_publishes_fresh_value_and_timestamp(
    fifo_cache: FifoRefreshableValue[float],
    fake_clock: Clock,
) -> None:
    now, advance = fake_clock
    advance(0.5)
    fifo_cache.set_success(1.23)
    snap = fifo_cache.snapshot()
    assert snap.state is CacheState.FRESH
    assert snap.value == 1.23
    assert snap.ts == now()


def test_mark_refreshing_preserves_last_known_value(
    fifo_cache: FifoRefreshableValue[float],
    fake_clock: Clock,
) -> None:
    now, advance = fake_clock
    fifo_cache.set_success(3.14)
    prev = fifo_cache.snapshot()
    advance(1.0)
    fifo_cache.mark_refreshing()
    snap = fifo_cache.snapshot()
    assert snap.state is CacheState.REFRESHING
    assert snap.value == 3.14                  # preserved
    assert snap.ts == now()
    assert snap is not prev                    # new Snapshot object


def test_set_failure_keeps_last_value_and_marks_error(
    fifo_cache: FifoRefreshableValue[float],
    fake_clock: Clock,
) -> None:
    now, advance = fake_clock
    fifo_cache.set_success(42.0)
    advance(2.0)
    fifo_cache.set_failure()
    snap = fifo_cache.snapshot()
    assert snap.state is CacheState.ERROR
    assert snap.value == 42.0                  # last known value retained
    assert snap.ts == now()


def test_state_transitions_sequence(
    fifo_cache: FifoRefreshableValue[float],
) -> None:
    # STALE -> REFRESHING -> FRESH -> REFRESHING -> ERROR
    assert fifo_cache.snapshot().state is CacheState.STALE
    fifo_cache.mark_refreshing()
    assert fifo_cache.snapshot().state is CacheState.REFRESHING
    fifo_cache.set_success(7.0)
    assert fifo_cache.snapshot().state is CacheState.FRESH
    fifo_cache.mark_refreshing()
    assert fifo_cache.snapshot().state is CacheState.REFRESHING
    fifo_cache.set_failure()
    assert fifo_cache.snapshot().state is CacheState.ERROR


def test_snapshot_is_immutable(
    fifo_cache: FifoRefreshableValue[float],
) -> None:
    fifo_cache.set_success(9.99)
    snap = fifo_cache.snapshot()
    with pytest.raises(FrozenInstanceError):
        # type: ignore[attr-defined]  # force a mutation attempt
        snap.value = 0.0  # type: ignore[misc]  # force a mutation attempt


def test_snapshot_object_changes_on_publish(
    fifo_cache: FifoRefreshableValue[float],
) -> None:
    a = fifo_cache.snapshot()
    fifo_cache.set_success(1.0)
    b = fifo_cache.snapshot()
    fifo_cache.mark_refreshing()
    c = fifo_cache.snapshot()
    fifo_cache.set_failure()
    d = fifo_cache.snapshot()

    # Each publish should replace the Snapshot object
    assert a is not b
    assert b is not c
    assert c is not d
