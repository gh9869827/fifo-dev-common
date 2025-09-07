from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventResultWithCID


TSuccess = TypeVar("TSuccess", bound=FifoEvent, covariant=True)
TFailure = TypeVar("TFailure", bound=FifoEventResultWithCID, covariant=True)


@dataclass(frozen=True)
class FifoEventCIDOutcome(Generic[TSuccess, TFailure]):
    """
    Result of a CID request chain.

    - ok: True on final success, False on any failure stage
    - event: The final event instance (success or failure)

    Use `success()` and `failure()` to retrieve the event as the
    expected generic type for each outcome.
    """
    ok: bool
    event: FifoEvent

    def success(self) -> TSuccess:
        """
        Return the final event as `TSuccess` (asserts `ok`).
        """
        assert self.ok, "Called success() on a failure outcome"
        return cast(TSuccess, self.event)

    def failure(self) -> TFailure:
        """
        Return the final event as `TFailure` (asserts `not ok`).
        """
        assert not self.ok, "Called failure() on a success outcome"
        return cast(TFailure, self.event)


__all__ = [
    "FifoEventCIDOutcome",
    "TSuccess",
    "TFailure",
]
