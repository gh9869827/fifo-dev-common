from __future__ import annotations

from typing import TypeVar, cast, Callable, Awaitable

from fifo_dev_common.logging.logger import get_logger

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventWithCID,
    FifoEventResultWithCID,
)
from fifo_dev_common.event.fifo_event_queue_network_handler import (
    FifoEventQueueNetworkAsyncHandlerCID,
    ExpectedEventClasses,
)
from fifo_dev_common.state.fifo_refreshable_value import FifoRefreshableValue
from fifo_dev_common.event.fifo_event_cid_outcome import (
    FifoEventCIDOutcome,
    TSuccess,
    TFailure,
)

logger = get_logger(__name__)


TValue = TypeVar("TValue")


class FifoEventCIDRefreshManager:
    """
    Helper to wire a CID request/response flow to a FifoRefreshableValue cache.

    On send: marks the cache as REFRESHING. On final success: extracts a value from
    the success event and publishes it via set_success(). On failure (any stage):
    publishes ERROR via set_failure().

    This is background-only; there is no per-request awaitable. Use alongside the
    network client by sending requests normally; registered callbacks perform the
    cache updates.

    Args:
        handler (FifoEventQueueNetworkAsyncHandlerCID):
            CID-capable handler that manages per-CID stage tracking and executes the
            registered success/failure callbacks.
    """

    _handler: FifoEventQueueNetworkAsyncHandlerCID

    def __init__(self, handler: FifoEventQueueNetworkAsyncHandlerCID) -> None:
        """
        Initialize a refresh manager bound to a CID-aware handler.

        Args:
            handler (FifoEventQueueNetworkAsyncHandlerCID):
                CID-capable handler that manages per-CID stage tracking and executes the
                registered success/failure callbacks.
        """
        self._handler = handler

    def register(
        self,
        event_cls: type[FifoEventWithCID],
        expected: ExpectedEventClasses,
        refreshable: FifoRefreshableValue[TValue],
        extract: Callable[[FifoEvent], TValue],
        *,
        success_types: tuple[type[FifoEvent], ...] | None = None,
        on_done: Callable[[FifoEventCIDOutcome[TSuccess, TFailure] | None, FifoEventWithCID],
                          Awaitable[None]] | None = None,
    ) -> None:
        """
        Register a refresh pipeline for an outbound CID-capable request class.

        Args:
            event_cls (type[FifoEventWithCID]):
                The outbound event class to register a template for. When instances of this
                class (or its subclasses) are sent, the template will be applied automatically.

            expected (ExpectedEventClasses):
                Sequence of expected response stages. Each stage can be either a single event
                class or a sequence of event classes (any of which can satisfy that stage).
                Example: [FifoEventAck, [FifoEventSuccess, FifoEventFailure]]

            refreshable (FifoRefreshableValue[TValue]):
                Cache to update (mark_refreshing/set_success/set_failure).

            extract (Callable[[FifoEvent], TValue]):
                Function mapping the final success event to the cached value.

            success_types (tuple[type[FifoEvent], ...] | None):
                Optional tuple of success event classes to guard against unexpected events
                in on_success.

            on_done (Callable[[FifoEventCIDOutcome[TSuccess, TFailure] | None, FifoEventWithCID], 
                              Awaitable[None]] | None):
                Optional per-call cleanup hook invoked by the handler's send helpers in all
                cases (success, failure, timeout, cancellation, error). To have this run,
                callers must send using `handler.send_and_wait(...)` or
                `handler.send_in_background(...)` so the helper can invoke it.
        """

        async def on_send(_req: FifoEventWithCID) -> None:
            refreshable.mark_refreshing()

        async def on_outcome(
                outcome: FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID],
                _src: FifoEventWithCID
        ) -> None:
            if outcome.ok:
                ev = outcome.success()
                try:
                    if (
                            success_types is not None 
                        and not any(isinstance(ev, t) for t in success_types)
                    ):
                        logger.error("refresh manager: unexpected success event %s for %s",
                                     type(ev).__name__, event_cls.__name__)
                        refreshable.set_failure()
                    else:
                        val = extract(ev)
                        refreshable.set_success(val)
                except (TypeError, AttributeError, ValueError) as exc:
                    logger.error("refresh manager extract failed for %s (%s)",
                                 event_cls.__name__, type(exc).__name__)
                    refreshable.set_failure()
            else:
                refreshable.set_failure()

        self._handler.register_cid_template(
            event_cls,
            expected,
            on_send=on_send,
            on_outcome=on_outcome,
            on_done=cast(
                Callable[
                    [
                        FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID] | None,
                        FifoEventWithCID
                    ],
                    Awaitable[None]
                ] | None,
                on_done
            ),
        )


__all__ = [
    # Request/background managers removed; keep only refresh manager
    "FifoEventCIDRefreshManager",
    "FifoEventCIDOutcome",
]
