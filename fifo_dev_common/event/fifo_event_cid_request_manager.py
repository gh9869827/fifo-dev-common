from __future__ import annotations

import asyncio
from uuid import UUID
from typing import Generic, TypeVar, cast, Callable, Awaitable
from dataclasses import dataclass

from fifo_dev_common.logging.logger import get_logger

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventWithCID,
    FifoEventResultWithCID,
)
from fifo_dev_common.event.fifo_event_protocols import SupportsFifoEventPut
from fifo_dev_common.event.fifo_event_queue_network_handler import (
    FifoEventQueueNetworkAsyncHandlerCID,
    ExpectedEventClasses,
)
from fifo_dev_common.state.fifo_refreshable_value import FifoRefreshableValue

logger = get_logger(__name__)


TSuccess = TypeVar("TSuccess", bound=FifoEvent)
TFailure = TypeVar("TFailure", bound=FifoEventResultWithCID)
TValue = TypeVar("TValue")


@dataclass(frozen=True)
class CIDOutcome(Generic[TSuccess, TFailure]):
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


class FifoEventCIDRequestManager:
    """
    Helper to coordinate request/response flows for CID-capable events.

    This class centralizes the boilerplate of tracking a per-request Future and wiring
    success/failure callbacks. It works with `FifoEventQueueNetworkAsyncHandlerCID` templates
    and provides a simple `send_and_wait()` helper used by higher-level APIs.

    Typical usage:
        manager = FifoEventCIDRequestManager(loop, handler)

        # Register expected response chains once per outbound request class
        manager.register(MyRequest, [MyAck, [MyDoneSuccess, MyDoneFailure]])

        # In your API method(s):
        req = MyRequest(...)
        result = await manager.send_and_wait(client, req, timeout=5.0)

    Notes:
        - The underlying handler (`FifoEventQueueNetworkAsyncHandlerCID`) keeps at most one
          template per outbound request class.
        - If the same request class must be used both as awaited and in background, use this
          manager: one registration supports `send_and_wait(...)` and `send_in_background(...)`.
        - If requests are only sent in background, prefer `FifoEventCIDBackgroundManager` for a
          leaner path (no Futures). If the goal is to update a cache, use
          `FifoEventCIDRefreshManager` to wire results into a `FifoRefreshableValue`.

    Args:
        loop (asyncio.AbstractEventLoop):
            Application event loop used to create and manage per-request Futures and schedule
            background tasks.

        handler (FifoEventQueueNetworkAsyncHandlerCID):
            CID-capable handler that manages per-CID stage tracking and invokes the registered
            success/failure callbacks; this manager wires those callbacks to resolve per-request
            Futures.
    """

    _loop: asyncio.AbstractEventLoop
    _handler: FifoEventQueueNetworkAsyncHandlerCID
    _futures: dict[UUID, asyncio.Future[CIDOutcome[FifoEvent, FifoEventResultWithCID]]]

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 handler: FifoEventQueueNetworkAsyncHandlerCID) -> None:
        """
        Initialize a CID request manager bound to an event handler and loop.

        Args:
            loop (asyncio.AbstractEventLoop):
                Application event loop used to create and manage per-request Futures and schedule
                background tasks.

            handler (FifoEventQueueNetworkAsyncHandlerCID):
                CID-capable handler that manages per-CID stage tracking and invokes the
                registered success/failure callbacks; this manager wires those callbacks to
                resolve per-request Futures.
        """
        self._loop = loop
        self._handler = handler
        self._futures = {}

    def register(self,
                 event_cls: type[FifoEventWithCID],
                 expected: ExpectedEventClasses) -> None:
        """
        Register the expected response chain for an outbound CID-capable request class.

        This wires `on_success` and `on_failure` callbacks into the underlying handler to
        resolve and clean up a per-CID Future representing the final outcome of the request.
        An `on_sent` callback is not required for this flow.

        Args:
            event_cls (type[FifoEventWithCID]):
                The outbound request event class to track (e.g., `MyRequest`). Instances of this
                class must carry a `correlation_id` and will be auto-matched on send.

            expected (ExpectedEventClasses):
                The expected response stages for this request class, expressed as a sequence of
                event types or nested choices per stage. Example:
                `[MyAck, [MyDoneSuccess, MyDoneFailure]]`.

        Returns:
            None
        """

        async def on_success(ev: FifoEventWithCID | FifoEventResultWithCID,
                             src: FifoEventWithCID) -> None:
            fut = self._futures.get(src.correlation_id)
            if fut is not None and not fut.done():
                fut.set_result(CIDOutcome(True, ev))
            # Cleanup after setting result (idempotent with send_and_wait finally)
            self._futures.pop(src.correlation_id, None)

        async def on_failure(_ev: FifoEventResultWithCID,
                             src: FifoEventWithCID) -> None:
            fut = self._futures.get(src.correlation_id)
            if fut is not None and not fut.done():
                # Return the failing event with ok=False to preserve details
                fut.set_result(CIDOutcome(False, _ev))
            # Cleanup after setting result (idempotent with send_and_wait finally)
            self._futures.pop(src.correlation_id, None)

        self._handler.register_cid_template(
            event_cls,
            expected,
            on_success=on_success,
            on_failure=on_failure,
        )

    async def send_and_wait(self,
                            transport: SupportsFifoEventPut,
                            req: FifoEventWithCID,
                            *,
                            timeout: float | None = None) -> CIDOutcome[FifoEvent,
                                                                        FifoEventResultWithCID]:
        """
        Send a CID-capable request and await the final result event.

        - Creates a per-request Future keyed by the request's correlation ID
        - Sends the request over `transport.put(req)`
        - Resolves when the handler's registered callbacks deliver a final success/failure

        Args:
            transport (SupportsFifoEventPut):
                Network client/server (or adapter) exposing an async `put(FifoEvent)` method.

            req (FifoEventWithCID):
                Outbound request event. Must carry a correlation_id (auto-assigned if None by
                the event constructor).

            timeout (float | None, optional):
                Optional timeout in seconds for awaiting the final result. If None, waits
                indefinitely.

        Returns:
            CIDOutcome[FifoEvent, FifoEventResultWithCID]:
                Wraps success flag and the final event (success or failure).
        """
        fut: asyncio.Future[
            CIDOutcome[FifoEvent, FifoEventResultWithCID]
        ] = self._loop.create_future()
        self._futures[req.correlation_id] = fut

        try:
            await transport.put(req)
        except Exception as e:  # pragma: no cover - transport failure path
            # Ensure we clean up and propagate the error to the awaiting caller
            self._futures.pop(req.correlation_id, None)
            if not fut.done():
                fut.set_exception(e)
            raise

        try:
            return await (asyncio.wait_for(fut, timeout) if timeout is not None else fut)
        finally:
            # Safety: callbacks pop on completion; this is idempotent
            self._futures.pop(req.correlation_id, None)

    def send_in_background(self,
                           transport: SupportsFifoEventPut,
                           req: FifoEventWithCID,
                           on_outcome: Callable[[CIDOutcome[FifoEvent, FifoEventResultWithCID],
                                                FifoEventWithCID],
                                                Awaitable[None]],
                           *,
                           timeout: float | None = None) -> asyncio.Task[None]:
        """
        Launch a background task to send a CID-capable request and invoke a callback
        with the final `CIDOutcome` (success or failure).

        Args:
            transport (SupportsFifoEventPut):
                Network client/server (or adapter) exposing an async `put(FifoEvent)` method.

            req (FifoEventWithCID):
                Outbound request event. Must carry a correlation_id (auto-assigned if None by
                the event constructor).

            on_outcome (Callable[[CIDOutcome, FifoEventWithCID], Awaitable[None]]):
                Async callback invoked with the final outcome and the original request once the
                request completes (either success or failure). Runs in a background task and must
                not block the event loop for long periods.

            timeout (float | None, optional):
                Optional timeout in seconds for awaiting the final result. If None, waits
                indefinitely.

        Returns:
            asyncio.Task[None]:
                The scheduled background task. You may await, cancel, or attach callbacks.
        """
        async def _runner() -> None:
            outcome = await self.send_and_wait(transport, req, timeout=timeout)
            await on_outcome(outcome, req)

        task = self._loop.create_task(_runner())

        def _done(t: asyncio.Task[None]) -> None:
            # Always-safe logging: avoid traceback and sensitive messages
            if t.cancelled():
                logger.warning("CID background task for %s cancelled", type(req).__name__)
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "CID background task failed for %s (%s)",
                    type(req).__name__,
                    type(exc).__name__,
                )

        task.add_done_callback(_done)
        return task


class FifoEventCIDBackgroundManager:
    """
    Background-only manager for CID-capable request/response flows without Futures.

    This manager registers class-level templates with the handler and invokes a provided
    `on_outcome(outcome, req)` callback for each request instance (final success or failure).
    There is no per-request awaiting or Future mapping.

    Use this when:
        - You only need background handling for a given request class, and
        - You prefer to register the outcome handler once for the class (at registration time),
          rather than passing a handler on each send; and you do not need to await a specific
          request. If you need to await a request or supply a different outcome handler per send,
          use `FifoEventCIDRequestManager` instead (`send_and_wait(...)` /
          `send_in_background(...)`).

    If you need to mix foreground (awaited) and background flows for the same request class,
    prefer `FifoEventCIDRequestManager` to avoid conflicting template registrations.

    Args:
        loop (asyncio.AbstractEventLoop):
            Application event loop on which background callbacks are scheduled.

        handler (FifoEventQueueNetworkAsyncHandlerCID):
            CID-capable handler that manages per-CID stage tracking and executes the
            registered success/failure callbacks. This manager only adapts its callbacks.
    """

    _loop: asyncio.AbstractEventLoop
    _handler: FifoEventQueueNetworkAsyncHandlerCID

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 handler: FifoEventQueueNetworkAsyncHandlerCID) -> None:
        """
        Initialize a background-only CID manager.

        Args:
            loop (asyncio.AbstractEventLoop):
                Application event loop on which background callbacks are scheduled.

            handler (FifoEventQueueNetworkAsyncHandlerCID):
                CID-capable handler that manages per-CID stage tracking and executes the
                registered success/failure callbacks.
        """
        self._loop = loop
        self._handler = handler

    def register(self,
                 event_cls: type[FifoEventWithCID],
                 expected: ExpectedEventClasses,
                 on_outcome: Callable[[CIDOutcome[TSuccess, TFailure],
                                      FifoEventWithCID], Awaitable[None]]) -> None:
        """
        Register a background outcome callback for an outbound CID-capable request class.

        The handler tracks stages per CID as usual. When the final stage succeeds or an
        intermediate failure occurs, `on_outcome` is invoked with a `CIDOutcome` and the
        original request.
        """

        async def _call_outcome(outcome: CIDOutcome[TSuccess, TFailure],
                                req: FifoEventWithCID) -> None:
            # Schedule user callback on the application loop; avoid blocking handler dispatcher
            async def _runner() -> None:
                await on_outcome(outcome, req)
            task = self._loop.create_task(_runner())

            def _done(t: asyncio.Task[None]) -> None:
                if t.cancelled():
                    logger.warning(
                        "CID background on_outcome cancelled for %s",
                        type(req).__name__
                    )
                    return
                exc = t.exception()
                if exc is not None:
                    logger.error(
                        "CID background on_outcome failed for %s (%s)",
                        type(req).__name__,
                        type(exc).__name__,
                    )

            task.add_done_callback(_done)

        async def on_success(ev: FifoEventWithCID | FifoEventResultWithCID,
                             src: FifoEventWithCID) -> None:
            await _call_outcome(CIDOutcome(True, ev), src)

        async def on_failure(ev: FifoEventResultWithCID,
                             src: FifoEventWithCID) -> None:
            await _call_outcome(CIDOutcome(False, ev), src)

        self._handler.register_cid_template(
            event_cls,
            expected,
            on_success=on_success,
            on_failure=on_failure,
        )

    async def send(self,
                   transport: SupportsFifoEventPut,
                   req: FifoEventWithCID) -> None:
        """
        Send a CID-capable request without awaiting the outcome. The registered
        `on_outcome` will be invoked when the flow completes.
        """
        await transport.put(req)


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
        """

        async def on_success(ev: FifoEventWithCID | FifoEventResultWithCID,
                             _src: FifoEventWithCID) -> None:
            try:
                if success_types is not None and not any(isinstance(ev, t) for t in success_types):
                    logger.error("refresh manager: unexpected success event %s for %s",
                                 type(ev).__name__, event_cls.__name__)
                    refreshable.set_failure()
                    return
                val = extract(ev)
                refreshable.set_success(val)
            except (
                TypeError, AttributeError, ValueError
            ) as exc:  # pragma: no cover - defensive path
                logger.error("refresh manager extract failed for %s (%s)",
                             event_cls.__name__, type(exc).__name__)
                refreshable.set_failure()

        async def on_failure(_ev: FifoEventResultWithCID, _src: FifoEventWithCID) -> None:
            refreshable.set_failure()

        async def on_send(_req: FifoEventWithCID) -> None:
            refreshable.mark_refreshing()

        self._handler.register_cid_template(
            event_cls,
            expected,
            on_success=on_success,
            on_failure=on_failure,
            on_send=on_send,
        )


__all__ = [
    "FifoEventCIDRequestManager",
    "FifoEventCIDBackgroundManager",
    "FifoEventCIDRefreshManager",
    "CIDOutcome",
]
