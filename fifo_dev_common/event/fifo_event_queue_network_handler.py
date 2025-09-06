from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Awaitable, Callable, Sequence, TypeAlias, cast
from uuid import UUID

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventResultWithCID,
    FifoEventShutdown,
    FifoEventWithCID,
    EErrorCode,
)
from fifo_dev_common.logging.logger import get_logger
from fifo_dev_common.event.fifo_event_cid_outcome import FifoEventCIDOutcome
from fifo_dev_common.event.fifo_event_protocols import SupportsFifoEventPut

logger = get_logger(__name__)

# User-facing alias: a sequence where each element represents one stage.
# An element can be either a single event class (single expected type for that stage)
# or a sequence of event classes (any of those types may satisfy the stage).
# Example accepted forms:
#   [A, B, C] => [[A], [B], [C]]
#   [A, [B, C], D] => [[A], [B, C], [D]]
# This keeps register() expressive: each position is a progression stage.
StageElement: TypeAlias = (
               type[FifoEventResultWithCID] | type[FifoEventWithCID]
    | Sequence[type[FifoEventResultWithCID] | type[FifoEventWithCID]]
)
ExpectedEventClasses: TypeAlias = Sequence[StageElement]

# Internal canonical form: always list of stages, each stage a list of types
NormalizedExpected: TypeAlias = list[list[type[FifoEventResultWithCID] | type[FifoEventWithCID]]]

# Callback aliases for readability
OnSendCallback: TypeAlias = Callable[
    [FifoEventWithCID],
    Awaitable[None]
]
OnSentCallback: TypeAlias = Callable[
    [FifoEventWithCID],
    Awaitable[None]
]
OnOutcomeCallback: TypeAlias = Callable[
    [FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID], FifoEventWithCID],
    Awaitable[None]
]
OnDoneCallback: TypeAlias = Callable[
    [FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID] | None, FifoEventWithCID],
    Awaitable[None]
]

class FifoEventQueueNetworkAsyncHandlerBase(ABC):
    """
    Base class for asynchronous network event handlers.

    Handlers can intercept and process events in three phases:

    - Incoming events (from the network, before enqueuing):
      Return the event (possibly modified) to enqueue, or None to suppress.

    - Outgoing events (before sending to the network):
      Return the event (possibly modified) to send, or None to suppress sending.

    - Sent events (after the event has been successfully written and flushed):
      Invoked with the event that was sent. This hook is for notification only
      and must not return a value.

    Subclasses must implement `process_incoming_event()`,
    `process_outgoing_event()`, and `process_sent_event()`.

    Note:
        `FifoEventShutdown` is always propagated. Handlers can observe it via
        `process_incoming_event()` and `process_outgoing_event()`, but their return
        values are ignored. The original shutdown event is always enqueued or sent
        exactly once. `process_sent_event()` is still invoked with the shutdown event
        after it has been sent.
    """

    @abstractmethod
    async def process_incoming_event(self, event: FifoEvent) -> FifoEvent | None:
        """
        Process an incoming event before it is enqueued.

        Called when an event is received from the network and before it is added to the 
        outgoing queue.

        Args:
            event (FifoEvent):
                Incoming event from the network.

        Returns:
            FifoEvent | None:
                The event to enqueue (possibly modified), or None to suppress enqueuing.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def process_outgoing_event(self, event: FifoEvent) -> FifoEvent | None:
        """
        Process an outgoing event before it is sent over the network.

        Called when an event is about to be serialized and written to the network.

        Args:
            event (FifoEvent):
                Outgoing event to be sent over the network.

        Returns:
            FifoEvent | None:
                The event to send (possibly modified), or None to suppress sending.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def process_sent_event(self, event: FifoEvent) -> None:
        """
        Called after an outgoing event has been successfully sent over the network.

        Implementations can perform post-send bookkeeping (logging, metrics,
        triggering side-effects, etc.). This hook is for notification only and
        must not return a value.

        Args:
            event (FifoEvent):
                The event that was successfully sent.
        """
        raise NotImplementedError  # pragma: no cover


class FifoEventQueueNetworkAsyncHandlerCID(FifoEventQueueNetworkAsyncHandlerBase):
    """
    Correlation ID-based event handler for managing request/response workflows.

    This handler matches events using correlation IDs and executes registered callbacks
    when specific event sequences are received. It supports **template-based registration**:

    - You register *event classes* exactly once along with the expected response stages and
      success/failure callbacks (see `register_cid_template`).
    - Each time a matching *event instance* (with a correlation ID) is **sent**, the handler
      automatically creates a per-CID registration based on the template.

    This avoids per-request manual registration: define the contract once; instances are
    auto-registered on send.

    Usage:
        1. Define a template for an outbound event class via `register_cid_template()`
        2. Send an instance of that class (must subclass `FifoEventWithCID`)
        3. The handler auto-registers the instance's CID
        4. Incoming events matching the stages trigger the appropriate callbacks

    Example:
        ```python
        handler = FifoEventQueueNetworkAsyncHandlerCID()

        # 1) Define the contract once at startup
        async def on_ok(ev: FifoEventWithCID | FifoEventResultWithCID) -> None: ...
        async def on_err(ev: FifoEventResultWithCID) -> None: ...

        handler.register_cid_template(
            FifoEventLoadMap,
            [FifoEventLoadMapAck, [FifoEventLoadMapDoneSuccess, FifoEventLoadMapDoneFailure]],
            on_success=on_ok,
            on_failure=on_err,
        )

        # 2) Later, send an instance -> auto-registered using its correlation_id
        req = FifoEventLoadMap(correlation_id=uuid4(), map_name="level1")
        await client.send(req)
        ```

    Thread Safety:
        This handler is thread-safe and designed for concurrent use in asyncio environments.
        Callbacks are executed sequentially on a dedicated background task to prevent
        blocking the network reader.
    """

    class QueueItemKind(Enum):
        """
        Kinds of items queued for the background callback dispatcher.

        - SEND: schedule the `on_send` callback for an outbound request event
        - SENT: schedule the `on_sent` callback after an outbound request was sent
        - OUTCOME: schedule a registered `on_outcome` callback (final result)
        - SHUTDOWN: signal the dispatcher loop to terminate
        - LISTENER: schedule an incoming-only listener callback
        """
        SEND = auto()
        SENT = auto()
        OUTCOME = auto()
        SHUTDOWN = auto()
        LISTENER = auto()

    # Payloads are uniform tuples; first element is the callback, followed by args.
    QueuePayloadSend: TypeAlias = tuple[OnSendCallback, FifoEventWithCID]
    QueuePayloadSent: TypeAlias = tuple[OnSentCallback, FifoEventWithCID]
    QueuePayloadOutcome: TypeAlias = tuple[OnOutcomeCallback,
                                           FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID],
                                           FifoEventWithCID]
    QueuePayloadShutdown: TypeAlias = tuple[()]
    QueuePayloadListener: TypeAlias = tuple[
        Callable[[FifoEvent], Awaitable[None]],
        FifoEvent,
    ]

    QueueItem: TypeAlias = tuple[
        QueueItemKind,
        QueuePayloadSend |
        QueuePayloadSent |
        QueuePayloadOutcome |
        QueuePayloadShutdown |
        QueuePayloadListener,
    ]

    TemplateContent: TypeAlias = tuple[
        NormalizedExpected,
        OnSendCallback | None,
        OnSentCallback | None,
        OnOutcomeCallback | None,
        OnDoneCallback | None,
    ]

    _queue: asyncio.Queue[QueueItem]

    _task: asyncio.Task[None]

    _registrations: dict[
        UUID,
        tuple[
            NormalizedExpected,
            int,
            FifoEventWithCID,
            asyncio.Future[FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID]],
        ],
    ]

    _templates: dict[type[FifoEventWithCID], TemplateContent]

    # Incoming-only listeners (non-CID events and shutdown)
    _listeners: dict[
        type[FifoEvent],
        list[tuple[Callable[[FifoEvent], Awaitable[None]], bool]],  # (callback, consume)
    ]

    def __init__(self) -> None:
        """
        Initialize the correlation ID handler and start the background task that
        dispatches callbacks.
        """
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

        # Per-CID active registrations created at send-time from templates.
        #   cid -> (normalized_expected, stage_index, request_event, future)
        self._registrations = {}

        # Class-level templates registered once.
        #   event_cls -> (normalized_expected, on_send, on_sent, on_outcome, on_done)
        self._templates = {}

        # Incoming-only listeners registered by event type (supports MRO lookup)
        self._listeners = {}

    async def _run(self) -> None:
        """
        Process queued callbacks until a shutdown event is received.
        """
        while True:
            kind, payload = await self._queue.get()
            if kind is self.QueueItemKind.SHUTDOWN:
                break
            try:
                if kind is self.QueueItemKind.SEND:
                    cb, ev = cast(
                        FifoEventQueueNetworkAsyncHandlerCID.QueuePayloadSend, payload
                    )
                    await cb(ev)
                elif kind is self.QueueItemKind.SENT:
                    cb, ev = cast(
                        FifoEventQueueNetworkAsyncHandlerCID.QueuePayloadSent, payload
                    )
                    await cb(ev)
                elif kind is self.QueueItemKind.OUTCOME:
                    cb_outcome, outcome, req = cast(
                        FifoEventQueueNetworkAsyncHandlerCID.QueuePayloadOutcome, payload
                    )
                    await cb_outcome(outcome, req)
                elif kind is self.QueueItemKind.LISTENER:
                    cb_listener, ev_any = cast(
                        FifoEventQueueNetworkAsyncHandlerCID.QueuePayloadListener, payload
                    )
                    await cb_listener(ev_any)
                else:  # pragma: no cover - defensive branch
                    logger.error("unknown queue item kind: %s", kind)
            except (TypeError, AttributeError, ValueError, RuntimeError, asyncio.CancelledError):
                logger.error("handler callback failed")
            except Exception:  # pragma: no cover # pylint: disable=broad-exception-caught
                # Fallback for any other unexpected exceptions
                logger.error("handler callback failed with unexpected exception")

    async def join(self) -> None:
        """
        Wait for the background task to finish.
        """
        await self._task

    async def _notify_listeners(self, event: FifoEvent) -> bool:
        """
        Enqueue all matching incoming-only listeners and return whether any consumes.

        Returns:
            bool:
                True if at least one matching listener is marked consume=True.
        """
        consumed = False
        for cls in type(event).mro():
            listeners = self._listeners.get(cls)
            if not listeners:
                continue
            for cb, consume in listeners:
                if consume:
                    consumed = True
                await self._queue.put((self.QueueItemKind.LISTENER, (cb, event)))
        return consumed

    def _normalize_expected(self, expected_cls: ExpectedEventClasses) -> NormalizedExpected:
        seq: NormalizedExpected = []
        for stage in expected_cls:
            if isinstance(stage, type):
                seq.append([stage])
            else:
                seq.append([cls for cls in stage])
        return seq

    def register_incoming_listener(
        self,
        event_cls: type[FifoEvent],
        on_event: Callable[[FifoEvent], Awaitable[None]],
        *,
        consume: bool = False,
    ) -> None:
        """
        Register an incoming-only listener for NON-CID events (and shutdown).

        The callback is invoked for incoming events that are not CID-capable
        (i.e., not instances of FifoEventWithCID or FifoEventResultWithCID), as
        well as for FifoEventShutdown. Listeners are observers by default and do
        not affect CID stage advancement. For non-CID events, if any matching
        listener is registered with `consume=True`, the event will not be
        propagated to the output queue, but all listeners will still be invoked.
        Note: FifoEventShutdown is always propagated to the output queue; the
        `consume` flag has no effect on shutdown delivery.

        Args:
            event_cls (type[FifoEvent]):
                Event class to listen for (supports base-class registration; MRO is used).

            on_event (Callable[[FifoEvent], Awaitable[None]]):
                Async callback invoked with the incoming event instance.

            consume (bool, optional):
                If True, this listener marks matching non-CID events as consumed,
                preventing their propagation to the output queue. Has no effect
                on FifoEventShutdown (which is always propagated). Defaults to False.
        
        Raises:
            ValueError:
                - If `event_cls` is a CID-capable class (listeners are for non-CID and shutdown).
                - If the same `(event_cls, on_event, consume)` listener is already registered.
        """
        # Guard: listeners are for non-CID classes and shutdown; disallow CID event classes
        if issubclass(event_cls, (FifoEventWithCID, FifoEventResultWithCID)):
            raise ValueError(
                f"Incoming listeners are for non-CID events; got CID class {event_cls.__name__}"
            )

        listeners = self._listeners.setdefault(event_cls, [])
        # Guard: prevent exact duplicate registration of the same callback/consume pair
        if any(cb is on_event and c == consume for cb, c in listeners):
            raise ValueError(
                f"Listener already registered for {event_cls.__name__} with consume={consume}"
            )
        listeners.append((on_event, consume))

    async def process_incoming_event(self, event: FifoEvent) -> FifoEvent | None:
        """
        Process an incoming event and apply CID stage rules and/or incoming listeners.

        Behavior by event kind:
            - FifoEventShutdown:
                Notify incoming listeners first, then schedule internal shutdown of the
                handler's dispatcher. The original shutdown event is returned; callers typically
                ignore this return value and enqueue the original shutdown event.

            - CID-capable events (FifoEventWithCID, FifoEventResultWithCID):
                Apply the registered CID template pipeline:
                  * A stage advances only if its matching event is classified as success.
                  * On failure at any stage, conclude the flow and resolve the per-CID Future
                    (if any) with a failure outcome.
                  * On the final stage success, conclude the flow and resolve the per-CID Future
                    (if any) with a success outcome.
                  * When the flow concludes (either failure at any stage or final success), if
                    the template registered an `on_outcome` callback, schedule it on the
                    dispatcher with a `FifoEventCIDOutcome(ok, event)` and the original request.
                Returns None when the event is consumed by the CID pipeline; otherwise returns
                the event to be propagated.

            - Non-CID events:
                Notify all matching incoming listeners (via MRO). If any matching listener was
                registered with `consume=True`, the event is considered consumed and None is
                returned; otherwise the event is returned to be propagated. Incoming listeners
                do not affect CID stage advancement.

        Returns:
            FifoEvent | None:
                The event to propagate further (possibly unchanged), or None to indicate the
                event has been consumed by either the CID pipeline or a consuming incoming
                listener.
        """
        if isinstance(event, FifoEventShutdown):
            # Notify listeners first, then schedule shutdown so callbacks run before exit
            await self._notify_listeners(event)
            await self._queue.put((self.QueueItemKind.SHUTDOWN, ()))
            # Cancel and clear all unresolved Futures and registrations
            for _cid, (_seq, _idx, _req, fut) in list(self._registrations.items()):
                if not fut.done():
                    fut.cancel()
            self._registrations.clear()
            return event

        # Non-CID events: notify listeners and optionally consume
        if not isinstance(event, (FifoEventWithCID, FifoEventResultWithCID)):
            consumed = await self._notify_listeners(event)
            return None if consumed else event

        cid = getattr(event, "correlation_id", None)
        if cid is None:
            return event

        registration = self._registrations.get(cid)
        if registration is None:
            return event

        seq, idx, request_event, fut = registration
        expected = seq[idx]
        if not any(isinstance(event, cls) for cls in expected):
            return event

        last_stage = idx == len(seq) - 1
        is_failure = False
        if isinstance(event, FifoEventResultWithCID):
            if event.code != EErrorCode.OK:
                is_failure = True

        if not last_stage and not is_failure:
            self._registrations[cid] = (seq, idx + 1, request_event, fut)
            return None

        # Flow concludes at this point: remove registration
        self._registrations.pop(cid, None)

        # Resolve built-in future
        outcome = cast(
            FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID],
            FifoEventCIDOutcome(not is_failure, event)
        )
        if not fut.done():
            fut.set_result(outcome)

        # Dispatch outcome callback using the template default, if any
        template: FifoEventQueueNetworkAsyncHandlerCID.TemplateContent | None = None
        for cls in type(request_event).mro():
            if cls in self._templates:
                template = self._templates[cls]
                break
        if template is not None:
            _exp, _osend, _osent, cb_outcome, _odone = template
            if cb_outcome is not None:
                await self._queue.put(
                    (self.QueueItemKind.OUTCOME, (cb_outcome, outcome, request_event))
                )

        return None

    async def process_outgoing_event(self, event: FifoEvent) -> FifoEvent | None:
        """
        Auto-register per-CID tracking for matching templates; otherwise pass through.

        If `event` is an instance of a registered template class and has a `correlation_id`,
        a new per-CID registration is created from the template. If a registration already exists
        for the same CID, it is left unchanged.

        Args:
            event (FifoEvent):
                Outgoing event to be sent over the network.

        Returns:
            FifoEvent | None:
                The input event is always returned as is.
        """
        if isinstance(event, FifoEventShutdown):
            return event

        # Only work with CID-capable events
        if not isinstance(event, FifoEventWithCID):
            return event

        # Find a template for this event class (supporting inheritance chains)
        template: FifoEventQueueNetworkAsyncHandlerCID.TemplateContent | None = None
        for cls in type(event).mro():  # search MRO for a registered base class
            if cls in self._templates:
                template = self._templates[cls]
                break

        if template is None:
            return event

        cid: UUID | None = getattr(event, "correlation_id", None)
        if cid is None:
            return event

        expected, on_send, _on_sent, _tmpl_on_outcome, _on_done = template

        # Schedule the on_send callback only if provided
        if on_send is not None:
            await self._queue.put((self.QueueItemKind.SEND, (on_send, event)))

        if cid not in self._registrations:
            # Fresh stage index 0 for this new request instance, store the request event
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._registrations[cid] = (expected, 0, event, fut)

        return event

    async def process_sent_event(self, event: FifoEvent) -> None:
        """
        Observe an event after it has been successfully sent over the network.

        This hook mirrors `process_outgoing_event()` but runs only after the
        event was written and flushed on the socket. For CID-capable events that
        match a registered template, it schedules the optional `on_sent`
        callback on the internal dispatcher task.

        Notes:
            - Non-CID events are ignored by this handler and no callback is
              scheduled.
            - `FifoEventShutdown` may be observed here as well; it is treated as
              any other event (no special handling beyond template lookup).

        Args:
            event (FifoEvent):
                The event instance that has just been sent.

        Returns:
            None
        """
        # Only work with CID-capable events
        if not isinstance(event, FifoEventWithCID):
            return None

        # Find a template for this event class (supporting inheritance chains)
        template: FifoEventQueueNetworkAsyncHandlerCID.TemplateContent | None = None
        for cls in type(event).mro():  # search MRO for a registered base class
            if cls in self._templates:
                template = self._templates[cls]
                break

        if template is None:
            return None

        _expected, _on_send, on_sent, _on_outcome, _on_done = template

        # Schedule the on_sent callback only if provided
        if on_sent is not None:
            await self._queue.put((self.QueueItemKind.SENT, (on_sent, event)))

        return None

    def register_cid_template(
        self,
        event_cls: type[FifoEventWithCID],
        expected_cls: ExpectedEventClasses,
        *,
        on_send: OnSendCallback | None = None,
        on_sent: OnSentCallback | None = None,
        on_outcome: OnOutcomeCallback | None = None,
        on_done: OnDoneCallback | None = None,
    ) -> None:
        """
        Register a CID template for an outbound event class with optional hooks.

        This focuses on stage matching and built-in Futures and allows registering
        default callbacks for class instances:
          - on_send: scheduled before the event is sent
          - on_sent: scheduled after the event has been sent
          - on_outcome: scheduled when the final outcome (success or failure) is reached
          - on_done: invoked by send helpers (send_and_wait / send_in_background) in all
            cases (including timeout/cancellation) with outcome=None when not available.

        Args:
            event_cls (type[FifoEventWithCID]):
                The outbound event class to register a template for.

            expected_cls (ExpectedEventClasses):
                Sequence of expected response stages per request.

            on_send (OnSendCallback | None):
                Optional hook invoked before sending the event instance.

            on_sent (OnSentCallback | None):
                Optional hook invoked after the event instance was sent.

            on_outcome (OnOutcomeCallback | None):
                Optional hook invoked when the request reaches a terminal outcome.

            on_done (OnDoneCallback | None):
                Optional hook invoked by send helpers in all cases (including timeout/cancel).

        Raises:
            ValueError: If a template is already registered for `event_cls`.
        """
        if event_cls in self._templates:
            raise ValueError(f"CID template already registered for {event_cls.__name__}")

        self._templates[event_cls] = (
            self._normalize_expected(expected_cls),
            on_send,
            on_sent,
            on_outcome,
            on_done,
        )

    # --- Awaiting helpers -------------------------------------------------

    def _resolve_on_done(
        self,
        req: FifoEventWithCID,
        on_done: OnDoneCallback | None,
    ) -> OnDoneCallback | None:
        """
        Resolve on_done by falling back to template defaults when absent.

        Args:
            req (FifoEventWithCID):
                Request instance used to look up the template.

            on_done (OnDoneCallback | None):
                Per-send on_done, if any.

        Returns:
            OnDoneCallback | None:
                Resolved on_done or template default.
        """
        if on_done is not None:
            return on_done

        template = None
        for cls in type(req).mro():
            if cls in self._templates:
                template = self._templates[cls]
                break
        if template is None:
            return on_done

        _exp, _osend, _osent, _default_on_outcome, default_on_done = template
        return default_on_done

    async def _wait_for_cid(
        self,
        cid: UUID,
        *,
        timeout: float | None = None,
    ) -> FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID]:
        """
        Await the final outcome for a given correlation ID.

        Requires that a registration was created for `cid` (typically via sending an
        instance of a registered outbound event class). If no registration exists, the wait
        may never complete.
        """
        reg = self._registrations.get(cid)
        if reg is None:
            raise ValueError(f"No registration for CID {cid}")
        _seq, _idx, _req, fut = reg

        try:
            return await (asyncio.wait_for(fut, timeout) if timeout is not None else fut)
        finally:
            # nothing to clean; registration is popped on completion in process_incoming_event
            pass

    async def send_and_wait(
        self,
        transport: SupportsFifoEventPut,
        req: FifoEventWithCID,
        *,
        timeout: float | None = None,
        on_done: OnDoneCallback | None = None,
    ) -> FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID]:
        """
        Send a CID-capable request and await the final outcome.

        Outcome delivery has two parts:
        - Template on_outcome: If the request class was registered with an
          `on_outcome` callback via `register_cid_template(...)`, that callback is
          scheduled on the handler's internal dispatcher when the flow concludes
          (failure at any stage or final success). It is not called here; it runs
          asynchronously on the dispatcher task.
        - Per-call on_done: This optional callback is invoked by this method in a
          `finally` clause with the resolved `FifoEventCIDOutcome` (or None when a
          timeout/cancellation/exception prevented completion). Use it for per-call
          cleanup (e.g., releasing a local gate/lock) regardless of outcome.

        Args:
            transport (SupportsFifoEventPut):
                Transport exposing an async put(FifoEvent) method.

            req (FifoEventWithCID):
                Outbound request carrying a correlation_id.

            timeout (float | None):
                Optional timeout for the await.

            on_done (OnDoneCallback | None):
                Per-call override; falls back to template.

        Returns:
            FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID]:
                The final outcome for this request (ok + event).
        """
        # Resolve on_done (fill from template if not provided)
        on_done = self._resolve_on_done(req, on_done)

        outcome: FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID] | None = None
        try:
            await transport.put(req)
            # No per-request on_outcome override; template handles outcome callback
            outcome = await self._wait_for_cid(req.correlation_id, timeout=timeout)
            return outcome
        finally:
            if on_done is not None:
                try:
                    await on_done(outcome, req)
                except Exception:  # pragma: no cover # pylint: disable=broad-exception-caught
                    logger.error("on_done callback failed for %s", type(req).__name__)

    def send_in_background(
        self,
        transport: SupportsFifoEventPut,
        req: FifoEventWithCID,
        *,
        timeout: float | None = None,
        on_done: OnDoneCallback | None = None,
    ) -> asyncio.Task[None]:
        """
        Launch a background task to send and await a CID-capable request.

        Outcome delivery mirrors `send_and_wait`:
        - Template on_outcome: If registered on the class, is scheduled on the
          handler's dispatcher when the flow concludes (not called here).
        - Per-call on_done: If provided, is invoked by `send_and_wait` in all
          cases (success, failure, timeout, cancellation, error).

        Args:
            transport (SupportsFifoEventPut):
                Transport exposing an async put(FifoEvent) method.

            req (FifoEventWithCID):
                Outbound request carrying a correlation_id.

            timeout (float | None):
                Optional timeout for the await.

            on_done (OnDoneCallback | None):
                Per-call override; falls back to template.

        Returns:
            asyncio.Task[None]:
                The background task running the send/await flow.
        """
        # Resolve on_done (fill from template if not provided)
        on_done = self._resolve_on_done(req, on_done)

        async def _runner() -> None:
            try:
                # send_and_wait will invoke on_done in all cases (including timeout)
                await self.send_and_wait(
                    transport,
                    req,
                    timeout=timeout,
                    on_done=on_done,
                )
            except asyncio.TimeoutError:
                # Timeout already delivered to on_done(None, req) by send_and_wait
                pass
            except Exception as exc:  # pragma: no cover # pylint: disable=broad-exception-caught
                logger.error(
                    "CID background task failed for %s (%s)",
                    type(req).__name__,
                    type(exc).__name__,
                )

        task = asyncio.create_task(_runner())
        return task
