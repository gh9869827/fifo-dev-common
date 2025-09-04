from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Awaitable, Callable, Dict, Sequence, TypeAlias, cast
from uuid import UUID

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventResultWithCID,
    FifoEventShutdown,
    FifoEventWithCID,
    EErrorCode,
)
from fifo_dev_common.logging.logger import get_logger

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
        - SUCCESS: schedule the success callback with the incoming event and original request
        - FAILURE: schedule the failure callback with the incoming event and original request
        - SHUTDOWN: signal the dispatcher loop to terminate
        - LISTENER: schedule an incoming-only listener callback
        """
        SEND = auto()
        SENT = auto()
        SUCCESS = auto()
        FAILURE = auto()
        SHUTDOWN = auto()
        LISTENER = auto()

    # Payloads are uniform tuples; first element is the callback, followed by args.
    QueuePayloadSend: TypeAlias = tuple[
        Callable[[FifoEventWithCID], Awaitable[None]],
        FifoEventWithCID,
    ]
    QueuePayloadSent: TypeAlias = tuple[
        Callable[[FifoEventWithCID], Awaitable[None]],
        FifoEventWithCID,
    ]
    QueuePayloadSuccess: TypeAlias = tuple[
        Callable[[FifoEventWithCID | FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
        FifoEventWithCID | FifoEventResultWithCID,
        FifoEventWithCID,
    ]
    QueuePayloadFailure: TypeAlias = tuple[
        Callable[[FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
        FifoEventResultWithCID,
        FifoEventWithCID,
    ]
    QueuePayloadShutdown: TypeAlias = tuple[()]
    QueuePayloadListener: TypeAlias = tuple[
        Callable[[FifoEvent], Awaitable[None]],
        FifoEvent,
    ]

    QueueItem: TypeAlias = tuple[
        QueueItemKind,
        QueuePayloadSend |
        QueuePayloadSent |
        QueuePayloadSuccess |
        QueuePayloadFailure |
        QueuePayloadShutdown |
        QueuePayloadListener,
    ]

    _queue: asyncio.Queue[QueueItem]

    _task: asyncio.Task[None]

    _registrations: Dict[
        UUID,
        tuple[
            NormalizedExpected,
            Callable[[FifoEventWithCID|FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
            Callable[[FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
            int,
            FifoEventWithCID,
        ],
    ]

    _templates: Dict[
        type[FifoEventWithCID],
        tuple[
            NormalizedExpected,
            Callable[[FifoEventWithCID|FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
            Callable[[FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
            Callable[[FifoEventWithCID], Awaitable[None]] | None,
            Callable[[FifoEventWithCID], Awaitable[None]] | None,
        ],
    ]

    # Incoming-only listeners (non-CID events and shutdown)
    _listeners: Dict[
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
        #   cid -> (normalized_expected, on_success, on_failure, stage_index, request_event)
        self._registrations = {}

        # Class-level templates registered once.
        #   event_cls -> (normalized_expected, on_success, on_failure, on_send, on_sent)
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
                elif kind is self.QueueItemKind.SUCCESS:
                    cb_success, ev_success, req_success = cast(
                        FifoEventQueueNetworkAsyncHandlerCID.QueuePayloadSuccess, payload
                    )
                    await cb_success(ev_success, req_success)
                elif kind is self.QueueItemKind.FAILURE:
                    cb_failure, ev_failure, req_failure = cast(
                        FifoEventQueueNetworkAsyncHandlerCID.QueuePayloadFailure, payload
                    )
                    await cb_failure(ev_failure, req_failure)
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

    # --- Registration API -------------------------------------------------

    def _normalize_expected(self, expected_cls: ExpectedEventClasses) -> NormalizedExpected:
        seq: NormalizedExpected = []
        for stage in expected_cls:
            if isinstance(stage, type):
                seq.append([stage])
            else:
                seq.append([cls for cls in stage])
        return seq

    def register_cid_template(
        self,
        event_cls: type[FifoEventWithCID],
        expected_cls: ExpectedEventClasses,
        on_success: Callable[[FifoEventWithCID | FifoEventResultWithCID, FifoEventWithCID],
                             Awaitable[None]],
        on_failure: Callable[[FifoEventResultWithCID, FifoEventWithCID],
                             Awaitable[None]],
        *,
        on_send: Callable[[FifoEventWithCID], Awaitable[None]] | None = None,
        on_sent: Callable[[FifoEventWithCID], Awaitable[None]] | None = None,
    ) -> None:
        """
        Register callbacks and expected stages for an *outbound event class*.

        Templates are applied automatically when instances of `event_cls` are sent via
        `process_outgoing_event` (i.e., on the send path). Each instance must carry a
        `correlation_id`; a per-CID registration is created at send-time.

        Args:
            event_cls (type[FifoEventWithCID]):
                The outbound event class to register a template for. When instances of this
                class (or its subclasses) are sent, the template will be applied automatically.

            expected_cls (ExpectedEventClasses):
                Sequence of expected response stages. Each stage can be either a single event
                class or a sequence of event classes (any of which can satisfy that stage).
                Example: [FifoEventAck, [FifoEventSuccess, FifoEventFailure]]

            on_success (Callable[[FifoEventWithCID | FifoEventResultWithCID, FifoEventWithCID],
                                 Awaitable[None]]):
                Callback invoked when an event is successfully processed. Receives the
                incoming event and the original request event. Called for:
                - Non-final stages with successful events (FifoEventResultWithCID with OK code
                  or non-result events)
                - Final stage with successful events

            on_failure (Callable[[FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]]):
                Callback invoked when a failure occurs. Receives the incoming event and the
                original request event. Called for any stage when a
                FifoEventResultWithCID is received with a non-OK error code.

            on_send (Callable[[FifoEventWithCID], Awaitable[None]] | None, optional):
                If provided, invoked when an instance of `event_cls` (or its subclass) is about to
                be sent over the network. If None, no callback is scheduled. The event instance is
                passed to the callback.

            on_sent (Callable[[FifoEventWithCID], Awaitable[None]] | None, optional):
                If provided, invoked after an instance of `event_cls` (or its subclass) has been
                successfully written and flushed to the network. If None, no callback is scheduled.
                The event instance is passed to the callback.
        """
        self._templates[event_cls] = (
            self._normalize_expected(expected_cls),
            on_success,
            on_failure,
            on_send,
            on_sent,
        )

    # --- Incoming-only listener API --------------------------------------

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
        """
        self._listeners.setdefault(event_cls, []).append((on_event, consume))

    # --- Pipeline hooks ---------------------------------------------------

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
                  * On failure at an intermediate stage, invoke the failure callback, stop
                    (do not advance) and remove the per-CID registration.
                  * On the final stage, always invoke the corresponding callback (success or
                    failure) and then remove the per-CID registration. Class-level templates
                    remain registered and continue to apply to future requests.
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

        seq, on_success, on_failure, idx, request_event = registration
        expected = seq[idx]
        if not any(isinstance(event, cls) for cls in expected):
            return event

        last_stage = idx == len(seq) - 1
        is_failure = False
        if isinstance(event, FifoEventResultWithCID):
            if event.code != EErrorCode.OK:
                is_failure = True

        if not last_stage and not is_failure:
            self._registrations[cid] = (seq, on_success, on_failure, idx + 1, request_event)
        else:
            self._registrations.pop(cid, None)

        if is_failure:
            await self._queue.put(
                (
                    self.QueueItemKind.FAILURE,
                    (
                        on_failure,
                        cast(FifoEventResultWithCID, event),
                        request_event,
                    ),
                )
            )
        elif last_stage:
            await self._queue.put(
                (
                    self.QueueItemKind.SUCCESS,
                    (
                        on_success,
                        event,
                        request_event,
                    ),
                )
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
        template: tuple[NormalizedExpected,
                        Callable[[FifoEventWithCID | FifoEventResultWithCID, FifoEventWithCID],
                                 Awaitable[None]],
                        Callable[[FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
                        Callable[[FifoEventWithCID], Awaitable[None]] | None,
                        Callable[[FifoEventWithCID], Awaitable[None]] | None] | None = None
        for cls in type(event).mro():  # search MRO for a registered base class
            if cls in self._templates:
                template = self._templates[cls]
                break

        if template is None:
            return event

        cid: UUID | None = getattr(event, "correlation_id", None)
        if cid is None:
            return event

        expected, on_success, on_failure, on_send, _on_sent = template

        # Schedule the on_send callback only if provided
        if on_send is not None:
            await self._queue.put((self.QueueItemKind.SEND, (on_send, event)))

        if cid not in self._registrations:
            # Fresh stage index 0 for this new request instance, store the request event
            self._registrations[cid] = (expected, on_success, on_failure, 0, event)

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
        template: tuple[NormalizedExpected,
                        Callable[[FifoEventWithCID | FifoEventResultWithCID, FifoEventWithCID],
                                 Awaitable[None]],
                        Callable[[FifoEventResultWithCID, FifoEventWithCID], Awaitable[None]],
                        Callable[[FifoEventWithCID], Awaitable[None]] | None,
                        Callable[[FifoEventWithCID], Awaitable[None]] | None] | None = None
        for cls in type(event).mro():  # search MRO for a registered base class
            if cls in self._templates:
                template = self._templates[cls]
                break

        if template is None:
            return None

        _expected, _on_success, _on_failure, _on_send, on_sent = template

        # Schedule the on_sent callback only if provided
        if on_sent is not None:
            await self._queue.put((self.QueueItemKind.SENT, (on_sent, event)))

        return None
