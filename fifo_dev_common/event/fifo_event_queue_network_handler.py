from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
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
    type[FifoEventResultWithCID] | Sequence[type[FifoEventResultWithCID]]
)
ExpectedEventClasses: TypeAlias = Sequence[StageElement]

# Internal canonical form: always list of stages, each stage a list of types
NormalizedExpected: TypeAlias = list[list[type[FifoEventResultWithCID]]]


async def _async_noop(_: FifoEvent) -> None:
    """
    Asynchronous no-op function used for shutdown.
    """
    return


class FifoEventQueueNetworkAsyncHandlerBase(ABC):
    """
    Base class for asynchronous network event handlers.

    Handlers can intercept and process events in both directions:
    - Incoming events (from network before enqueuing): return the event (possibly modified) to add
      to the outgoing queue, or None to suppress.
    - Outgoing events (before sending to network): return the event (possibly modified) to send, or
      None to suppress sending.

    Subclasses must implement process_incoming_event() and process_outgoing_event().

    Note:
        FifoEventShutdown is always propagated. Handlers can observe it via both
        process_incoming_event() and process_outgoing_event(), but their return values
        are ignored. The original shutdown event is always enqueued or sent exactly once.
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
        raise NotImplementedError

    @abstractmethod
    async def process_outgoing_event(self, event: FifoEvent) -> FifoEvent | None:
        """
        Process an outgoing event before it is sent over the network.

        Called when an event is about to be sent over the network.

        Args:
            event (FifoEvent):
                Outgoing event to be sent over the network.

        Returns:
            FifoEvent | None:
                The event to send (possibly modified), or None to suppress sending.
        """
        raise NotImplementedError


class FifoEventQueueNetworkAsyncHandlerCID(FifoEventQueueNetworkAsyncHandlerBase):
    """
    Correlation ID-based event handler for managing request/response workflows.

    This handler matches events using correlation IDs and executes registered callbacks
    when specific event sequences are received. It supports **template-based registration**:

    - You register *event classes* exactly once along with the expected response stages and
      success/failure callbacks (see `register_template`).
    - Each time a matching *event instance* (with a correlation ID) is **sent**, the handler
      automatically creates a per-CID registration based on the template.

    This avoids per-request manual registration: define the contract once; instances are
    auto-registered on send.

    Usage:
        1. Define a template for an outbound event class via `register_template()`
        2. Send an instance of that class (must subclass `FifoEventWithCID`)
        3. The handler auto-registers the instance's CID
        4. Incoming events matching the stages trigger the appropriate callbacks

    Example:
        ```python
        handler = FifoEventQueueNetworkAsyncHandlerCID()

        # 1) Define the contract once at startup
        async def on_ok(ev: FifoEvent) -> None: ...
        async def on_err(ev: FifoEventResultWithCID) -> None: ...

        handler.register_template(
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

    _queue: asyncio.Queue[tuple[Callable[[FifoEvent], Awaitable[None]], FifoEvent]]

    _task: asyncio.Task[None]

    _registrations: Dict[
        UUID,
        tuple[
            NormalizedExpected,
            Callable[[FifoEvent], Awaitable[None]],
            Callable[[FifoEventResultWithCID], Awaitable[None]],
            int,
        ],
    ]

    _templates: Dict[
        type[FifoEventWithCID],
        tuple[
            NormalizedExpected,
            Callable[[FifoEvent], Awaitable[None]],
            Callable[[FifoEventResultWithCID], Awaitable[None]],
        ],
    ]

    def __init__(self) -> None:
        """
        Initialize the correlation ID handler and start the background task that
        dispatches callbacks.
        """
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

        # Per-CID active registrations created at send-time from templates.
        #   cid -> (normalized_expected, on_success, on_failure, stage_index)
        self._registrations = {}

        # Class-level templates registered once.
        #   event_cls -> (normalized_expected, on_success, on_failure)
        self._templates = {}

    async def _run(self) -> None:
        """
        Process queued callbacks until a shutdown event is received.
        """
        while True:
            callback, event = await self._queue.get()
            if isinstance(event, FifoEventShutdown):
                break
            try:
                await callback(event)
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

    # --- Registration API -------------------------------------------------

    def _normalize_expected(self, expected_cls: ExpectedEventClasses) -> NormalizedExpected:
        seq: NormalizedExpected = []
        for stage in expected_cls:
            if isinstance(stage, type):
                seq.append([stage])
            else:
                seq.append([cls for cls in stage])
        return seq

    def register_template(
        self,
        event_cls: type[FifoEventWithCID],
        expected_cls: ExpectedEventClasses,
        on_success: Callable[[FifoEvent], Awaitable[None]],
        on_failure: Callable[[FifoEventResultWithCID], Awaitable[None]],
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

            on_success (Callable[[FifoEvent], Awaitable[None]]):
                Callback invoked when an event is successfully processed. Called for:
                - Non-final stages with successful events (FifoEventResultWithCID with OK code
                  or non-result events)
                - Final stage with successful events

            on_failure (Callable[[FifoEventResultWithCID], Awaitable[None]]):
                Callback invoked when a failure occurs. Called for any stage when a
                FifoEventResultWithCID is received with a non-OK error code.
        """
        self._templates[event_cls] = (
            self._normalize_expected(expected_cls), on_success, on_failure
        )

    # --- Pipeline hooks ---------------------------------------------------

    async def process_incoming_event(self, event: FifoEvent) -> FifoEvent | None:
        """
        Process an incoming event applying success/failure stage rules.

        Rules:
            - A stage advances only if its matching event is classified as success.
            - On failure (intermediate stage), invoke failure callback, stop (do not advance) and
              remove the registration.
            - On the final stage, always invoke the corresponding callback (success or failure)
              and then remove the registration.
        
        Returns:
            FifoEvent | None:
                If an event was not consumed to advance the stages or to invoke a callback, the
                event itself is returned; otherwise None is returned.
        """
        if isinstance(event, FifoEventShutdown):
            await self._queue.put((_async_noop, event))
            return event

        cid = getattr(event, "correlation_id", None)
        if cid is None:
            return event

        registration = self._registrations.get(cid)
        if registration is None:
            return event

        seq, on_success, on_failure, idx = registration
        expected = seq[idx]
        if not any(isinstance(event, cls) for cls in expected):
            return event

        last_stage = idx == len(seq) - 1
        is_failure = False
        if isinstance(event, FifoEventResultWithCID):
            if event.code != EErrorCode.OK:
                is_failure = True

        if not last_stage and not is_failure:
            self._registrations[cid] = (seq, on_success, on_failure, idx + 1)
        else:
            self._registrations.pop(cid, None)

        if is_failure:
            await self._queue.put((cast(Callable[[FifoEvent], Awaitable[None]], on_failure), event))
        else:
            await self._queue.put((on_success, event))

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
                        Callable[[FifoEvent], Awaitable[None]],
                        Callable[[FifoEventResultWithCID], Awaitable[None]]] | None = None
        for cls in type(event).mro():  # search MRO for a registered base class
            if cls in self._templates:
                template = self._templates[cls]
                break

        if template is None:
            return event

        cid: UUID | None = getattr(event, "correlation_id", None)
        if cid is None:
            return event

        if cid not in self._registrations:
            expected, on_success, on_failure = template
            # Fresh stage index 0 for this new request instance
            self._registrations[cid] = (expected, on_success, on_failure, 0)

        return event
