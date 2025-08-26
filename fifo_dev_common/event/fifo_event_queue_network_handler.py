from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, Sequence
from uuid import UUID

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventResultWithCID,
    FifoEventShutdown,
    FifoEventWithCID,
)
from fifo_dev_common.logging.logger import get_logger

logger = get_logger(__name__)


async def _async_noop(_: FifoEvent) -> None:
    """
    Asynchronous no-op function used for shutdown.
    """
    return


class FifoEventQueueNetworkAsyncHandlerBase(ABC):
    """Base class for asynchronous network event handlers.

    This handler processes callbacks on a background task so that the network
    reader is never blocked. Derived classes should implement :py:meth:`register`
    and may override :py:meth:`process_event`.
    """

    def __init__(self) -> None:
        """
        Initialize the handler and start the background processing task.
        """
        self._queue: asyncio.Queue[tuple[Callable[[FifoEvent], Awaitable[None]], FifoEvent]]
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            callback, event = await self._queue.get()
            if isinstance(event, FifoEventShutdown):
                break
            try:
                await callback(event)
            except Exception:  # pragma: no cover
                logger.exception("handler callback failed")

    async def join(self) -> None:
        """
        Wait for the handler background task to finish.
        """
        await self._task

    @abstractmethod
    def register(self,
                 event: FifoEventWithCID,
                 expected_cls: Sequence[type[FifoEventResultWithCID]] | Sequence[Sequence[type[FifoEventResultWithCID]]],
                 callback: Callable[[FifoEventResultWithCID], Awaitable[bool]]) -> None:
        """
        Register a callback for a specific incoming event.

        Args:
            event (FifoEventWithCID):
                Event object being sent and for which a response is expected.

            expected_cls (Sequence[type[FifoEventResultWithCID]] | Sequence[Sequence[type[FifoEventResultWithCID]]]):
                One or more sequences of expected event classes. Each inner sequence
                represents a stage of possible events processed in order.

            callback (Callable[[FifoEventResultWithCID], Awaitable[bool]]):
                Asynchronous function invoked when the matching event is received. It
                returns True to continue to the next stage or False to stop further
                processing for the correlation ID.
        """

    async def process_event(self, event: FifoEvent) -> bool:
        """
        Process an incoming event.

        Args:
            event (FifoEvent):
                Incoming event from the network.

        Returns:
            bool:
                True if the event has been processed by the handler and should
                not be inserted into the out queue. False if it was not handled
                and should be queued for further processing.
        """
        if isinstance(event, FifoEventShutdown):
            await self._queue.put((_async_noop, event))
            return True
        return False


class FifoEventQueueNetworkAsyncHandlerCID(FifoEventQueueNetworkAsyncHandlerBase):
    """Handler that matches events using correlation IDs."""

    def __init__(self) -> None:
        """
        Initialize the correlation ID handler.
        """
        super().__init__()
        self._registrations: Dict[
            UUID,
            tuple[
                list[list[type[FifoEventResultWithCID]]],
                Callable[[FifoEventResultWithCID], Awaitable[bool]],
                int,
            ],
        ]
        self._registrations = {}

    def register(self,
                 event: FifoEventWithCID,
                 expected_cls: Sequence[type[FifoEventResultWithCID]] | Sequence[Sequence[type[FifoEventResultWithCID]]],
                 callback: Callable[[FifoEventResultWithCID], Awaitable[bool]]) -> None:
        """
        Register a callback for an expected event with the same correlation ID.

        Args:
            event (FifoEventWithCID):
                The event that was sent and from which the correlation ID is taken.

            expected_cls (Sequence[type[FifoEventResultWithCID]] | Sequence[Sequence[type[FifoEventResultWithCID]]]):
                One or more sequences of event classes that may be received in order.

            callback (Callable[[FifoEventResultWithCID], Awaitable[bool]]):
                Asynchronous method to call when the matching event is received.
                It returns True to continue to the next stage or False to stop
                processing further events for this correlation ID.
        """
        cid = getattr(event, "correlation_id")
        if expected_cls and isinstance(expected_cls[0], type):  # type: ignore[index]
            seq = [list(expected_cls)]  # type: ignore[list-item]
        else:
            seq = [list(s) for s in expected_cls]  # type: ignore[arg-type]
        self._registrations[cid] = (seq, callback, 0)

    async def process_event(self, event: FifoEvent) -> bool:
        if await super().process_event(event):
            return True

        cid = getattr(event, "correlation_id", None)
        if cid is None:
            return False

        registration = self._registrations.get(cid)
        if registration is None:
            return False

        seq, callback, idx = registration
        expected = seq[idx]
        if not any(isinstance(event, cls) for cls in expected):
            return False

        async def _wrapper(ev: FifoEvent) -> None:
            cont = await callback(ev)  # type: ignore[arg-type]
            if not cont:
                self._registrations.pop(cid, None)

        if idx + 1 < len(seq):
            self._registrations[cid] = (seq, callback, idx + 1)
        else:
            self._registrations.pop(cid, None)

        await self._queue.put((_wrapper, event))
        return True
