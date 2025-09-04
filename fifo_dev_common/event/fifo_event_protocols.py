from __future__ import annotations
from typing import Protocol, runtime_checkable
from enum import Enum

from fifo_dev_common.event.fifo_event import FifoEvent


@runtime_checkable
class SupportsFifoEventPut(Protocol):
    """
    Protocol for accepting FifoEvent instances via put.
    """

    async def put(self, item: FifoEvent) -> None:
        """
        Asynchronously enqueue a FifoEvent.

        Args:
            item (FifoEvent):
                Event to enqueue.
        """

class SendStatus(Enum):
    """
    Result of attempting to send a `FifoEvent`.

    - SENT: the event was actually written (or shutdown was propagated)
    - SUPPRESSED: the handler suppressed the event (returned None for a non-shutdown event),
      or a handler hook failed and the event was discarded before writing
    """
    SENT = 0
    SUPPRESSED = 1


@runtime_checkable
class SupportsFifoEventSend(Protocol):
    """
    Protocol for transports that can send an event and report suppression.

    Implementations should raise on network/serialization errors; the return value only
    indicates whether the event was written vs. suppressed by a handler or if a handler hook failed
    and the event was discarded.
    """

    async def send(self, event: FifoEvent) -> SendStatus:
        """
        Attempt to send a `FifoEvent`.

        Args:
            event (FifoEvent):
                Event to send.

        Returns:
            SendStatus:
                `SendStatus.SENT` if written (or shutdown propagated).
                `SendStatus.SUPPRESSED` if the handler suppressed the event (returned None for a
                non-shutdown event) or if a handler hook failed and the event was discarded.
        """
        ...  # pylint: disable=unnecessary-ellipsis


__all__ = ["SupportsFifoEventPut", "SendStatus", "SupportsFifoEventSend"]
