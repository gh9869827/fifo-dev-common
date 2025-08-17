from __future__ import annotations
from typing import Protocol, runtime_checkable

from fifo_dev_common.event.fifo_event import FifoEvent


@runtime_checkable
class SupportsFifoEventPut(Protocol):
    """
    Protocol for accepting FifoEvent instances via put.
    """

    async def put(self, event: FifoEvent) -> None:
        """
        Asynchronously enqueue a FifoEvent.

        Args:
            event (FifoEvent):
                Event to enqueue.
        """
        ...


__all__ = ["SupportsFifoEventPut"]
