"""
Common asyncio connector utilities for FIFO event queues.

The module consolidates the shared asyncio logic previously embedded in the TCP client so it can
be reused by multiple transports (network, serial, etc.).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Final

from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_protocols import SupportsFifoEventPut, SendStatus
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerBase,
)
from fifo_dev_common.logging.logger import get_logger

logger = get_logger(__name__)


async def bounded_close_and_wait_closed_writer(
    writer: asyncio.StreamWriter,
    *,
    timeout: float,
    label: str,
) -> None:
    """
    Close an asyncio StreamWriter and wait for it to finish closing with timeout protection.

    The helper mirrors the previous network implementation but is shared across
    connector types. It performs a graceful shutdown of the writer by calling
    close() and then waiting for wait_closed() to complete while bounding the
    wait time.

    Args:
        writer (asyncio.StreamWriter):
            The asyncio writer to close and wait for.

        timeout (float):
            Maximum seconds to wait for wait_closed().

        label (str):
            Label used in log messages for additional context.
    """

    peer: Final[tuple[str, int] | None] = writer.get_extra_info("peername")
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "[%s] wait_closed() timed out after %.1fs (peer=%s); proceeding.",
            label,
            timeout,
            peer,
        )
    except (BrokenPipeError, ConnectionResetError) as exc:
        logger.debug(
            "[%s] wait_closed() benign connection error: %r (peer=%s); proceeding.",
            label,
            exc,
            peer,
        )


class FifoEventQueueConnectorAsyncMixin:
    """
    Mixin with common asyncio connector behaviour.

    The mixin encapsulates the logic for sending and receiving FifoEvent instances
    over an asyncio stream pair. It assumes subclasses provide _reader, _writer,
    _out_queue and _task attributes as well as an optional handler used to
    intercept events.
    """

    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _task: asyncio.Task[None]
    _handler: FifoEventQueueConnectorAsyncHandlerBase | None

    async def stop(self) -> None:
        """
        Signal the connection to stop by sending a shutdown event.

        This method sends a FifoEventShutdown event which will cause the background
        connector-to-queue task to terminate after processing the shutdown signal.
        Call join() after this method to wait for the actual shutdown to complete.

        Raises:
            ConnectionError: If the connection is already closed or there is a transport error.
        """

        await self.send(FifoEventShutdown())

    async def _network_to_queue(self) -> None:
        """
        Background task that continuously receives events from the connector and enqueues them.

        This method runs in a background asyncio task and continuously deserializes events
        from the connector stream. If a handler is configured, incoming events are processed
        through the handler's process_incoming_event() method, which may modify or suppress
        them. Only non-None events are placed into the output queue. The task terminates when
        a FifoEventShutdown event is received.

        Note:
            When a FifoEventShutdown is received, process_incoming_event() is still invoked so the
            handler can observe it, but its return value is ignored. The original shutdown event is
            always enqueued exactly once.

        Handler errors are logged; the failing event is discarded (except shutdown, which still
        propagates).

        Raises:
            ConnectionError: If the connector stream is lost during operation.
        """

        while True:
            try:
                event = await FifoEvent.deserialize_from_stream_async(self._reader)
            except (RuntimeError, TypeError, ValueError) as exc:
                role = "client" if "Client" in type(self).__name__ else "server"
                logger.error("[%s] Error receiving event: %r", role, type(exc))
                continue

            if self._handler is not None:
                try:
                    event_to_enqueue = await self._handler.process_incoming_event(event)
                except Exception:  # pylint: disable=broad-exception-caught
                    role = "client" if "Client" in type(self).__name__ else "server"
                    logger.error(
                        "[%s] process_incoming_event handler failed. Discarding event.",
                        role,
                    )
                    event_to_enqueue = None

                if not isinstance(event, FifoEventShutdown):
                    if event_to_enqueue is None:
                        continue
                    event = event_to_enqueue

            await self._out_queue.put(event)
            if isinstance(event, FifoEventShutdown):
                break

    async def send(self, event: FifoEvent) -> SendStatus:
        """
        Send a FifoEvent over the connector.

        This method processes the event through the handler's process_outgoing_event() method (if a
        handler is configured), which may modify or suppress the event. If the handler returns a
        non-None event, it is serialized and sent over the connector. The event is automatically
        flushed to ensure delivery.

        If the write completes without raising an exception, the handler's process_sent_event()
        method is invoked (if configured) with the event that has been successfully sent.

        Note:
            When a FifoEventShutdown is sent, process_outgoing_event() is still invoked so the
            handler can observe it, but its return value is ignored. The original shutdown event is
            always sent instead of the return value.

        Handler errors are logged; the failing event is discarded (except shutdown, which is still
        sent).

        Args:
            event (FifoEvent):
                The event to send over the connector.

        Returns:
            SendStatus:
                SendStatus.SENT if the event was written (or shutdown propagated).
                SendStatus.SUPPRESSED if the handler suppressed the event (returned None for a
                non-shutdown event) or if a handler hook failed and the event was discarded.

        Raises:
            ConnectionError: If the connector is closed or a transport error occurs.
        """

        if self._handler is not None:
            try:
                event_to_send = await self._handler.process_outgoing_event(event)
            except Exception:  # pylint: disable=broad-exception-caught
                role = "client" if "Client" in type(self).__name__ else "server"
                logger.error(
                    "[%s] process_outgoing_event handler failed. Discarding event.",
                    role,
                )
                event_to_send = None

            if not isinstance(event, FifoEventShutdown):
                if event_to_send is None:
                    return SendStatus.SUPPRESSED
                event = event_to_send

        await event.serialize_to_stream_async(self._writer)

        if self._handler is not None:
            try:
                await self._handler.process_sent_event(event)
            except Exception:  # pylint: disable=broad-exception-caught
                role = "client" if "Client" in type(self).__name__ else "server"
                logger.error(
                    "[%s] process_sent_event handler failed.",
                    role,
                )

        return SendStatus.SENT

    async def put(self, item: FifoEvent) -> None:
        """
        Queue-like alias that forwards to send().

        Args:
            item (FifoEvent):
                The event to enqueue for sending.
        """

        await self.send(item)


class FifoEventQueueConnectorAsyncClient(
    FifoEventQueueConnectorAsyncMixin,
    SupportsFifoEventPut,
):
    """
    Base asyncio client shared by connector implementations.

    Subclasses provide the logic that creates the StreamReader/StreamWriter pair, typically via a
    classmethod like `connect()`. Once instantiated the client mirrors the behaviour of the previous
    network-only implementation: outbound events are written immediately when `send()` is called,
    while inbound events are received in a background task and queued in `_out_queue`.

    Attributes:
        _reader (asyncio.StreamReader):
            Asyncio stream reader used to pull serialized events from the transport.

        _writer (asyncio.StreamWriter):
            Asyncio stream writer used to push serialized events to the transport.

        _out_queue (asyncio.PriorityQueue[FifoEvent]):
            Priority queue populated with incoming events in the order they are received.

        _task (asyncio.Task[None]):
            Background task created in `__init__` that runs `_network_to_queue()`.

        _handler (FifoEventQueueConnectorAsyncHandlerBase | None):
            Optional handler invoked for incoming, outgoing, and sent events.
    """

    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _task: asyncio.Task[None]
    _handler: FifoEventQueueConnectorAsyncHandlerBase | None

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        out_queue: asyncio.PriorityQueue[FifoEvent] | None,
        handler: FifoEventQueueConnectorAsyncHandlerBase | None = None,
    ) -> None:
        """
        Initialize a connector client with existing asyncio stream objects.

        Args:
            reader (asyncio.StreamReader):
                Stream reader that supplies serialized events.

            writer (asyncio.StreamWriter):
                Stream writer that receives serialized events.

            out_queue (asyncio.PriorityQueue[FifoEvent] | None):
                Optional priority queue to receive inbound events. When `None`, a fresh queue is
                created so the client owns its own buffer.

            handler (FifoEventQueueConnectorAsyncHandlerBase | None, optional):
                Handler used to intercept and process events. If None, events are
                queued and sent directly without additional processing. When provided,
                the handler can:
                - process incoming events before they are enqueued,
                - process outgoing events before they are sent, and
                - observe sent events after they have been successfully sent.
        """
        self._reader = reader
        self._writer = writer
        self._out_queue = out_queue or asyncio.PriorityQueue()
        self._handler = handler
        self._task = asyncio.create_task(self._network_to_queue())

    async def join(self, timeout: float = 5.0) -> None:
        """
        Wait for the connector to finish shutting down and close the writer.

        This method waits for the background receive task to finish, cancelling it if the
        timeout elapses. After the task completes (or is cancelled), the writer is closed using
        _bounded_close_and_wait_closed_writer() to ensure a graceful shutdown.

        Args:
            timeout (float):
                Maximum seconds to wait for the receive task.

        Raises:
            ConnectionError: If the connector encounters an error while shutting down.
        """

        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[client] join() timed out after %.1fs; cancelling background task.",
                timeout,
            )
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        await bounded_close_and_wait_closed_writer(
            self._writer,
            timeout=3.0,
            label="client",
        )
