"""
Asyncio-based serial transport for `FifoEvent` objects using `serial_asyncio`.
"""

from __future__ import annotations

import asyncio
from typing import Any
from serial_asyncio import ( # pyright: ignore[reportMissingTypeStubs]
    open_serial_connection # pyright: ignore[reportUnknownVariableType]
)

from fifo_dev_common.event.fifo_event import FifoEvent
from fifo_dev_common.event.fifo_event_queue_connector import (
    FifoEventQueueConnectorAsyncClient,
)
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerBase,
)
from fifo_dev_common.logging.logger import get_logger

logger = get_logger(__name__)


class FifoEventQueueSerialAsyncClient(FifoEventQueueConnectorAsyncClient):
    """
    Asyncio-based serial client for sending and receiving `FifoEvent` objects.

    This client mirrors the behaviour of the TCP client but targets serial links managed by
    `serial_asyncio`. Outbound events are serialized and written to the serial stream as soon as
    `send()` is called, while inbound events are deserialized in a background task and placed into
    an asyncio priority queue for consumption by the application.

    The client inherits connection lifecycle handling, handler hooks, and queue management from
    `FifoEventQueueConnectorAsyncClient`. It adds a convenience `connect()` constructor tailored to
    serial ports and emits explicit logging so operators are aware of the lack of transport-level
    encryption/authentication on typical serial links.
    """

    @classmethod
    async def connect(
        cls,
        port: str,
        *,
        baudrate: int = 115_200,
        out_queue: asyncio.PriorityQueue[FifoEvent] | None = None,
        handler: FifoEventQueueConnectorAsyncHandlerBase | None = None,
        **serial_kwargs: Any,
    ) -> FifoEventQueueSerialAsyncClient:
        """
        Open an asyncio serial connection and create a serial client instance.

        Args:
            port (str):
                Path or URL of the serial device (e.g., `/dev/ttyUSB0`, `COM3`, or a virtual port
                understood by `serial_asyncio`).

            baudrate (int, optional):
                Baud rate used when opening the serial connection. Defaults to `115_200`.

            out_queue (asyncio.PriorityQueue[FifoEvent] | None, optional):
                Optional priority queue that receives inbound events. When `None`, a new queue is
                created so the client remains self-contained.

            handler (FifoEventQueueConnectorAsyncHandlerBase | None, optional):
                Optional handler used to intercept incoming, outgoing, and sent events. The handler
                API matches the TCP implementation and is invoked in the same situations.

            **serial_kwargs:
                Additional keyword arguments forwarded to
                `serial_asyncio.open_serial_connection()` (e.g., `bytesize`, `parity`,
                `stopbits`). The `url` and `baudrate` parameters are reserved by this helper and
                must not be supplied here.

        Returns:
            FifoEventQueueSerialAsyncClient:
                Newly constructed serial client that wraps the opened serial connection.

        Raises:
            serial.SerialException: If the serial connection cannot be opened.
            ValueError: If invalid parameters are provided to the underlying serial implementation.
        """

        if "url" in serial_kwargs or "baudrate" in serial_kwargs:
            raise ValueError(
                "serial_kwargs must not contain 'url' or 'baudrate'; use the dedicated "
                "parameters instead."
            )

        logger.warning(
            "[serial-client] Opening serial connection on %s with baudrate %s",
            port,
            baudrate,
        )

        reader, writer = await open_serial_connection(
            url=port,
            baudrate=baudrate,
            **serial_kwargs,
        )

        return cls(reader, writer, out_queue, handler)
