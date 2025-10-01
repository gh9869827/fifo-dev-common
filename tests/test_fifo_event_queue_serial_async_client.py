from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fifo_dev_common.event.fifo_event import FifoEvent
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerBase,
)
from fifo_dev_common.event.fifo_event_queue_serial import (
    FifoEventQueueSerialAsyncClient,
)


# Accessing private members on purpose to verify internal state in tests
# pyright: reportPrivateUsage=false
# pylint: disable=protected-access


@pytest.mark.asyncio
async def test_connect_opens_serial_connection(monkeypatch: pytest.MonkeyPatch,
                                               caplog: pytest.LogCaptureFixture):
    """connect() should open the serial link and initialize the client."""

    fake_reader = MagicMock(spec=asyncio.StreamReader)
    fake_writer = MagicMock(spec=asyncio.StreamWriter)

    open_mock = AsyncMock(return_value=(fake_reader, fake_writer))
    monkeypatch.setattr(
        "fifo_dev_common.event.fifo_event_queue_serial.open_serial_connection",
        open_mock,
    )

    async def dummy_network_to_queue(_self: Any):  # pragma: no cover - trivial coroutine
        return None

    monkeypatch.setattr(
        FifoEventQueueSerialAsyncClient,
        "_network_to_queue",
        dummy_network_to_queue,
    )

    queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    handler = MagicMock(spec=FifoEventQueueConnectorAsyncHandlerBase)

    caplog.set_level("WARNING", logger="fifo_dev_common.event.fifo_event_queue_serial")

    client = await FifoEventQueueSerialAsyncClient.connect(
        "/dev/ttyUSB0",
        baudrate=57_600,
        out_queue=queue,
        handler=handler,
        bytesize=7,
    )

    open_mock.assert_awaited_once_with(url="/dev/ttyUSB0", baudrate=57_600, bytesize=7)
    assert isinstance(client, FifoEventQueueSerialAsyncClient)
    assert client._reader is fake_reader
    assert client._writer is fake_writer
    assert client._out_queue is queue
    assert client._handler is handler

    # ensure the insecurity warning is emitted so operators stay informed
    assert any(
        "Opening serial connection" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_connect_rejects_reserved_serial_kwargs():
    with pytest.raises(ValueError):
        await FifoEventQueueSerialAsyncClient.connect(
            "/dev/ttyUSB1",
            url="loop://",
        )
