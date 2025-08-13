from __future__ import annotations
import asyncio
import ssl
from dataclasses import dataclass, field
from typing import ClassVar

import pytest
import trustme

from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_queue_network import (
    FifoEventQueueNetworkAsyncClient,
    FifoEventQueueNetworkAsyncServer,
)
from fifo_dev_common.serialization.fifo_serialization import serializable

# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyEvent(FifoEvent):
    """Simple event used for round-trip testing."""
    event_id: ClassVar[int] = 5000
    value: int = field(metadata={"format": "i"})

    def __init__(self, value: int, priority: int = -1):
        super().__init__(priority=priority)
        self.value = value


@pytest.mark.asyncio
@pytest.mark.parametrize("use_tls", [False, True])
async def test_client_server_roundtrip(use_tls: bool, unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_ctx = client_ctx = None
    if use_tls:
        ca = trustme.CA()
        cert = ca.issue_cert("localhost")
        server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert.configure_cert(server_ctx)
        server_ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ca.configure_trust(client_ctx)
        client_ctx.minimum_version = ssl.TLSVersion.TLSv1_3

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(
            host, port, ssl_ctx=server_ctx
        )
    )
    await asyncio.sleep(0.01)

    client = await FifoEventQueueNetworkAsyncClient.connect(
        host,
        port,
        ssl_ctx=client_ctx,
        server_hostname="localhost" if use_tls else None,
    )
    server = await server_task

    await client.send(DummyEvent(value=1))
    recv = await server._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 1

    await server.send(DummyEvent(value=2))
    recv = await client._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 2

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)

    await asyncio.gather(client.join(), server.join())


@pytest.mark.asyncio
async def test_require_tls_without_context_raises(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port
    with pytest.raises(ValueError):
        await FifoEventQueueNetworkAsyncClient.connect(host, port, require_tls=True)
    with pytest.raises(ValueError):
        await FifoEventQueueNetworkAsyncServer.accept(host, port, require_tls=True)
