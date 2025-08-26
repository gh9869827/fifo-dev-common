from __future__ import annotations
import asyncio
import ssl
from dataclasses import dataclass, field
from typing import ClassVar

import pytest
import trustme

from fifo_dev_common.event.fifo_event import (
    EErrorCode,
    FifoEvent,
    FifoEventResultWithCID,
    FifoEventShutdown,
    FifoEventWithCID,
)
from fifo_dev_common.event.fifo_event_queue_network import (
    FifoEventQueueNetworkAsyncClient,
    FifoEventQueueNetworkAsyncServer,
)
from fifo_dev_common.serialization.fifo_serialization import serializable

# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

@pytest.fixture(autouse=True)
def ensure_fifo_event_shutdown_registered():
    if FifoEventShutdown.event_id not in FifoEvent._registry:
        FifoEvent.register(FifoEventShutdown)

@pytest.fixture(autouse=True)
def ensure_fifo_event_dummy_registered():
    if DummyEvent.event_id not in FifoEvent._registry:
        FifoEvent.register(DummyEvent)


@pytest.fixture(autouse=True)
def ensure_fifo_event_cid_registered():
    if DummyCID.event_id not in FifoEvent._registry:
        FifoEvent.register(DummyCID)
    if DummyAck.event_id not in FifoEvent._registry:
        FifoEvent.register(DummyAck)
    if DummyAckFail.event_id not in FifoEvent._registry:
        FifoEvent.register(DummyAckFail)
    if DummyDoneSuccess.event_id not in FifoEvent._registry:
        FifoEvent.register(DummyDoneSuccess)
    if DummyDoneFailure.event_id not in FifoEvent._registry:
        FifoEvent.register(DummyDoneFailure)


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


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyCID(FifoEventWithCID):
    """Event carrying a correlation ID."""
    event_id: ClassVar[int] = 5001
    value: int = field(metadata={"format": "i"})

    def __init__(self, value: int, correlation_id=None, priority: int = -1):
        super().__init__(correlation_id=correlation_id, priority=priority)
        self.value = value


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyAck(FifoEventResultWithCID):
    """Acknowledgement event with correlation ID."""
    event_id: ClassVar[int] = 5002

    def __init__(self,
                 code: EErrorCode,
                 correlation_id: UUID,
                 message: str | None = None,
                 priority: int = -1):
        super().__init__(
            code=code, correlation_id=correlation_id, priority=priority, message=message
        )


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyAckFail(FifoEventResultWithCID):
    """Negative acknowledgement with correlation ID."""
    event_id: ClassVar[int] = 5003

    def __init__(self,
                 code: EErrorCode,
                 correlation_id: UUID,
                 message: str | None = None,
                 priority: int = -1):
        super().__init__(
            code=code, correlation_id=correlation_id, priority=priority, message=message
        )


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyDoneSuccess(FifoEventResultWithCID):
    """Completion event indicating success."""
    event_id: ClassVar[int] = 5004

    def __init__(self,
                 code: EErrorCode,
                 correlation_id: UUID,
                 message: str | None = None,
                 priority: int = -1):
        super().__init__(
            code=code, correlation_id=correlation_id, priority=priority, message=message
        )


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyDoneFailure(FifoEventResultWithCID):
    """Completion event indicating failure."""
    event_id: ClassVar[int] = 5005

    def __init__(self,
                 code: EErrorCode,
                 correlation_id: UUID,
                 message: str | None = None,
                 priority: int = -1):
        super().__init__(
            code=code, correlation_id=correlation_id, priority=priority, message=message
        )


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
        # Pylance: Type of "configure_cert" is partially unknown
        cert.configure_cert(server_ctx) # type: ignore[reportUnknownMemberType]
        server_ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        # Pylance: Type of "configure_trust" is partially unknown
        ca.configure_trust(client_ctx) # type: ignore[reportUnknownMemberType]
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

    await client.put(DummyEvent(value=1))
    recv = await server._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 1

    await client.send(DummyEvent(value=2))
    recv = await server._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 2

    await server.put(DummyEvent(value=3))
    recv = await client._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 3

    await server.send(DummyEvent(value=4))
    recv = await client._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 4

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


@pytest.mark.asyncio
async def test_cid_handler_consumes_event(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    client = await FifoEventQueueNetworkAsyncClient.connect(host, port)
    server = await server_task

    received: asyncio.Future[FifoEventResultWithCID] = asyncio.Future()

    req = DummyCID(value=5)

    async def _cb(ev: FifoEventResultWithCID) -> bool:
        received.set_result(ev)
        return True

    client.register(req, [DummyAck], _cb)

    await client.send(req)
    srv_req = await server._out_queue.get()
    await server.send(DummyAck(code=EErrorCode.OK, correlation_id=srv_req.correlation_id))

    ack = await asyncio.wait_for(received, 1.0)
    assert isinstance(ack, DummyAck)
    assert client._out_queue.empty()

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())


@pytest.mark.asyncio
async def test_cid_handler_chain_consumes_events(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    client = await FifoEventQueueNetworkAsyncClient.connect(host, port)
    server = await server_task

    ack_event = asyncio.Event()
    done_event = asyncio.Event()

    req = DummyCID(value=6)

    async def _cb(ev: FifoEventResultWithCID) -> bool:
        if isinstance(ev, DummyAck):
            ack_event.set()
            return True
        done_event.set()
        return False

    client.register(
        req,
        [[DummyAck], [DummyDoneSuccess, DummyDoneFailure]],
        _cb,
    )

    await client.send(req)
    srv_req = await server._out_queue.get()
    await server.send(DummyAck(code=EErrorCode.OK, correlation_id=srv_req.correlation_id))
    await asyncio.wait_for(ack_event.wait(), 1.0)
    await server.send(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=srv_req.correlation_id))
    await asyncio.wait_for(done_event.wait(), 1.0)
    assert client._out_queue.empty()

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())


@pytest.mark.asyncio
async def test_cid_handler_chain_stops_on_failure(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    client = await FifoEventQueueNetworkAsyncClient.connect(host, port)
    server = await server_task

    ack_event = asyncio.Event()

    req = DummyCID(value=7)

    async def _cb(ev: FifoEventResultWithCID) -> bool:
        if isinstance(ev, DummyAck):
            ack_event.set()
            return True
        if isinstance(ev, DummyAckFail):
            ack_event.set()
            return False
        return False

    client.register(
        req,
        [[DummyAck, DummyAckFail], [DummyDoneSuccess, DummyDoneFailure]],
        _cb,
    )

    await client.send(req)
    srv_req = await server._out_queue.get()
    await server.send(DummyAckFail(code=EErrorCode.ERROR, correlation_id=srv_req.correlation_id))
    await asyncio.wait_for(ack_event.wait(), 1.0)

    await server.send(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=srv_req.correlation_id))
    recv = await asyncio.wait_for(client._out_queue.get(), 1.0)
    assert isinstance(recv, DummyDoneSuccess)

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())
