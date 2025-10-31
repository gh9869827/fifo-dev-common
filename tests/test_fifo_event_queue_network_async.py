from __future__ import annotations
import asyncio
import ssl
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID

import pytest
import trustme

from fifo_dev_common.event.fifo_event import (
    EErrorCode,
    FifoEvent,
    FifoEventResultWithCID,
    FifoEventShutdown,
    FifoEventWithCID,
)
from fifo_dev_common.event.fifo_event_cid_outcome import FifoEventCIDOutcome
from fifo_dev_common.event.fifo_event_queue_network import (
    FifoEventQueueNetworkAsyncClient,
    FifoEventQueueNetworkAsyncServer,
    FifoEventQueueNetworkAsyncHub,
    FifoEventQueueNetworkAsyncHubClientContext,
)
from fifo_dev_common.event.fifo_event_queue_serial import (
    FifoEventQueueSerialAsyncClient,
)
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerCID,
)
from fifo_dev_common.serialization.fifo_serialization import serializable
from fifo_dev_common.event.fifo_event_protocols import SendStatus

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

    def __init__(self, value: int, correlation_id: UUID | None=None, priority: int = -1):
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

    assert await client.send(DummyEvent(value=2)) is SendStatus.SENT
    recv = await server._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 2

    await server.put(DummyEvent(value=3))
    recv = await client._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 3

    assert await server.send(DummyEvent(value=4)) is SendStatus.SENT
    recv = await client._out_queue.get()
    assert isinstance(recv, DummyEvent)
    assert recv.value == 4

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)

    await asyncio.gather(client.join(), server.join())


@pytest.mark.asyncio
async def test_hub_handles_multiple_clients(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    events: asyncio.Queue[tuple[int, int, int]] = asyncio.Queue()
    broadcasts: asyncio.Queue[dict[int, SendStatus]] = asyncio.Queue()

    async def on_event(event: DummyEvent,
                       context: FifoEventQueueNetworkAsyncHubClientContext) -> None:
        assert context.is_active
        count = context.data.get("count", 0) + 1
        context.data["count"] = count
        await events.put((context.client_id, event.value, count))

        if event.value == 10:
            await context.send(DummyEvent(value=event.value + 1))
        if event.value == 20:
            statuses = await context.broadcast(DummyEvent(value=event.value + 2))
            await broadcasts.put(statuses)

    hub = await FifoEventQueueNetworkAsyncHub.listen(host, port, on_event)

    client1 = await FifoEventQueueNetworkAsyncClient.connect(host, port)
    client2 = await FifoEventQueueNetworkAsyncClient.connect(host, port)

    await client1.put(DummyEvent(value=1))
    await client2.put(DummyEvent(value=2))
    await client1.put(DummyEvent(value=10))
    await client2.put(DummyEvent(value=20))

    received: list[tuple[int, int, int]] = []
    for _ in range(4):
        received.append(await asyncio.wait_for(events.get(), timeout=2.0))

    assert (1, 1, 1) in received
    assert (2, 2, 1) in received
    assert (1, 10, 2) in received
    assert (2, 20, 2) in received

    reply = await asyncio.wait_for(client1._out_queue.get(), timeout=1.0)
    assert isinstance(reply, DummyEvent)
    assert reply.value == 11

    broadcast_status = await asyncio.wait_for(broadcasts.get(), timeout=1.0)
    assert broadcast_status == {1: SendStatus.SENT, 2: SendStatus.SENT}

    broadcast_reply1 = await asyncio.wait_for(client1._out_queue.get(), timeout=1.0)
    broadcast_reply2 = await asyncio.wait_for(client2._out_queue.get(), timeout=1.0)
    assert isinstance(broadcast_reply1, DummyEvent)
    assert isinstance(broadcast_reply2, DummyEvent)
    assert broadcast_reply1.value == 22
    assert broadcast_reply2.value == 22

    await hub.stop()

    shutdown1 = await asyncio.wait_for(client1._out_queue.get(), timeout=1.0)
    shutdown2 = await asyncio.wait_for(client2._out_queue.get(), timeout=1.0)
    assert isinstance(shutdown1, FifoEventShutdown)
    assert isinstance(shutdown2, FifoEventShutdown)

    await hub.join()
    await asyncio.gather(client1.join(), client2.join())


@pytest.mark.asyncio
async def test_serial_client_uses_connector_base(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    reader, writer = await asyncio.open_connection(host, port)
    client = FifoEventQueueSerialAsyncClient(reader, writer, None)
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
    await asyncio.gather(client.join(), server.join())


@pytest.mark.asyncio
async def test_ensure_ssl_ctx_without_context_raises(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port
    with pytest.raises(ValueError):
        await FifoEventQueueNetworkAsyncClient.connect(host, port, ensure_ssl_ctx=True)
    with pytest.raises(ValueError):
        await FifoEventQueueNetworkAsyncServer.accept(host, port, ensure_ssl_ctx=True)


@pytest.mark.asyncio
async def test_cid_handler_consumes_event(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    handler = FifoEventQueueConnectorAsyncHandlerCID()
    client = await FifoEventQueueNetworkAsyncClient.connect(host, port, handler=handler)
    server = await server_task

    received: asyncio.Future[FifoEvent] = asyncio.Future()

    async def on_outcome(outcome: FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID], _req: FifoEventWithCID) -> None:
        received.set_result(outcome.event)

    # Register template for DummyCID events
    handler.register_cid_template(DummyCID, [DummyAck], on_outcome=on_outcome)

    req = DummyCID(value=5)
    assert await client.send(req) is SendStatus.SENT
    srv_req = await server._out_queue.get()
    assert isinstance(srv_req, FifoEventWithCID)
    assert await server.send(DummyAck(code=EErrorCode.OK, correlation_id=srv_req.correlation_id)) is SendStatus.SENT

    ack = await asyncio.wait_for(received, 1.0)
    assert isinstance(ack, DummyAck)
    assert client._out_queue.empty()

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())
    await handler.join()


@pytest.mark.asyncio
async def test_cid_handler_on_send_callback_invoked(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    handler = FifoEventQueueConnectorAsyncHandlerCID()
    client = await FifoEventQueueNetworkAsyncClient.connect(host, port, handler=handler)
    server = await server_task

    send_called = asyncio.Event()
    seen: list[FifoEvent] = []

    async def on_send(ev: FifoEvent) -> None:
        seen.append(ev)
        send_called.set()

    # Register template with on_send callback
    handler.register_cid_template(
        DummyCID,
        [DummyAck],
        on_send=on_send,
    )

    req = DummyCID(value=42)
    assert await client.send(req) is SendStatus.SENT

    # Ensure the event is sent over the wire
    srv_req = await asyncio.wait_for(server._out_queue.get(), 1.0)
    assert isinstance(srv_req, FifoEventWithCID)

    # Verify on_send callback was invoked with the same event instance
    await asyncio.wait_for(send_called.wait(), 1.0)
    assert seen and seen[0] is req

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())
    await handler.join()


@pytest.mark.asyncio
async def test_cid_handler_on_sent_callback_invoked(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    handler = FifoEventQueueConnectorAsyncHandlerCID()
    client = await FifoEventQueueNetworkAsyncClient.connect(host, port, handler=handler)
    server = await server_task

    sent_called = asyncio.Event()
    seen: list[FifoEvent] = []

    async def on_sent(ev: FifoEvent) -> None:
        seen.append(ev)
        sent_called.set()

    # Register template with on_sent callback
    handler.register_cid_template(
        DummyCID,
        [DummyAck],
        on_sent=on_sent,
    )

    req = DummyCID(value=99)
    assert await client.send(req) is SendStatus.SENT

    # Ensure the event is sent over the wire
    srv_req = await asyncio.wait_for(server._out_queue.get(), 1.0)
    assert isinstance(srv_req, FifoEventWithCID)

    # Verify on_sent callback was invoked with the same event instance
    await asyncio.wait_for(sent_called.wait(), 1.0)
    assert seen and seen[0] is req

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())
    await handler.join()


@pytest.mark.asyncio
async def test_handler_logs_on_sent_callback_failure(caplog: pytest.LogCaptureFixture):
    import logging

    # Capture error logs from the handler module
    caplog.set_level(logging.ERROR, logger="fifo_dev_common.event.fifo_event_queue_connector_handler")

    handler = FifoEventQueueConnectorAsyncHandlerCID()

    async def on_sent_raises(_ev: FifoEvent) -> None:
        # Raise a whitelisted exception to trigger the error log path
        raise TypeError("boom")

    # Register a template with an on_sent callback that raises
    handler.register_cid_template(
        DummyCID,
        [DummyAck],
        on_sent=on_sent_raises,
    )

    # Trigger the on_sent path: process_outgoing first, then sent for same instance
    ev = DummyCID(value=1)
    await handler.process_outgoing_event(ev)
    await handler.process_sent_event(ev)

    # Give the handler loop a moment to process the queued callback
    await asyncio.sleep(0.02)

    # Shut down the handler loop cleanly
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()

    # Validate that the failure was logged as expected
    assert any("handler callback failed" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_handler_logs_on_callback_failure(caplog: pytest.LogCaptureFixture):
    import logging

    # Capture error logs from the handler module
    caplog.set_level(logging.ERROR, logger="fifo_dev_common.event.fifo_event_queue_connector_handler")

    handler = FifoEventQueueConnectorAsyncHandlerCID()

    async def on_send_raises(_ev: FifoEvent) -> None:
        # Raise a whitelisted exception to trigger the error log path
        raise TypeError("boom")

    # Register a template with an on_send callback that raises
    handler.register_cid_template(
        DummyCID,
        [DummyAck],
        on_send=on_send_raises,
    )

    # Trigger the on_send path without needing a network connection
    await handler.process_outgoing_event(DummyCID(value=1))

    # Give the handler loop a moment to process the queued callback
    await asyncio.sleep(0.02)

    # Shut down the handler loop cleanly
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()

    # Validate that the failure was logged as expected
    assert any("handler callback failed" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_cid_handler_chain_consumes_events(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    handler = FifoEventQueueConnectorAsyncHandlerCID()
    client = await FifoEventQueueNetworkAsyncClient.connect(host, port, handler=handler)
    server = await server_task

    ack_event = asyncio.Event()
    done_event = asyncio.Event()

    async def on_outcome(outcome: FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID], _req: FifoEventWithCID) -> None:
        if isinstance(outcome.event, DummyAck):
            ack_event.set()
        elif isinstance(outcome.event, DummyDoneSuccess):
            done_event.set()

    # Register template for DummyCID events with two-stage response
    handler.register_cid_template(
        DummyCID,
        [DummyAck, [DummyDoneSuccess, DummyDoneFailure]],
        on_outcome=on_outcome,
    )

    req = DummyCID(value=6)
    assert await client.send(req) is SendStatus.SENT
    srv_req = await server._out_queue.get()
    assert isinstance(srv_req, FifoEventWithCID)
    assert await server.send(DummyAck(code=EErrorCode.OK, correlation_id=srv_req.correlation_id)) is SendStatus.SENT
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(ack_event.wait(), 1.0)
    assert await server.send(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=srv_req.correlation_id)) is SendStatus.SENT
    await asyncio.wait_for(done_event.wait(), 1.0)
    assert client._out_queue.empty()

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())
    await handler.join()


@pytest.mark.asyncio
async def test_cid_handler_chain_stops_on_failure(unused_tcp_port: int):
    host = "127.0.0.1"
    port = unused_tcp_port

    server_task = asyncio.create_task(
        FifoEventQueueNetworkAsyncServer.accept(host, port)
    )
    await asyncio.sleep(0.01)

    handler = FifoEventQueueConnectorAsyncHandlerCID()
    client = await FifoEventQueueNetworkAsyncClient.connect(host, port, handler=handler)
    server = await server_task

    failure_event = asyncio.Event()

    async def on_outcome(outcome: FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID], _req: FifoEventWithCID) -> None:
        if not outcome.ok:
            failure_event.set()

    # Register template for DummyCID events
    handler.register_cid_template(
        DummyCID,
        [DummyAck, [DummyDoneSuccess, DummyDoneFailure]],
        on_outcome=on_outcome,
    )

    req = DummyCID(value=7)
    assert await client.send(req) is SendStatus.SENT
    srv_req = await server._out_queue.get()
    assert isinstance(srv_req, FifoEventWithCID)

    # Send a failure ACK - this should trigger on_failure and stop the chain
    assert await server.send(DummyAck(code=EErrorCode.ERROR, correlation_id=srv_req.correlation_id)) is SendStatus.SENT
    await asyncio.wait_for(failure_event.wait(), 1.0)

    # Send a subsequent success event - this should go to the client queue since the
    # handler chain stopped
    assert await server.send(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=srv_req.correlation_id)) is SendStatus.SENT
    recv = await asyncio.wait_for(client._out_queue.get(), 1.0)
    assert isinstance(recv, DummyDoneSuccess)

    await client.stop()
    await server.stop()
    assert isinstance(await server._out_queue.get(), FifoEventShutdown)
    assert isinstance(await client._out_queue.get(), FifoEventShutdown)
    await asyncio.gather(client.join(), server.join())
    await handler.join()
