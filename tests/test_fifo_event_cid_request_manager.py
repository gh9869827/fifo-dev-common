from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID, uuid4

import pytest

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventWithCID,
    FifoEventResultWithCID,
    FifoEventShutdown,
    EErrorCode,
)
from fifo_dev_common.event.fifo_event_cid_request_manager import (
    FifoEventCIDRequestManager,
    CIDOutcome,
)
from fifo_dev_common.event.fifo_event_queue_network_handler import (
    FifoEventQueueNetworkAsyncHandlerCID,
)
from fifo_dev_common.serialization.fifo_serialization import serializable


# pylint: disable=protected-access
# pyright: reportPrivateUsage=false


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class _TestSuccess(FifoEvent):
    """Minimal success event for CIDOutcome tests."""
    event_id: ClassVar[int] = 65001


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class _TestFailure(FifoEventResultWithCID):
    """Minimal failure event for CIDOutcome tests."""
    event_id: ClassVar[int] = 65002


def test_cid_outcome_success_getter_and_assertion():
    ev = _TestSuccess()
    outcome: CIDOutcome[_TestSuccess, _TestFailure] = CIDOutcome(True, ev)

    # success() returns the event as the success type
    got = outcome.success()
    assert got is ev
    assert isinstance(got, _TestSuccess)

    # failure() asserts when called on a success outcome
    with pytest.raises(AssertionError):
        _ = outcome.failure()


def test_cid_outcome_failure_getter_and_assertion():
    ev = _TestFailure(code=EErrorCode.ERROR, correlation_id=uuid4())
    outcome: CIDOutcome[_TestSuccess, _TestFailure] = CIDOutcome(False, ev)

    # failure() returns the event as the failure type
    got = outcome.failure()
    assert got is ev
    assert isinstance(got, _TestFailure)

    # success() asserts when called on a failure outcome
    with pytest.raises(AssertionError):
        _ = outcome.success()


@pytest.fixture(autouse=True)
def ensure_fifo_event_shutdown_registered():
    if FifoEventShutdown.event_id not in FifoEvent._registry:
        FifoEvent.register(FifoEventShutdown)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyCID(FifoEventWithCID):
    event_id: ClassVar[int] = 5101
    value: int = field(metadata={"format": "i"})

    def __init__(self, value: int, correlation_id: UUID | None = None, priority: int = -1):
        super().__init__(correlation_id=correlation_id, priority=priority)
        self.value = value


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyAck(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5102

    def __init__(self, code: EErrorCode, correlation_id: UUID, message: str | None = None, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, message=message, priority=priority)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyDoneSuccess(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5103

    def __init__(self, code: EErrorCode, correlation_id: UUID, message: str | None = None, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, message=message, priority=priority)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyDoneFailure(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5104

    def __init__(self, code: EErrorCode, correlation_id: UUID, message: str | None = None, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, message=message, priority=priority)


class _DummyTransport:
    """Minimal transport that only runs the handler's outgoing hook.

    This avoids using real sockets in tests while still exercising template
    auto-registration on the send path.
    """

    def __init__(self, handler: FifoEventQueueNetworkAsyncHandlerCID) -> None:
        self._handler = handler

    async def put(self, item: FifoEvent) -> None:  # SupportsFifoEventPut
        await self._handler.process_outgoing_event(item)


@pytest.mark.asyncio
async def test_cid_request_manager_success_flow():
    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRequestManager(asyncio.get_event_loop(), handler)
    mgr.register(DummyCID, [DummyAck, [DummyDoneSuccess, DummyDoneFailure]])

    transport = _DummyTransport(handler)

    req = DummyCID(value=1)
    task = asyncio.create_task(mgr.send_and_wait(transport, req, timeout=1.0))
    # Ensure the handler has registered this CID before injecting responses
    while req.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)

    # Simulate incoming ACK (success) then final DONE success
    await handler.process_incoming_event(DummyAck(code=EErrorCode.OK, correlation_id=req.correlation_id))
    await handler.process_incoming_event(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=req.correlation_id))

    outcome = await asyncio.wait_for(task, 1.0)
    assert isinstance(outcome, CIDOutcome)
    assert outcome.ok is True
    assert isinstance(outcome.event, DummyDoneSuccess)

    # Shutdown handler loop
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_cid_request_manager_failure_on_ack():
    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRequestManager(asyncio.get_event_loop(), handler)
    mgr.register(DummyCID, [DummyAck, [DummyDoneSuccess, DummyDoneFailure]])

    transport = _DummyTransport(handler)

    req = DummyCID(value=2)
    task = asyncio.create_task(mgr.send_and_wait(transport, req, timeout=1.0))
    # Ensure the handler has registered this CID before injecting responses
    while req.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)

    # Failure at ACK stage should resolve immediately with ok=False
    await handler.process_incoming_event(DummyAck(code=EErrorCode.ERROR, correlation_id=req.correlation_id))

    outcome = await asyncio.wait_for(task, 1.0)
    assert outcome.ok is False
    assert isinstance(outcome.event, DummyAck)

    # Shutdown handler loop
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_cid_request_manager_timeout():
    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRequestManager(asyncio.get_event_loop(), handler)
    mgr.register(DummyCID, [DummyAck, [DummyDoneSuccess, DummyDoneFailure]])

    transport = _DummyTransport(handler)

    req = DummyCID(value=3)

    with pytest.raises(asyncio.TimeoutError):
        await mgr.send_and_wait(transport, req, timeout=0.05)

    # Shutdown handler loop
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_send_in_background_success_calls_callback():
    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRequestManager(asyncio.get_event_loop(), handler)
    mgr.register(DummyCID, [DummyAck, [DummyDoneSuccess, DummyDoneFailure]])

    transport = _DummyTransport(handler)

    called = asyncio.Event()
    seen: list[CIDOutcome[FifoEvent, FifoEventResultWithCID]] = []

    async def on_outcome(outcome: CIDOutcome[FifoEvent, FifoEventResultWithCID], _req: FifoEventWithCID) -> None:
        seen.append(outcome)
        called.set()

    req = DummyCID(value=10)
    task = mgr.send_in_background(transport, req, on_outcome, timeout=1.0)
    assert isinstance(task, asyncio.Task)

    # lock should become held shortly after scheduling
    # spin until handler registers CID
    while req.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)

    # Simulate successful ACK then DONE success
    await handler.process_incoming_event(DummyAck(code=EErrorCode.OK, correlation_id=req.correlation_id))
    await handler.process_incoming_event(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=req.correlation_id))

    await asyncio.wait_for(called.wait(), 1.0)
    assert seen and seen[0].ok is True
    assert isinstance(seen[0].event, DummyDoneSuccess)

    # yield once to let done-callback run
    await asyncio.sleep(0)

    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_send_in_background_failure_calls_callback():
    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRequestManager(asyncio.get_event_loop(), handler)
    mgr.register(DummyCID, [DummyAck, [DummyDoneSuccess, DummyDoneFailure]])

    transport = _DummyTransport(handler)

    called = asyncio.Event()
    seen: list[CIDOutcome[FifoEvent, FifoEventResultWithCID]] = []

    async def on_outcome(outcome: CIDOutcome[FifoEvent, FifoEventResultWithCID], _req: FifoEventWithCID) -> None:
        seen.append(outcome)
        called.set()

    req = DummyCID(value=12)
    task2 = mgr.send_in_background(transport, req, on_outcome, timeout=1.0)
    assert isinstance(task2, asyncio.Task)

    while req.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)

    # Simulate failure at ACK stage
    await handler.process_incoming_event(DummyAck(code=EErrorCode.ERROR, correlation_id=req.correlation_id))

    await asyncio.wait_for(called.wait(), 1.0)
    assert seen and seen[0].ok is False
    assert isinstance(seen[0].event, DummyAck)

    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_send_in_background_logs_on_callback_exception(caplog: pytest.LogCaptureFixture):
    import logging

    caplog.set_level(logging.ERROR, logger="fifo_dev_common.event.fifo_event_cid_request_manager")

    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRequestManager(asyncio.get_event_loop(), handler)
    mgr.register(DummyCID, [DummyAck, [DummyDoneSuccess, DummyDoneFailure]])

    transport = _DummyTransport(handler)

    async def on_outcome_raises(_outcome: CIDOutcome[FifoEvent, FifoEventResultWithCID], _req: FifoEventWithCID) -> None:
        raise RuntimeError("boom")

    req = DummyCID(value=13)
    task3 = mgr.send_in_background(transport, req, on_outcome_raises, timeout=1.0)
    assert isinstance(task3, asyncio.Task)

    while req.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)
    await handler.process_incoming_event(DummyAck(code=EErrorCode.OK, correlation_id=req.correlation_id))
    await handler.process_incoming_event(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=req.correlation_id))

    # Allow task to finish and log
    await asyncio.sleep(0.05)

    # Shut down the handler loop cleanly
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()

    assert any("CID background task failed" in rec.getMessage() for rec in caplog.records)
