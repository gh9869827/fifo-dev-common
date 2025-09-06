from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID

import pytest

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventWithCID,
    FifoEventResultWithCID,
    FifoEventShutdown,
    EErrorCode,
)
from fifo_dev_common.event.fifo_event_cid_outcome import FifoEventCIDOutcome
from fifo_dev_common.event.fifo_event_queue_network_handler import (
    FifoEventQueueNetworkAsyncHandlerCID,
)
from fifo_dev_common.serialization.fifo_serialization import serializable


# pylint: disable=protected-access
# pyright: reportPrivateUsage=false


@pytest.fixture(autouse=True)
def ensure_fifo_event_shutdown_registered():
    if FifoEventShutdown.event_id not in FifoEvent._registry:
        FifoEvent.register(FifoEventShutdown)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyCID(FifoEventWithCID):
    event_id: ClassVar[int] = 5201
    value: int = field(metadata={"format": "i"})

    def __init__(self, value: int, correlation_id: UUID | None = None, priority: int = -1):
        super().__init__(correlation_id=correlation_id, priority=priority)
        self.value = value


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyAck(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5202

    def __init__(self, code: EErrorCode, correlation_id: UUID, message: str | None = None, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, message=message, priority=priority)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class DummyDoneSuccess(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5203

    def __init__(self, code: EErrorCode, correlation_id: UUID, message: str | None = None, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, message=message, priority=priority)


@pytest.mark.asyncio
async def test_background_manager_success_and_failure():
    handler = FifoEventQueueNetworkAsyncHandlerCID()

    outcomes: list[tuple[FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID], FifoEventWithCID]] = []
    called = asyncio.Event()

    async def on_outcome(outcome: FifoEventCIDOutcome[FifoEvent, FifoEventResultWithCID],
                         req: FifoEventWithCID) -> None:
        outcomes.append((outcome, req))
        # Signal when we have at least one outcome; subsequent outcomes gathered later
        called.set()

    # register default on_outcome at template level
    handler.register_cid_template(DummyCID, [DummyAck, DummyDoneSuccess], on_outcome=on_outcome)

    # Minimal transport that only triggers template auto-registration
    class _Transport:
        def __init__(self, h: FifoEventQueueNetworkAsyncHandlerCID) -> None:
            self.h = h

        async def put(self, item: FifoEvent) -> None:
            await self.h.process_outgoing_event(item)

    transport = _Transport(handler)

    # Send two concurrent requests; both should call the same on_outcome
    req1 = DummyCID(value=1)
    req2 = DummyCID(value=2)
    await transport.put(req1)
    await transport.put(req2)

    # Wait until both CIDs are registered
    while any(req.correlation_id not in handler._registrations  # type: ignore[attr-defined]
              for req in (req1, req2)):
        await asyncio.sleep(0)

    # Drive one failure (ACK error) and one success (ACK ok + DONE)
    await handler.process_incoming_event(DummyAck(code=EErrorCode.ERROR, correlation_id=req1.correlation_id))
    await handler.process_incoming_event(DummyAck(code=EErrorCode.OK, correlation_id=req2.correlation_id))
    await handler.process_incoming_event(DummyDoneSuccess(code=EErrorCode.OK, correlation_id=req2.correlation_id))

    # Wait for at least one callback, then give time for the second
    await asyncio.wait_for(called.wait(), 1.0)
    for _ in range(10):
        if len(outcomes) >= 2:
            break
        await asyncio.sleep(0.01)

    # Validate we observed both a failure and a success, tied to req1/req2
    kinds = {(o.ok, isinstance(o.event, DummyDoneSuccess)) for o, _r in outcomes}
    assert (False, False) in kinds  # failure path
    assert (True, True) in kinds    # success path

    # Ensure the reqs we got in outcomes match one of the sent reqs
    req_ids = {r.correlation_id for _o, r in outcomes}
    assert req1.correlation_id in req_ids and req2.correlation_id in req_ids

    # Shutdown handler
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()
