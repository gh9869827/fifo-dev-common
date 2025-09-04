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
from fifo_dev_common.event.fifo_event_cid_request_manager import (
    FifoEventCIDRefreshManager,
)
from fifo_dev_common.event.fifo_event_queue_network_handler import (
    FifoEventQueueNetworkAsyncHandlerCID,
)
from fifo_dev_common.state.fifo_refreshable_value import FifoRefreshableValue, CacheState
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
class Rq(FifoEventWithCID):
    event_id: ClassVar[int] = 5301

    def __init__(self, correlation_id: UUID | None = None, priority: int = -1):
        super().__init__(correlation_id=correlation_id, priority=priority)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class Ack(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5302

    def __init__(self, code: EErrorCode, correlation_id: UUID, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, priority=priority)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class Done(FifoEventResultWithCID):
    event_id: ClassVar[int] = 5303
    value: int = field(metadata={"format": "i"})

    def __init__(self, code: EErrorCode, correlation_id: UUID, value: int, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, priority=priority)
        self.value = value


@pytest.mark.asyncio
async def test_refresh_manager_updates_cache():
    handler = FifoEventQueueNetworkAsyncHandlerCID()
    mgr = FifoEventCIDRefreshManager(handler)

    cache: FifoRefreshableValue[int] = FifoRefreshableValue()

    # Register refresh pipeline: extract value from Done
    mgr.register(Rq, [Ack, Done], cache, extract=lambda ev: ev.value, success_types=(Done,)) # type: ignore

    # Minimal transport to trigger template auto-registration
    class _Transport:
        def __init__(self, h: FifoEventQueueNetworkAsyncHandlerCID) -> None:
            self.h = h

        async def put(self, item: FifoEvent) -> None:
            await self.h.process_outgoing_event(item)

    tr = _Transport(handler)

    req = Rq()
    await tr.put(req)

    # After send, wait for the async on_send callback to mark refreshing
    for _ in range(50):
        if cache.snapshot().state.name == CacheState.REFRESHING.name:
            break
        await asyncio.sleep(0)
    assert cache.snapshot().state.name == CacheState.REFRESHING.name

    # Wait until registration exists
    while req.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)

    # Deliver success path
    await handler.process_incoming_event(Ack(code=EErrorCode.OK, correlation_id=req.correlation_id))
    await handler.process_incoming_event(Done(code=EErrorCode.OK, correlation_id=req.correlation_id, value=7))

    # Cache should eventually be fresh with value 7
    for _ in range(50):
        snap = cache.snapshot()
        if snap.state.name == CacheState.FRESH.name and snap.value == 7:
            break
        await asyncio.sleep(0)
    snap = cache.snapshot()
    assert snap.state.name == CacheState.FRESH.name and snap.value == 7

    # Drive failure path updates ERROR while retaining last value
    req2 = Rq()
    await tr.put(req2)
    while req2.correlation_id not in handler._registrations:  # type: ignore[attr-defined]
        await asyncio.sleep(0)
    await handler.process_incoming_event(Ack(code=EErrorCode.ERROR, correlation_id=req2.correlation_id))
    # Wait for failure to be reflected
    for _ in range(50):
        snap2 = cache.snapshot()
        if snap2.state.name == CacheState.ERROR.name:
            break
        await asyncio.sleep(0)
    snap2 = cache.snapshot()
    assert snap2.state.name == CacheState.ERROR.name and snap2.value == 7

    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()
