from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventShutdown,
    FifoEventWithCID,
)
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerCID,
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
class _ListenerDummy(FifoEvent):
    event_id: ClassVar[int] = 60100
    value: int = field(metadata={"format": "i"})

    def __init__(self, value: int, priority: int = -1):
        super().__init__(priority=priority)
        self.value = value


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class _DummyCID(FifoEventWithCID):
    event_id: ClassVar[int] = 60101

    def __init__(self, priority: int = -1):
        super().__init__(priority=priority)


@pytest.mark.asyncio
async def test_incoming_listener_fires_for_non_cid():
    handler = FifoEventQueueConnectorAsyncHandlerCID()
    evt = asyncio.Event()
    seen: list[FifoEvent] = []

    async def on_event(ev: FifoEvent) -> None:
        seen.append(ev)
        evt.set()

    handler.register_incoming_listener(_ListenerDummy, on_event)

    # Process a non-CID event
    await handler.process_incoming_event(_ListenerDummy(value=123))
    await asyncio.wait_for(evt.wait(), 1.0)
    assert isinstance(seen[0], _ListenerDummy)

    # Clean shutdown
    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_incoming_listener_receives_shutdown():
    handler = FifoEventQueueConnectorAsyncHandlerCID()
    evt = asyncio.Event()

    async def on_shutdown(_ev: FifoEvent) -> None:
        evt.set()

    # Listen specifically for shutdown
    handler.register_incoming_listener(FifoEventShutdown, on_shutdown)

    await handler.process_incoming_event(FifoEventShutdown())
    await asyncio.wait_for(evt.wait(), 1.0)
    await handler.join()


@pytest.mark.asyncio
async def test_incoming_listener_does_not_fire_for_cid():
    handler = FifoEventQueueConnectorAsyncHandlerCID()
    evt = asyncio.Event()

    async def on_any(_ev: FifoEvent) -> None:
        evt.set()

    # Register a broad listener, but CID events should not trigger it
    handler.register_incoming_listener(FifoEvent, on_any)

    await handler.process_incoming_event(_DummyCID())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(evt.wait(), 0.05)

    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_incoming_listener_consume_suppresses_propagation():
    handler = FifoEventQueueConnectorAsyncHandlerCID()
    evt_called = asyncio.Event()

    async def on_event(_ev: FifoEvent) -> None:
        evt_called.set()

    # consuming listener
    handler.register_incoming_listener(_ListenerDummy, on_event, consume=True)

    # process event and verify handler returns None (consumed)
    result = await handler.process_incoming_event(_ListenerDummy(value=7))
    assert result is None
    await asyncio.wait_for(evt_called.wait(), 1.0)

    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()


@pytest.mark.asyncio
async def test_multiple_listeners_with_one_consumer_both_called_but_consumed():
    handler = FifoEventQueueConnectorAsyncHandlerCID()
    called = [False, False]

    async def l1(_ev: FifoEvent) -> None:
        called[0] = True

    async def l2(_ev: FifoEvent) -> None:
        called[1] = True

    handler.register_incoming_listener(_ListenerDummy, l1, consume=True)
    handler.register_incoming_listener(_ListenerDummy, l2, consume=False)

    result = await handler.process_incoming_event(_ListenerDummy(value=9))
    assert result is None  # consumed

    # Give dispatcher time to run callbacks
    await asyncio.sleep(0.01)
    assert all(called)

    await handler.process_incoming_event(FifoEventShutdown())
    await handler.join()
