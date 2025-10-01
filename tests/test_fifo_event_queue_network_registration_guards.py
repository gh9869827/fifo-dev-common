from __future__ import annotations

from uuid import UUID
from dataclasses import dataclass
from typing import ClassVar
import pytest

from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventWithCID,
    FifoEventResultWithCID,
    FifoEventShutdown,
    EErrorCode,
)
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerCID,
)
from fifo_dev_common.serialization.fifo_serialization import serializable


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class _Req(FifoEventWithCID):
    event_id: ClassVar[int] = 64001


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class _Ack(FifoEventResultWithCID):
    event_id: ClassVar[int] = 64002

    def __init__(self, code: EErrorCode, correlation_id: UUID, priority: int = -1):
        super().__init__(code=code, correlation_id=correlation_id, priority=priority)


@FifoEvent.register
@serializable
@dataclass(kw_only=True)
class _NonCID(FifoEvent):
    event_id: ClassVar[int] = 64003


@pytest.mark.asyncio
async def test_duplicate_cid_template_registration_raises():
    h = FifoEventQueueConnectorAsyncHandlerCID()

    h.register_cid_template(_Req, [_Ack])
    with pytest.raises(ValueError):
        h.register_cid_template(_Req, [_Ack])

    # cleanup
    await h.process_incoming_event(FifoEventShutdown())
    await h.join()


@pytest.mark.asyncio
async def test_incoming_listener_guards_and_dedup():
    h = FifoEventQueueConnectorAsyncHandlerCID()

    async def cb(_e: FifoEvent):
        pass

    # Cannot register listener for CID class
    with pytest.raises(ValueError):
        h.register_incoming_listener(_Req, cb)

    # First registration ok
    h.register_incoming_listener(_NonCID, cb)

    # Duplicate (same callback + consume) raises
    with pytest.raises(ValueError):
        h.register_incoming_listener(_NonCID, cb)

    # Different consume flag is allowed
    h.register_incoming_listener(_NonCID, cb, consume=True)

    # cleanup
    await h.process_incoming_event(FifoEventShutdown())
    await h.join()
