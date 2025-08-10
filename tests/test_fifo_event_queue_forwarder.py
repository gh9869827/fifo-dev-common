from __future__ import annotations
import asyncio
import logging
import multiprocessing
import pytest
from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_queue_forwarder import FifoEventQueueForwarderMpToAsync
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from multiprocessing.queues import Queue as MpQueue
else:
    MpQueue = multiprocessing.Queue  # type: ignore[misc]


# pylint: disable=protected-access
# pyright: reportPrivateUsage=false


@pytest.mark.asyncio
async def test_forwarder_forwards_events_and_stops():
    mp_queue: MpQueue[FifoEvent] = MpQueue()
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    loop = asyncio.get_running_loop()
    forwarder = FifoEventQueueForwarderMpToAsync(mp_queue, async_queue, loop)

    forwarder.start()

    mp_queue.put(FifoEvent(priority=2))
    mp_queue.put(FifoEvent(priority=3))
    mp_queue.put(FifoEvent(priority=1))

    forwarder.stop()
    forwarder.join()

    received = [await async_queue.get() for _ in range(3)]
    assert [e.priority for e in received] == [1, 2, 3]
    assert all(not isinstance(e, FifoEventShutdown) for e in received)
    assert forwarder.stopped()


@pytest.mark.asyncio
async def test_forwarder_forward_shutdown_event_when_configured():
    mp_queue: MpQueue[FifoEvent] = MpQueue()
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    loop = asyncio.get_running_loop()
    forwarder = FifoEventQueueForwarderMpToAsync(
        mp_queue, async_queue, loop, forward_shutdown_event=True
    )

    forwarder.start()

    mp_queue.put(FifoEventShutdown())

    forwarder.join()

    event = await async_queue.get()
    assert isinstance(event, FifoEventShutdown)
    assert forwarder.stopped()


def test_forwarder_start_requires_running_loop():
    mp_queue: MpQueue[FifoEvent] = MpQueue()
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    loop = asyncio.new_event_loop()
    forwarder = FifoEventQueueForwarderMpToAsync(mp_queue, async_queue, loop)

    try:
        with pytest.raises(RuntimeError):
            forwarder.start()
    finally:
        loop.close()


def test_enqueue_async_drops_event_when_loop_closed(caplog: pytest.LogCaptureFixture):
    mp_queue: MpQueue[FifoEvent] = MpQueue()
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    loop = asyncio.new_event_loop()
    forwarder = FifoEventQueueForwarderMpToAsync(mp_queue, async_queue, loop)
    loop.close()

    with caplog.at_level(
        logging.ERROR, logger="fifo_dev_common.event.fifo_event_queue_forwarder"
    ):
        forwarder._enqueue_async("t", FifoEvent())
        assert any(
            "Event loop closed; dropping event" in record.message
            for record in caplog.records
        )

    assert async_queue.qsize() == 0


@pytest.mark.asyncio
async def test_start_warns_when_thread_alive(caplog: pytest.LogCaptureFixture):
    mp_queue: MpQueue[FifoEvent] = MpQueue()
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    loop = asyncio.get_running_loop()
    forwarder = FifoEventQueueForwarderMpToAsync(mp_queue, async_queue, loop)

    forwarder.start()
    with caplog.at_level(
        logging.WARNING, logger="fifo_dev_common.event.fifo_event_queue_forwarder"
    ):
        forwarder.start()
        assert any(
            "start() called but thread already started" in record.message
            for record in caplog.records
        )

    forwarder.stop()
    forwarder.join()


def test_join_warns_when_not_started(caplog: pytest.LogCaptureFixture):
    mp_queue: MpQueue[FifoEvent] = MpQueue()
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    loop = asyncio.new_event_loop()
    forwarder = FifoEventQueueForwarderMpToAsync(mp_queue, async_queue, loop)

    with caplog.at_level(
        logging.WARNING, logger="fifo_dev_common.event.fifo_event_queue_forwarder"
    ):
        forwarder.join()
        assert any(
            "join() called but thread never started" in record.message
            for record in caplog.records
        )

    loop.close()


def test_run_logs_error_when_queue_closed(caplog: pytest.LogCaptureFixture):
    class DummyQueue:
        def get(self) -> FifoEvent:
            raise EOFError()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    forwarder = FifoEventQueueForwarderMpToAsync(DummyQueue(), async_queue, loop) # type: ignore
    try:
        with caplog.at_level(
            logging.ERROR, logger="fifo_dev_common.event.fifo_event_queue_forwarder"
        ):
            forwarder._run()
            assert any(
                "Source queue closed (EOFError); stopping" in record.message
                and record.levelno == logging.ERROR
                for record in caplog.records
            )
        assert forwarder.stopped()
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_run_logs_warning_on_queue_closed_after_stop_requested(caplog: pytest.LogCaptureFixture):
    class DummyQueue:
        def get(self) -> FifoEvent:
            raise OSError()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    forwarder = FifoEventQueueForwarderMpToAsync(DummyQueue(), async_queue, loop) # type: ignore
    forwarder._stop_requested.set()
    try:
        with caplog.at_level(
            logging.WARNING, logger="fifo_dev_common.event.fifo_event_queue_forwarder"
        ):
            forwarder._run()
            assert any(
                "Source queue closed (OSError); stopping" in record.message
                and record.levelno == logging.WARNING
                for record in caplog.records
            )
        assert forwarder.stopped()
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_run_logs_unexpected_exception(caplog: pytest.LogCaptureFixture):
    class DummyQueue:
        def get(self) -> FifoEvent:
            raise ValueError()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_queue: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
    forwarder = FifoEventQueueForwarderMpToAsync(DummyQueue(), async_queue, loop) # type: ignore
    try:
        with caplog.at_level(
            logging.ERROR, logger="fifo_dev_common.event.fifo_event_queue_forwarder"
        ):
            forwarder._run()
            assert any(
                "Unexpected ValueError; stopping" in record.message
                and record.levelno == logging.ERROR
                for record in caplog.records
            )
        assert forwarder.stopped()
    finally:
        asyncio.set_event_loop(None)
        loop.close()
