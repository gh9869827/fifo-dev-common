from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
import time
import pytest
from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventPoison
from fifo_dev_common.process.utils import (
    FifoProcessManager,
    FifoAsyncProcessWorkerCallback,
    FifoSyncProcessWorkerCallback
)

if TYPE_CHECKING:
    from multiprocessing.queues import Queue  # pragma: nocover
else:
    from multiprocessing import Queue


class DemoFifoAsyncProcessWorkerCallback(FifoAsyncProcessWorkerCallback):
    def __init__(self, timeout: float = -1):
        self._timeout = timeout

    def initialize(self):
        pass

    def finalize(self):
        pass

    async def loop(self,
                   incoming_event: FifoEvent | None,
                   incoming_queue_size: int,
                   outgoing_queue: asyncio.PriorityQueue[FifoEvent]):

        if incoming_event is not None:
            await outgoing_queue.put(incoming_event)
        else:
            await asyncio.sleep(0.01)

    def get_timeout(self) -> float:
        return self._timeout


class DemoFifoSyncProcessWorkerCallback(FifoSyncProcessWorkerCallback):

    def initialize(self):
        pass

    def finalize(self):
        pass

    def process_event(self,
                      incoming_event: FifoEvent,
                      incoming_queue_size: int,
                      outgoing_queue: Queue[FifoEvent]):
        outgoing_queue.put(incoming_event)

    def process_task(self, outgoing_queue: Queue[FifoEvent]):
        time.sleep(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, 0, 0.5])
async def test_async_process_manager_echo(timeout: float):
    loop = asyncio.get_event_loop()
    process = FifoProcessManager(loop, DemoFifoAsyncProcessWorkerCallback(timeout=timeout))
    process.start()

    await process.send(FifoEvent(priority=2))
    await process.send(FifoEvent(priority=3))
    await process.send(FifoEvent(priority=1))

    received: list[int] = []
    for _ in range(3):
        event = await process.receive()
        received.append(event.priority)

    await asyncio.sleep(1)

    await process.stop()
    process.join()

    # Events should be echoed back in the order they were sent
    assert sorted(received) == [1, 2, 3]


def test_sync_process_manager_echo():
    # Run sync worker in a subprocess, communicate via queues
    loop = asyncio.new_event_loop()
    process = FifoProcessManager(loop, DemoFifoSyncProcessWorkerCallback())
    process.start()

    async def send_and_receive():
        await process.send(FifoEvent(priority=2))
        await process.send(FifoEvent(priority=3))
        await process.send(FifoEvent(priority=1))

        received: list[int] = []
        for _ in range(3):
            event = await process.receive()
            received.append(event.priority)

        await process.stop()
        process.join()
        return received

    received = loop.run_until_complete(send_and_receive())
    assert sorted(received) == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_process_manager_poison_event():
    loop = asyncio.get_event_loop()
    process = FifoProcessManager(loop, DemoFifoAsyncProcessWorkerCallback())
    process.start()

    await process.send(FifoEventPoison())
    await process.stop()
    process.join()
