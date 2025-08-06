from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
import time
from uuid import uuid4
import pytest
from fifo_dev_common.event.fifo_event import (
    FifoEvent,
    FifoEventShutdown,
    FifoEventException,
    FifoEventWithCID,
    FifoEventResultWithCID,
    EErrorCode,
)
from fifo_dev_common.process.utils import (
    FifoProcessManager,
    FifoAsyncProcessWorkerCallback,
    FifoSyncProcessWorkerCallback
)

# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

@pytest.fixture(autouse=True)
def ensure_fifo_event_exception_registered():
    if FifoEventException.event_id not in FifoEvent._registry:
        FifoEvent.register(FifoEventException)

if TYPE_CHECKING:
    from multiprocessing.queues import Queue  # pragma: nocover
else:
    from multiprocessing import Queue


class DemoFifoAsyncProcessWorkerCallback(FifoAsyncProcessWorkerCallback):
    def __init__(self, timeout: float = -1):
        self._timeout = timeout

    def initialize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        pass

    def finalize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
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

    def initialize(self, outgoing_queue: Queue[FifoEvent]):
        pass

    def finalize(self, outgoing_queue: Queue[FifoEvent]):
        pass

    def process_event(self,
                      incoming_event: FifoEvent,
                      incoming_queue_size: int,
                      outgoing_queue: Queue[FifoEvent]):
        outgoing_queue.put(incoming_event)

    def process_task(self, outgoing_queue: Queue[FifoEvent]):
        time.sleep(1)


class ErrorAsyncCallback(FifoAsyncProcessWorkerCallback):
    def initialize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        pass

    def finalize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        pass

    async def loop(self, incoming_event: FifoEvent | None, incoming_queue_size: int,
                   outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        raise RuntimeError("boom")

    def get_timeout(self) -> float:
        return -1


class ErrorSyncCallback(FifoSyncProcessWorkerCallback):
    def initialize(self, outgoing_queue: Queue[FifoEvent]):
        pass

    def finalize(self, outgoing_queue: Queue[FifoEvent]):
        pass

    def process_event(self, incoming_event: FifoEvent, incoming_queue_size: int,
                      outgoing_queue: Queue[FifoEvent]):
        raise RuntimeError("boom")

    def process_task(self, outgoing_queue: Queue[FifoEvent]):
        pass


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
async def test_async_process_manager_shutdown_event():
    loop = asyncio.get_event_loop()
    process = FifoProcessManager(loop, DemoFifoAsyncProcessWorkerCallback())
    process.start()

    await process.send(FifoEventShutdown())
    await process.stop()
    process.join()


@pytest.mark.asyncio
async def test_async_exception_event_propagation():
    loop = asyncio.get_event_loop()
    process = FifoProcessManager(loop, ErrorAsyncCallback())
    process.start()

    await process.send(FifoEvent())
    event = await process.receive()

    assert isinstance(event, FifoEventException)
    assert event.class_name == "RuntimeError"
    assert event.message == "boom"

    await process.stop()
    process.join()


def test_sync_exception_event_propagation():
    loop = asyncio.new_event_loop()
    process = FifoProcessManager(loop, ErrorSyncCallback())
    process.start()

    async def send_and_receive():
        await process.send(FifoEvent())
        event = await process.receive()
        await process.stop()
        process.join()
        return event

    event = loop.run_until_complete(send_and_receive())
    assert isinstance(event, FifoEventException)
    assert event.class_name == "RuntimeError"
    assert event.message == "boom"


@FifoEvent.register
class TestRequest(FifoEventWithCID):
    event_id = 1100
    default_priority = 10


@FifoEvent.register
class TestAck(FifoEventResultWithCID):
    event_id = 1101
    default_priority = 10


@FifoEvent.register
class TestDone(FifoEventResultWithCID):
    event_id = 1102
    default_priority = 10


@pytest.mark.asyncio
async def test_update_received_correlation_id_unmatched():
    loop = asyncio.get_event_loop()
    manager = FifoProcessManager(loop, DemoFifoSyncProcessWorkerCallback())
    event = TestAck(code=EErrorCode.OK, correlation_id=uuid4())

    await manager._update_received_correlation_id(event)
    queued = await manager._async_out.get()
    assert queued is event


@pytest.mark.asyncio
async def test_send_and_wait_response_ack_done():
    loop = asyncio.get_event_loop()
    manager = FifoProcessManager(loop, DemoFifoSyncProcessWorkerCallback())
    request = TestRequest()

    task = asyncio.create_task(
        manager.send_and_wait_response(request, TestAck, TestDone)
    )
    await asyncio.sleep(0)
    await manager._update_received_correlation_id(
        TestAck(code=EErrorCode.OK, correlation_id=request.correlation_id)
    )
    assert not task.done()
    await manager._update_received_correlation_id(
        TestDone(code=EErrorCode.OK, correlation_id=request.correlation_id)
    )
    result = await task
    assert isinstance(result, TestDone)
    assert manager._received_cid == {}


@pytest.mark.asyncio
async def test_send_and_wait_response_ack_error():
    loop = asyncio.get_event_loop()
    manager = FifoProcessManager(loop, DemoFifoSyncProcessWorkerCallback())
    request = TestRequest()

    task = asyncio.create_task(
        manager.send_and_wait_response(request, TestAck, TestDone)
    )
    await asyncio.sleep(0)
    err_ack = TestAck(code=EErrorCode.ERROR, correlation_id=request.correlation_id)
    await manager._update_received_correlation_id(err_ack)
    result = await task
    assert result is err_ack
    assert manager._received_cid == {}


class Worker(FifoSyncProcessWorkerCallback):
    def initialize(self, outgoing_queue: Queue[FifoEvent]):
        pass

    def finalize(self, outgoing_queue: Queue[FifoEvent]):
        pass

    def process_event(
        self,
        incoming_event: FifoEvent,
        incoming_queue_size: int,
        outgoing_queue: Queue[FifoEvent],
    ):
        if isinstance(incoming_event, TestRequest):
            outgoing_queue.put(
                TestAck(code=EErrorCode.OK, correlation_id=incoming_event.correlation_id)
            )
            outgoing_queue.put(
                TestDone(code=EErrorCode.OK, correlation_id=incoming_event.correlation_id)
            )
        else:
            outgoing_queue.put(incoming_event)

    def process_task(self, outgoing_queue: Queue[FifoEvent]):
        pass


@pytest.mark.asyncio
async def test_send_and_wait_response_end_to_end():
    """Verify send_and_wait_response works with a running process manager."""
    loop = asyncio.get_event_loop()

    manager = FifoProcessManager(loop, Worker())
    manager.start()

    done = await manager.send_and_wait_response(TestRequest(), TestAck, TestDone)
    assert isinstance(done, TestDone)
    assert manager._received_cid == {}

    await manager.stop()
    manager.join()
