from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import threading
import queue
from threading import Thread
from multiprocessing import Process
from typing import TYPE_CHECKING, Type
from uuid import UUID
from fifo_dev_common.event.fifo_event import (
    EErrorCode,
    FifoEventShutdown,
    FifoEvent,
    FifoEventException,
    FifoEventResultWithCID
)
from fifo_dev_common.event.fifo_event_protocols import SupportsFifoEventPut
from fifo_dev_common.logging.logger import get_logger


if TYPE_CHECKING:
    from multiprocessing.queues import Queue  # pragma: nocover
else:
    from multiprocessing import Queue


logger = get_logger(__name__)

_trace = logger.trace

class FifoProcessWorkerBase(ABC):
    """
    Abstract base class for workers that run in a separate process from the main application.
    Communication between the worker and the main process is handled via interprocess queues.

    This approach enables parallelism for CPU-bound tasks: each worker process can run on its
    own CPU core, avoiding contention with the main process caused by Python's Global Interpreter
    Lock (GIL).
    Note that a single worker process will only use up to one core, but multiple workers can be
    distributed across cores. Running workers in isolated processes also helps contain failures,
    minimizing their impact on the main process.

    Important:

        Each instance of this class (and its managers) handles a single worker process.
        To utilize multiple processes, create multiple instances.

    This class should be subclassed to implement either synchronous or asynchronous
    event processing logic. It provides a consistent interface for exchanging events
    between the main process and worker processes.

    Attributes:
        _in_queue (Queue[FifoEvent]):
            Interprocess queue for incoming events (from main process to worker process).

        _out_queue (Queue[FifoEvent]):
            Interprocess queue for outgoing events (from the worker process to the main process).

    Usage:
        - Subclass this base class to define custom event handling.
        - Use the provided queues to transfer FifoEvent objects between processes.
        - These queues are not priority queues; for event prioritization, use a local
          priority queue within the worker, fed by the interprocess queue.
    """
    _in_queue: Queue[FifoEvent]
    _out_queue: Queue[FifoEvent]

    def __init__(self, in_queue: Queue[FifoEvent], out_queue: Queue[FifoEvent]):
        """
        Initializes the utility class with input and output queues.

        Args:
            in_queue (Queue[FifoEvent]):
                Interprocess queue for incoming events (from main process to worker process).

            out_queue (Queue[FifoEvent]):
                Interprocess queue for outgoing events (from the worker process to the main
                process).
        """
        self._in_queue = in_queue
        self._out_queue = out_queue

    @abstractmethod
    def run_until_complete(self) -> None:
        """
        Abstract method to be implemented by subclasses.

        Should contain the main event processing loop for the worker process.
        """


class FifoAsyncProcessWorkerCallback(ABC):
    """
    Abstract base class for defining asynchronous event processing logic for a worker process.

    Subclass this and implement the `loop` method to handle incoming events and produce outgoing
    events. An instance of this callback is passed to FifoAsyncProcessWorker, which calls `loop`
    repeatedly within the worker process.

    The callback provides a flexible interface for custom event handling, allowing you to process
    events as they arrive and send results or responses back to the main process.

    Lifecycle hooks:
        - `initialize()` is called once before any event or task processing begins, allowing
          setup of resources, connections, or state needed by the worker.
        - `finalize()` is called once after all processing is complete, allowing cleanup of
          resources, connections, or state before the worker exits.

    Usage:
        - Implement the `loop` method to define how each event should be processed.
        - Use the provided outgoing queue to send events/results back to the main process.
        - The method is called repeatedly; when no event is available, `incoming_event` will be 
          `None`.  
          If `get_timeout()` returns `0`, the loop runs continuously and may busy-wait unless you
          include `await asyncio.sleep(...)` in your implementation.
    """

    @abstractmethod
    def initialize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        """
        Called once before any event or task processing begins.

        Use this method to set up resources, connections, or state needed by the worker.
        This method is invoked in the worker process before any threads are created or started,
        ensuring all initialization is complete before event or task processing begins.

        This method is called after the asyncio loop has been created.

        Args:
            outgoing_queue (asyncio.PriorityQueue[FifoEvent]):
                Queue for sending events/results back to the main process.
        """

    @abstractmethod
    def finalize(self, outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        """
        Called once after all event and task processing is complete.

        Use this method to clean up resources, connections, or state before the worker exits.
        This method is invoked in the worker process after all threads have finished and all
        event/task processing is complete, just before the process terminates.

        The asyncio loop remains active when this method is called.

        Args:
            outgoing_queue (asyncio.PriorityQueue[FifoEvent]):
                Queue for sending events/results back to the main process.
        """

    @abstractmethod
    async def loop(self,
                   incoming_event: FifoEvent | None,
                   incoming_queue_size: int,
                   outgoing_queue: asyncio.PriorityQueue[FifoEvent]):
        """
        Called repeatedly by the worker process to handle events.

        Args:
            incoming_event (FifoEvent | None):
                The next event from the input queue, or None if the queue is empty.

            incoming_queue_size (int):
                The current number of events waiting in the input queue.
                It is obtained by calling `qsize`, but is only an approximation since the queue
                size may change between the call and its use in the `loop` function.

            outgoing_queue (asyncio.PriorityQueue[FifoEvent]):
                Queue for sending events/results back to the main process.

        Notes:
            - If `incoming_event` is None, the queue was empty at the time of call.
            - Use `outgoing_queue.put()` to send events back to the main process.
            - This method should be implemented as a coroutine and may include
              `await` expressions for asynchronous operations.
        """

    @abstractmethod
    def get_timeout(self) -> float:
        """
        Return the timeout for queue.get():
            -1: wait forever
             0: non-blocking (nowait)
            >0: wait for up to timeout seconds

        Warning:
            When idle, if this method returns 0 (non-blocking), your `loop` implementation must
            include a sleep (e.g., `await asyncio.sleep(...)`) to avoid busy-waiting and wasting
            CPU cycles.
        """


class FifoAsyncProcessWorker(FifoProcessWorkerBase):
    """
    Asynchronous worker that runs in a separate process and handles event processing using asyncio.

    This class manages two internal asyncio priority queues for incoming and outgoing events.
    It launches threads to move events between interprocess queues and these internal queues,
    enabling asynchronous event handling within the worker process.

    The worker repeatedly calls the provided FifoAsyncProcessWorkerCallback's `loop` method,
    passing each event and the current queue size, allowing for custom asynchronous event
    processing.

    Internal threads:
        - _in_queue_puller: Moves events from the interprocess input queue
          to the internal async queue.
        - _out_queue_pusher: Moves events from the internal async output queue
          to the interprocess output queue.

    Usage:
        - Instantiate with input/output queues and a callback implementing
          FifoAsyncProcessWorkerCallback.
        - Call run_until_complete() to start processing events until a `FifoEventShutdown`
          is received.

    Attributes:

        _async_in (asyncio.PriorityQueue[FifoEvent]):
            Internal async queue for incoming events (from main process to worker process).

        _async_out (asyncio.PriorityQueue[FifoEvent]):
            Internal async queue for outgoing events (from the worker process to the main process).

        _loop (asyncio.AbstractEventLoop | None):
            The event loop used for async operations.

        _event_queue_pusher_done (asyncio.Event):
            Signals when the output pusher thread is done.

        _callback (FifoAsyncProcessWorkerCallback):
            The callback for event processing logic.
    """

    _async_in: asyncio.PriorityQueue[FifoEvent]
    _async_out: asyncio.PriorityQueue[FifoEvent]
    _loop: asyncio.AbstractEventLoop | None
    _event_queue_pusher_done: asyncio.Event
    _callback: FifoAsyncProcessWorkerCallback

    def __init__(self,
                 in_queue: Queue[FifoEvent],
                 out_queue: Queue[FifoEvent],
                 callback: FifoAsyncProcessWorkerCallback):
        """
        Initialize the asynchronous worker with input/output queues and a callback.

        Args:
            in_queue (Queue[FifoEvent]):
                Interprocess queue for incoming events (from main process to worker process).

            out_queue (Queue[FifoEvent]):
                Interprocess queue for outgoing events (from the worker process to the main
                process).

            callback (FifoAsyncProcessWorkerCallback):
                Callback for custom event processing.
        """
        super().__init__(in_queue, out_queue)
        self._async_in: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
        self._async_out: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
        self._loop = None
        self._event_queue_pusher_done = asyncio.Event()
        self._callback = callback

    def _in_queue_puller(self) -> None:
        """
        Thread target: Moves events from the interprocess input queue to the internal async queue.

        Stops when a FifoEventShutdown is received, which is forwarded to the internal async queue
        to cascade the shutdown process.
        """
        assert self._loop is not None

        _trace("[FifoAsyncProcessWorker.thread:_in_queue_puller] Thread running")
        while True:
            event = self._in_queue.get()
            asyncio.run_coroutine_threadsafe(self._async_in.put(event), self._loop)
            _trace("[FifoAsyncProcessWorker.thread:_in_queue_puller] Event pulled from main process")  # pylint: disable=line-too-long
            if isinstance(event, FifoEventShutdown):
                _trace("[FifoAsyncProcessWorker.thread:_in_queue_puller] Shutdown event received; stopping thread")  # pylint: disable=line-too-long
                break

    def _out_queue_pusher(self) -> None:
        """
        Thread target: Moves events from the internal async output queue to the interprocess
        output queue.

        Stops when a FifoEventShutdown is received, which is forwarded to the interprocess output
        queue to cascade the shutdown process.
        """
        assert self._loop is not None

        _trace("[FifoAsyncProcessWorker.thread:_out_queue_pusher] Thread running")
        while True:
            event = asyncio.run_coroutine_threadsafe(self._async_out.get(), self._loop).result()
            self._out_queue.put(event)
            _trace("[FifoAsyncProcessWorker.thread:_out_queue_pusher] Event sent to main process")
            if isinstance(event, FifoEventShutdown):
                _trace("[FifoAsyncProcessWorker.thread:_out_queue_pusher] Shutdown event sent; stopping thread")  # pylint: disable=line-too-long
                break

        self._loop.call_soon_threadsafe(self._event_queue_pusher_done.set)

    async def _process_loop(self) -> None:
        """
        Main asynchronous event loop.

        Continuously retrieves events from the internal async input queue and calls the callback's
        loop method. Stops when a FifoEventShutdown is received and waits for the output pusher
        thread to finish, signaling that there are no more asyncio operations being processed or
        pending.
        """
        _trace("[FifoAsyncProcessWorker.fct:_process_loop] Event loop running")
        while True:
            timeout = self._callback.get_timeout()
            if timeout == -1:
                event = await self._async_in.get()
            elif timeout == 0:
                try:
                    event = self._async_in.get_nowait()
                except asyncio.QueueEmpty:
                    event = None
            else:
                try:
                    event = await asyncio.wait_for(self._async_in.get(), timeout)
                except asyncio.TimeoutError:
                    event = None

            if isinstance(event, FifoEventShutdown):
                await self._async_out.put(event)
                _trace("[FifoAsyncProcessWorker.fct:_process_loop] Shutdown event processed; shutting down event loop")  # pylint: disable=line-too-long
                break

            try:
                await self._callback.loop(event, self._async_in.qsize(), self._async_out)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("[FifoAsyncProcessWorker.loop] Unhandled exception")
                await self._async_out.put(
                    FifoEventException(exception=e, source="FifoAsyncProcessWorker.loop")
                )

        # need to wait for the pusher thread to complete so that we exit the loop process
        # only when no more asyncio operation are processing or pending
        _trace("[FifoAsyncProcessWorker.fct:_process_loop] Waiting for _out_queue_pusher thread to signal completion")  # pylint: disable=line-too-long
        await self._event_queue_pusher_done.wait()
        _trace("[FifoAsyncProcessWorker.fct:_process_loop] Event loop stopped")

    def run_until_complete(self) -> None:
        """
        Starts the worker process event loop and associated threads.

        Runs until a Shutdown event is received, then joins threads and closes the event loop.
        """
        _trace("[FifoAsyncProcessWorker] Initializing async worker")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._callback.initialize(self._async_out)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[FifoAsyncProcessWorker.initialize] Unhandled exception")
            self._out_queue.put(
                FifoEventException(exception=e, source="FifoAsyncProcessWorker.initialize")
            )

        puller_thread = threading.Thread(target=self._in_queue_puller)
        pusher_thread = threading.Thread(target=self._out_queue_pusher)
        puller_thread.start()
        pusher_thread.start()
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] Worker threads started")
        self._loop.run_until_complete(self._process_loop())
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] Event loop run complete")
        puller_thread.join()
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] _in_queue_puller thread joined")
        pusher_thread.join()
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] _out_queue_pusher thread joined")

        try:
            self._callback.finalize(self._async_out)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[FifoAsyncProcessWorker.finalize] Unhandled exception")
            self._out_queue.put(
                FifoEventException(exception=e, source="FifoAsyncProcessWorker.finalize")
            )
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] Callback finalized")
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] Async worker shutdown complete")

        self._loop.close()
        _trace("[FifoAsyncProcessWorker.fct:run_until_complete] Event loop closed")
        self._loop = None


class FifoSyncProcessWorkerCallback(ABC):
    """
    Abstract base class for defining synchronous event and task processing logic for a worker
    process.

    Subclass this and implement both `process_event` and `process_task` methods to handle incoming
    events and perform periodic tasks. An instance of this callback is passed to
    FifoSyncProcessWorker, which runs each method in its own dedicated thread within the worker
    process.

    Threading model:
        - The event processing thread waits for incoming events and calls `process_event` whenever
          an event arrives. This thread blocks until an event is available.
        - The task processing thread runs `process_task` continuously. It must process a single task
          and return promptly, so that the thread can be interrupted and exit when a shutdown event
          is received.

    Lifecycle hooks:
        - `initialize()` is called once before any event or task processing begins, allowing
          setup of resources, connections, or state needed by the worker.
        - `finalize()` is called once after all processing is complete, allowing cleanup of
          resources, connections, or state before the worker exits.

    Usage:
        - Implement `process_event` to define how each incoming event should be handled.
        - Implement `process_task` to perform periodic or background tasks.
        - Both methods receive the outgoing queue to send results/events back to the main process.
        - There is no timeout mechanism, as each callback runs in its own thread and blocking is
          handled by the thread itself.
    """

    @abstractmethod
    def initialize(self, outgoing_queue: Queue[FifoEvent]):
        """
        Called once before event and task processing begins.

        Use this method to set up resources, connections, or state needed by the worker.
        This method is invoked in the worker process before any events or tasks are processed.

        Args:
            outgoing_queue (Queue[FifoEvent]):
                Queue for sending events/results back to the main process.
        """

    @abstractmethod
    def finalize(self, outgoing_queue: Queue[FifoEvent]):
        """
        Called once after all event and task processing is complete.

        Use this method to clean up resources, connections, or state before the worker exits.
        This method is invoked in the worker process after all threads have finished.

        Args:
            outgoing_queue (Queue[FifoEvent]):
                Queue for sending events/results back to the main process.
        """

    @abstractmethod
    def process_event(self,
                      incoming_event: FifoEvent,
                      incoming_queue_size: int,
                      outgoing_queue: Queue[FifoEvent]):
        """
        Called by the event processing thread for each event received from the priority queue.

        Should process the event and optionally put results in the outgoing queue.

        Args:
            incoming_event (FifoEvent):
                The event received from the main process, delivered to this method via the
                worker's local priority queue.

            incoming_queue_size (int):
                The current number of events waiting in the priority queue.
                It is obtained by calling `qsize`, but is only an approximation since the queue
                size may change between the call and its use in the `process_event` function.

            outgoing_queue (Queue[FifoEvent]):
                Queue for sending events/results back to the main process.
        """

    @abstractmethod
    def process_task(self, outgoing_queue: Queue[FifoEvent]):
        """
        Called repeatedly by the task processing thread to perform one unit of work.

        Should process a single task and return promptly, so the thread can exit when
        a shutdown event is received.

        Args:
            outgoing_queue (Queue[FifoEvent]):
                Queue for sending events/results back to the main process.
        """


class FifoSyncProcessWorker:
    """
    Synchronous worker that runs in a separate process and manages event and task processing
    using threads.

    More precisely, this worker launches three threads:
        - One thread reads events from the interprocess input queue and puts them into a local
          thread-safe priority queue.
        - One thread processes events from the local priority queue by calling the callback's
          `process_event` method.
        - One thread continuously calls the callback's `process_task` method to perform periodic
          or background tasks.

    The worker stops when a shutdown event is received, ensuring all threads exit cleanly.

    Usage:
        - Instantiate with input/output queues and a callback implementing
          FifoSyncProcessWorkerCallback.
        - Call `run_until_complete()` to start processing events and tasks until a shutdown event is
          received.

    Attributes:

        _local_priority_queue (queue.PriorityQueue[FifoEvent]):
            Thread-safe priority queue used internally by the worker to store incoming events.
            The interprocess queue is not priority-based, so this local priority queue is required
            to ensure events are processed in priority order within the worker.
    """

    _local_priority_queue: queue.PriorityQueue[FifoEvent]

    def __init__(self,
                 in_queue: Queue[FifoEvent],
                 out_queue: Queue[FifoEvent],
                 callback: FifoSyncProcessWorkerCallback):
        """
        Initialize the synchronous worker with input/output queues and a callback.

        Args:
            in_queue (Queue[FifoEvent]):
                Interprocess queue for incoming events (from main process to worker process).

            out_queue (Queue[FifoEvent]):
                Interprocess queue for outgoing events (from the worker process to the main
                process).

            callback (FifoSyncProcessWorkerCallback):
                Callback for custom event and task processing.
        """
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._local_priority_queue = queue.PriorityQueue()
        self._callback = callback
        self._stop_event = threading.Event()

    def _priority_in_reader(self):
        """
        Thread target: Reads events from the interprocess input queue and puts them into the local
        priority queue.

        Stops when a FifoEventShutdown is received, which is forwarded to the local priority queue
        to cascade the shutdown process.
        """
        _trace("[FifoSyncProcessWorker.thread:_priority_in_reader] Thread running")
        while True:
            event = self._in_queue.get()
            self._local_priority_queue.put(event)
            _trace("[FifoSyncProcessWorker.thread:_priority_in_reader] Event queued for processing")
            if isinstance(event, FifoEventShutdown):
                _trace("[FifoSyncProcessWorker.thread:_priority_in_reader] Shutdown event received; stopping thread")  # pylint: disable=line-too-long
                break

    def _priority_event_processor(self):
        """
        Thread target: Processes events from the local priority queue by calling the callback's
        `process_event` method.

        Stops when a FifoEventShutdown is received, signals the stop event.
        
        The shutdown event is **not** forwarded immediately to `_out_queue` to avoid
        triggering an early cascade shutdown. Forwarding it too soon would cause
        `_out_queue_puller` to stop before all pending events were drained, leaving
        subsequent events stuck in the queue. 

        Instead, `_stop_event` is set, causing `_out_task_loop` (which runs
        `process_task`) to exit and allowing `run_until_complete` to complete the join
        sequence.

        As the final action, `run_until_complete` enqueues a `FifoEventShutdown` to
        `_out_queue`. This ensures that any events emitted by `process_task` after the
        shutdown signal are still read by the main process: `_out_queue_puller`
        terminates upon receiving this final shutdown event, which is by design the
        last event in the queue.
        """
        _trace("[FifoSyncProcessWorker.thread:_priority_event_processor] Thread running")
        while True:
            event = self._local_priority_queue.get()
            if isinstance(event, FifoEventShutdown):
                self._stop_event.set()
                _trace("[FifoSyncProcessWorker.thread:_priority_event_processor] Shutdown event processed; stopping thread")  # pylint: disable=line-too-long
                break
            try:
                self._callback.process_event(
                    event, self._local_priority_queue.qsize(), self._out_queue
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("[FifoSyncProcessWorker.process_event] Unhandled exception")
                self._out_queue.put(
                    FifoEventException(exception=e, source="FifoSyncProcessWorker.process_event")
                )

    def _out_task_loop(self):
        """
        Thread target: Continuously calls the callback's `process_task` method to perform periodic
        or background tasks.

        Stops when the stop event is set (after a shutdown event is received).
        """
        _trace("[FifoSyncProcessWorker.thread:_out_task_loop] Thread running")
        while not self._stop_event.is_set():
            try:
                self._callback.process_task(self._out_queue)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("[FifoSyncProcessWorker.process_task] Unhandled exception")
                self._out_queue.put(
                    FifoEventException(exception=e, source="FifoSyncProcessWorker.process_task")
                )
        _trace("[FifoSyncProcessWorker.thread:_out_task_loop] Thread stopping")

    def run_until_complete(self):
        """
        Starts all threads and processes events and tasks until a shutdown event is received.

        Joins all threads to ensure clean shutdown.
        """
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] Initializing sync worker")

        try:
            self._callback.initialize(self._out_queue)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[FifoSyncProcessWorker.initialize] Unhandled exception")
            self._out_queue.put(
                FifoEventException(exception=e, source="FifoSyncProcessWorker.initialize")
            )

        reader_thread = threading.Thread(target=self._priority_in_reader)
        processor_thread = threading.Thread(target=self._priority_event_processor)
        out_task_thread = threading.Thread(target=self._out_task_loop)

        reader_thread.start()
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] _priority_in_reader thread started")
        processor_thread.start()
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] _priority_event_processor thread started")  # pylint: disable=line-too-long
        out_task_thread.start()
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] _out_task_loop thread started")

        reader_thread.join()
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] _priority_in_reader thread joined")
        processor_thread.join()
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] _priority_event_processor thread joined")  # pylint: disable=line-too-long
        out_task_thread.join()
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] _out_task_loop thread joined")

        _trace("[FifoSyncProcessWorker.fct:run_until_complete] Worker threads joined")

        try:
            self._callback.finalize(self._out_queue)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[FifoSyncProcessWorker.finalize] Unhandled exception")
            self._out_queue.put(
                FifoEventException(exception=e, source="FifoSyncProcessWorker.finalize")
            )
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] Callback finalized")
        self._out_queue.put(FifoEventShutdown())
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] Shutdown event sent to _out_queue")
        _trace("[FifoSyncProcessWorker.fct:run_until_complete] Sync worker shutdown complete")

def _runner_async(in_queue: Queue[FifoEvent],
                  out_queue: Queue[FifoEvent],
                  callback: FifoAsyncProcessWorkerCallback):
    FifoAsyncProcessWorker(in_queue, out_queue, callback).run_until_complete()

def _runner_sync(in_queue: Queue[FifoEvent],
                 out_queue: Queue[FifoEvent],
                 callback: FifoSyncProcessWorkerCallback):
    FifoSyncProcessWorker(in_queue, out_queue, callback).run_until_complete()


@dataclass
class _ReceivedCID:
    """
    Internal helper class for tracking the state of pending correlation IDs.

    This class is used to keep track of which ACK and/or DONE events have been received
    for a given correlation ID, as well as to store the future that will be completed
    when all expected responses are received.

    Attributes:
        future (asyncio.Future[FifoEventResultWithCID]):
            The future to be completed when the expected response(s) are received.

        cls_ack (Type[FifoEventResultWithCID] | None):
            The class of the expected ACK event, or None if not expected.

        cls_done (Type[FifoEventResultWithCID] | None):
            The class of the expected DONE event, or None if not expected.

        ack_received (bool):
            Whether the ACK event has been received.

        done_received (bool):
            Whether the DONE event has been received.
    """

    future: asyncio.Future[FifoEventResultWithCID]
    cls_ack: Type[FifoEventResultWithCID] | None = None
    cls_done: Type[FifoEventResultWithCID] | None = None
    ack_received: bool = False
    done_received: bool = False

    def done(self) -> bool:
        """
        Return True if all expected responses (ACK and/or DONE) have been received.

        Returns:
            bool:
                True if the ACK (if expected) and DONE (if expected) have both been received.
        """
        return (
            (self.cls_ack is None or self.ack_received)
            and
            (self.cls_done is None or self.done_received)
        )


class FifoProcessManager:
    """
    Manager class for running a worker in a separate OS process and handling interprocess
    communication.

    This class abstracts the creation, startup, and shutdown of a worker process (either async or
    sync), and manages the threads that transfer events in both directions between the main process
    and the worker process.

    It provides async methods for sending events, and ensures clean shutdown by
    propagating shutdown events and joining all threads and the process.

    Usage:
        - Instantiate with an event loop, a worker callback (async or sync), and an async
          output queue.
        - Call `start()` to launch the worker process and communication threads.
        - Use `send()` and `send_and_wait_response()` to asynchronously send events.
        - Call `stop()` to shut down the worker process.
        - Call `join()` to wait for all threads and the process to exit.

    Attributes:
        _in_queue (Queue[FifoEvent]):
            Interprocess queue for incoming events (from main process to worker process).

        _out_queue (Queue[FifoEvent]):
            Interprocess queue for outgoing events (from worker process to main process).

        _loop (asyncio.AbstractEventLoop):
            The event loop used for async operations in the main process.

        _proc (Process):
            The worker process instance.

        _pusher_thread (Thread):
            Thread that pushes events from the async input queue to the interprocess input queue.

        _puller_thread (Thread):
            Thread that pulls events from the interprocess output queue to the async output queue.

        _async_in (asyncio.PriorityQueue[FifoEvent]):
            Async priority queue for incoming events in the main process.

        _async_out (SupportsFifoEventPut):
            Object to receive outgoing events from the main process, such as an asyncio
            priority queue or a `FifoEventQueueNetworkAsyncServer/Client`.

        _lock_cid (asyncio.Lock):
            Lock used to protect concurrent access to the `_received_cid` dictionary from multiple
            coroutines. Ensures that updates to correlation ID tracking are safe when accessed from
            both async code and threads via `run_coroutine_threadsafe`.

        _received_cid (dict[UUID, _ReceivedCID]):
            Dictionary mapping correlation IDs to their tracking state (`_ReceivedCID`).
            Used to manage pending requests and match incoming ACK/DONE events to their
            corresponding futures.

        _event_stopped (asyncio.Event):
            Event signaling the worker has been stopped.
    """

    _in_queue: Queue[FifoEvent]
    _out_queue: Queue[FifoEvent]
    _loop: asyncio.AbstractEventLoop
    _proc: Process
    _pusher_thread: Thread
    _puller_thread: Thread
    _async_in: asyncio.PriorityQueue[FifoEvent]
    _async_out: SupportsFifoEventPut
    _lock_cid: asyncio.Lock
    _received_cid: dict[UUID, _ReceivedCID]
    _event_stopped: asyncio.Event

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 callback: FifoAsyncProcessWorkerCallback | FifoSyncProcessWorkerCallback,
                 async_out: SupportsFifoEventPut
                 ) -> None:
        """
        Initialize the process manager with an event loop, worker callback, and async
        output queue.

        Args:
            loop (asyncio.AbstractEventLoop):
                The event loop used for async operations in the main process.

            callback (FifoAsyncProcessWorkerCallback | FifoSyncProcessWorkerCallback):
                The worker callback (async or sync) to run in the worker process.

            async_out (SupportsFifoEventPut):
                Object to receive outgoing events from the main process, such as an asyncio
                priority queue or a `FifoEventQueueNetworkAsyncServer/Client`.
        """
        self._loop = loop

        self._lock_cid = asyncio.Lock()
        self._received_cid = {}

        self._in_queue = Queue()
        self._out_queue = Queue()

        self._async_in: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
        self._async_out = async_out

        if isinstance(callback, FifoAsyncProcessWorkerCallback):
            self._proc = Process(target=_runner_async,
                                 args=(self._in_queue, self._out_queue, callback))
        else: # isinstance(callback, FifoSyncProcessWorkerCallback):
            self._proc = Process(target=_runner_sync,
                                 args=(self._in_queue, self._out_queue, callback))

        self._pusher_thread = Thread(target=self._in_queue_pusher)
        self._puller_thread = Thread(target=self._out_queue_puller)

        self._event_stopped = asyncio.Event()

    def start(self) -> None:
        """
        Start the worker process and the communication threads.

        This launches the worker process and starts the threads that transfer events in both 
        directions between the main process and the worker process.
        """
        _trace("[FifoProcessManager.fct:start] Starting worker process and communication threads")
        self._proc.start()
        _trace("[FifoProcessManager.fct:start] Worker process started")
        self._pusher_thread.start()
        _trace("[FifoProcessManager.fct:start] _pusher_thread started")
        self._puller_thread.start()
        _trace("[FifoProcessManager.fct:start] _puller_thread started")

    async def stop(self) -> None:
        """
        Stop the worker process by sending a shutdown event and waiting for completion.

        Sends a shutdown event to the worker process and waits until it is echoed back,
        indicating that shutdown has completed successfully.
        """
        _trace("[FifoProcessManager.fct:stop] Sending FifoEventShutdown to request shutdown")
        await self._async_in.put(FifoEventShutdown())

        _trace("[FifoProcessManager.fct:stop] Awaiting worker shutdown confirmation")
        await self._event_stopped.wait()
        _trace("[FifoProcessManager.fct:stop] Received FifoEventShutdown confirmation from worker")

    def join(self) -> None:
        """
        Wait for the worker process and communication threads to exit.

        Joins the worker process and both communication threads to ensure clean shutdown.
        """
        _trace("[FifoProcessManager.fct:join] Joining worker process and communication threads")
        self._proc.join()
        _trace("[FifoProcessManager.fct:join] Worker process joined")
        self._pusher_thread.join()
        _trace("[FifoProcessManager.fct:join] _pusher_thread joined")
        self._puller_thread.join()
        _trace("[FifoProcessManager.fct:join] _puller_thread joined")
        _trace("[FifoProcessManager.fct:join] Worker process and communication threads joined")

    async def send(self, event: FifoEvent) -> None:
        """
        Asynchronously send an event to the worker process.

        Args:
            event (FifoEvent):
                The event to send to the worker process.
        """
        _trace("[FifoProcessManager.fct:send] Dispatching event to worker")
        await self._async_in.put(event)
        _trace("[FifoProcessManager.fct:send] Event dispatched")

    async def send_and_wait_response(
            self, event: FifoEvent,
            cls_ack: Type[FifoEventResultWithCID] | None = None,
            cls_done: Type[FifoEventResultWithCID] | None = None) -> FifoEventResultWithCID:
        """
        Asynchronously send an event to the worker process and wait for an ACK and/or DONE response.

        This method sends the given event to the worker process and waits for a response event
        matching the specified ACK and/or DONE result classes. The correlation ID of the event
        is used to track and match the response. This is useful for request/response workflows
        where confirmation or completion events are expected.

        Args:
            event (FifoEvent):
                The event to send to the worker process.

            cls_ack (Type[FifoEventResultWithCID] | None, optional):
                The class of the expected ACK response event. If None, no ACK is expected.

            cls_done (Type[FifoEventResultWithCID] | None, optional):
                The class of the expected DONE response event. If None, no DONE is expected.

        Returns:
            FifoEventResultWithCID:
                The received ACK or DONE response event, depending on which is requested and
                received last.

        Raises:
            ValueError: If neither ACK nor DONE result classes are specified.
            RuntimeError: If the event does not have a correlation ID, or if a duplicate 
                          correlation ID is detected.
        """
        if cls_ack is None and cls_done is None:
            raise ValueError("At least one of cls_ack or cls_done must be specified.")

        await self.send(event)

        correlation_id = getattr(event, "correlation_id", None)
        if correlation_id is None:
            raise RuntimeError("Missing correlation id")
        async with self._lock_cid:
            if correlation_id in self._received_cid:
                raise RuntimeError("Duplicated correlation id")
            future=self._loop.create_future()
            self._received_cid[correlation_id] = _ReceivedCID(
                future=future,
                cls_ack=cls_ack,
                cls_done=cls_done
            )
        _trace("[FifoProcessManager.fct:send] Waiting on result (ack/done)")
        result = await future
        _trace("[FifoProcessManager.fct:send] Result received (ack/done)")

        return result

    def _in_queue_pusher(self) -> None:
        """
        Thread target: Moves events from the async input queue to the interprocess input queue.

        Stops when a FifoEventShutdown is received, which is forwarded to the interprocess input
        queue to cascade the shutdown process.
        """
        _trace("[FifoProcessManager.thread:_in_queue_pusher] Thread running")
        while True:
            event = asyncio.run_coroutine_threadsafe(self._async_in.get(), self._loop).result()
            self._in_queue.put(event)
            _trace("[FifoProcessManager.thread:_in_queue_pusher] Forwarded event to worker process")
            if isinstance(event, FifoEventShutdown):
                _trace("[FifoProcessManager.thread:_in_queue_pusher] Shutdown event forwarded; stopping thread")  # pylint: disable=line-too-long
                break

    async def _update_received_correlation_id(self, event: FifoEventResultWithCID):
        """
        Update the state of a pending correlation ID entry when a result event is received.

        This method is called by the `_out_queue_puller` thread when it receives a
        `FifoEventResultWithCID` event from the worker process (via the interprocess output queue).
        In this case, the event is handled directly and is **not** placed into the `self._async_out`
        queue by the `_out_queue_puller` thread.

        If the event's correlation ID does not match any pending request, the event is inserted
        into the async output queue (`self._async_out`) to ensure it is not discarded. This allows
        unexpected or unsolicited result events to still be processed by the main application.

        The method updates the corresponding entry in `_received_cid` to track whether the expected
        ACK and/or DONE events have been received for a given correlation ID. If all required
        responses have been received, the associated future is completed and the entry is removed
        from the tracking dictionary.

        If an ACK event is received with an error code (i.e., `event.code is not ErrorCode.OK`),
        the method will immediately complete the future and remove the entry, without waiting for
        a DONE event. This ensures that tasks which failed to start or were not accepted do not
        block waiting for a completion event that will never arrive.

        Args:
            event (FifoEventResultWithCID):
                The result event received from the worker process, containing a correlation ID.
        """
        async with self._lock_cid:

            received_cid = self._received_cid.get(event.correlation_id)

            if received_cid is None:
                # this event does not match to any request, we insert it to the output queue
                # in order to not discard it
                await self._async_out.put(event)
                return

            if event.__class__ == received_cid.cls_ack:
                if event.code is not EErrorCode.OK:
                    # there has been an error, then we do not wait for the DONE event as the task
                    # was not received / started successfully.
                    received_cid.future.set_result(event)
                    del self._received_cid[event.correlation_id]
                    return

                received_cid.ack_received = True
            if event.__class__ == received_cid.cls_done:
                received_cid.done_received = True
            if received_cid.done():
                received_cid.future.set_result(event)
                del self._received_cid[event.correlation_id]

    def _out_queue_puller(self) -> None:
        """
        Thread target: Moves events from the interprocess output queue to the async output queue.

        For most events, this thread inserts them directly into the async output queue
        (`self._async_out`). However, events of type `FifoEventResultWithCID` are **not** inserted
        into the queue by this thread. Instead, they are forwarded to the 
        `_update_received_correlation_id` coroutine for correlation ID tracking.
        If a `FifoEventResultWithCID` does not match any pending request (i.e., its correlation ID
        is not found), `_update_received_correlation_id` will insert it into the async output queue
        to ensure it is not discarded.

        Stops when a `FifoEventShutdown` is received, which is forwarded to the async output queue
        to cascade the shutdown process.
        """
        _trace("[FifoProcessManager.thread:_out_queue_puller] Thread running")
        while True:
            event = self._out_queue.get()
            if isinstance(event, FifoEventResultWithCID):
                asyncio.run_coroutine_threadsafe(
                    self._update_received_correlation_id(event), self._loop
                )
            else:
                asyncio.run_coroutine_threadsafe(self._async_out.put(event), self._loop)
            _trace("[FifoProcessManager.thread:_out_queue_puller] Received event from worker process")  # pylint: disable=line-too-long
            if isinstance(event, FifoEventShutdown):
                self._loop.call_soon_threadsafe(self._event_stopped.set)
                asyncio.run_coroutine_threadsafe(self._async_out.put(event), self._loop)
                _trace("[FifoProcessManager.thread:_out_queue_puller] Shutdown event received; stopping thread")  # pylint: disable=line-too-long
                break
