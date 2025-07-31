from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import threading
import queue
from threading import Thread
from multiprocessing import Process
from typing import TYPE_CHECKING
from fifo_dev_common.event.fifo_event import FifoEventPoison, FifoEvent
from fifo_dev_common.logging.logger import get_logger


if TYPE_CHECKING:
    from multiprocessing.queues import Queue  # pragma: nocover
else:
    from multiprocessing import Queue


logger = get_logger(__name__)


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
    def initialize(self):
        """
        Called once before any event or task processing begins.

        Use this method to set up resources, connections, or state needed by the worker.
        This method is invoked in the worker process before any threads are created or started,
        ensuring all initialization is complete before event or task processing begins.

        This method is called after the asyncio loop has been created.
        """

    @abstractmethod
    def finalize(self):
        """
        Called once after all event and task processing is complete.

        Use this method to clean up resources, connections, or state before the worker exits.
        This method is invoked in the worker process after all threads have finished and all
        event/task processing is complete, just before the process terminates.

        The asyncio loop remains active when this method is called.
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
        - Call run_until_complete() to start processing events until a `FifoEventPoison`
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

        Stops when a FifoEventPoison is received, which is forwarded to the internal async queue
        to cascade the shutdown process.
        """
        assert self._loop is not None

        while True:
            event = self._in_queue.get()
            asyncio.run_coroutine_threadsafe(self._async_in.put(event), self._loop)
            if isinstance(event, FifoEventPoison):
                break

    def _out_queue_pusher(self) -> None:
        """
        Thread target: Moves events from the internal async output queue to the interprocess
        output queue.

        Stops when a FifoEventPoison is received, which is forwarded to the interprocess output
        queue to cascade the shutdown process.
        """
        assert self._loop is not None

        while True:
            event = asyncio.run_coroutine_threadsafe(self._async_out.get(), self._loop).result()
            self._out_queue.put(event)
            if isinstance(event, FifoEventPoison):
                break

        self._loop.call_soon_threadsafe(self._event_queue_pusher_done.set)

    async def _process_loop(self) -> None:
        """
        Main asynchronous event loop.

        Continuously retrieves events from the internal async input queue and calls the callback's
        loop method. Stops when a FifoEventPoison is received and waits for the output pusher
        thread to finish, signaling that there are no more asyncio operations being processed or
        pending.
        """
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

            if isinstance(event, FifoEventPoison):
                await self._async_out.put(event)
                break

            await self._callback.loop(event, self._async_in.qsize(), self._async_out)

        # need to wait for the pusher thread to complete so that we exit the loop process
        # only when no more asyncio operation are processing or pending
        await self._event_queue_pusher_done.wait()

    def run_until_complete(self) -> None:
        """
        Starts the worker process event loop and associated threads.

        Runs until a poison event is received, then joins threads and closes the event loop.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._callback.initialize()

        puller_thread = threading.Thread(target=self._in_queue_puller)
        pusher_thread = threading.Thread(target=self._out_queue_pusher)
        puller_thread.start()
        pusher_thread.start()
        self._loop.run_until_complete(self._process_loop())
        puller_thread.join()
        pusher_thread.join()

        self._callback.finalize()

        self._loop.close()
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
          and return promptly, so that the thread can be interrupted and exit when a poison event is
          received.

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
    def initialize(self):
        """
        Called once before event and task processing begins.

        Use this method to set up resources, connections, or state needed by the worker.
        This method is invoked in the worker process before any events or tasks are processed.
        """

    @abstractmethod
    def finalize(self):
        """
        Called once after all event and task processing is complete.

        Use this method to clean up resources, connections, or state before the worker exits.
        This method is invoked in the worker process after all threads have finished.
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
        a poison event is received.

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

    The worker stops when a poison event is received, ensuring all threads exit cleanly.

    Usage:
        - Instantiate with input/output queues and a callback implementing
          FifoSyncProcessWorkerCallback.
        - Call `run_until_complete()` to start processing events and tasks until a poison event is
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

        Stops when a FifoEventPoison is received, which is forwarded to the local priority queue
        to cascade the shutdown process.
        """
        while True:
            event = self._in_queue.get()
            self._local_priority_queue.put(event)
            if isinstance(event, FifoEventPoison):
                break

    def _priority_event_processor(self):
        """
        Thread target: Processes events from the local priority queue by calling the callback's
        `process_event` method.

        Stops when a FifoEventPoison is received, signals the stop event, and forwards the poison
        event to the output queue to cascade the shutdown process.
        """
        while True:
            event = self._local_priority_queue.get()
            if isinstance(event, FifoEventPoison):
                self._stop_event.set()
                self._out_queue.put(event)
                break
            self._callback.process_event(event, self._local_priority_queue.qsize(), self._out_queue)

    def _out_task_loop(self):
        """
        Thread target: Continuously calls the callback's `process_task` method to perform periodic
        or background tasks.

        Stops when the stop event is set (after a poison event is received).
        """
        while not self._stop_event.is_set():
            self._callback.process_task(self._out_queue)

    def run_until_complete(self):
        """
        Starts all threads and processes events and tasks until a poison event is received.

        Joins all threads to ensure clean shutdown.
        """
        self._callback.initialize()

        reader_thread = threading.Thread(target=self._priority_in_reader)
        processor_thread = threading.Thread(target=self._priority_event_processor)
        out_task_thread = threading.Thread(target=self._out_task_loop)

        reader_thread.start()
        processor_thread.start()
        out_task_thread.start()
        reader_thread.join()
        processor_thread.join()
        out_task_thread.join()

        self._callback.finalize()

def _runner_async(in_queue: Queue[FifoEvent],
                  out_queue: Queue[FifoEvent],
                  callback: FifoAsyncProcessWorkerCallback):
    FifoAsyncProcessWorker(in_queue, out_queue, callback).run_until_complete()

def _runner_sync(in_queue: Queue[FifoEvent],
                 out_queue: Queue[FifoEvent],
                 callback: FifoSyncProcessWorkerCallback):
    FifoSyncProcessWorker(in_queue, out_queue, callback).run_until_complete()

class FifoProcessManager:
    """
    Manager class for running a worker in a separate OS process and handling interprocess
    communication.

    This class abstracts the creation, startup, and shutdown of a worker process (either async or
    sync), and manages the threads that transfer events in both directions between the main process
    and the worker process.

    It provides async methods for sending and receiving events, and ensures clean shutdown by
    propagating poison events and joining all threads and the process.

    Usage:
        - Instantiate with an event loop and a worker callback (async or sync).
        - Call `start()` to launch the worker process and communication threads.
        - Use `send()` and `receive()` to asynchronously exchange events.
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

        _async_out (asyncio.PriorityQueue[FifoEvent]):
            Async priority queue for outgoing events in the main process.
    """

    _in_queue: Queue[FifoEvent]
    _out_queue: Queue[FifoEvent]
    _loop: asyncio.AbstractEventLoop
    _proc: Process
    _pusher_thread: Thread
    _puller_thread: Thread
    _async_in: asyncio.PriorityQueue[FifoEvent]
    _async_out: asyncio.PriorityQueue[FifoEvent]

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 callback: FifoAsyncProcessWorkerCallback | FifoSyncProcessWorkerCallback,
                 async_out: asyncio.PriorityQueue[FifoEvent] | None = None
                 ) -> None:
        """
        Initialize the process manager with an event loop, worker callback, and optional async
        output queue. If no async output queue is provided, a new one is created.

        Args:
            loop (asyncio.AbstractEventLoop):
                The event loop used for async operations in the main process.

            callback (FifoAsyncProcessWorkerCallback | FifoSyncProcessWorkerCallback):
                The worker callback (async or sync) to run in the worker process.

            async_out (asyncio.PriorityQueue[FifoEvent] | None):
                Optional async priority queue for outgoing events in the main process.
        """
        self._loop = loop

        self._in_queue = Queue()
        self._out_queue = Queue()

        self._async_in: asyncio.PriorityQueue[FifoEvent] = asyncio.PriorityQueue()
        self._async_out = asyncio.PriorityQueue() if async_out is None else async_out

        if isinstance(callback, FifoAsyncProcessWorkerCallback):
            self._proc = Process(target=_runner_async,
                                 args=(self._in_queue, self._out_queue, callback))
        else: # isinstance(callback, FifoSyncProcessWorkerCallback):
            self._proc = Process(target=_runner_sync,
                                 args=(self._in_queue, self._out_queue, callback))

        self._pusher_thread = Thread(target=self._in_queue_pusher)
        self._puller_thread = Thread(target=self._out_queue_puller)

    def start(self) -> None:
        """
        Start the worker process and the communication threads.

        This launches the worker process and starts the threads that transfer events in both 
        directions between the main process and the worker process.
        """
        logger.trace("[PROCESS:MAIN] Start the worker process and the communication threads")
        self._proc.start()
        self._pusher_thread.start()
        self._puller_thread.start()

    async def stop(self) -> None:
        """
        Stop the worker process by sending a poison event and waiting for shutdown.

        Sends a poison event to the worker process and waits until the poison event is received
        back from the worker, indicating shutdown is complete.
        """
        logger.trace("[PROCESS:MAIN] Sent `FifoEventPoison` to request shutdown")
        await self._async_in.put(FifoEventPoison())

        logger.trace("[PROCESS:MAIN] Awaiting worker confirmation...")
        while True:
            event: FifoEvent = await self._async_out.get()
            if isinstance(event, FifoEventPoison):
                break
        logger.trace("[PROCESS:MAIN] Received `FifoEventPoison` confirmation from worker")

    def join(self) -> None:
        """
        Wait for the worker process and communication threads to exit.

        Joins the worker process and both communication threads to ensure clean shutdown.
        """
        self._proc.join()
        self._pusher_thread.join()
        self._puller_thread.join()

    async def send(self, event: FifoEvent) -> None:
        """
        Asynchronously send an event to the worker process.

        Args:
            event (FifoEvent):
                The event to send to the worker process.
        """
        await self._async_in.put(event)

    async def receive(self) -> FifoEvent:
        """
        Asynchronously receive the next event from the worker process.

        Returns:
            FifoEvent:
                The next event produced by the worker process.
        """
        return await self._async_out.get()

    def _in_queue_pusher(self) -> None:
        """
        Thread target: Moves events from the async input queue to the interprocess input queue.

        Stops when a FifoEventPoison is received, which is forwarded to the interprocess input
        queue to cascade the shutdown process.
        """
        while True:
            event = asyncio.run_coroutine_threadsafe(self._async_in.get(), self._loop).result()
            self._in_queue.put(event)
            if isinstance(event, FifoEventPoison):
                break

    def _out_queue_puller(self) -> None:
        """
        Thread target: Moves events from the interprocess output queue to the async output queue.

        Stops when a FifoEventPoison is received, which is forwarded to the async output queue to
        cascade the shutdown process.
        """
        while True:
            event = self._out_queue.get()
            asyncio.run_coroutine_threadsafe(self._async_out.put(event), self._loop)
            if isinstance(event, FifoEventPoison):
                break
