from __future__ import annotations
import multiprocessing
import threading
import logging
import asyncio
from typing import TYPE_CHECKING
from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown

if TYPE_CHECKING:
    from multiprocessing.queues import Queue as MpQueue
else:
    MpQueue = multiprocessing.Queue  # type: ignore[misc]

logger = logging.getLogger(__name__)


class FifoEventQueueForwarderMpToAsync:
    """
    Forwards events from a multiprocessing queue to an asyncio queue using a background thread.
    
    This class connects multiprocessing and asyncio by running a background thread
    that continuously reads events from a multiprocessing queue and forwards them to an asyncio
    priority queue. The forwarding stops when a FifoEventShutdown event is received.

    The thread is designed to handle the synchronization between different concurrency models:
    - Multiprocessing (blocking queue operations)
    - Asyncio (non-blocking, event loop based)

    Notes:
        Ordering is **priority-based only**. If two events have the same priority and your
        `FifoEvent.__lt__` does not provide a tie-breaker, ordering among equal-priority items
        is not guaranteed (heapq is not stable).

    Attributes:
        _mp_queue (MpQueue[FifoEvent]):
            The source multiprocessing queue from which events are read.

        _asyncio_queue (asyncio.PriorityQueue[FifoEvent]):
            The destination asyncio priority queue to which events are forwarded.

        _loop (asyncio.AbstractEventLoop):
            The asyncio event loop used to schedule the forwarding operations thread-safely.
            
        _thread (threading.Thread):
            The background thread that performs the actual event forwarding.

        _started (bool):
            Flag indicating whether the forwarder thread has been started and is expected
            to be running. This is reset to False after a successful `join()`.

        _stop_requested (threading.Event):
            Set when shutdown has been requested (via `stop()` or on receiving
            `FifoEventShutdown`); cleared in `start()`.

        _forward_shutdown_event (bool):
            Whether FifoEventShutdown events should be forwarded to the asyncio queue (True)
            or only used to stop the forwarder (False).
    """

    _mp_queue: MpQueue[FifoEvent]
    _asyncio_queue: asyncio.PriorityQueue[FifoEvent]
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _forward_shutdown_event: bool
    _started: bool
    _stop_requested: threading.Event
    _stopped_event: threading.Event

    def __init__(
        self,
        mp_queue: MpQueue[FifoEvent],
        asyncio_queue: asyncio.PriorityQueue[FifoEvent],
        loop: asyncio.AbstractEventLoop,
        forward_shutdown_event: bool = False,
        daemon: bool = False,
    ):
        """
        Initialize the event queue forwarder.
        
        Args:
            mp_queue (MpQueue[FifoEvent]):
                The multiprocessing queue from which to read events.
                
            asyncio_queue (asyncio.PriorityQueue[FifoEvent]):
                The asyncio priority queue to which events will be forwarded.
                
            loop (asyncio.AbstractEventLoop):
                The asyncio event loop that will be used for thread-safe operations.
                
            forward_shutdown_event (bool, optional):
                Whether to forward FifoEventShutdown events to the asyncio queue.
                If False (default), shutdown events are only used to stop the forwarder.
                If True, shutdown events are also forwarded before stopping.

            daemon (bool, optional):
                Whether the forwarding thread should be a daemon. Defaults to False.
        """
        self._mp_queue = mp_queue
        self._asyncio_queue = asyncio_queue
        self._loop = loop
        self._forward_shutdown_event = forward_shutdown_event

        self._thread = threading.Thread(
            target=self._run,
            daemon=daemon,
            name="fifo-forwarder",
        )
        self._started = False
        self._stopped_event = threading.Event()
        self._stop_requested = threading.Event()

    def _enqueue_async(self, thread_name: str, event: FifoEvent) -> None:
        """
        Schedule an async put; use put_nowait fast path, fall back to await if full.
        """
        def _cb() -> None:
            try:
                self._asyncio_queue.put_nowait(event)  # O(1), no Task allocation
            except asyncio.QueueFull:
                # Only allocate a Task when saturated; let loop await capacity
                asyncio.create_task(self._asyncio_queue.put(event))

        # Short-circuit if the loop is already closed (avoids noisy RuntimeError)
        if self._loop.is_closed():
            logger.error("[%s] Event loop closed; dropping event %s",
                        thread_name, type(event).__name__)
            return

        try:
            self._loop.call_soon_threadsafe(_cb)
        except RuntimeError:
            # The loop may have closed between is_closed() check and this call.
            # We can't schedule the callback anymore, so we drop the event.
            logger.error("[%s] Event loop closed; dropping event %s",
                         thread_name, type(event).__name__)

    def _run(self) -> None:
        """
        Main thread function that continuously forwards events.
        
        This method runs in a background thread and:
        1. Continuously reads events from the multiprocessing queue (blocking operation)
        2. Checks for shutdown events to determine when to stop
        3. Forwards non-shutdown events to the asyncio queue using thread-safe operations
        4. Optionally forwards shutdown events if configured to do so
        5. Breaks the loop and exits when a shutdown event is received
        
        The forwarding is done using `call_soon_threadsafe` to ensure proper
        synchronization with the asyncio event loop.
        """
        thread_name = threading.current_thread().name
        logger.debug("[%s] Thread started", thread_name)
        try:
            while True:
                try:
                    event = self._mp_queue.get()
                except (EOFError, OSError) as e:
                    level = logging.WARNING if self._stop_requested.is_set() else logging.ERROR
                    logger.log(level, "[%s] Source queue closed (%s); stopping",
                               thread_name, type(e).__name__)
                    break

                if isinstance(event, FifoEventShutdown):
                    logger.debug("[%s] Shutdown event received, stopping", thread_name)
                    self._stop_requested.set()
                    if self._forward_shutdown_event:
                        self._enqueue_async(thread_name, event)
                    break

                self._enqueue_async(thread_name, event)
        except BaseException as e:  # pylint: disable=broad-exception-caught
            # Keep interrupts/exits intact; sanitize everything else
            if isinstance(e, (SystemExit, KeyboardInterrupt, GeneratorExit)):
                raise
            logger.error("[%s] Unexpected %s; stopping", thread_name, type(e).__name__)
        finally:
            self._stopped_event.set()
            logger.debug("[%s] Thread Stopped", thread_name)

    def start(self) -> None:
        """
        Start (or restart) the background forwarding thread.

        If the thread is already alive, logs a warning and returns.
        If it previously finished, this recreates the internal thread and stopped-event,
        and starts it again.

        The background thread will begin forwarding events from the multiprocessing queue 
        to the asyncio queue. The thread will continue running until a FifoEventShutdown 
        event is received.

        Raises:
            RuntimeError: if the asyncio event loop is not running.
        """
        if not self._loop.is_running():
            raise RuntimeError("Event loop must be running before start()")

        if self._thread.is_alive():
            logger.warning("[Main] start() called but thread already started")
            return

        # Recreate synchronization event and thread for a clean restart
        self._stopped_event = threading.Event()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=self._thread.daemon,
            name=self._thread.name or "fifo-forwarder",
        )

        logger.debug("[Main] Starting thread")
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """
        Signal the background forwarding thread to stop by enqueuing a shutdown event.

        This method sends a FifoEventShutdown event to the multiprocessing queue,
        which will cause the background thread to exit its forwarding loop.
        """
        logger.debug("[Main] Stopping thread")
        self._stop_requested.set()
        try:
            self._mp_queue.put(FifoEventShutdown())
        except (ValueError, OSError) as e:
            logger.error("[Main] Could not enqueue shutdown: %s", type(e).__name__)

    def join(self, timeout: float | None = None) -> None:
        """
        Wait for the background thread to finish execution.

        This method blocks until the background thread has completely stopped.
        It should be called after stop() to ensure clean shutdown.

        If the thread was never started, returns immediately with a warning.
        """
        if not self._started:
            logger.warning("[Main] join() called but thread never started")
            return

        logger.debug("[Main] Joining thread")
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("[Main] Thread did not stop within timeout")
        else:
            logger.debug("[Main] Thread joined")
            self._started = False

    def stopped(self) -> bool:
        """
        Check if the thread has already stopped.

        This method does not wait; it returns immediately with True if the
        thread has finished execution, or False otherwise.

        Returns:
            bool:
                True if the thread has finished execution, or False otherwise.
        """
        return self._stopped_event.is_set()
