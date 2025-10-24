import asyncio


from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_protocols import SupportsFifoEventSend


class FifoEventRateLimiter:
    """
    Asynchronous rate limiter for FifoEvent transmission with latest-wins queuing.

    This class wraps a FifoEvent connection and enforces a maximum transmission rate
    by queuing events and sending them at controlled intervals. When multiple events
    arrive faster than the rate limit allows, only the most recent event is retained
    (latest-wins strategy), preventing queue buildup while ensuring timely delivery
    of the freshest data.

    The rate limiter runs a background task that manages event timing and transmission.
    Events are sent immediately if sufficient time has elapsed since the last send,
    or queued and sent after the minimum required interval.

    Special handling for FifoEventShutdown:
        - If `stop_on_shutdown=True`:
          - Once a FifoEventShutdown is queued, it cannot be replaced by any
            subsequent events (including other shutdown events).
          - The sender stops automatically after sending a FifoEventShutdown.
        - If `stop_on_shutdown=False`, FifoEventShutdown is treated as a normal event
          and the sender continues running.

    Shutdown behavior:
        - When `stop()` is called or the sender auto-stops after sending FifoEventShutdown,
          any remaining queued event is discarded (not sent).
        - This ensures immediate shutdown without waiting for rate limiting intervals.

    Example timeline (max_rate=100 events/sec → 10ms intervals):
        ```
        Time (ms):  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
        Events:        A     B     C     D                    E
        
        t=1:  A sent immediately
        t=3:  B queued (< 10ms since A)
        t=5:  C replaces B in queue (< 10ms since A)
        t=7:  D replaces C in queue (< 10ms since A)
        t=11: D sent (10ms elapsed since A)
        t=14: E queued
        t=24: E sent (10ms after D)
        ```

    Attributes:
        _connection (SupportsFifoEventSend):
            The underlying connection used for sending events.

        _stop_on_shutdown (bool):
            If True, the sender automatically stops after sending a FifoEventShutdown.
            If False, shutdown events are treated as normal payload.

        _task_sender (asyncio.Task[None]):
            Background task that manages event timing and transmission.

        _shutdown_event (asyncio.Event):
            Signals the sender task to gracefully terminate.

        _new_event (asyncio.Event):
            Signals that a new event has been queued for sending.

        _event_to_send (FifoEvent | None):
            The currently queued event, or None if no event is pending.

        _max_rate (float):
            Maximum transmission rate in events per second.

        _time_last_sent (float):
            Timestamp (from event loop) of the last successful send.

        _min_interval (float):
            Minimum time interval between sends, computed as 1.0 / max_rate.

    Usage:
        ```python
        # Create rate limiter with 100 events/second max rate
        limiter = FifoEventRateLimiter(connection, max_rate=100.0)

        # Send events (latest-wins if sent too quickly)
        limiter.send(event1)
        limiter.send(event2)  # May replace event1 if queued
        limiter.send(event3)  # May replace event2 if queued

        # Stop and wait for completion (any queued event is discarded)
        limiter.stop()
        await limiter.join()
        ```

        ```python
        # Auto-stop on shutdown event
        limiter = FifoEventRateLimiter(
            connection,
            max_rate=50.0,
            stop_on_shutdown=True
        )

        limiter.send(normal_event)
        limiter.send(FifoEventShutdown())  # Sender will auto-stop after sending
        await limiter.join()
        ```

    Raises:
        ValueError:
            If max_rate is not positive (raised in __init__).
    """
    _connection: SupportsFifoEventSend
    _stop_on_shutdown: bool
    _task_sender: asyncio.Task[None]
    _shutdown_event: asyncio.Event
    _new_event: asyncio.Event
    _event_to_send: FifoEvent | None
    _max_rate: float  # event per seconds
    _time_last_sent: float
    _min_interval: float

    def __init__(self,
                 connection: SupportsFifoEventSend,
                 max_rate: float,
                 stop_on_shutdown: bool = False):
        """
        Initialize the rate limiter and start the background sender task.

        Args:
            connection (SupportsFifoEventSend):
                The connection object used to send events. Must implement the
                SupportsFifoEventSend protocol with an async `send()` method.

            max_rate (float):
                Maximum transmission rate in events per second. Must be positive.
                For example, max_rate=100.0 allows up to 100 events per second,
                enforcing a minimum 10ms interval between sends.

            stop_on_shutdown (bool, optional):
                Controls behavior when a FifoEventShutdown is sent:
                - If True: the sender automatically stops after sending the shutdown event.
                - If False: the shutdown event is treated as normal payload and the
                  sender continues running.
                Defaults to False.

        Raises:
            ValueError:
                If max_rate is not positive.
        """
        if max_rate <= 0:
            raise ValueError("max_rate must be > 0")

        self._max_rate = max_rate
        self._connection = connection
        self._stop_on_shutdown = stop_on_shutdown

        self._shutdown_event = asyncio.Event()
        self._new_event = asyncio.Event()
        self._event_to_send = None
        self._time_last_sent = 0.0
        self._min_interval = 1.0 / max_rate

        # Start after all fields are initialized to avoid races
        self._task_sender = asyncio.create_task(self._sender(), name="FifoEventRateLimiter")

    def send(self, event: FifoEvent):
        """
        Queue an event for rate-limited transmission.

        This method implements a latest-wins strategy: if an event is already queued
        and insufficient time has passed to send it, the new event replaces the
        queued one. This prevents queue buildup and ensures the most recent data
        is transmitted.

        Special handling for FifoEventShutdown:
            - If `stop_on_shutdown=True`:
              - Once a FifoEventShutdown is queued, it cannot be replaced by any
                subsequent events (including other shutdown events).
              - New events sent after a shutdown is queued are silently ignored.

        Args:
            event (FifoEvent):
                The event to send. Can be any FifoEvent subclass, including
                FifoEventShutdown for graceful termination signaling.

        Note:
            This method returns immediately after queuing. Actual transmission
            occurs asynchronously in the background sender task, subject to
            rate limiting constraints.
        """
        if self._shutdown_event.is_set():
            return
        # If a shutdown is already pending, ignore all new events
        if self._stop_on_shutdown and isinstance(self._event_to_send, FifoEventShutdown):
            return

        # Replace the pending event (normal or shutdown)
        self._event_to_send = event
        self._new_event.set()

    def stop(self):
        """
        Request graceful shutdown of the sender task.

        This method signals the background sender to stop processing events and exit.
        Any event currently queued is discarded (not sent) to ensure immediate shutdown.

        The sender task will complete its current send operation (if any) and then
        terminate. Use `join()` to wait for the task to fully complete.

        Note:
            This method returns immediately. Use `await join()` to wait for the
            sender task to finish.
        """
        self._shutdown_event.set()
        self._new_event.set()  # wake sender if waiting

    async def join(self):
        """
        Wait for the background sender task to complete.

        This method blocks until the sender task has fully terminated, either due to
        a call to `stop()`, sending a FifoEventShutdown (when `stop_on_shutdown=True`),
        or an unhandled exception in the sender.

        Usage:
            ```python
            limiter.stop()
            await limiter.join()  # Wait for clean shutdown
            ```

        Raises:
            Any exception raised by the sender task will propagate when awaiting join().
        """
        await self._task_sender

    async def _sender(self):
        """
        Background task that manages event timing and transmission.

        This coroutine runs continuously until shutdown is requested, monitoring
        for new events and enforcing rate limits. It implements the following logic:

        1. Wait for a new event to be queued (via `send()` or `stop()`)
        2. Calculate time since last send and enforce minimum interval
        3. Send the queued event when the rate limit allows
        4. Handle shutdown conditions (any remaining queued event is discarded)

        The sender ensures events are transmitted at most once per minimum interval
        (1.0 / max_rate seconds), while always sending the most recent queued event
        (latest-wins strategy).

        When shutdown is triggered (via `stop()` or after sending FifoEventShutdown
        when `stop_on_shutdown=True`), any remaining queued event is discarded to
        ensure immediate termination without waiting for rate limiting intervals.

        This method should not be called directly; it is started automatically
        during initialization.

        Note:
            Exceptions from `connection.send()` are caught and suppressed to prevent
            the sender from crashing. The timestamp is still updated to maintain
            rate limiting even when sends fail.
        """
        loop = asyncio.get_running_loop()
        min_interval = self._min_interval

        while not self._shutdown_event.is_set():
            # Wait for a new event or shutdown
            if self._event_to_send is None:
                self._new_event.clear()
                await self._new_event.wait()  # wakes on send() or stop()
                if self._shutdown_event.is_set():
                    break
                # Defensive: woke but nothing queued
                if self._event_to_send is None:
                    continue

            current_time = loop.time()
            time_since_last = current_time - self._time_last_sent

            # Enforce max rate by waiting if too soon
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=wait_time)
                    # Shutdown triggered
                    break
                except asyncio.TimeoutError:
                    pass  # continue sending

            # Snapshot the event and clear slot (latest-wins)
            event_to_send = self._event_to_send
            self._event_to_send = None

            try:
                await self._connection.send(event_to_send)
            except Exception:
                # Send failed; continue but still advance timestamp
                pass
            finally:
                self._time_last_sent = loop.time()

            # If we just sent a shutdown event and policy says to stop, stop.
            if self._stop_on_shutdown and isinstance(event_to_send, FifoEventShutdown):
                self._shutdown_event.set()
                break
