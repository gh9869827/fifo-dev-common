import asyncio

import pytest

from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_protocols import SendStatus, SupportsFifoEventSend
from fifo_dev_common.event.fifo_event_rate_limiter import FifoEventRateLimiter


class DummyEvent(FifoEvent):
    """Minimal event type for exercising the rate limiter."""

    event_id = 42
    default_priority = 0


class FakeConnection(SupportsFifoEventSend):
    def __init__(self) -> None:
        self.sent: list[tuple[FifoEvent, float]] = []
        self._event = asyncio.Event()

    async def send(self, event: FifoEvent) -> SendStatus:  # type: ignore[override]
        loop = asyncio.get_running_loop()
        self.sent.append((event, loop.time()))
        self._event.set()
        return SendStatus.SENT

    async def wait_for_sent(self, count: int, timeout: float = 1.0) -> list[tuple[FifoEvent, float]]:
        loop = asyncio.get_running_loop()
        end_time = loop.time() + timeout
        while len(self.sent) < count:
            remaining = end_time - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"Timed out waiting for {count} sends (have {len(self.sent)}).")
            await asyncio.wait_for(self._event.wait(), remaining)
            self._event.clear()
        return self.sent


@pytest.mark.asyncio
async def test_latest_event_wins_when_rate_limited():
    connection = FakeConnection()
    limiter = FifoEventRateLimiter(connection, max_rate=10.0)  # 100 ms minimum interval
    try:
        first = DummyEvent()
        limiter.send(first)
        await connection.wait_for_sent(1)

        second = DummyEvent()
        third = DummyEvent()
        limiter.send(second)
        await asyncio.sleep(0.01)  # 10 ms, still within the 100 ms interval
        limiter.send(third)

        sent = await connection.wait_for_sent(2)
        events = [event for event, _ in sent]
        assert events[0] is first
        assert events[1] is third

        interval = sent[1][1] - sent[0][1]
        assert interval >= 0.09  # allow small scheduling jitter
    finally:
        limiter.stop()
        await limiter.join()


@pytest.mark.asyncio
async def test_shutdown_event_triggers_auto_stop():
    connection = FakeConnection()
    limiter = FifoEventRateLimiter(connection, max_rate=100.0, stop_on_shutdown=True)

    first = DummyEvent()
    limiter.send(first)
    await connection.wait_for_sent(1)

    limiter.send(FifoEventShutdown())
    limiter.send(DummyEvent())  # ignored because shutdown is pending

    sent = await connection.wait_for_sent(2)
    assert [type(event) for event, _ in sent] == [DummyEvent, FifoEventShutdown]

    await limiter.join()
    assert limiter._shutdown_event.is_set()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stop_discards_pending_event():
    connection = FakeConnection()
    limiter = FifoEventRateLimiter(connection, max_rate=5.0)  # 200 ms interval

    first = DummyEvent()
    limiter.send(first)
    await connection.wait_for_sent(1)

    limiter.send(DummyEvent())
    await asyncio.sleep(0.01)

    limiter.stop()
    await limiter.join()

    assert len(connection.sent) == 1
