import logging
import pytest

from fifo_dev_common.logging.logger import get_logger, TRACE_LEVEL_NUM

def test_trace_level_logs(caplog: pytest.LogCaptureFixture):
    logger = get_logger("test_logger")
    with caplog.at_level(TRACE_LEVEL_NUM):
        logger.trace("trace message %d", 123)
    # Check that the trace message is in the captured logs
    assert any("trace message 123" in record.message and record.levelname == "TRACE"
               for record in caplog.records)

def test_trace_level_not_logged_by_default(caplog: pytest.LogCaptureFixture):
    logger = get_logger("test_logger2")
    with caplog.at_level(logging.INFO):
        logger.trace("should not appear")
    # TRACE is lower than INFO, so it should not be logged
    assert all(record.levelname != "TRACE" for record in caplog.records)

def test_get_logger_returns_fifo_logger():
    logger = get_logger("test_logger3")
    # Should have the trace method
    assert hasattr(logger, "trace")
    # Should be an instance of logging.Logger (actually _FifoLogger)
    assert isinstance(logger, logging.Logger)
