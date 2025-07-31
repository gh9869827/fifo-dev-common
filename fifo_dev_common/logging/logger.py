import logging
from typing import Optional


TRACE_LEVEL_NUM = 5


class _FifoLogger(logging.Logger):
    """
    Internal Logger subclass that adds support for the TRACE level.

    Use `logger.trace(...)` to log at TRACE level (level 5).
    """

    def trace(self, msg: str, *args: object, **kwargs: object) -> None:
        """
        Log 'msg % args' with severity 'TRACE'.

        Parameters:
            msg (str):
                The message format string.

            *args:
                Arguments for the message format string.

            **kwargs:
                Additional keyword arguments passed to the logger.
        """
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            # type ignore is used because Pylance/mypy can't validate **kwargs types for _log()
            self._log(TRACE_LEVEL_NUM, msg, args, **kwargs)  # type: ignore[arg-type]


# Register TRACE level with logging
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

# Use FifoLogger as the global logger class
logging.setLoggerClass(_FifoLogger)

def get_logger(name: Optional[str] = None) -> _FifoLogger:
    """
    Return a logger instance with TRACE level support.

    Parameters:
        name (str | None):
            Logger name. `None` returns the root logger.

    Returns:
        FifoLogger:
            Logger instance with `.trace()` method and TRACE level enabled.
    """
    logger = logging.getLogger(name)
    assert isinstance(logger, _FifoLogger)
    return logger
