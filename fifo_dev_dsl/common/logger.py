import logging

TRACE_LEVEL_NUM = 5

# Register TRACE level with logging if not already registered
if not hasattr(logging, 'TRACE'):
    logging.addLevelName(TRACE_LEVEL_NUM, 'TRACE')
    def trace(self, message, *args, **kws):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kws)
    logging.Logger.trace = trace
    logging.TRACE = TRACE_LEVEL_NUM


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger with TRACE level support.

    Args:
        name: Optional logger name. ``None`` returns the root logger.

    Returns:
        logging.Logger: Logger instance configured with TRACE level support.
    """
    return logging.getLogger(name)
