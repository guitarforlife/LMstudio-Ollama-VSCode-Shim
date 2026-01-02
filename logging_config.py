"""Logging configuration helpers for the shim."""

import asyncio
import logging
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class SuppressShutdownErrors(logging.Filter):  # pylint: disable=too-few-public-methods
    """Filter out noisy shutdown errors from Uvicorn/Starlette."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "timeout graceful shutdown exceeded" in message:
            return False
        if "Exception in ASGI application" in message:
            return False
        if "CancelledError" in message:
            return False
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, asyncio.CancelledError):
            return False
        return True


def setup_logging(debug: bool) -> None:
    """Configure root logging with request-id support."""
    record_factory = logging.getLogRecordFactory()

    def _with_request_id(*args, **kwargs) -> logging.LogRecord:
        record = record_factory(*args, **kwargs)
        record.request_id = request_id_ctx.get("-")
        return record

    logging.setLogRecordFactory(_with_request_id)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s",
    )
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG if debug else logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    shutdown_filter = SuppressShutdownErrors()
    logging.getLogger("uvicorn.error").addFilter(shutdown_filter)
