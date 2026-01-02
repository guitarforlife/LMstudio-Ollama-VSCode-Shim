"""Logging configuration helpers for the shim."""

import logging
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
logger = logging.getLogger("lmstudio_shim")


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
