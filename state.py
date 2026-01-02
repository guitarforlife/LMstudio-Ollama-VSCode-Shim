"""Application state shared across routers."""

from __future__ import annotations

import logging

from config import Settings
from constants import SHIM_VERSION as DEFAULT_SHIM_VERSION
from logging_config import logger, setup_logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pylint: disable=broad-exception-caught
    pass

try:
    settings = Settings()
except ValueError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("lmstudio_shim").error("Invalid configuration: %s", exc)
    raise

setup_logging(settings.debug)

logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
logging.getLogger("httpx").setLevel(logging.DEBUG if settings.debug else logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SHUTDOWN_IN_PROGRESS = False

LMSTUDIO_OPENAI_BASE = settings.lmstudio_base.rstrip("/")
LMSTUDIO_REST_BASE = settings.rest_base.rstrip("/")
OLLAMA_VERSION = settings.ollama_version
SHIM_VERSION = DEFAULT_SHIM_VERSION
STREAM_CONTENT_TYPE = "text/event-stream"
