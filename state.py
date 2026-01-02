"""Application state shared across routers."""

from __future__ import annotations

import logging

from config import Settings
from constants import SHIM_VERSION as DEFAULT_SHIM_VERSION
from logging_config import setup_logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pylint: disable=broad-exception-caught
    pass

logger = logging.getLogger(__name__)

try:
    settings = Settings()
except ValueError as exc:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    logger.error("Invalid configuration: %s", exc)
    raise

setup_logging(settings.debug)

SHUTDOWN_IN_PROGRESS = False

LMSTUDIO_OPENAI_BASE = settings.lmstudio_base.rstrip("/")
LMSTUDIO_REST_BASE = settings.rest_base.rstrip("/")
OLLAMA_VERSION = settings.ollama_version
SHIM_VERSION = DEFAULT_SHIM_VERSION
STREAM_CONTENT_TYPE = "text/event-stream"
