"""Configuration for the LM Studio shim."""

from __future__ import annotations

import re
from typing import Any, List, Optional

import httpx
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from constants import DEFAULT_LMSTUDIO_BASE, DEFAULT_OLLAMA_VERSION
from constants import DEFAULT_TEMPERATURE


def _derive_rest_base(openai_base: str) -> str:
    """Derive REST base from an OpenAI-compatible base URL."""
    base = openai_base.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")] + "/api/v0"
    if re.search(r"/v\\d+$", base):
        return re.sub(r"/v\\d+$", "/api/v0", base)
    return base + "/api/v0"


def _validate_base_url(value: str, name: str) -> str:
    """Validate a base URL for LM Studio configuration."""
    try:
        url = httpx.URL(value)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ValueError(f"{name} is not a valid URL: {value}") from exc
    if not url.scheme or not url.host:
        raise ValueError(f"{name} must include scheme and host: {value}")
    return value


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All environment variables are expected to be prefixed with ``SHIM_``.
    For example, ``SHIM_LMSTUDIO_BASE`` overrides the default LM Studio base URL.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="SHIM_",
        protected_namespaces=("settings_",),
    )

    lmstudio_base: str = DEFAULT_LMSTUDIO_BASE
    lmstudio_rest_base: Optional[str] = None

    ollama_version: str = DEFAULT_OLLAMA_VERSION
    http_timeout: Optional[float] = 300.0
    debug: bool = False
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost", "http://127.0.0.1"]
    )
    api_key: Optional[str] = None

    # Default TTL applied to all proxied requests if the client doesn't specify keep_alive/ttl.
    # Set to 0 or empty to disable injection (use LM Studio defaults).
    default_ttl_seconds: Optional[int] = 0

    # If a client asks to "unload immediately" (keep_alive=0), we can't force an immediate unload
    # via REST; we approximate by setting ttl to this many seconds for that request.
    # Set to 0 or empty to disable unload requests and keep models loaded.
    unload_ttl_seconds: int = 0
    host: str = "0.0.0.0"
    port: int = 11434

    max_connections: int = 50
    max_keepalive_connections: int = 20
    request_retries: int = 1
    request_retry_backoff: float = 0.25
    verify_ssl: bool = True
    max_request_bytes: Optional[int] = None
    model_cache_ttl_seconds: float = 30.0
    default_temperature: float = DEFAULT_TEMPERATURE
    default_system_prompt: Optional[str] = None
    default_stop: Optional[str] = None

    @property
    def rest_base(self) -> str:
        """Return the derived REST base URL."""
        return self.lmstudio_rest_base or _derive_rest_base(self.lmstudio_base)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [origin.strip() for origin in value if str(origin).strip()]
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        raise ValueError("ALLOWED_ORIGINS must be a comma-separated string")

    @field_validator("http_timeout", mode="before")
    @classmethod
    def _validate_timeout(cls, value: Any) -> Optional[float]:
        if value is None or value == "":
            return 300.0
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("HTTP_TIMEOUT must be a number") from exc
        if parsed == 0:
            return None
        if parsed < 0.1:
            raise ValueError("HTTP_TIMEOUT must be >= 0.1 or 0 for no limit")
        return parsed

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    @field_validator("default_ttl_seconds", mode="before")
    @classmethod
    def _validate_default_ttl(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("DEFAULT_TTL_SECONDS must be an integer") from exc
        if parsed < 0:
            raise ValueError("DEFAULT_TTL_SECONDS must be >= 0")
        return parsed

    @field_validator("lmstudio_base", mode="before")
    @classmethod
    def _validate_lmstudio_base(cls, value: Any) -> str:
        if value is None or value == "":
            return DEFAULT_LMSTUDIO_BASE
        if not isinstance(value, str):
            raise ValueError("LMSTUDIO_BASE must be a string URL")
        return value

    @field_validator("lmstudio_rest_base", mode="before")
    @classmethod
    def _normalize_lmstudio_rest_base(cls, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError("LMSTUDIO_REST_BASE must be a string URL")
        return value

    @field_validator("host", mode="before")
    @classmethod
    def _normalize_host(cls, value: Any) -> str:
        if value is None or value == "":
            return "0.0.0.0"
        if not isinstance(value, str):
            raise ValueError("HOST must be a string")
        return value

    @field_validator("port", mode="before")
    @classmethod
    def _normalize_port(cls, value: Any) -> int:
        if value is None or value == "":
            return 11434
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("PORT must be an integer") from exc
        if parsed <= 0:
            raise ValueError("PORT must be > 0")
        return parsed

    @field_validator("max_connections", "max_keepalive_connections", mode="before")
    @classmethod
    def _normalize_connection_limits(cls, value: Any) -> int:
        if value is None or value == "":
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Connection limits must be integers") from exc
        if parsed < 0:
            raise ValueError("Connection limits must be >= 0")
        return parsed

    @field_validator("request_retries", mode="before")
    @classmethod
    def _normalize_retries(cls, value: Any) -> int:
        if value is None or value == "":
            return 1
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("REQUEST_RETRIES must be an integer") from exc
        if parsed < 0:
            raise ValueError("REQUEST_RETRIES must be >= 0")
        return parsed

    @field_validator("request_retry_backoff", mode="before")
    @classmethod
    def _normalize_backoff(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.25
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("REQUEST_RETRY_BACKOFF must be a number") from exc
        if parsed < 0:
            raise ValueError("REQUEST_RETRY_BACKOFF must be >= 0")
        return parsed

    @field_validator("max_request_bytes", mode="before")
    @classmethod
    def _normalize_max_request_bytes(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("MAX_REQUEST_BYTES must be an integer") from exc
        if parsed <= 0:
            raise ValueError("MAX_REQUEST_BYTES must be > 0")
        return parsed

    @field_validator("model_cache_ttl_seconds", mode="before")
    @classmethod
    def _normalize_model_cache_ttl(cls, value: Any) -> float:
        if value is None or value == "":
            return 30.0
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("MODEL_CACHE_TTL_SECONDS must be a number") from exc
        if parsed < 0:
            raise ValueError("MODEL_CACHE_TTL_SECONDS must be >= 0")
        return parsed

    @field_validator("default_temperature", mode="before")
    @classmethod
    def _normalize_default_temperature(cls, value: Any) -> float:
        if value is None or value == "":
            return DEFAULT_TEMPERATURE
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("DEFAULT_TEMPERATURE must be a number") from exc
        if parsed < 0:
            raise ValueError("DEFAULT_TEMPERATURE must be >= 0")
        return parsed

    @field_validator("default_system_prompt", mode="before")
    @classmethod
    def _normalize_default_system_prompt(cls, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        return str(value)

    @field_validator("default_stop", mode="before")
    @classmethod
    def _normalize_default_stop(cls, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        return str(value)

    @model_validator(mode="after")
    def _finalize_bases(self) -> "Settings":
        rest_base = self.rest_base
        _validate_base_url(self.lmstudio_base, "SHIM_LMSTUDIO_BASE")
        _validate_base_url(rest_base, "SHIM_LMSTUDIO_REST_BASE")
        return self
