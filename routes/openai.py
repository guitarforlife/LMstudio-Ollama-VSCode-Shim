"""OpenAI-compatible endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend import BackendClient, _extract_model_id, openai_stream_error
from backend_api import api as backend_api
from deps import get_client, get_model_cache
from state import STREAM_CONTENT_TYPE, logger, settings
from utils.http import inject_ttl_if_missing

router = APIRouter()


class ChatCompletionRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """OpenAI-compatible chat completion request body."""

    model: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    temperature: float = Field(default_factory=lambda: settings.default_temperature)
    stream: bool = True
    ttl: Optional[int] = None
    keep_alive: Optional[Any] = None
    model_config = ConfigDict(extra="allow")


@router.get("/v1/models")
async def openai_models(
    client: httpx.AsyncClient = Depends(get_client),
    model_cache=Depends(get_model_cache),
) -> Dict[str, Any]:
    """Return OpenAI-compatible model list."""
    models = await backend_api.models(client, model_cache)
    created = int(time.time())
    data = []
    seen = set()
    for entry in models:
        model_id = _extract_model_id(entry)
        if not model_id or model_id in seen:
            continue
        data.append(
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": entry.get("owned_by", "lmstudio"),
            }
        )
        seen.add(model_id)
    logger.info("openai_models count=%s", len(data))
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
async def openai_chat(
    req: ChatCompletionRequest,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache=Depends(get_model_cache),
) -> Response:
    """OpenAI-compatible chat completions endpoint (streaming supported)."""
    logger.debug(
        "openai_chat request",
        extra={"model": req.model, "keep_alive": req.keep_alive, "stream": req.stream},
    )
    body = req.model_dump(exclude_none=True)

    body["model"] = await backend_api.ensure_selected(client, model_cache, req.model)
    body = inject_ttl_if_missing(body, req.keep_alive)
    body.pop("keep_alive", None)

    if not body.get("stream", False):
        result = await backend_api.post_openai_json(
            client,
            model_cache,
            "/chat/completions",
            body,
        )
        return JSONResponse(result)

    await backend_api.preflight(client)
    backend_client = BackendClient(client, model_cache)

    def on_error(message: str) -> Iterable[bytes]:
        return [openai_stream_error(message), b"data: [DONE]\n\n"]

    return StreamingResponse(
        backend_api.stream_post_raw(
            client,
            backend_client.openai_url("/chat/completions"),
            body,
            log_label="openai_chat",
            on_error=on_error,
        ),
        media_type=STREAM_CONTENT_TYPE,
    )


class CompletionRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """OpenAI-compatible completion request body."""

    model: str
    prompt: Optional[Any] = None
    temperature: float = Field(default_factory=lambda: settings.default_temperature)
    stop: Optional[Any] = None
    stream: bool = True
    ttl: Optional[int] = None
    keep_alive: Optional[Any] = None
    model_config = ConfigDict(extra="allow")


@router.post("/v1/completions")
async def openai_completions(
    req: CompletionRequest,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache=Depends(get_model_cache),
) -> Response:
    """OpenAI-compatible text completions endpoint (streaming supported)."""
    body = req.model_dump(exclude_none=True)
    if "stop" not in body and settings.default_stop is not None:
        body["stop"] = settings.default_stop

    body["model"] = await backend_api.ensure_selected(client, model_cache, req.model)
    body = inject_ttl_if_missing(body, req.keep_alive)
    body.pop("keep_alive", None)

    if not body.get("stream", False):
        result = await backend_api.post_openai_json(client, model_cache, "/completions", body)
        return JSONResponse(result)

    await backend_api.preflight(client)
    backend_client = BackendClient(client, model_cache)

    def on_error(message: str) -> Iterable[bytes]:
        return [openai_stream_error(message), b"data: [DONE]\n\n"]

    return StreamingResponse(
        backend_api.stream_post_raw(
            client,
            backend_client.openai_url("/completions"),
            body,
            log_label="openai_completions",
            on_error=on_error,
        ),
        media_type=STREAM_CONTENT_TYPE,
    )
