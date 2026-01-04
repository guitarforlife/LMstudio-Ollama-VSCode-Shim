"""Ollama-compatible endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend import (
    BackendClient,
    ModelCache,
    _entry_is_loaded,
    _extract_model_id,
    _ollama_model_name,
    ollama_base_fields,
    ollama_done,
    ollama_error,
)
from backend_api import api as backend_api
from deps import get_client, get_model_cache
from state import settings
from utils import json
from utils.model_selection import prepare_payload
from utils.pydantic import OllamaBaseModel
from utils.time import now

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_sse_line(line_text: str) -> tuple[Optional[str], bool, bool]:
    """Parse an SSE line into (payload, done, is_data)."""
    if not line_text:
        return None, False, False
    if line_text.startswith("data:"):
        data = line_text[5:].strip()
        if not data:
            return None, False, True
        if data == "[DONE]":
            return None, True, True
        return data, False, True
    if line_text.startswith("{") and "\"error\"" in line_text:
        return line_text + "\n", False, False
    return None, False, False


def _generate_from_payload(payload: str, response_model: str) -> Optional[str]:
    """Convert a completion SSE payload into an Ollama NDJSON line."""
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("generate stream skipped non-JSON data: %s", payload)
        return None
    delta = chunk["choices"][0].get("text")
    if not delta:
        return None
    return (
        json.dumps(
            {
                **ollama_base_fields(response_model),
                "response": delta,
                "done": False,
            }
        )
        + "\n"
    )


def _chat_from_payload(
    payload: str,
    response_model: str,
    role_sent: bool,
) -> tuple[Optional[str], bool]:
    """Convert a chat SSE payload into an Ollama NDJSON line."""
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("chat stream skipped non-JSON data: %s", payload)
        return None, role_sent
    delta = chunk["choices"][0].get("delta", {})
    if not delta:
        return None, role_sent
    if not role_sent and "role" not in delta:
        delta = {"role": "assistant", **delta}
        role_sent = True
    elif "role" in delta:
        role_sent = True
    return (
        json.dumps(
            {
                **ollama_base_fields(response_model),
                "message": delta,
                "done": False,
            }
        )
        + "\n",
        role_sent,
    )


async def _generate_stream(
    client: httpx.AsyncClient,
    backend_client: BackendClient,
    payload: Dict[str, Any],
    response_model: str,
    shutdown_event: asyncio.Event,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Stream generate responses as NDJSON."""
    buffer = b""

    def on_error(message: str) -> Iterable[bytes]:
        return [ollama_error(response_model, message).encode()]

    async for chunk in backend_api.stream_post_raw(
        client,
        backend_client.openai_url("/completions"),
        payload,
        log_label="generate",
        on_error=on_error,
    ):
        # Abort if client disconnects (e.g., user pressed Stop)
        if await request.is_disconnected():
            logger.debug("Client disconnected – aborting generate stream")
            return
        if shutdown_event.is_set():
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line_text = line.decode("utf-8", errors="replace").strip()
            payload_text, done, is_data = _parse_sse_line(line_text)
            if done:
                yield ollama_done(response_model)
                return
            if payload_text and not is_data:
                yield payload_text
                continue
            if payload_text:
                result = _generate_from_payload(payload_text, response_model)
                if result:
                    yield result

    if buffer:
        line_text = buffer.decode("utf-8", errors="replace").strip()
        payload_text, done, is_data = _parse_sse_line(line_text)
        if done:
            yield ollama_done(response_model)
        elif payload_text and not is_data:
            yield payload_text
        elif payload_text:
            result = _generate_from_payload(payload_text, response_model)
            if result:
                yield result


def _build_chat_messages(req: "ChatRequest") -> List[Dict[str, Any]]:
    """Build chat messages for the backend."""
    messages = [message.model_dump() for message in req.messages]
    if req.system:
        messages.insert(0, {"role": "system", "content": req.system})
    return messages


async def _chat_stream(
    client: httpx.AsyncClient,
    backend_client: BackendClient,
    payload: Dict[str, Any],
    response_model: str,
    shutdown_event: asyncio.Event,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Stream chat responses as NDJSON."""
    buffer = b""
    role_sent = False

    def on_error(message: str) -> Iterable[bytes]:
        return [ollama_error(response_model, message).encode()]

    async for chunk in backend_api.stream_post_raw(
        client,
        backend_client.openai_url("/chat/completions"),
        payload,
        log_label="chat",
        on_error=on_error,
    ):
        # Abort if client disconnects (e.g., user pressed Stop)
        if await request.is_disconnected():
            logger.debug("Client disconnected – aborting chat stream")
            return
        if shutdown_event.is_set():
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line_text = line.decode("utf-8", errors="replace").strip()
            payload_text, done, is_data = _parse_sse_line(line_text)
            if done:
                yield ollama_done(response_model)
                return
            if payload_text and not is_data:
                yield payload_text
                continue
            if payload_text:
                result, role_sent = _chat_from_payload(
                    payload_text,
                    response_model,
                    role_sent,
                )
                if result:
                    yield result

    if buffer:
        line_text = buffer.decode("utf-8", errors="replace").strip()
        payload_text, done, is_data = _parse_sse_line(line_text)
        if done:
            yield ollama_done(response_model)
        elif payload_text and not is_data:
            yield payload_text
        elif payload_text:
            result, _ = _chat_from_payload(payload_text, response_model, role_sent)
            if result:
                yield result

class TagItem(BaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible tag entry."""

    name: str
    model: str
    modified_at: str
    size: int = 0
    digest: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class TagResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible tag list response."""

    models: List[TagItem]


@router.get("/api/tags", response_model=TagResponse)
async def tags(
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> Dict[str, Any]:
    """Return Ollama-compatible tag list (models)."""
    models = await backend_api.models(client, model_cache)
    tags_list = []
    for entry in models:
        model_id = _extract_model_id(entry)
        if not model_id:
            continue
        name = model_id if ":" in model_id else f"{model_id}:latest"
        details = entry.details or {"family": model_id, "state": entry.state}
        tags_list.append(
            {
                "name": name,
                "model": name,
                "modified_at": now(),
                "size": int(entry.size or 0),
                "digest": entry.digest or "",
                "details": details,
            }
        )
    return {"models": tags_list}


class ShowRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible show request body."""

    name: Optional[str] = None
    model: Optional[str] = None
    id: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class ShowResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible show response body."""

    license: str
    modelfile: str
    parameters: Dict[str, Any]
    template: str
    capabilities: List[str]
    details: Dict[str, Any]
    model_info: Dict[str, Any]
    model_config = ConfigDict(protected_namespaces=())


@router.post("/api/show", response_model=ShowResponse)
async def show(req: ShowRequest) -> Response:
    """Return Ollama-compatible model detail payload."""
    name = req.name or req.model or req.id or ""
    if not name:
        return JSONResponse({"error": "missing model name"}, status_code=400)
    model = _ollama_model_name(name)
    return JSONResponse(
        {
            "license": "unknown",
            "modelfile": "",
            "parameters": {},
            "template": "",
            "capabilities": ["completion", "chat", "tools"],
            "details": {"family": model},
            "model_info": {},
        }
    )


class GenerateRequest(OllamaBaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible generate request body."""

    model: str
    prompt: str
    temperature: float = Field(default_factory=lambda: settings.default_temperature)
    stop: Optional[str] = None
    stream: bool = True
    keep_alive: Optional[Any] = None

class GenerateResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible generate response body."""

    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str


@router.post("/api/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    request: Request,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> Response:
    """Generate text from a prompt (Ollama-compatible)."""
    response_model = req.model or _ollama_model_name(req.model)
    stop_value = req.stop if req.stop is not None else settings.default_stop
    payload: Dict[str, Any] = {
        "model": req.model,
        "prompt": req.prompt,
        "temperature": req.temperature,
        "stop": stop_value,
        "stream": req.stream,
    }
    payload = await prepare_payload(client, model_cache, req.model, payload, req.keep_alive)

    if not payload["stream"]:
        r = await backend_api.post_openai_json(client, model_cache, "/completions", payload)
        text = r["choices"][0].get("text", "")
        return JSONResponse(
            {
                **ollama_base_fields(response_model),
                "response": text,
                "done": True,
                "done_reason": "stop",
            }
        )

    await backend_api.preflight(client)
    backend_client = BackendClient(client, model_cache)

    return StreamingResponse(
        _generate_stream(
            client,
            backend_client,
            payload,
            response_model,
            request.app.state.shutdown_event,
            request,
        ),
        media_type="application/x-ndjson",
    )


class ChatMessage(BaseModel):  # pylint: disable=too-few-public-methods
    """Chat message payload."""

    role: str
    content: str


class ChatRequest(OllamaBaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible chat request body."""

    model: str
    messages: List[ChatMessage] = Field(default_factory=list)
    system: Optional[str] = None
    temperature: float = Field(default_factory=lambda: settings.default_temperature)
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None

@router.post("/api/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> Response:
    """Chat with a model using messages (Ollama-compatible)."""
    response_model = req.model or _ollama_model_name(req.model)
    messages = _build_chat_messages(req)
    if req.system is None and settings.default_system_prompt:
        messages.insert(0, {"role": "system", "content": settings.default_system_prompt})

    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "temperature": req.temperature,
        "stream": req.stream,
        "tools": req.tools,
        "tool_choice": req.tool_choice,
    }
    payload = await prepare_payload(client, model_cache, req.model, payload, req.keep_alive)

    if not payload["stream"]:
        r = await backend_api.post_openai_json(client, model_cache, "/chat/completions", payload)
        msg = r["choices"][0].get("message", {})
        return JSONResponse(
            {
                **ollama_base_fields(response_model),
                "message": msg,
                "done": True,
                "done_reason": "stop",
            }
        )

    await backend_api.preflight(client)
    backend_client = BackendClient(client, model_cache)

    return StreamingResponse(
        _chat_stream(
            client,
            backend_client,
            payload,
            response_model,
            request.app.state.shutdown_event,
            request,
        ),
        media_type="application/x-ndjson",
    )


class EmbeddingsRequest(OllamaBaseModel):  # pylint: disable=too-few-public-methods
    """Ollama-compatible embeddings request body."""

    model: str
    prompt: str
    keep_alive: Optional[Any] = None

@router.post("/api/embeddings")
async def embeddings(
    req: EmbeddingsRequest,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> Dict[str, Any]:
    """Generate embeddings for a prompt (Ollama-compatible)."""
    payload: Dict[str, Any] = {"model": req.model, "input": req.prompt}
    payload = await prepare_payload(client, model_cache, req.model, payload, req.keep_alive)

    r = await backend_api.post_openai_json(client, model_cache, "/embeddings", payload)
    data = r["data"][0]["embedding"]
    return {"embedding": data}


@router.get("/api/ps")
async def ps(
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> Dict[str, Any]:
    """Return best-effort process status (loaded models)."""
    models = await backend_api.models(client, model_cache)
    loaded = []
    for entry in models:
        if _entry_is_loaded(entry):
            mid = _extract_model_id(entry)
            if mid:
                loaded.append({"name": mid})
    return {"models": loaded}


@router.post("/api/create")
async def create() -> JSONResponse:
    """Stub: create a model (always success)."""
    return JSONResponse({"status": "success"}, status_code=200)


@router.post("/api/pull")
async def pull() -> JSONResponse:
    """Stub: pull a model (always success)."""
    return JSONResponse({"status": "success"}, status_code=200)


@router.post("/api/push")
async def push() -> JSONResponse:
    """Stub: push a model (always success)."""
    return JSONResponse({"status": "success"}, status_code=200)


class DeleteRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Best-effort delete request body."""

    name: Optional[str] = None
    model: Optional[str] = None
    id: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


@router.post("/api/delete")
async def delete(
    req: DeleteRequest,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> JSONResponse:
    """Best-effort delete/unload (maps to a short TTL request)."""
    unload_ttl = int(settings.unload_ttl_seconds or 0)
    name = req.name or req.model or req.id or ""
    if name and unload_ttl > 0:
        model = await backend_api.ensure_selected(client, model_cache, name)
        payload = {
            "model": model,
            "prompt": "",
            "max_tokens": 0,
            "stream": False,
            "ttl": unload_ttl,
        }
        try:
            await backend_api.post_openai_json(client, model_cache, "/completions", payload)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    return JSONResponse({"status": "success"}, status_code=200)
