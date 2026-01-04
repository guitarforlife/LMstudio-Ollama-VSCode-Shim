"""Ollama-compatible endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend import (
    BackendClient,
    ModelEntry,
    ModelCache,
    _entry_is_loaded,
    _extract_model_id,
    _ollama_model_name,
    ollama_base_fields,
    ollama_done,
    ollama_error,
)
from backend_api import api as backend_api
from client import BackendError, RequestOptions, request_json
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


_SIZE_RE = re.compile(
    r"^(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>[KMGTP]?B)?$",
    re.IGNORECASE,
)


def _coerce_size_bytes(  # pylint: disable=too-many-return-statements
    value: Any,
) -> Optional[int]:
    """Best-effort parse of size values into bytes."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        match = _SIZE_RE.match(text)
        if not match:
            return None
        number = float(match.group("value"))
        unit = (match.group("unit") or "B").upper()
        scale = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
            "PB": 1024**5,
        }.get(unit)
        if scale is None:
            return None
        return int(number * scale)
    return None


def _coerce_int(value: Any) -> Optional[int]:  # pylint: disable=too-many-return-statements
    """Coerce basic numeric types and strings to int."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
    return None


def _extract_context_length(*sources: Optional[Dict[str, Any]]) -> Optional[int]:
    """Extract a context length from available payloads."""
    for source in sources:
        if not source:
            continue
        for key in ("context_length", "max_context_length", "loaded_context_length"):
            parsed = _coerce_int(source.get(key))
            if parsed is not None and parsed > 0:
                return parsed
    return None


def _normalize_family_key(value: Any) -> Optional[str]:
    """Normalize a family/arch label into a model_info prefix."""
    if not isinstance(value, str):
        return None
    slug = re.sub(r"[^a-z0-9]", "", value.lower())
    return slug or None


def _extract_size_from_details(details: Optional[Dict[str, Any]]) -> int:
    """Find a non-zero size in common LM Studio fields."""
    details = details or {}
    for value in (
        details.get("size"),
        details.get("size_bytes"),
        details.get("bytes"),
        details.get("model_size"),
        details.get("file_size"),
    ):
        parsed = _coerce_size_bytes(value)
        if parsed is not None and parsed > 0:
            return parsed
    return 0


def _extract_size_bytes(entry: ModelEntry, details: Optional[Dict[str, Any]]) -> int:
    """Find a non-zero size in model entry fields and details."""
    parsed = _coerce_size_bytes(entry.size)
    if parsed is not None and parsed > 0:
        return parsed
    return _extract_size_from_details(details)


async def _fetch_model_details(
    client: httpx.AsyncClient,
    backend_client: BackendClient,
    model_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch per-model metadata from the REST API."""
    if not model_id:
        return None
    safe_id = quote(model_id, safe="")
    url = backend_client.rest_url(f"/models/{safe_id}")
    try:
        return await request_json(
            client,
            "GET",
            url,
            options=RequestOptions(
                retries=settings.request_retries,
                backoff=settings.request_retry_backoff,
            ),
        )
    except BackendError as exc:
        if exc.status_code == 404:
            return None
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return None


def _build_details_payload(entry: Optional[ModelEntry]) -> Dict[str, Any]:
    """Build a details payload from the model entry."""
    if entry is None:
        return {}
    details = entry.details or entry.model_dump(exclude_none=True, exclude={"details"})
    model_id = _extract_model_id(entry)
    if model_id:
        details.setdefault("family", model_id)
    if entry.state is not None:
        details.setdefault("state", entry.state)
    return details


async def _generate_stream(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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


async def _chat_stream(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
    backend_client = BackendClient(client, model_cache)
    tags_list = []
    for entry in models:
        model_id = _extract_model_id(entry)
        if not model_id:
            continue
        name = model_id if ":" in model_id else f"{model_id}:latest"
        details = _build_details_payload(entry)
        size = _extract_size_bytes(entry, details)
        if size == 0:
            details_payload = await _fetch_model_details(client, backend_client, model_id)
            if isinstance(details_payload, dict):
                size = _extract_size_from_details(details_payload)
                if size == 0 and isinstance(details_payload.get("details"), dict):
                    size = _extract_size_from_details(details_payload.get("details"))
        digest = entry.digest or hashlib.sha256(name.encode("utf-8")).hexdigest()
        tags_list.append(
            {
                "name": name,
                "model": name,
                "modified_at": now(),
                "size": size,
                "digest": digest,
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
    parameters: str
    template: str
    capabilities: List[str]
    details: Dict[str, Any]
    model_info: Dict[str, Any]
    model_config = ConfigDict(protected_namespaces=())


@router.post("/api/show", response_model=ShowResponse)
async def show(  # pylint: disable=too-many-locals,too-many-branches
    req: ShowRequest,
    client: httpx.AsyncClient = Depends(get_client),
    model_cache: Optional[ModelCache] = Depends(get_model_cache),
) -> Response:
    """Return Ollama-compatible model detail payload."""
    name = req.name or req.model or req.id or ""
    if not name:
        return JSONResponse({"error": "missing model name"}, status_code=400)
    model = _ollama_model_name(name)
    entry: Optional[ModelEntry] = None
    try:
        models = await backend_api.models(client, model_cache)
    except Exception:  # pylint: disable=broad-exception-caught
        models = ()
    for candidate in models:
        mid = _extract_model_id(candidate)
        if not mid:
            continue
        if mid in {name, model} or _ollama_model_name(mid) == model:
            entry = candidate
            break
    backend_client = BackendClient(client, model_cache)
    details = _build_details_payload(entry)
    details_payload = await _fetch_model_details(
        client,
        backend_client,
        _extract_model_id(entry) if entry else name,
    )
    if isinstance(details_payload, dict):
        nested = details_payload.get("details")
        if isinstance(nested, dict):
            details.update(nested)
        details.update({k: v for k, v in details_payload.items() if k != "details"})
    context_length = _extract_context_length(details_payload, details)
    model_info: Dict[str, Any] = {}
    architecture = details.get("arch")
    if isinstance(architecture, str) and architecture:
        model_info["general.architecture"] = architecture
        if context_length is not None:
            model_info[f"{architecture}.context_length"] = context_length
    model_name_key = model.split("/", 1)[-1]
    if model_name_key and context_length is not None:
        model_info[f"{model_name_key}.context_length"] = context_length
    entry_id = _extract_model_id(entry) if entry else ""
    family_value = details.get("family")
    if isinstance(family_value, str) and family_value in {name, model, entry_id}:
        family_value = None
    family_key = _normalize_family_key(family_value)
    if family_key and context_length is not None:
        model_info[f"{family_key}.context_length"] = context_length
    if context_length is not None:
        model_info["context_length"] = context_length
    loaded_context_length = _coerce_int(details.get("loaded_context_length"))
    if loaded_context_length is not None:
        model_info["loaded_context_length"] = loaded_context_length
    max_context_length = _coerce_int(details.get("max_context_length"))
    if max_context_length is not None:
        model_info["max_context_length"] = max_context_length
    return JSONResponse(
        {
            "license": "unknown",
            "modelfile": "",
            "parameters": "",
            "template": "",
            "capabilities": ["completion", "chat", "tools"],
            "details": {"family": model, **details},
            "model_info": model_info,
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
