# [shim](README.md) – Ollama ↔ LM Studio Bridge

## Overview
`shim` is a lightweight **FastAPI‑based shim** that translates requests from an Ollama‑compatible client into calls against a **LM Studio** backend.  It provides the same HTTP interface that Ollama expects while delegating all model‑related work to LM Studio's REST and OpenAI‑compatible endpoints.  The project is deliberately minimal: it contains only the routing, request handling, caching, and streaming logic required to make an Ollama client think it is talking directly to a model server.

---
## High‑Level Architecture
```
+----------------------+      +--------------------------+
|   Ollama Client     | ---> |  shim (FastAPI)        |
+----------------------+      +-----------+--------------+
                                      | 1. HTTP request (GET/POST)
                                      v
                               +------+-------------------+
                               |  routes/*.py (routers)   |
                               +------+-------------------+
                                      | 2. FastAPI calls backend
                                      v
+----------------------+      +------+-------------------+
|   LM Studio API    | <--- |  backend.py (core logic) |
+----------------------+      +--------------------------+
```

* **FastAPI entry point** – `main.py` creates the FastAPI app, installs middleware (CORS, request‑size limiting, request‑id handling, API‑key checking), and starts a Uvicorn server.
* **Routing** – The `routes/` package defines three routers:
  * `health_router` – health‑check endpoint.
  * `openai_router` – OpenAI‑compatible endpoints (`/v1/models`, `/v1/chat/completions`, …).
  * `ollama_router` – Ollama‑style endpoints (`/api/generate`, `/api/chat`).
* **Backend core** – `backend.py` implements:
  * **ModelCache** – in‑memory TTL cache for model listings.
  * **ModelSelector** – LRU cache and concurrency‑safe resolution of the active model.
  * **Streaming utilities** – `_stream_post_raw` streams raw bytes from LM Studio to the client, handling chunking and error conversion.
  * **Error handling** – `BackendError` / `BackendUnavailableError` provide consistent JSON error responses.
* **Facade** – `backend_api.py` offers a thin wrapper (`BackendAPI`) that the test suite and external callers can import without pulling in FastAPI internals.
* **Utilities** – Helper modules under `utils/` provide JSON handling, HTTP request wrappers with retry logic (`proxy_request`), and type definitions.
* **Configuration** – All runtime settings live in `state.py` (host, port, cache TTL, request‑retry policy, etc.) and can be overridden via environment variables.

---
## Core Modules Explained
| File | Purpose |
|------|--------|
| `main.py` | FastAPI application factory, lifecycle management (`lifespan`), server start (`run`). |
| `backend.py` | Model cache, model selector, streaming logic, error handling, LM Studio request helpers. |
| `backend_api.py` | Public facade exposing async methods (`models`, `ensure_selected`, `post_openai_json`, `preflight`, `stream_post_raw`). |
| `client.py` | Thin wrapper around `httpx.AsyncClient` with retry and timeout defaults. |
| `routes/*.py` | FastAPI routers that map HTTP endpoints to the backend functions. |
| `utils/*` | JSON helpers, request retry logic, type definitions (`BackendLike`). |
| `state.py` | Global configuration object (read from environment variables) and shutdown flag. |
| `constants.py` | Centralised error‑code strings used in JSON responses. |

---
## Request Flow Example (Chat Completion)
1. **Client request** – An Ollama client POSTs to `/api/chat` with a JSON payload.
2. **FastAPI router** – `ollama_router` receives the request and forwards it to `backend_api.stream_post_raw`.
3. **Backend API** – The wrapper builds a streaming request to LM Studio's OpenAI‑compatible endpoint (`/v1/chat/completions`).
4. **Streaming** – `_stream_post_raw` opens an HTTPX stream, yields chunks to the caller while converting any transport errors into SSE‑style error messages.
5. **Response** – The FastAPI endpoint streams the bytes back to the client, preserving Ollama's expected SSE format.

---
## Caching & Model Selection
* **ModelCache** – Stores a tuple of `ModelEntry` objects for a configurable TTL (`state.settings.model_cache_ttl_seconds`).
* **ModelSelector** – Provides an async lock‑protected LRU cache (`_cache`) to avoid repeated model resolution.  The selector also logs changes in the active model.

---
## Error Handling & Logging
* Errors from LM Studio are wrapped in `BackendError` (status‑code, error string, detail) or `BackendUnavailableError` for connectivity problems.
* All log messages include a request‑ID context (`request_id_ctx`) to correlate logs with individual API calls.
* The `backend_error_handler` and `backend_unavailable_handler` FastAPI exception handlers return a consistent JSON error schema.

---
## Testing Strategy
The `tests/` directory contains pytest suites that exercise:
* Model cache behaviour (`test_ttl_processor.py`).
* API endpoint correctness (`test_health.py`, `test_openai_router.py`).
* Error handling paths and request‑size limits.
Running `pytest` in the repository executes all unit tests; they currently pass (`exit code 0`).

---
## Dependencies & Runtime Requirements
| Dependency | Reason |
|-----------|--------|
| `fastapi` | HTTP framework with async support. |
| `uvicorn` | ASGI server; optional `uvloop` for performance. |
| `httpx` | Async HTTP client used to talk to LM Studio. |
| `pydantic` | Data validation for model payloads (`ModelEntry`, `ModelsPayload`). |
| `pytest` | Test runner. |
| `ruff` / `black` (dev) | Code style enforcement.

---
## Running the Shim
```bash
# Install dependencies (prefer a venv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: pip install uvloop   # for faster event loop

# Start the server (default host=0.0.0.0, port=8000)
python -m shim.main   # or `shim/main.py` directly
```
Environment variables that affect behaviour are documented in `state.py` – e.g. `LMSTUDIO_OPENAI_BASE`, `MODEL_CACHE_TTL_SECONDS`, `WORKERS`.

---
## Extensibility Points
* **Middleware** – Additional FastAPI middleware can be added to `main.py` (e.g., authentication, rate limiting).
* **Client factory** – Override `client_factory` to inject custom HTTP client settings (proxy, TLS config).
* **Model selector cache size** – Adjust via `MODEL_SELECTOR_CACHE_SIZE` env var.
* **Streaming chunk size** – Controlled by `STREAM_CHUNK_SIZE` in the settings module.

---
## Summary
`shim` provides a thin, well‑structured bridge that lets existing Ollama clients talk to an LM Studio server without any code changes on the client side.  Its design isolates concerns cleanly:
* **Routing** – FastAPI routers map the public API.
* **Backend logic** – All interaction with LM Studio lives in `backend.py`.
* **Caching & selection** – ModelCache + ModelSelector keep repeated lookups cheap.
* **Streaming** – Efficient chunked streaming with robust error handling.
The project is ready for production use, but can be extended with additional authentication layers, metrics collection, or alternative back‑ends if needed.

---
*Generated by GitHub Copilot using the Ollama Language Model Extension, this Shim, and LM Studio hosting OpenAI's gpt-oss-safeguard-120b-mlx:latest model on 2026‑01‑02.*