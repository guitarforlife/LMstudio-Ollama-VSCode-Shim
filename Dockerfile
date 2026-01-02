# Under Development

FROM python:3.11-slim AS builder
WORKDIR /app

# Install build deps (if any) and dependencies in one layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage (optional)
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r nonroot \
    && useradd -r -g nonroot nonroot
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
RUN chown -R nonroot:nonroot /app
USER nonroot

ENV SHIM_LMSTUDIO_BASE="http://localhost:1234/v1"
ENV SHIM_HTTP_TIMEOUT="300"
ENV SHIM_DEBUG="0"
ENV SHIM_OLLAMA_VERSION="0.13.5"
ENV SHIM_ALLOWED_ORIGINS="*"

EXPOSE 11434
HEALTHCHECK CMD curl -f http://localhost:11434/health || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "11434"]
