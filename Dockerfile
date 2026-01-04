FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN python -m pip wheel --wheel-dir /wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN groupadd -r nonroot \
    && useradd -r -g nonroot nonroot \
    && chown -R nonroot:nonroot /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels
COPY --chown=nonroot:nonroot . .

USER nonroot
ENV SHIM_LMSTUDIO_BASE="http://host.docker.internal:1234/v1"
EXPOSE 11434
HEALTHCHECK CMD python -c "import urllib.request, sys; urllib.request.urlopen('http://localhost:11434/health').read(); sys.exit(0)"

CMD ["python", "main.py"]
