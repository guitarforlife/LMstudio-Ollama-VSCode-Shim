FROM python:3.11-slim AS builder
WORKDIR /app

# Install build deps (if any) and dependencies in one layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
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

ENV SHIM_LMSTUDIO_BASE="http://host.docker.internal:1234/v1"

EXPOSE 11434
HEALTHCHECK CMD curl -f http://localhost:11434/health || exit 1

CMD ["python", "main.py"]
