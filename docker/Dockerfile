FROM python:3.15-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYYAML_FORCE_LIBYAML=0

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels ".[api,ui,reporting,forecasting]"

FROM python:3.15-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMNIFORGE_ENVIRONMENT=production \
    OMNIFORGE_WORKSPACE_DIR=/var/lib/omniforge \
    OMNIFORGE_API_HOST=0.0.0.0 \
    OMNIFORGE_API_PORT=8000 \
    OMNIFORGE_UI_HOST=0.0.0.0 \
    OMNIFORGE_UI_PORT=8501

RUN groupadd --gid 1000 omniforge && \
    useradd --uid 1000 --gid omniforge --create-home --shell /bin/bash omniforge && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY configs /app/configs
COPY src /app/src
COPY docker/entrypoint.sh /usr/local/bin/omniforge-entrypoint.sh
RUN chmod +x /usr/local/bin/omniforge-entrypoint.sh && \
    mkdir -p /var/lib/omniforge && \
    chown -R omniforge:omniforge /app /var/lib/omniforge

USER omniforge

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

ENTRYPOINT ["/usr/local/bin/omniforge-entrypoint.sh"]
CMD ["api"]
