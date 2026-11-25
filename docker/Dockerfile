ARG PYTHON_IMAGE=python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
FROM ${PYTHON_IMAGE} AS builder
ARG TARGETARCH
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY docker/requirements-build.lock docker/requirements-${TARGETARCH}.lock ./
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi && \
    python -m venv /opt/venv && \
    python -m pip install --require-hashes --only-binary=:all: -r requirements-build.lock && \
    /opt/venv/bin/python -m pip install --require-hashes --only-binary=:all: -r "requirements-${TARGETARCH}.lock"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN --network=none python -m pip wheel --no-build-isolation --no-deps --wheel-dir=/wheels . && \
    /opt/venv/bin/python -m pip install --no-deps /wheels/*.whl && \
    /opt/venv/bin/python -m pip check

FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/var/lib/omniforge/matplotlib \
    OMNIFORGE_ENVIRONMENT=production \
    OMNIFORGE_WORKSPACE_DIR=/var/lib/omniforge \
    OMNIFORGE_PATHS__WORKSPACE_DIR=/var/lib/omniforge \
    OMNIFORGE_API_HOST=0.0.0.0 \
    OMNIFORGE_API_PORT=8000 \
    OMNIFORGE_API_WORKERS=1 \
    OMNIFORGE_UI_HOST=0.0.0.0 \
    OMNIFORGE_UI_PORT=8501

RUN groupadd --gid 1000 omniforge && \
    useradd --uid 1000 --gid omniforge --create-home --shell /bin/bash omniforge && \
    install -d -o omniforge -g omniforge /var/lib/omniforge
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs /app/configs
RUN chmod -R a+rX /app/configs
COPY --chmod=0644 docker/healthcheck.py /app/healthcheck.py
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/omniforge-entrypoint.sh

USER omniforge
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 CMD ["python", "/app/healthcheck.py"]
ENTRYPOINT ["/usr/local/bin/omniforge-entrypoint.sh"]
CMD ["api"]
