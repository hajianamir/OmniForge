#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-api}"
mkdir -p "${OMNIFORGE_WORKSPACE_DIR:-/var/lib/uap}"

case "$MODE" in
  api)
    exec uvicorn omniforge.api.main:app \
      --host "${OMNIFORGE_API_HOST:-0.0.0.0}" \
      --port "${OMNIFORGE_API_PORT:-8000}" \
      --workers "${OMNIFORGE_API_WORKERS:-2}"
    ;;
  ui)
    exec python -m omniforge.launcher --mode ui \
      --host "${OMNIFORGE_UI_HOST:-0.0.0.0}" \
      --ui-port "${OMNIFORGE_UI_PORT:-8501}" \
      --no-browser
    ;;
  worker)
    exec uap monitor --interval "${OMNIFORGE_MONITOR_INTERVAL:-60}" --max-cycles "${OMNIFORGE_MONITOR_MAX_CYCLES:-100000}"
    ;;
  cli)
    shift || true
    exec uap "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
