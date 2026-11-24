#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-api}"
WORKSPACE_DIR="${OMNIFORGE_PATHS__WORKSPACE_DIR:-${OMNIFORGE_WORKSPACE_DIR:-/var/lib/omniforge}}"
mkdir -p "${WORKSPACE_DIR}"

case "$MODE" in
  api)
    # Default to 1 worker because Workbench sessions and job execution are process-local.
    # Job status is SQLite-backed; multi-worker workspaces need sticky or shared state.
    exec uvicorn omniforge.api.main:app \
      --host "${OMNIFORGE_API_HOST:-0.0.0.0}" \
      --port "${OMNIFORGE_API_PORT:-8000}" \
      --workers "${OMNIFORGE_API_WORKERS:-1}"
    ;;
  ui)
    exec python -m omniforge.launcher --mode ui \
      --host "${OMNIFORGE_UI_HOST:-0.0.0.0}" \
      --ui-port "${OMNIFORGE_UI_PORT:-8501}" \
      --no-browser
    ;;
  worker)
    exec omniforge monitor \
      --interval "${OMNIFORGE_MONITOR_INTERVAL:-60}" \
      --max-cycles "${OMNIFORGE_MONITOR_MAX_CYCLES:-100000}"
    ;;
  cli)
    shift || true
    exec omniforge "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
