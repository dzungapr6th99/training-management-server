#!/usr/bin/env bash
# Training Studio (Linux/macOS): set up on the first run, create the database, start the web app -> http://127.0.0.1:8700
#   ./studio.sh            run        ./studio.sh initdb   only create/refresh the DB        ./studio.sh resetdb   recreate the DB
#   Needs Python 3.10+, internet (first run), and Node 18+ only when frontend/dist is missing.
#   Env: STUDIO_HOST (127.0.0.1) STUDIO_PORT (8700) STUDIO_DB STUDIO_VENV (backend/.venv). The app has NO login; keep it on 127.0.0.1.
#   Training jobs need the Windows training environment (env\python.exe); on Linux/macOS the app is a data viewer/manager.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Settings from .env (copy .env.example). Variables already set in the environment win; empty values are skipped.
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r k v || [ -n "$k" ]; do
    k="${k%%[[:space:]]*}"; v="${v%$'\r'}"
    case "$k" in ''|\#*) continue ;; esac
    [ -n "$v" ] && [ -z "${!k+x}" ] && export "$k=$v"
  done < "$ROOT/.env"
fi
cd "$ROOT/backend"
VENV="${STUDIO_VENV:-$ROOT/backend/.venv}"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "[studio] first run: creating the Python environment in $VENV"
  python3 -m venv "$VENV"
  "$PY" -m pip install -q -r requirements.txt
fi
if [ ! -f "$ROOT/frontend/dist/index.html" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[studio] building the frontend..."
    (cd "$ROOT/frontend" && npm install --no-audit --no-fund && npm run build)
  else
    echo "[studio] frontend/dist is missing and Node/npm is not installed: the API will run but there is no web page."
  fi
fi
case "${1:-run}" in
  resetdb) exec "$PY" -m app.init_db --reset ;;
  initdb)  exec "$PY" -m app.init_db ;;
esac
"$PY" -m app.init_db
# Training environment (torch + isaacsim + lerobot): installs what is missing; STUDIO_AUTO_SETUP_TRAIN_ENV=0 only checks.
TRAIN_ENV="${STUDIO_TRAIN_ENV:-$ROOT/env_isaacsim}"
case "$TRAIN_ENV" in /*) ;; *) TRAIN_ENV="$ROOT/$TRAIN_ENV" ;; esac
TRAIN_MODE=ensure; [ "${STUDIO_AUTO_SETUP_TRAIN_ENV:-1}" = 0 ] && TRAIN_MODE=check
if bash "$ROOT/setup_train_env.sh" "$TRAIN_MODE"; then
  export STUDIO_TRAIN_PYTHON="${STUDIO_TRAIN_PYTHON:-$TRAIN_ENV/bin/python}"
  export STUDIO_ISAAC_SIM_PYTHON="${STUDIO_ISAAC_SIM_PYTHON:-$STUDIO_TRAIN_PYTHON}"
else
  echo "[studio] training environment not ready: train jobs may be unavailable, see README_STUDIO.md"
fi
HOST="${STUDIO_HOST:-127.0.0.1}"; PORT="${STUDIO_PORT:-8700}"
echo "[studio] http://$HOST:$PORT   (Ctrl+C to stop)"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
