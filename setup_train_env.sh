#!/usr/bin/env bash
# Training environment for Training Studio jobs: Python 3.12 with torch (CUDA), Isaac Sim (pip) and lerobot.
# Installs only what is missing, so it is safe to rerun.
#   ./setup_train_env.sh          install what is missing (several GB the first time, needs internet, NVIDIA driver, GLIBC 2.35+)
#   ./setup_train_env.sh ensure   quick check; installs only if torch / isaacsim / lerobot is missing (studio.sh calls this)
#   ./setup_train_env.sh check    only report, install nothing
# Target Python: STUDIO_TRAIN_PYTHON if set and existing (missing packages go into THAT env), else the venv STUDIO_TRAIN_ENV
# (default ./env_isaacsim). Versions: STUDIO_TORCH_INDEX (cu128), STUDIO_TORCH_VERSION (2.11.0),
# STUDIO_TORCHVISION_VERSION (0.26.0), STUDIO_ISAACSIM_VERSION (6.0.1.0). Settings come from the environment or .env.
# Never installs into an Isaac Sim install's own python.sh. Does not accept the NVIDIA Omniverse EULA for you.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r k v || [ -n "$k" ]; do
    k="${k%%[[:space:]]*}"; v="${v%$'\r'}"
    case "$k" in ''|\#*) continue ;; esac
    [ -n "$v" ] && [ -z "${!k+x}" ] && export "$k=$v"
  done < "$ROOT/.env"
fi
MODE="${1:-install}"
ENV_DIR="${STUDIO_TRAIN_ENV:-$ROOT/env_isaacsim}"
case "$ENV_DIR" in /*) ;; *) ENV_DIR="$ROOT/$ENV_DIR" ;; esac
TORCH_INDEX="${STUDIO_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_V="${STUDIO_TORCH_VERSION:-2.11.0}"
TV_V="${STUDIO_TORCHVISION_VERSION:-0.26.0}"
ISAAC_V="${STUDIO_ISAACSIM_VERSION:-6.0.1.0}"

if [ -n "${STUDIO_TRAIN_PYTHON:-}" ]; then case "$STUDIO_TRAIN_PYTHON" in /*) ;; *) STUDIO_TRAIN_PYTHON="$ROOT/$STUDIO_TRAIN_PYTHON" ;; esac; fi
PY="$ENV_DIR/bin/python"; OWN_VENV=1
if [ -n "${STUDIO_TRAIN_PYTHON:-}" ] && [ -e "$STUDIO_TRAIN_PYTHON" ]; then PY="$STUDIO_TRAIN_PYTHON"; OWN_VENV=0; fi
case "$PY" in *python.sh) echo "[train-env] $PY is an Isaac Sim install's python.sh: packages are never installed there."; exit 1 ;; esac

probe() {  # sets HAVE_TORCH HAVE_ISAAC HAVE_LEROBOT HAVE_ALL MISSING
  HAVE_TORCH=0; HAVE_ISAAC=0; HAVE_LEROBOT=0; HAVE_ALL=0; MISSING=""
  if [ ! -x "$PY" ]; then MISSING=" python torch isaacsim lerobot"; return 0; fi
  "$PY" -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('torch').split('+')[0]=='$TORCH_V' else 1)" >/dev/null 2>&1 && HAVE_TORCH=1
  "$PY" -c "import importlib.metadata as m; m.version('isaacsim')" >/dev/null 2>&1 && HAVE_ISAAC=1
  "$PY" -c "import importlib.metadata as m; m.version('lerobot')" >/dev/null 2>&1 && HAVE_LEROBOT=1
  [ "$HAVE_TORCH" = 1 ] || MISSING="$MISSING torch==$TORCH_V"
  [ "$HAVE_ISAAC" = 1 ] || MISSING="$MISSING isaacsim"
  [ "$HAVE_LEROBOT" = 1 ] || MISSING="$MISSING lerobot"
  if [ "$HAVE_TORCH$HAVE_ISAAC$HAVE_LEROBOT" = 111 ]; then HAVE_ALL=1; fi
  return 0
}

report() {
  [ -x "$PY" ] || { echo "[train-env] no Python at $PY: run ./setup_train_env.sh"; exit 1; }
  "$PY" -c "import sys, importlib.metadata as m; have={(d.metadata['Name'] or '').lower(): d.version for d in m.distributions()}; print('[train-env]', sys.executable, '| python', sys.version.split()[0], *[f'| {n} {have.get(n, \"MISSING\")}' for n in ('torch', 'isaacsim', 'lerobot')])"
  [ "$HAVE_ALL" = 1 ] || { echo "[train-env] missing:$MISSING"; exit 1; }
  echo "[train-env] OK. studio.sh uses it automatically; to pin it put in $ROOT/.env:"
  echo "    STUDIO_TRAIN_PYTHON=$PY"
  echo "    STUDIO_ISAAC_SIM_PYTHON=$PY"
}

probe
case "$MODE" in
  check) report; exit 0 ;;
  ensure) [ "$HAVE_ALL" = 1 ] && exit 0
          echo "[train-env] the training environment is missing:$MISSING - installing it now (set STUDIO_AUTO_SETUP_TRAIN_ENV=0 to skip)" ;;
esac
trap 'echo "[train-env] FAILED (see the output above). Rerun to resume: installed packages are kept."' ERR

echo "[train-env] target python: $PY"
command -v nvidia-smi >/dev/null 2>&1 || echo "[train-env] WARNING: nvidia-smi not found: install the NVIDIA driver before training."
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
[ -n "$GLIBC" ] && echo "[train-env] GLIBC $GLIBC (Isaac Sim needs 2.35+)"

if [ ! -x "$PY" ]; then
  [ "$OWN_VENV" = 1 ] || { echo "[train-env] $PY not found"; exit 1; }
  command -v python3.12 >/dev/null 2>&1 || { echo "[train-env] python3.12 not found: install Python 3.12 and retry"; exit 1; }
  echo "[train-env] creating Python 3.12 venv in $ENV_DIR"
  python3.12 -m venv "$ENV_DIR"
  "$PY" -m pip install --upgrade pip
fi
"$PY" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" || { echo "[train-env] $PY is not Python 3.12 (Isaac Sim 6 needs 3.12)"; exit 1; }
probe

CONSTRAINTS="$(mktemp)"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_V" "$TV_V" > "$CONSTRAINTS"
if [ "$HAVE_TORCH" = 1 ]; then echo "[train-env] torch $TORCH_V: already installed"; else
  echo "[train-env] installing torch $TORCH_V + torchvision $TV_V"
  "$PY" -m pip install "torch==$TORCH_V" "torchvision==$TV_V" --index-url "$TORCH_INDEX"
fi
if [ "$HAVE_ISAAC" = 1 ]; then echo "[train-env] isaacsim: already installed"; else
  echo "[train-env] installing isaacsim $ISAAC_V (large download)"
  "$PY" -m pip install "isaacsim[all,extscache]==$ISAAC_V" --extra-index-url https://pypi.nvidia.com -c "$CONSTRAINTS"
fi
if [ "$HAVE_LEROBOT" = 1 ]; then echo "[train-env] lerobot: already installed"; else
  if [ -f "$ROOT/vendor/lerobot/pyproject.toml" ]; then
    echo "[train-env] installing lerobot[smolvla] from vendor/lerobot"
    "$PY" -m pip install -e "$ROOT/vendor/lerobot[smolvla]" -c "$CONSTRAINTS"
  else
    echo "[train-env] installing lerobot[smolvla] 0.6.2 from PyPI (vendor/lerobot not found)"
    "$PY" -m pip install "lerobot[smolvla]==0.6.2" -c "$CONSTRAINTS"
  fi
fi
rm -f "$CONSTRAINTS"
probe
report
