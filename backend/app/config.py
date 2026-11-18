"""Settings for the studio backend.

Every value can be set with an environment variable or in `training-management-server/.env` (copy `.env.example`).
Real environment variables win over `.env`. Defaults are derived from this file's location (no hard-coded D:\\ paths),
so an unchanged checkout keeps working. Relative paths in `.env` are resolved against training-management-server/."""
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("studio.config")

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent                      # training-management-server/
ENV_FILE = Path(os.environ.get("STUDIO_ENV_FILE", ROOT / ".env"))


def load_env_file(path):
    """Read KEY=VALUE lines into os.environ without overriding variables that are already set. Returns the count loaded."""
    if not path.is_file():
        return 0
    n = 0
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1]
            elif " #" in val:
                val = val.split(" #", 1)[0].rstrip()
            if key and key not in os.environ:
                os.environ[key] = val
                n += 1
    except OSError as e:
        log.error("cannot read %s: %s", path, e)
        return 0
    return n


load_env_file(ENV_FILE)


def _path(name, default):
    """Path from env var `name` (relative values are taken from ROOT), else `default`."""
    v = os.environ.get(name, "").strip()
    if not v:
        return Path(default)
    p = Path(os.path.expandvars(os.path.expanduser(v)))
    return p if p.is_absolute() else (ROOT / p).resolve()


def _int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        log.error("%s must be an integer; using %s", name, default)
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        log.error("%s must be a number; using %s", name, default)
        return default


# robot description folders: <design>/<design_dir>/ holds each robot's URDF/USD (paths in the DB are relative to it)
DESIGN = _path("STUDIO_DESIGN_DIR", ROOT / "design")
# Isaac Sim install (folder with python.bat / python.sh); falls back to the ISAAC_SIM_DIR used by the sim scripts
ISAAC_SIM_DIR = _path("STUDIO_ISAAC_SIM_DIR", os.environ.get("ISAAC_SIM_DIR", "").strip() or ROOT / "isaacsim")
# Python that can `import isaacsim`: the install's python.bat/python.sh, or a pip venv (setup_train_env.bat) via STUDIO_ISAAC_SIM_PYTHON
ISAAC_SIM_PYTHON = _path("STUDIO_ISAAC_SIM_PYTHON", ISAAC_SIM_DIR / ("python.bat" if sys.platform == "win32" else "python.sh"))

DATASETS = _path("STUDIO_DATASETS_DIR", ROOT / "datasets" / "lerobot")
OUTPUTS = _path("STUDIO_OUTPUTS_DIR", ROOT / "outputs")
RESULTS = _path("STUDIO_RESULTS_DIR", ROOT / "results")
LOGS = _path("STUDIO_LOGS_DIR", ROOT / "logs")
MODELS = _path("STUDIO_MODELS_DIR", ROOT / "models")
HUMANOID_JSON = _path("STUDIO_HUMANOID_JSON", ROOT / "scripts" / "data" / "humanoid.json")

DB_PATH = _path("STUDIO_DB", BACKEND / "data" / "studio.db")
FRONTEND_DIST = _path("STUDIO_FRONTEND_DIST", ROOT / "frontend" / "dist")

# ---- training jobs
_DEFAULT_TRAIN_PY = ROOT / "env" / ("python.exe" if sys.platform == "win32" else "bin/python")
TRAIN_PYTHON = _path("STUDIO_TRAIN_PYTHON", _DEFAULT_TRAIN_PY)
HF_HOME = _path("STUDIO_HF_HOME", MODELS / "hf_home")
# short path on Windows: long cache lock names exceed 260 chars (WinError 206)
HF_DATASETS_CACHE = _path("STUDIO_HF_DATASETS_CACHE", Path(ROOT.anchor) / "hfds")
HF_HUB_OFFLINE = os.environ.get("STUDIO_HF_HUB_OFFLINE", "1").strip() or "1"
SMOLVLA_BASE = {16: _path("STUDIO_SMOLVLA_BASE_16D", MODELS / "smolvla_base_humanoid"),
                8: _path("STUDIO_SMOLVLA_BASE_8D", MODELS / "smolvla_base_yellowarm")}
GPU_MIN_FREE_GIB = _float("STUDIO_GPU_MIN_FREE_GIB", 4.0)
TRAIN_NUM_WORKERS = _int("STUDIO_TRAIN_NUM_WORKERS", 4)
TRAIN_MAX_STEPS = _int("STUDIO_TRAIN_MAX_STEPS", 200000)
TRAIN_MAX_BATCH = _int("STUDIO_TRAIN_MAX_BATCH", 64)
POLICY_DEVICE = os.environ.get("STUDIO_POLICY_DEVICE", "cuda").strip() or "cuda"

# folders whose files may be stored in the DB and registered as models
DATA_ROOTS = (ROOT, DATASETS, OUTPUTS, RESULTS, LOGS, MODELS)


def rel(p, base=None):
    """Path as stored in the DB/JSON: relative (posix) to `base` (default ROOT) when inside it, else absolute posix.
    Readers join it back with `base / value`, which also works for absolute values."""
    p = Path(p)
    try:
        return p.relative_to(base or ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def inside(p, *bases):
    """True when `p` lies inside one of `bases`."""
    rp = Path(p).resolve()
    for b in bases:
        try:
            rp.relative_to(Path(b).resolve())
            return True
        except ValueError:
            continue
    return False


def summary():
    """Effective settings for logs/the overview page (paths only, no secrets)."""
    return {"env_file": str(ENV_FILE) if ENV_FILE.is_file() else None, "root": str(ROOT),
            "design": str(DESIGN), "isaac_sim": str(ISAAC_SIM_DIR), "isaac_sim_python": str(ISAAC_SIM_PYTHON), "datasets": str(DATASETS), "outputs": str(OUTPUTS), "results": str(RESULTS),
            "logs": str(LOGS), "models": str(MODELS), "db": str(DB_PATH), "train_python": str(TRAIN_PYTHON),
            "hf_home": str(HF_HOME), "hf_datasets_cache": str(HF_DATASETS_CACHE)}
