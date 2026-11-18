"""SQLite storage (stdlib sqlite3). The index tables are rebuilt from disk on every reindex; only `assignment`
and `assignment_history` hold user decisions and survive a reindex."""
import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

INDEX_TABLES = ["robot", "task", "dataset", "train_run", "model", "eval_run", "meta"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS robot (
  id TEXT PRIMARY KEY, name TEXT, kind TEXT, status TEXT, description TEXT,
  state_dim INTEGER, action_dim INTEGER, cameras TEXT, design_dir TEXT, urdf TEXT, usd TEXT,
  embodiment TEXT, notes TEXT, source TEXT DEFAULT 'registry', hidden INTEGER DEFAULT 0, usd_ros TEXT
);
CREATE TABLE IF NOT EXISTS task (
  id TEXT PRIMARY KEY, name TEXT, robot_id TEXT, status TEXT, description TEXT,
  levels TEXT, splits TEXT, seeds TEXT, sim_script TEXT
);
CREATE TABLE IF NOT EXISTS dataset (
  id TEXT PRIMARY KEY, path TEXT, robot_id TEXT, task_id TEXT, robot_type TEXT, status TEXT, note TEXT,
  episodes INTEGER, frames INTEGER, fps REAL, n_tasks INTEGER, state_dim INTEGER, action_dim INTEGER,
  cameras TEXT, feature_names TEXT, size_bytes INTEGER, video_files INTEGER, stats TEXT
);
CREATE TABLE IF NOT EXISTS train_run (
  id TEXT PRIMARY KEY, model_type TEXT, dataset_id TEXT, init TEXT, steps INTEGER, batch_size INTEGER,
  finished INTEGER, wall_hours REAL, peak_mem_gb REAL, final_loss REAL, loss_first REAL,
  checkpoints TEXT, last_checkpoint TEXT, command TEXT, log TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS model (
  id TEXT PRIMARY KEY, run_id TEXT, checkpoint TEXT, model_type TEXT, dataset_id TEXT, robot_id TEXT,
  path TEXT, state_dim INTEGER, action_dim INTEGER, cameras TEXT, chunk_size INTEGER, n_action_steps INTEGER,
  status TEXT, size_bytes INTEGER, name TEXT, notes TEXT, source TEXT DEFAULT 'disk', hidden INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS eval_run (
  tag TEXT PRIMARY KEY, kind TEXT, model_id TEXT, run_id TEXT, checkpoint TEXT, task_id TEXT, level INTEGER,
  split TEXT, n INTEGER, k INTEGER, rate REAL, lo REAL, hi REAL, complete INTEGER, exec_steps INTEGER,
  episode_seconds REAL, seed_first INTEGER, seed_last INTEGER, smooth INTEGER, infer_ms REAL, infer_ms_max REAL,
  vram_gib REAL, wall_seconds REAL, started TEXT, policy_type TEXT, file TEXT, wilson_ok INTEGER
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS assignment (
  robot_id TEXT, slot TEXT, ref_id TEXT, note TEXT, updated TEXT, PRIMARY KEY (robot_id, slot)
);
CREATE TABLE IF NOT EXISTS robot_user (
  id TEXT PRIMARY KEY, data TEXT, hidden INTEGER DEFAULT 0, custom INTEGER DEFAULT 0, updated TEXT
);
CREATE TABLE IF NOT EXISTS model_user (
  id TEXT PRIMARY KEY, data TEXT, hidden INTEGER DEFAULT 0, custom INTEGER DEFAULT 0, updated TEXT
);
CREATE TABLE IF NOT EXISTS job (
  id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, name TEXT, status TEXT, command TEXT, params TEXT, gpu INTEGER,
  pid INTEGER, log TEXT, created TEXT, started TEXT, ended TEXT, exit_code INTEGER, result TEXT
);
CREATE TABLE IF NOT EXISTS assignment_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, robot_id TEXT, slot TEXT, ref_id TEXT, note TEXT, at TEXT
);
"""


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


SCHEMA_VERSION = 3  # 2: robot/model got source+hidden(+name,notes) and the user-layer tables; 3: robot.usd_ros


def init():
    with session() as con:
        if con.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            for t in INDEX_TABLES:  # index tables are rebuilt from disk, so they can simply be recreated
                con.execute(f"DROP TABLE IF EXISTS {t}")
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        con.executescript(SCHEMA)


def rows(con, sql, args=()):
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def one(con, sql, args=()):
    r = con.execute(sql, args).fetchone()
    return dict(r) if r else None


def jload(v, default=None):
    if v is None or v == "":
        return default
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def jdump(v):
    return json.dumps(v, ensure_ascii=False)
