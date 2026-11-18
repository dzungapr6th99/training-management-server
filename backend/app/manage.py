"""User-managed robots and models (create / edit / delete / restore).

The index tables (robot, model) are rebuilt from disk on every reindex, so user decisions live in `robot_user` / `model_user`:
  * custom=1  -> an entry the user created (robot: assets under design/<dir>/; model: a pretrained_model folder to register)
  * custom=0  -> overrides on top of an indexed item (registry robot or checkpoint found on disk)
  * hidden=1  -> "deleted": the item disappears from the app and can be restored. Files on disk are NEVER deleted by this module.

Robot assets: `urdf` and `usd` are REQUIRED for a new robot and must exist as files under design/<design_dir>/;
`usd_ros` (the USD with the ROS 2 bridge graph for Isaac Sim) is optional and may be null."""
import re
import time

from . import config as C
from . import db, indexer
from .registry import ROBOTS

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")
USD_EXT = (".usd", ".usda", ".usdc", ".usdz")
ROBOT_STATUS = ("active", "design", "component", "deprecated")
MODEL_STATUS = ("final", "intermediate", "custom", "archived")


class Invalid(ValueError):
    """A user-facing validation problem (HTTP 400)."""


class Conflict(ValueError):
    """HTTP 409."""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _file_under(base, rel, exts, label):
    """Resolve `rel` (relative to `base`) and require an existing file inside the design folder with one of `exts`."""
    if not rel or not str(rel).strip():
        raise Invalid(f"{label} là bắt buộc")
    p = (base / str(rel).strip().replace("\\", "/")).resolve()
    if not C.inside(p, base):
        raise Invalid(f"{label}: đường dẫn phải nằm trong thư mục design của robot")
    if not p.is_file():
        raise Invalid(f"{label}: không tìm thấy file '{rel}' trong design/{C.rel(base, C.DESIGN)}")
    if not p.name.lower().endswith(exts):
        raise Invalid(f"{label}: phải là file {'/'.join(exts)}")
    return str(rel).strip().replace("\\", "/")


def validate_robot_data(data, require_assets=True):
    """Validate the editable robot fields; return a cleaned dict."""
    out = dict(data)
    if "name" in out and not str(out["name"] or "").strip():
        raise Invalid("tên robot là bắt buộc")
    if "status" in out and out["status"] not in ROBOT_STATUS:
        raise Invalid(f"status phải là một trong {ROBOT_STATUS}")
    for k in ("state_dim", "action_dim"):
        if out.get(k) is not None and (not isinstance(out[k], int) or out[k] < 1 or out[k] > 512):
            raise Invalid(f"{k} phải là số nguyên 1–512 hoặc để trống")
    if "cameras" in out:
        cams = out["cameras"] or []
        if not all(isinstance(c, str) and re.match(r"^[A-Za-z0-9_]+$", c) for c in cams):
            raise Invalid("tên camera chỉ gồm chữ, số, _")
        out["cameras"] = cams
    dd = out.get("design_dir")
    assets_given = any(k in out for k in ("urdf", "usd", "usd_ros"))
    if require_assets or assets_given or dd:
        if not dd:
            raise Invalid("design_dir (thư mục trong design/) là bắt buộc để khai báo URDF/USD")
        base = (C.DESIGN / dd).resolve()
        try:
            base.relative_to(C.DESIGN.resolve())
        except ValueError:
            raise Invalid("design_dir phải nằm trong design/")
        if not base.is_dir():
            raise Invalid(f"design/{dd} không tồn tại")
        if require_assets or "urdf" in out:
            out["urdf"] = _file_under(base, out.get("urdf"), (".urdf",), "URDF")
        if require_assets or "usd" in out:
            out["usd"] = _file_under(base, out.get("usd"), USD_EXT, "USD")
        if out.get("usd_ros"):
            out["usd_ros"] = _file_under(base, out["usd_ros"], USD_EXT, "USD có ROS")
        elif "usd_ros" in out:
            out["usd_ros"] = None  # optional: null is allowed
    return out


def _user_row(con, table, id_):
    return db.one(con, f"SELECT * FROM {table} WHERE id=?", (id_,))


# ------------------------------------------------------------------------------------------------ robots
def robot_configured(con, rid):
    """Registry defaults + user overrides = the values the form edits (kept even when a file is currently missing)."""
    u = _user_row(con, "robot_user", rid)
    base = dict(ROBOTS.get(rid, {}))
    if u:
        base.update(db.jload(u["data"], {}))
    return base


def create_robot(data):
    if not ID_RE.match(data.get("id") or ""):
        raise Invalid("id gồm chữ thường, số, _ hoặc - (2–41 ký tự)")
    given = {k: v for k, v in data.items() if k != "id" and (v is not None or k == "usd_ros")}  # unset optional fields are simply absent
    clean = validate_robot_data(given, require_assets=True)
    if not str(clean.get("name") or "").strip():
        raise Invalid("tên robot là bắt buộc")
    clean.setdefault("status", "design")
    clean.setdefault("kind", "other")
    with db.session() as con:
        if con.execute("SELECT 1 FROM robot WHERE id=?", (data["id"],)).fetchone() or _user_row(con, "robot_user", data["id"]):
            raise Conflict(f"robot '{data['id']}' đã tồn tại")
        con.execute("INSERT INTO robot_user (id, data, hidden, custom, updated) VALUES (?,?,0,1,?)",
                    (data["id"], db.jdump(clean), now()))
    indexer.reindex()
    return data["id"]


def update_robot(rid, patch):
    with db.session() as con:
        exists = con.execute("SELECT 1 FROM robot WHERE id=?", (rid,)).fetchone()
        if not exists:
            return None
        u = _user_row(con, "robot_user", rid)
        custom = bool(u and u["custom"])
        cur = robot_configured(con, rid)
        merged = {**cur, **patch}
        # urdf and usd must never be cleared; usd_ros may be null
        for k in ("urdf", "usd"):
            if k in patch and not (patch[k] or "").strip():
                raise Invalid(f"{k.upper()} là bắt buộc, không thể để trống")
        check = {k: merged.get(k) for k in ("name", "status", "state_dim", "action_dim", "cameras", "design_dir", "urdf", "usd", "usd_ros")
                 if k in merged and (merged.get(k) is not None or k in patch or k not in ("urdf", "usd", "usd_ros"))}
        # only re-validate assets when the user touched them (or the robot is custom, which must always be complete)
        touched = any(k in patch for k in ("urdf", "usd", "usd_ros", "design_dir"))
        clean = validate_robot_data(check, require_assets=custom or touched and bool(merged.get("urdf") or merged.get("usd")))
        edits = {k: (clean[k] if k in clean else patch[k]) for k in patch if k in indexer.ROBOT_FIELDS}  # free-text fields pass through
        data = {**(db.jload(u["data"], {}) if u else {}), **edits}
        if u:
            con.execute("UPDATE robot_user SET data=?, updated=? WHERE id=?", (db.jdump(data), now(), rid))
        else:
            con.execute("INSERT INTO robot_user (id, data, hidden, custom, updated) VALUES (?,?,0,0,?)", (rid, db.jdump(data), now()))
    indexer.reindex()
    return rid


def delete_robot(rid):
    """Custom robots are removed from the studio; built-in ones are hidden (restorable). Assets on disk are untouched."""
    with db.session() as con:
        if not con.execute("SELECT 1 FROM robot WHERE id=?", (rid,)).fetchone():
            return None
        u = _user_row(con, "robot_user", rid)
        con.execute("DELETE FROM assignment WHERE robot_id=?", (rid,))
        if u and u["custom"]:
            con.execute("DELETE FROM robot_user WHERE id=?", (rid,))
            mode = "removed"
        else:
            if u:
                con.execute("UPDATE robot_user SET hidden=1, updated=? WHERE id=?", (now(), rid))
            else:
                con.execute("INSERT INTO robot_user (id, data, hidden, custom, updated) VALUES (?,?,1,0,?)", (rid, "{}", now()))
            mode = "hidden"
    indexer.reindex()
    return mode


def restore_robot(rid):
    with db.session() as con:
        u = _user_row(con, "robot_user", rid)
        if not u or not u["hidden"]:
            return None
        con.execute("UPDATE robot_user SET hidden=0, updated=? WHERE id=?", (now(), rid))
    indexer.reindex()
    return rid


# ------------------------------------------------------------------------------------------------ models
def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "model"


def _check_refs(con, robot_id, dataset_id):
    if robot_id and not con.execute("SELECT 1 FROM robot WHERE id=? AND hidden=0", (robot_id,)).fetchone():
        raise Invalid(f"robot '{robot_id}' không tồn tại")
    if dataset_id and not con.execute("SELECT 1 FROM dataset WHERE id=?", (dataset_id,)).fetchone():
        raise Invalid(f"dataset '{dataset_id}' không tồn tại")


def register_model(data):
    name = (data.get("name") or "").strip()
    if not name:
        raise Invalid("tên model là bắt buộc")
    rel = (data.get("path") or "").strip().replace("\\", "/")
    if not rel:
        raise Invalid("path (thư mục pretrained_model) là bắt buộc")
    p = (C.ROOT / rel).resolve()
    if not C.inside(p, *C.DATA_ROOTS):
        raise Invalid("path phải nằm trong training-management-server/ hoặc thư mục outputs/models đã cấu hình")
    info = indexer.read_model_dir(p) if p.is_dir() else None
    if not info:
        raise Invalid("path không phải thư mục model LeRobot (cần config.json, thường là .../pretrained_model)")
    status = data.get("status") or "custom"
    if status not in MODEL_STATUS:
        raise Invalid(f"status phải là một trong {MODEL_STATUS}")
    with db.session() as con:
        _check_refs(con, data.get("robot_id"), data.get("dataset_id"))
        mid = "custom/" + _slug(name)
        if con.execute("SELECT 1 FROM model WHERE id=?", (mid,)).fetchone() or _user_row(con, "model_user", mid):
            raise Conflict(f"model '{mid}' đã tồn tại; đổi tên khác")
        con.execute("INSERT INTO model_user (id, data, hidden, custom, updated) VALUES (?,?,0,1,?)",
                    (mid, db.jdump({"path": C.rel(p), "name": name, "notes": data.get("notes"),
                                    "robot_id": data.get("robot_id"), "dataset_id": data.get("dataset_id"), "status": status}), now()))
    indexer.reindex()
    return mid


def update_model(mid, patch):
    allowed = {k: patch[k] for k in indexer.MODEL_EDIT if k in patch}
    if "status" in allowed and allowed["status"] not in MODEL_STATUS:
        raise Invalid(f"status phải là một trong {MODEL_STATUS}")
    if "name" in allowed and not (allowed["name"] or "").strip():
        raise Invalid("tên model không được để trống")
    with db.session() as con:
        if not con.execute("SELECT 1 FROM model WHERE id=?", (mid,)).fetchone():
            return None
        _check_refs(con, allowed.get("robot_id"), allowed.get("dataset_id"))
        u = _user_row(con, "model_user", mid)
        data = {**(db.jload(u["data"], {}) if u else {}), **allowed}
        if u:
            con.execute("UPDATE model_user SET data=?, updated=? WHERE id=?", (db.jdump(data), now(), mid))
        else:
            con.execute("INSERT INTO model_user (id, data, hidden, custom, updated) VALUES (?,?,0,0,?)", (mid, db.jdump(data), now()))
    indexer.reindex()
    return mid


def delete_model(mid):
    with db.session() as con:
        if not con.execute("SELECT 1 FROM model WHERE id=?", (mid,)).fetchone():
            return None
        u = _user_row(con, "model_user", mid)
        con.execute("DELETE FROM assignment WHERE slot='model' AND ref_id=?", (mid,))
        if u and u["custom"]:
            con.execute("DELETE FROM model_user WHERE id=?", (mid,))
            mode = "removed"
        else:
            if u:
                con.execute("UPDATE model_user SET hidden=1, updated=? WHERE id=?", (now(), mid))
            else:
                con.execute("INSERT INTO model_user (id, data, hidden, custom, updated) VALUES (?,?,1,0,?)", (mid, "{}", now()))
            mode = "hidden"
    indexer.reindex()
    return mode


def restore_model(mid):
    with db.session() as con:
        u = _user_row(con, "model_user", mid)
        if not u or not u["hidden"]:
            return None
        con.execute("UPDATE model_user SET hidden=0, updated=? WHERE id=?", (now(), mid))
    indexer.reindex()
    return mid
