"""Training management studio backend. Reads the project files; writes only (a) robot assignments in the studio DB and
(b) the output of train jobs the user starts (outputs/<run>, results/train_<run>.json, logs/studio/job_<id>.log).
Run:  backend\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8700   (from backend\\)"""
import pathlib
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C
from . import db, episodes, indexer, jobs, manage

app = FastAPI(title="Training Management Studio", version="0.2.0")


@app.on_event("startup")
def _startup():
    db.init()
    with db.session() as con:
        n = con.execute("SELECT COUNT(*) c FROM robot").fetchone()["c"]
    if n == 0:
        indexer.reindex()
    jobs.start()


def _fix_robot(r):
    r["cameras"] = db.jload(r.get("cameras"), [])
    r["embodiment"] = db.jload(r.get("embodiment"))
    return r


def _fix_dataset(d):
    for k in ("cameras", "feature_names", "stats"):
        d[k] = db.jload(d.get(k))
    return d


def _fix_model(m):
    m["cameras"] = db.jload(m.get("cameras"), [])
    return m


def _fix_run(r):
    r["checkpoints"] = db.jload(r.get("checkpoints"), [])
    r["finished"] = None if r.get("finished") is None else bool(r["finished"])
    return r


# ------------------------------------------------------------------------------------------------ compatibility
def compat_dataset(con, robot, ds):
    problems = []
    if ds["robot_id"] != robot["id"]:
        problems.append(f"dataset thuộc robot '{ds['robot_id']}', không phải '{robot['id']}'")
    if robot["state_dim"] and ds["state_dim"] and robot["state_dim"] != ds["state_dim"]:
        problems.append(f"state {ds['state_dim']}-D ≠ robot {robot['state_dim']}-D")
    if robot["action_dim"] and ds["action_dim"] and robot["action_dim"] != ds["action_dim"]:
        problems.append(f"action {ds['action_dim']}-D ≠ robot {robot['action_dim']}-D")
    if ds["status"] in ("superseded",):
        problems.append("dataset là bản cũ đã được thay thế")
    return problems


def compat_model(con, robot, m):
    problems = []
    if not m["robot_id"]:
        problems.append("model chưa gắn robot nào (bấm Sửa model để chọn robot hoặc dataset)")
    elif m["robot_id"] != robot["id"]:
        problems.append(f"model thuộc robot '{m['robot_id']}', không phải '{robot['id']}'")
    if robot["state_dim"] and m["state_dim"] and robot["state_dim"] != m["state_dim"]:
        problems.append(f"state {m['state_dim']}-D ≠ robot {robot['state_dim']}-D")
    if robot["action_dim"] and m["action_dim"] and robot["action_dim"] != m["action_dim"]:
        problems.append(f"action {m['action_dim']}-D ≠ robot {robot['action_dim']}-D")
    rc = set(db.jload(robot.get("cameras"), []))
    mc = set(db.jload(m.get("cameras"), []))
    if rc and mc and rc != mc:
        problems.append(f"camera model {sorted(mc)} ≠ robot {sorted(rc)}")
    return problems


# ------------------------------------------------------------------------------------------------ read endpoints
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/overview")
def overview():
    with db.session() as con:
        meta = {r["key"]: r["value"] for r in db.rows(con, "SELECT * FROM meta")}
        evals = db.rows(con, "SELECT kind, COUNT(*) n FROM eval_run GROUP BY kind")
        ds_size = con.execute("SELECT COALESCE(SUM(size_bytes),0) s FROM dataset").fetchone()["s"]
        latest = db.rows(con, "SELECT tag, kind, model_id, level, split, n, k, rate, lo, hi, started FROM eval_run "
                              "WHERE started IS NOT NULL ORDER BY started DESC LIMIT 8")
        assigned = db.rows(con, "SELECT robot_id, slot, ref_id FROM assignment")
        return {"indexed_at": meta.get("indexed_at"), "counts": db.jload(meta.get("counts"), {}),
                "evals_by_kind": evals, "dataset_bytes": ds_size, "latest_evals": latest, "assignments": assigned,
                "root": str(C.ROOT), "can_train": jobs.can_train()}


@app.post("/api/reindex")
def reindex():
    return indexer.reindex()


def _assets(r):
    """The three Isaac Sim assets of a robot. urdf and usd are required; usd_ros (USD with the ROS 2 graph) is optional."""
    a = {"urdf": r.get("urdf"), "usd": r.get("usd"), "usd_ros": r.get("usd_ros")}
    r["assets"] = a
    r["assets_complete"] = bool(a["urdf"] and a["usd"])
    r["missing_assets"] = [k for k in ("urdf", "usd") if not a[k]]
    return r


@app.get("/api/robots")
def robots(include_hidden: bool = False):
    with db.session() as con:
        out = [_assets(_fix_robot(r)) for r in db.rows(
            con, "SELECT * FROM robot WHERE hidden=0 OR ? ORDER BY hidden, (status='active') DESC, id", (int(include_hidden),))]
        for r in out:
            r.pop("embodiment", None)
            r["n_datasets"] = con.execute("SELECT COUNT(*) c FROM dataset WHERE robot_id=?", (r["id"],)).fetchone()["c"]
            r["n_models"] = con.execute("SELECT COUNT(*) c FROM model WHERE robot_id=? AND hidden=0", (r["id"],)).fetchone()["c"]
            r["assignment"] = {a["slot"]: a["ref_id"] for a in db.rows(
                con, "SELECT slot, ref_id FROM assignment WHERE robot_id=?", (r["id"],))}
        return out


class RobotBody(BaseModel):
    id: str | None = None
    name: str | None = None
    kind: str | None = None
    status: str | None = None
    description: str | None = None
    state_dim: int | None = None
    action_dim: int | None = None
    cameras: list[str] | None = None
    design_dir: str | None = None
    urdf: str | None = None
    usd: str | None = None
    usd_ros: str | None = None
    notes: str | None = None


@app.get("/api/design/dirs")
def design_dirs():
    """Folders under design/ with the URDF / USD files they contain, to fill the robot form (paths are relative to the folder)."""
    out = []
    if not C.DESIGN.is_dir():
        return out
    for d in sorted(p for p in C.DESIGN.iterdir() if p.is_dir() and not p.name.startswith("_")):
        urdf, usd = [], []
        for root, dirs, files in __import__("os").walk(d):
            depth = len(pathlib.Path(root).relative_to(d).parts)
            dirs[:] = [x for x in dirs if x not in ("meshes", "Textures", "textures", "__pycache__", "tasks", "cad_tools") and depth < 4]
            for f in files:
                rel = str((pathlib.Path(root) / f).relative_to(d)).replace("\\", "/")
                if f.lower().endswith(".urdf"):
                    urdf.append(rel)
                elif f.lower().endswith(manage.USD_EXT):
                    usd.append(rel)
        out.append({"dir": d.name, "urdf": sorted(urdf)[:60], "usd": sorted(usd)[:60]})
    return out


def _bad(e):
    return HTTPException(409 if isinstance(e, manage.Conflict) else 400, str(e))


@app.post("/api/robots")
def robot_create(body: RobotBody):
    try:
        return {"id": manage.create_robot(body.model_dump())}
    except (manage.Invalid, manage.Conflict) as e:
        raise _bad(e)


@app.put("/api/robots/{rid}")
def robot_update(rid: str, body: RobotBody):
    patch = {k: v for k, v in body.model_dump().items() if k in body.model_fields_set and k != "id"}
    try:
        r = manage.update_robot(rid, patch)
    except (manage.Invalid, manage.Conflict) as e:
        raise _bad(e)
    if r is None:
        raise HTTPException(404, "robot not found")
    return {"id": r}


@app.delete("/api/robots/{rid}")
def robot_delete(rid: str):
    mode = manage.delete_robot(rid)
    if mode is None:
        raise HTTPException(404, "robot not found")
    return {"result": mode, "note": "Chỉ bỏ khỏi studio; file URDF/USD/CAD trong design/ không bị xoá."}


@app.post("/api/robots/{rid}/restore")
def robot_restore(rid: str):
    if manage.restore_robot(rid) is None:
        raise HTTPException(404, "robot không ở trạng thái đã xoá")
    return {"id": rid}


@app.get("/api/robots/{rid}")
def robot_detail(rid: str):
    with db.session() as con:
        r = db.one(con, "SELECT * FROM robot WHERE id=?", (rid,))
        if not r:
            raise HTTPException(404, "robot not found")
        r = _assets(_fix_robot(r))
        configured = manage.robot_configured(con, rid)
        datasets = [_fix_dataset(d) for d in db.rows(con, "SELECT * FROM dataset WHERE robot_id=? ORDER BY id", (rid,))]
        for d in datasets:
            d["compat"] = compat_dataset(con, r, d)
            d.pop("stats", None)
        models = [_fix_model(m) for m in db.rows(con, "SELECT * FROM model WHERE robot_id=? AND hidden=0 ORDER BY run_id, checkpoint", (rid,))]
        for m in models:
            m["compat"] = compat_model(con, r, m)
        tasks = db.rows(con, "SELECT * FROM task WHERE robot_id=?", (rid,))
        assignment = {a["slot"]: a for a in db.rows(con, "SELECT * FROM assignment WHERE robot_id=?", (rid,))}
        history = db.rows(con, "SELECT * FROM assignment_history WHERE robot_id=? ORDER BY id DESC LIMIT 30", (rid,))
        return {"robot": r, "configured": configured, "datasets": datasets, "models": models, "tasks": tasks,
                "assignment": assignment, "history": history}


class AssignBody(BaseModel):
    slot: str          # "dataset" | "model"
    ref_id: str | None  # None = clear
    note: str | None = None


@app.post("/api/robots/{rid}/assign")
def assign(rid: str, body: AssignBody):
    if body.slot not in ("dataset", "model"):
        raise HTTPException(400, "slot phải là 'dataset' hoặc 'model'")
    with db.session() as con:
        r = db.one(con, "SELECT * FROM robot WHERE id=? AND hidden=0", (rid,))
        if not r:
            raise HTTPException(404, "robot not found")
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if body.ref_id is None:
            con.execute("DELETE FROM assignment WHERE robot_id=? AND slot=?", (rid, body.slot))
        else:
            if body.slot == "dataset":
                x = db.one(con, "SELECT * FROM dataset WHERE id=?", (body.ref_id,))
                problems = compat_dataset(con, r, x) if x else ["dataset không tồn tại"]
            else:
                x = db.one(con, "SELECT * FROM model WHERE id=?", (body.ref_id,))
                problems = compat_model(con, r, x) if x else ["model không tồn tại"]
            if problems:
                raise HTTPException(409, {"message": "Không tương thích", "problems": problems})
            con.execute("INSERT OR REPLACE INTO assignment VALUES (?,?,?,?,?)", (rid, body.slot, body.ref_id, body.note, now))
        con.execute("INSERT INTO assignment_history (robot_id, slot, ref_id, note, at) VALUES (?,?,?,?,?)",
                    (rid, body.slot, body.ref_id, body.note, now))
    return {"ok": True}


@app.get("/api/tasks")
def tasks():
    with db.session() as con:
        out = db.rows(con, "SELECT * FROM task ORDER BY (status='active') DESC, id")
        for t in out:
            for k in ("levels", "splits", "seeds"):
                t[k] = db.jload(t[k])
            t["n_datasets"] = con.execute("SELECT COUNT(*) c FROM dataset WHERE task_id=?", (t["id"],)).fetchone()["c"]
            t["n_evals"] = con.execute("SELECT COUNT(*) c FROM eval_run WHERE task_id=?", (t["id"],)).fetchone()["c"]
        return out


@app.get("/api/datasets")
def datasets(robot: str | None = None, task: str | None = None, include_old: bool = True):
    sql, args = "SELECT * FROM dataset WHERE 1=1", []
    if robot:
        sql += " AND robot_id=?"; args.append(robot)
    if task:
        sql += " AND task_id=?"; args.append(task)
    if not include_old:
        sql += " AND status='ok'"
    with db.session() as con:
        out = [_fix_dataset(d) for d in db.rows(con, sql + " ORDER BY (status='ok') DESC, id", args)]
        for d in out:
            d.pop("feature_names", None)
            d.pop("stats", None)
        return out


@app.get("/api/datasets/{did}")
def dataset_detail(did: str):
    with db.session() as con:
        d = db.one(con, "SELECT * FROM dataset WHERE id=?", (did,))
        if not d:
            raise HTTPException(404, "dataset not found")
        d = _fix_dataset(d)
        d["trained_runs"] = db.rows(con, "SELECT id, model_type, steps, final_loss FROM train_run WHERE dataset_id=?", (did,))
        return d


@app.get("/api/runs")
def runs():
    with db.session() as con:
        return [_fix_run(r) for r in db.rows(con, "SELECT * FROM train_run ORDER BY (finished=1) DESC, id")]


@app.get("/api/runs/{rid}")
def run_detail(rid: str):
    with db.session() as con:
        r = db.one(con, "SELECT * FROM train_run WHERE id=?", (rid,))
        if not r:
            raise HTTPException(404, "run not found")
        r = _fix_run(r)
        r["models"] = [_fix_model(m) for m in db.rows(con, "SELECT * FROM model WHERE run_id=? ORDER BY checkpoint", (rid,))]
        r["evals"] = db.rows(con, "SELECT tag, kind, level, split, n, k, rate, lo, hi, exec_steps, checkpoint FROM eval_run "
                                  "WHERE run_id=? ORDER BY kind, level, split", (rid,))
        return r


@app.get("/api/runs/{rid}/loss")
def run_loss(rid: str):
    return {"run": rid, "points": indexer.loss_curve(rid)}


@app.get("/api/models")
def models(robot: str | None = None, only_final: bool = False, include_hidden: bool = False):
    sql, args = "SELECT * FROM model WHERE (hidden=0 OR ?)", [int(include_hidden)]
    if robot:
        sql += " AND robot_id=?"; args.append(robot)
    if only_final:
        sql += " AND status IN ('final','custom')"
    with db.session() as con:
        out = [_fix_model(m) for m in db.rows(con, sql + " ORDER BY hidden, run_id, checkpoint, id", args)]
        for m in out:
            m["n_evals"] = con.execute("SELECT COUNT(*) c FROM eval_run WHERE model_id=?", (m["id"],)).fetchone()["c"]
            m["assigned_to"] = [u["robot_id"] for u in db.rows(
                con, "SELECT robot_id FROM assignment WHERE slot='model' AND ref_id=?", (m["id"],))]
        return out


class ModelBody(BaseModel):
    path: str | None = None
    name: str | None = None
    notes: str | None = None
    status: str | None = None
    robot_id: str | None = None
    dataset_id: str | None = None


@app.post("/api/models")
def model_register(body: ModelBody):
    try:
        return {"id": manage.register_model(body.model_dump())}
    except (manage.Invalid, manage.Conflict) as e:
        raise _bad(e)


@app.get("/api/model-detail")
def model_detail(id: str):
    with db.session() as con:
        m = db.one(con, "SELECT * FROM model WHERE id=?", (id,))
        if not m:
            raise HTTPException(404, "model not found")
        m = _fix_model(m)
        evals = db.rows(con, "SELECT tag, kind, task_id, level, split, n, k, rate, lo, hi, exec_steps FROM eval_run WHERE model_id=? "
                             "ORDER BY kind, level, split", (id,))
        run = db.one(con, "SELECT id, model_type, steps, final_loss, finished FROM train_run WHERE id=?", (m["run_id"],)) if m["run_id"] else None
        robots_ = db.rows(con, "SELECT * FROM robot WHERE hidden=0 ORDER BY (status='active') DESC, id")
        fit = [{"id": r["id"], "name": r["name"], "problems": compat_model(con, r, m)} for r in robots_]
        assigned = [a["robot_id"] for a in db.rows(con, "SELECT robot_id FROM assignment WHERE slot='model' AND ref_id=?", (id,))]
        return {"model": m, "evals": evals, "run": run, "robots": fit, "assigned_to": assigned}


@app.put("/api/model-edit")
def model_update(id: str, body: ModelBody):
    patch = {k: v for k, v in body.model_dump().items() if k in body.model_fields_set}
    try:
        r = manage.update_model(id, patch)
    except (manage.Invalid, manage.Conflict) as e:
        raise _bad(e)
    if r is None:
        raise HTTPException(404, "model not found")
    return {"id": r}


@app.delete("/api/model-edit")
def model_delete(id: str):
    mode = manage.delete_model(id)
    if mode is None:
        raise HTTPException(404, "model not found")
    return {"result": mode, "note": "Chỉ bỏ khỏi studio; checkpoint trên đĩa không bị xoá."}


@app.post("/api/model-restore")
def model_restore(id: str):
    if manage.restore_model(id) is None:
        raise HTTPException(404, "model không ở trạng thái đã xoá")
    return {"id": id}


@app.get("/api/evals")
def evals(kind: str | None = None, model: str | None = None, task: str | None = None, level: int | None = None,
          split: str | None = None):
    sql, args = "SELECT * FROM eval_run WHERE 1=1", []
    for col, val in (("kind", kind), ("model_id", model), ("task_id", task), ("level", level), ("split", split)):
        if val is not None:
            sql += f" AND {col}=?"; args.append(val)
    with db.session() as con:
        return db.rows(con, sql + " ORDER BY kind, task_id, level, split, tag", args)


@app.get("/api/eval/compare")
def eval_compare(task: str = "humanoid_pallet", level: int | None = None, kind: str = "eval"):
    """Evaluation rows of one task that may be placed side by side. Rows are grouped by the conditions that must match for a
    fair comparison (CLAUDE.md): same seed range, same episode_seconds, same level. Groups with a single condition set are
    comparable; the response flags every row whose conditions differ from the majority of its level."""
    sql, args = "SELECT * FROM eval_run WHERE task_id=? AND kind=?", [task, kind]
    if level is not None:
        sql += " AND level=?"; args.append(level)
    with db.session() as con:
        rows = db.rows(con, sql + " ORDER BY level, split, model_id", args)
    groups = {}
    for r in rows:
        key = (r["level"], r["seed_first"], r["seed_last"], r["episode_seconds"])
        groups.setdefault(key, []).append(r["tag"])
    biggest = {}
    for (lvl, a, b, es), tags in groups.items():
        if lvl not in biggest or len(tags) > len(groups[biggest[lvl]]):
            biggest[lvl] = (lvl, a, b, es)
    for r in rows:
        key = (r["level"], r["seed_first"], r["seed_last"], r["episode_seconds"])
        r["comparable"] = key == biggest.get(r["level"])
        r["conditions"] = {"seeds": [r["seed_first"], r["seed_last"]], "episode_seconds": r["episode_seconds"]}
    return {"task": task, "kind": kind, "rows": rows,
            "reference_conditions": {str(k): {"seeds": [v[1], v[2]], "episode_seconds": v[3]} for k, v in biggest.items()}}


# ------------------------------------------------------------------------------------------------ episodes / media
@app.get("/api/datasets/{did}/episodes")
def dataset_episodes(did: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), q: str | None = None):
    r = episodes.list_episodes(did, offset, limit, q)
    if r is None:
        raise HTTPException(404, "dataset not found")
    return r


@app.get("/api/datasets/{did}/episodes/{ep}")
def dataset_episode(did: str, ep: int):
    r = episodes.episode_detail(did, ep)
    if r is None or "error" in r:
        raise HTTPException(404, "episode not found")
    return r


@app.get("/api/datasets/{did}/episodes/{ep}/series")
def dataset_episode_series(did: str, ep: int):
    r = episodes.series(did, ep)
    if r is None:
        raise HTTPException(404, "episode not found")
    return r


@app.get("/api/media/{did}/{camera}/{chunk}/{file}")
def media(did: str, camera: str, chunk: int, file: int):
    p = episodes.video_path(did, camera, chunk, file)
    if p is None:
        raise HTTPException(404, "video not found")
    return FileResponse(p, media_type="video/mp4")  # Starlette serves HTTP Range requests, so the browser can seek


# ------------------------------------------------------------------------------------------------ jobs (S3)
class TrainBody(BaseModel):
    dataset_id: str
    model_type: str = "act"
    steps: int = 100
    batch_size: int = 8
    run_name: str


def _fix_job(j):
    j["command"] = db.jload(j.get("command"))
    j["params"] = db.jload(j.get("params"), {})
    j["result"] = db.jload(j.get("result"))
    j["gpu"] = bool(j.get("gpu"))
    return j


@app.post("/api/jobs/train")
def start_train(body: TrainBody):
    try:
        cmd, params = jobs.build_train(body.dataset_id, body.model_type, body.steps, body.batch_size, body.run_name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    with db.session() as con:
        dup = db.one(con, "SELECT id FROM job WHERE type='train' AND status IN ('queued','running') AND "
                          "json_extract(params, '$.run')=?", (body.run_name,))
    if dup:
        raise HTTPException(409, f"đã có job train '{body.run_name}' đang chờ/chạy (#{dup['id']})")
    return {"id": jobs.submit("train", body.run_name, cmd, params)}


@app.get("/api/jobs")
def list_jobs():
    with db.session() as con:
        out = [_fix_job(j) for j in db.rows(con, "SELECT * FROM job ORDER BY id DESC LIMIT 100")]
    for j in out:
        j.pop("command")
    return out


@app.get("/api/jobs/{jid}")
def job_detail(jid: int):
    with db.session() as con:
        j = db.one(con, "SELECT * FROM job WHERE id=?", (jid,))
    if not j:
        raise HTTPException(404, "job not found")
    return _fix_job(j)


@app.get("/api/jobs/{jid}/log")
def job_log(jid: int, tail: int = Query(80, ge=1, le=500)):
    return {"lines": jobs.tail_log(jid, tail)}


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: int):
    r = jobs.cancel(jid)
    if r is None:
        raise HTTPException(404, "job not found")
    return {"status": r}


# ------------------------------------------------------------------------------------------------ static frontend
if (C.FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=C.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = C.FRONTEND_DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(C.FRONTEND_DIST / "index.html")
