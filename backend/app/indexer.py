"""Scan the project on disk and rebuild the index tables. Read-only with respect to the project: nothing under
datasets/, outputs/, results/ or design/ is ever modified. Success rates etc. are copied from results/*.json as written by
eval_harness.py; they are never recomputed (Wilson is only re-derived to flag mismatches)."""
import json
import math
import os
import re
import time
from pathlib import Path

import yaml

from . import config as C
from . import db
from .registry import ROBOTS, ROBOT_TYPE_TO_ROBOT, SIM_TO_TASK, TASKS, task_for_dataset

CKPT_RE = re.compile(r"outputs[\\/]+([^\\/]+)[\\/]+checkpoints[\\/]+([^\\/]+)[\\/]", re.I)


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def dir_size(path: Path, pattern="*"):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def exists_rel(base: Path, rel):
    return bool(rel) and (base / rel).exists()


# ----------------------------------------------------------------------------------------------- robots / tasks
ROBOT_FIELDS = ("name", "kind", "status", "description", "state_dim", "action_dim", "cameras", "design_dir", "urdf", "usd", "usd_ros", "notes")


def _insert_robot(con, rid, m, source, hidden):
    d = C.DESIGN / m["design_dir"] if m.get("design_dir") else None
    urdf = m.get("urdf") if d and exists_rel(d, m.get("urdf")) else None
    usd = m.get("usd") if d and exists_rel(d, m.get("usd")) else None
    usd_ros = m.get("usd_ros") if d and exists_rel(d, m.get("usd_ros")) else None
    emb = None
    if d:
        for name in ("embodiment.yaml", "robot.yaml"):
            p = d / name
            if p.exists():
                try:
                    emb = yaml.safe_load(p.read_text(encoding="utf-8"))
                except yaml.YAMLError:
                    emb = None
                break
    if rid == "humanoid_yellowarm":
        emb = read_json(C.HUMANOID_JSON)
    rel = lambda p: Path(p).as_posix()          # asset paths stay relative to <design>/<design_dir>/
    con.execute(
        "INSERT OR REPLACE INTO robot VALUES (" + ",".join("?" * 16) + ")",
        (rid, m["name"], m.get("kind") or "other", m.get("status") or "design", m.get("description"), m.get("state_dim"),
         m.get("action_dim"), db.jdump(m.get("cameras") or []),
         (m["design_dir"] if d and d.exists() else None),
         rel(urdf) if urdf else None, rel(usd) if usd else None, db.jdump(emb) if emb else None, m.get("notes"),
         source, int(hidden), rel(usd_ros) if usd_ros else None))


def index_robots(con):
    user = {r["id"]: r for r in db.rows(con, "SELECT * FROM robot_user")}
    for rid, m in ROBOTS.items():
        u = user.get(rid)
        merged = {**m, **(db.jload(u["data"], {}) if u else {})}
        _insert_robot(con, rid, merged, "registry+user" if u and u["data"] else "registry", bool(u and u["hidden"]))
    for rid, u in user.items():  # robots created by the user
        if u["custom"]:
            _insert_robot(con, rid, db.jload(u["data"], {}), "user", bool(u["hidden"]))


def index_tasks(con):
    for tid, t in TASKS.items():
        con.execute("INSERT OR REPLACE INTO task VALUES (?,?,?,?,?,?,?,?,?)",
                    (tid, t["name"], t["robot_id"], t["status"], t["description"], db.jdump(t["levels"]),
                     db.jdump(t["splits"]), db.jdump(t["seeds"]), t["sim_script"]))


# ----------------------------------------------------------------------------------------------- datasets
def index_datasets(con):
    stats = (read_json(C.RESULTS / "dataset_stats.json") or {}).get("sets", {})
    if not C.DATASETS.exists():
        return
    for d in sorted(p for p in C.DATASETS.iterdir() if p.is_dir()):
        info = read_json(d / "meta" / "info.json")
        if not info:
            continue
        feats = info.get("features", {})
        st = feats.get("observation.state", {})
        ac = feats.get("action", {})
        cams = [k.split(".")[-1] for k in feats if k.startswith("observation.images.")]
        rtype = info.get("robot_type")
        status, note = "ok", None
        if d.name.endswith("_dropframes"):
            status, note = "superseded", "Bản cũ: bộ encode video streaming làm rơi frame (ảnh lệch 1–2 frame); đã có bản sửa."
        task_id = task_for_dataset(d.name)
        if task_id and TASKS[task_id]["status"] == "deprecated" and status == "ok":
            status = "deprecated"
        key = d.name.replace("yellowarm_pallet_", "")
        dstat = stats.get(key) if d.name.startswith("yellowarm_pallet_") and not d.name.endswith("_dropframes") else None
        n_vid = sum(1 for _ in (d / "videos").rglob("*.mp4")) if (d / "videos").exists() else 0
        con.execute(
            "INSERT OR REPLACE INTO dataset VALUES (" + ",".join("?" * 18) + ")",
            (d.name, C.rel(d), ROBOT_TYPE_TO_ROBOT.get(rtype), task_id, rtype, status, note,
             info.get("total_episodes"), info.get("total_frames"), info.get("fps"), info.get("total_tasks"),
             (st.get("shape") or [None])[0], (ac.get("shape") or [None])[0], db.jdump(cams),
             db.jdump({"state": st.get("names"), "action": ac.get("names")}), dir_size(d), n_vid,
             db.jdump(dstat) if dstat else None))


# ----------------------------------------------------------------------------------------------- train runs / models
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def index_train_and_models(con):
    jsons = {}
    for p in C.RESULTS.glob("train_*.json"):
        j = read_json(p)
        if isinstance(j, dict) and j.get("run"):
            jsons[j["run"]] = (p, j)
    runs = {}
    if C.OUTPUTS.exists():
        for d in sorted(p for p in C.OUTPUTS.iterdir() if p.is_dir()):
            runs[d.name] = d
    for name in sorted(set(runs) | set(jsons)):
        d = runs.get(name)
        pj = jsons.get(name)
        j = pj[1] if pj else {}
        ckpts = sorted(c.name for c in (d / "checkpoints").iterdir() if c.is_dir() and c.name.isdigit()) if d and (d / "checkpoints").exists() else []  # run folder removed from disk: the train summary stays, checkpoints are gone
        last = ckpts[-1] if ckpts else None
        # config of the last checkpoint: model type, dataset, steps
        cfg = read_json(d / "checkpoints" / last / "pretrained_model" / "config.json") if d and last else None
        tcfg = read_json(d / "checkpoints" / last / "pretrained_model" / "train_config.json") if d and last else None
        mtype = j.get("model") or (cfg or {}).get("type")
        ds = None
        if j.get("dataset"):
            ds = Path(str(j["dataset"]).replace("\\", "/")).name
        elif tcfg:
            ds = ((tcfg.get("dataset") or {}).get("repo_id") or "").split("/")[-1] or None
        steps = j.get("steps") or (tcfg or {}).get("steps")
        bs = j.get("batch_size") or (tcfg or {}).get("batch_size")
        con.execute("INSERT OR REPLACE INTO train_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (name, mtype, ds, j.get("init"), steps, bs, int(bool(j.get("finished"))) if j else None,
                     j.get("wall_hours"), j.get("peak_mem_gb_logged"), j.get("final_loss_mean_last5_logs"),
                     j.get("loss_first_log"), db.jdump(ckpts), last, j.get("command"), j.get("log"),
                     C.rel(pj[0]) if pj else None))
        # one model entry per numeric checkpoint
        for step in ckpts:
            pm = d / "checkpoints" / step / "pretrained_model"
            c = read_json(pm / "config.json") or {}
            inp, outp = c.get("input_features", {}), c.get("output_features", {})
            sd = ((inp.get("observation.state") or {}).get("shape") or [None])[0]
            ad = ((outp.get("action") or {}).get("shape") or [None])[0]
            cams = [k.split(".")[-1] for k in inp if k.startswith("observation.images.")]
            dsr = con.execute("SELECT robot_id FROM dataset WHERE id=?", (ds,)).fetchone() if ds else None
            sf = pm / "model.safetensors"
            con.execute("INSERT OR REPLACE INTO model VALUES (" + ",".join("?" * 18) + ")",
                        (f"{name}/{step}", name, step, c.get("type") or mtype, ds, dsr["robot_id"] if dsr else None,
                         C.rel(pm), sd, ad, db.jdump(cams), c.get("chunk_size"),
                         c.get("n_action_steps"), "final" if step == last else "intermediate",
                         sf.stat().st_size if sf.exists() else None, None, None, "disk", 0))



MODEL_EDIT = ("name", "notes", "status", "robot_id", "dataset_id")


def read_model_dir(path: Path):
    """Describe a pretrained_model directory from its config.json (type, dims, cameras, chunk)."""
    c = read_json(path / "config.json")
    if not isinstance(c, dict):
        return None
    inp, outp = c.get("input_features", {}), c.get("output_features", {})
    sf = path / "model.safetensors"
    return {"model_type": c.get("type"), "state_dim": ((inp.get("observation.state") or {}).get("shape") or [None])[0],
            "action_dim": ((outp.get("action") or {}).get("shape") or [None])[0],
            "cameras": [k.split(".")[-1] for k in inp if k.startswith("observation.images.")],
            "chunk_size": c.get("chunk_size"), "n_action_steps": c.get("n_action_steps"),
            "size_bytes": sf.stat().st_size if sf.exists() else None}


def apply_model_user_layer(con):
    for u in db.rows(con, "SELECT * FROM model_user"):
        data = db.jload(u["data"], {})
        if u["custom"]:
            info = read_model_dir(C.ROOT / data["path"]) if data.get("path") else None
            if not info:
                continue  # the folder disappeared; keep the entry in model_user so it comes back when the folder does
            dsr = con.execute("SELECT robot_id FROM dataset WHERE id=?", (data.get("dataset_id"),)).fetchone() if data.get("dataset_id") else None
            con.execute("INSERT OR REPLACE INTO model VALUES (" + ",".join("?" * 18) + ")",
                        (u["id"], None, None, info["model_type"], data.get("dataset_id"),
                         data.get("robot_id") or (dsr["robot_id"] if dsr else None), data["path"], info["state_dim"],
                         info["action_dim"], db.jdump(info["cameras"]), info["chunk_size"], info["n_action_steps"],
                         data.get("status") or "custom", info["size_bytes"], data.get("name"), data.get("notes"), "user",
                         int(bool(u["hidden"]))))
        else:
            sets = {k: data[k] for k in MODEL_EDIT if k in data}
            sets["hidden"] = int(bool(u["hidden"]))
            con.execute("UPDATE model SET " + ",".join(f"{k}=?" for k in sets) + " WHERE id=?", (*sets.values(), u["id"]))
            if data:
                con.execute("UPDATE model SET source='disk+user' WHERE id=?", (u["id"],))


def loss_curve(run_id: str, max_points=400):
    """Parse `step:20K smpl:.. loss:0.109` lines from logs/train_<run>*.log (plus *_resume). Returns [[step, loss], ...]."""
    pat = re.compile(rb"step:(\d+(?:\.\d+)?)([KM]?)\s+smpl:\S+\s+ep:\S+\s+epch:\S+\s+loss:([0-9.eE+-]+)")
    pts = {}
    files = [p for p in sorted(C.LOGS.glob(f"train_{run_id}*.log")) if not p.name.startswith(f"train_{run_id}_try")]
    j = read_json(C.RESULTS / f"train_{run_id}.json")  # runs started from the studio log to logs/studio/job_<id>.log
    if j and j.get("log") and (C.ROOT / j["log"]).exists() and (C.ROOT / j["log"]) not in files:
        files.append(C.ROOT / j["log"])
    for p in files:
        for m in pat.finditer(p.read_bytes()):
            step = float(m.group(1)) * {b"": 1, b"K": 1000, b"M": 1_000_000}[m.group(2)]
            pts[int(step)] = float(m.group(3))
    arr = sorted(pts.items())
    if len(arr) > max_points:
        stride = len(arr) / max_points
        arr = [arr[int(i * stride)] for i in range(max_points)] + [arr[-1]]
    return [[s, l] for s, l in arr]


# ----------------------------------------------------------------------------------------------- eval runs
def _kind(tag: str):
    t = tag.lower()
    for prefix, kind in (("val_", "validation"), ("demo_", "demo"), ("expert_", "expert")):
        if t.startswith(prefix):
            return kind
    if any(w in t for w in ("smoke", "verify", "test_", "reorg", "recheck")):
        return "smoke"
    return "eval"


def _args(extra, flag):
    extra = extra or []
    for i, a in enumerate(extra):
        if a == flag and i + 1 < len(extra):
            return extra[i + 1]
    return None


def index_evals(con):
    for p in sorted(C.RESULTS.glob("*.json")):
        s = p.read_text(encoding="utf-8", errors="replace")
        try:
            r = json.loads(s)
        except ValueError:
            r = None
        if isinstance(r, dict) and "n_episodes" in r and "wilson95" in r:
            cfg = r.get("config", {})
            m = CKPT_RE.search(str(cfg.get("checkpoint", "")) + "\\")
            run, step = (m.group(1), m.group(2)) if m else (None, None)
            if run and step == "last":
                row = con.execute("SELECT last_checkpoint FROM train_run WHERE id=?", (run,)).fetchone()
                step = row["last_checkpoint"] if row and row["last_checkpoint"] else step
            extra = cfg.get("extra_sim_args") or []
            lvl = _args(extra, "--level")
            split = _args(extra, "--split")
            task = SIM_TO_TASK.get(Path(str(cfg.get("sim_script", "yellowarm_sim.py")).replace("\\", "/")).name,
                                   "cube_in_box" if not cfg.get("sim_script") else None)
            lo, hi = r["wilson95"]
            elo, ehi = wilson(r.get("n_success"), r["n_episodes"])
            ok = int(elo is not None and abs(elo - lo) < 2e-3 and abs(ehi - hi) < 2e-3)
            seeds = cfg.get("seeds") or [cfg.get("seed"), None]
            con.execute("INSERT OR REPLACE INTO eval_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r.get("tag") or p.stem, _kind(p.stem), f"{run}/{step}" if run else None, run, step, task,
                         int(lvl) if lvl else None, split, r["n_episodes"], r.get("n_success"), r.get("success_rate"), lo, hi,
                         int(bool(r.get("complete"))), cfg.get("exec_steps"), cfg.get("episode_seconds"), seeds[0],
                         seeds[1] if len(seeds) > 1 else None, int("--smooth" in extra), r.get("infer_ms_mean"),
                         r.get("infer_ms_max"), r.get("vram_peak_gib"), r.get("wall_seconds"), r.get("started"),
                         (r.get("server_info") or {}).get("policy_type"), p.name, ok))
            continue
        # expert files: JSON lines, one object per try ({"try","seed","success",...})
        lines = [ln for ln in s.splitlines() if ln.strip().startswith("{")]
        try:
            recs = [json.loads(ln) for ln in lines]
        except ValueError:
            continue
        if recs and all(isinstance(x, dict) and "success" in x and "seed" in x for x in recs):
            k = sum(1 for x in recs if x["success"])
            n = len(recs)
            lo, hi = wilson(k, n)
            seeds = [x["seed"] for x in recs]
            con.execute("INSERT OR REPLACE INTO eval_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (p.stem, "expert", None, None, None, "humanoid_pallet", recs[0].get("level"), recs[0].get("split"),
                         n, k, k / n, lo, hi, 1, None, None, min(seeds), max(seeds), 0, None, None, None, None, None,
                         "expert", p.name, None))


# ----------------------------------------------------------------------------------------------- entry
def reindex():
    t0 = time.time()
    with db.session() as con:
        for t in db.INDEX_TABLES:
            con.execute(f"DELETE FROM {t}")
        index_robots(con)
        index_tasks(con)
        index_datasets(con)
        index_train_and_models(con)
        index_evals(con)
        apply_model_user_layer(con)
        counts = {t: con.execute(f"SELECT COUNT(*) c FROM {t}" + (" WHERE hidden=0" if t in ("robot", "model") else "")).fetchone()["c"]
                  for t in ("robot", "task", "dataset", "train_run", "model", "eval_run")}
        con.execute("INSERT OR REPLACE INTO meta VALUES ('indexed_at', ?)", (time.strftime("%Y-%m-%dT%H:%M:%S"),))
        con.execute("INSERT OR REPLACE INTO meta VALUES ('counts', ?)", (db.jdump(counts),))
    return {"seconds": round(time.time() - t0, 2), "counts": counts}
