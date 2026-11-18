"""Minimal job runner (S3): one worker thread runs queued jobs one at a time, which also acts as the GPU lock (train and
Isaac eval never overlap, as required by CLAUDE.md). Each job is a subprocess whose stdout/stderr go to logs/studio/job_<id>.log.
Only `train` jobs exist so far. A finished train job writes results/train_<run>.json (same fields as train_pipeline.py) and
triggers a reindex so the new run/checkpoints appear in the UI."""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

from . import config as C
from . import db, indexer

JOB_LOGS = C.LOGS / "studio"
ENV_PY = C.TRAIN_PYTHON                     # STUDIO_TRAIN_PYTHON
TRAIN_MODULE = [str(ENV_PY), "-m", "lerobot.scripts.lerobot_train"]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,59}$")
# SmolVLA base checkpoints per robot state dimension (STUDIO_SMOLVLA_BASE_16D / _8D)
SMOLVLA_BASE = C.SMOLVLA_BASE
IS_WINDOWS = sys.platform == "win32"

_wake = threading.Event()
_procs: dict[int, subprocess.Popen] = {}
_cancel: set[int] = set()
_lock = threading.Lock()
_started = False


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def can_train():
    """True when the training interpreter exists (a viewer-only install has none)."""
    return ENV_PY.exists()


def hf_env():
    env = dict(os.environ)
    env.update({"HF_HOME": str(C.HF_HOME), "HF_HUB_OFFLINE": C.HF_HUB_OFFLINE,
                "HF_DATASETS_CACHE": str(C.HF_DATASETS_CACHE),   # keep it short on Windows: avoids WinError 206
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"})
    return env


# ------------------------------------------------------------------------------------------------ submit / control
def build_train(dataset_id, model_type, steps, batch_size, run_name):
    """Return (cmd, params) or raise ValueError with a user-facing message."""
    if not can_train():
        raise ValueError(f"máy này chưa có môi trường train (thiếu {ENV_PY}; đặt STUDIO_TRAIN_PYTHON trong .env): app chỉ dùng để xem dữ liệu, không chạy được job train")
    with db.session() as con:
        ds = db.one(con, "SELECT * FROM dataset WHERE id=?", (dataset_id,))
    if not ds:
        raise ValueError(f"dataset '{dataset_id}' không tồn tại")
    if not (C.ROOT / ds["path"] / "data").exists():
        raise ValueError(f"dataset '{dataset_id}' trên máy này chỉ có metadata (không có data/video) nên không train được")
    if ds["status"] == "superseded":
        raise ValueError("dataset là bản cũ đã được thay thế")
    if model_type not in ("act", "smolvla"):
        raise ValueError("model_type hiện chỉ hỗ trợ 'act' và 'smolvla'")
    if not NAME_RE.match(run_name or ""):
        raise ValueError("tên run phải gồm chữ/số/_/- (3–60 ký tự)")
    if (C.OUTPUTS / run_name).exists():
        raise ValueError(f"outputs/{run_name} đã tồn tại; chọn tên khác (app không ghi đè kết quả cũ)")
    if not 1 <= steps <= C.TRAIN_MAX_STEPS or not 1 <= batch_size <= C.TRAIN_MAX_BATCH:
        raise ValueError(f"steps phải 1–{C.TRAIN_MAX_STEPS} và batch_size 1–{C.TRAIN_MAX_BATCH}")
    out = C.OUTPUTS / run_name
    common = [f"--dataset.repo_id=local/{dataset_id}", f"--dataset.root={C.ROOT / ds['path']}", "--dataset.video_backend=pyav",
              "--policy.push_to_hub=false", f"--batch_size={batch_size}", f"--num_workers={C.TRAIN_NUM_WORKERS}",
              f"--log_freq={max(1, min(200, steps // 10))}", "--wandb.enable=false", f"--steps={steps}",
              f"--save_freq={max(1, min(steps, 10000))}", f"--output_dir={out}", f"--job_name={run_name}"]
    init = "scratch (lerobot defaults)"
    if model_type == "act":
        head = ["--policy.type=act", f"--policy.device={C.POLICY_DEVICE}"]
    else:
        base = SMOLVLA_BASE.get(ds["state_dim"])
        if not base or not base.exists():
            raise ValueError(f"không có base SmolVLA cho state {ds['state_dim']}-D")
        head = [f"--policy.path={base}"]
        init = C.rel(base)
    return [*TRAIN_MODULE, *head, *common], {"run": run_name, "dataset": dataset_id, "model": model_type, "steps": steps,
                                              "batch_size": batch_size, "init": init, "dataset_path": ds["path"]}


def submit(jtype, name, cmd, params, gpu=True):
    with db.session() as con:
        cur = con.execute("INSERT INTO job (type, name, status, command, params, gpu, created) VALUES (?,?,?,?,?,?,?)",
                          (jtype, name, "queued", db.jdump(cmd), db.jdump(params), int(gpu), now()))
        jid = cur.lastrowid
    _wake.set()
    return jid


def cancel(jid):
    with db.session() as con:
        j = db.one(con, "SELECT status FROM job WHERE id=?", (jid,))
        if not j:
            return None
        if j["status"] == "queued":
            con.execute("UPDATE job SET status='cancelled', ended=? WHERE id=?", (now(), jid))
            return "cancelled"
    if j["status"] == "running":
        with _lock:
            _cancel.add(jid)
            p = _procs.get(jid)
        if p and p.poll() is None:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(p.pid, signal.SIGTERM)
        return "cancelling"
    return j["status"]


def tail_log(jid, n=80):
    p = JOB_LOGS / f"job_{jid}.log"
    if not p.exists():
        return []
    raw = p.read_bytes()[-200_000:].decode("utf-8", errors="replace")
    lines = [ln.rstrip() for part in raw.replace("\r", "\n").split("\n") for ln in [part] if ln.strip()]
    # collapse tqdm progress spam: keep only the last progress line
    out, last_bar = [], None
    for ln in lines:
        if ln.lstrip().startswith("Training:") or "it/s]" in ln or "step/s]" in ln:
            last_bar = ln
        else:
            out.append(ln)
    if last_bar:
        out.append(last_bar[-200:])
    return out[-n:]


# ------------------------------------------------------------------------------------------------ worker
def _gpu_free_gib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        used, total = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return (total - used) / 1024
    except Exception:
        return None


def _run(job):
    jid = job["id"]
    cmd = db.jload(job["command"])
    params = db.jload(job["params"], {})
    JOB_LOGS.mkdir(parents=True, exist_ok=True)
    log = JOB_LOGS / f"job_{jid}.log"
    if job["gpu"]:
        free = _gpu_free_gib()
        if free is not None and free < C.GPU_MIN_FREE_GIB:
            log.write_text(f"[studio] GPU chỉ còn {free:.1f} GiB trống (cần ≥ {C.GPU_MIN_FREE_GIB:g}): có tiến trình khác đang dùng GPU. Không chạy.\n", encoding="utf-8")
            _finish(job, 3, "failed", None)
            return
    t0 = time.time()
    with open(log, "ab") as lf:
        lf.write(f"[studio] {now()} job {jid} start: {' '.join(map(str, cmd))}\n".encode("utf-8"))
        lf.flush()
        spawn = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if IS_WINDOWS
                 else {"start_new_session": True})
        p = subprocess.Popen(cmd, cwd=str(C.ROOT), env=hf_env(), stdout=lf, stderr=subprocess.STDOUT, **spawn)
        with _lock:
            _procs[jid] = p
        with db.session() as con:
            con.execute("UPDATE job SET status='running', pid=?, started=?, log=? WHERE id=?",
                        (p.pid, now(), C.rel(log), jid))
        code = p.wait()
    with _lock:
        _procs.pop(jid, None)
        was_cancelled = jid in _cancel
        _cancel.discard(jid)
    status = "cancelled" if was_cancelled else ("succeeded" if code == 0 else "failed")
    _finish(job, code, status, time.time() - t0)


def _finish(job, code, status, seconds):
    jid = job["id"]
    params = db.jload(job["params"], {})
    summary = None
    if job["type"] == "train" and status in ("succeeded", "failed", "cancelled") and seconds is not None:
        summary = _write_train_summary(job, params, code, seconds)
    with db.session() as con:
        con.execute("UPDATE job SET status=?, exit_code=?, ended=?, result=? WHERE id=?",
                    (status, code, now(), db.jdump(summary) if summary else None, jid))
    if job["type"] == "train":
        indexer.reindex()


def _write_train_summary(job, params, code, seconds):
    run = params["run"]
    log = JOB_LOGS / f"job_{job['id']}.log"
    pts = []
    if log.exists():
        pat = re.compile(rb"step:(\d+(?:\.\d+)?)([KM]?)\s+smpl:\S+\s+ep:\S+\s+epch:\S+\s+loss:([0-9.eE+-]+)")
        for m in pat.finditer(log.read_bytes()):
            pts.append((float(m.group(1)) * {b"": 1, b"K": 1000, b"M": 1_000_000}[m.group(2)], float(m.group(3))))
    ck_dir = C.OUTPUTS / run / "checkpoints"
    ckpts = sorted(c.name for c in ck_dir.iterdir() if c.is_dir() and c.name.isdigit()) if ck_dir.exists() else []
    finished = code == 0 and bool(ckpts) and int(ckpts[-1]) >= params["steps"]
    summary = {
        "generated_by": "backend/app/jobs.py (Training Studio)", "run": run, "model": params["model"], "init": params["init"],
        "dataset": f"datasets/lerobot/{params['dataset']}", "steps": params["steps"], "batch_size": params["batch_size"],
        "mode": "studio", "exit_code": code, "finished": finished, "wall_hours": round(seconds / 3600, 3),
        "steps_per_s": None, "peak_mem_gb_logged": None,
        "last_logged_step": f"{int(pts[-1][0])}" if pts else None,
        "final_loss_mean_last5_logs": round(sum(v for _, v in pts[-5:]) / len(pts[-5:]), 4) if pts else None,
        "loss_first_log": pts[0][1] if pts else None, "checkpoints": ckpts,
        "command": " ".join(map(str, db.jload(job["command"]))), "log": C.rel(log),
    }
    (C.RESULTS / f"train_{run}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"summary": f"results/train_{run}.json", "finished": finished, "checkpoints": ckpts,
            "final_loss": summary["final_loss_mean_last5_logs"]}


def _loop():
    while True:
        _wake.wait(timeout=2)
        _wake.clear()
        while True:
            with db.session() as con:
                job = db.one(con, "SELECT * FROM job WHERE status='queued' ORDER BY id LIMIT 1")
            if not job:
                break
            try:
                _run(job)
            except Exception as e:  # keep the worker alive whatever a single job does
                with db.session() as con:
                    con.execute("UPDATE job SET status='failed', ended=?, result=? WHERE id=?",
                                (now(), db.jdump({"error": str(e)}), job["id"]))


def start():
    global _started
    if _started:
        return
    _started = True
    with db.session() as con:  # jobs that were running when the server stopped are not tracked any more
        con.execute("UPDATE job SET status='interrupted', ended=? WHERE status='running'", (now(),))
    threading.Thread(target=_loop, daemon=True, name="job-worker").start()
