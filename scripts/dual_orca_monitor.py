"""Live web monitor for dual_orca_pipeline.py (Isaac stays headless): http://localhost:8765

  env\\python.exe scripts\\dual_orca_monitor.py [--port 8765] [--set v1]      (or run_dual_orca_monitor.bat)

Shows, refreshed every 2 s:
  * every running Isaac worker (expert data generation or policy eval): latest outside view + the 3 policy cameras
    (written by dual_orca_pick_sim.py --live-dir logs/live/<job>), seed, try, successes, last results
  * data jobs: recorded episodes / target, expert success rate (logs/dual_orca_gen_<job>.jsonl)
  * training: loss curve and progress parsed from logs/train_act_dual_orca_pick_<set>.log
  * eval: per-episode results (logs/eval_<tag>.episodes.jsonl) and the final results/<tag>.json with Wilson CI
  * recorded episodes: click one to watch its 3 cameras (mp4 encoded on first request, cached in logs/dual_orca_review)
Read-only: it never starts, stops or changes anything.
"""
import argparse
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mhh_paths import ROOT

LOGS, RAW, RES = ROOT / "logs", ROOT / "datasets" / "raw", ROOT / "results"
LIVE = LOGS / "live"
REVIEW = LOGS / "dual_orca_review"
PAGE = Path(__file__).with_name("dual_orca_monitor_page.html")
_gpu = {"t": 0.0, "v": None}
_vid_lock = threading.Lock()


def tail(p, n=40):
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def jsonl(p):
    out = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def gpu():
    if time.time() - _gpu["t"] > 5:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            u, mu, mt, t = [float(x) for x in r.stdout.strip().split(",")[:4]]
            _gpu["v"] = {"util": u, "mem_used_gb": round(mu / 1024, 1), "mem_total_gb": round(mt / 1024, 1), "temp": t}
        except Exception:  # noqa: BLE001 (no GPU tool: just leave it empty)
            _gpu["v"] = None
        _gpu["t"] = time.time()
    return _gpu["v"]


def num(s):
    m = {"K": 1e3, "M": 1e6}
    return float(s[:-1]) * m[s[-1]] if s and s[-1] in m else float(s)


def train_info(st):
    p = LOGS / f"train_act_dual_orca_pick_{st}.log"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    pts = []
    for m in re.finditer(r"step:(\S+) .*?loss:([0-9.]+)", txt):
        pts.append([num(m.group(1)), float(m.group(2))])
    prog = re.findall(r"Training:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([^<\]]*)<([^,\]]*),\s*([0-9.]+)\s*step/s", txt)
    last = prog[-1] if prog else None
    ck = ROOT / "outputs" / f"act_dual_orca_pick_{st}" / "checkpoints"
    return {"log": p.name, "points": pts[-400:], "age_s": round(time.time() - p.stat().st_mtime, 1),
            "step": int(last[0]) if last else None, "total": int(last[1]) if last else None,
            "elapsed": last[2] if last else None, "eta": last[3] if last else None,
            "rate": float(last[4]) if last else None,
            "checkpoints": sorted(d.name for d in ck.iterdir() if d.name != "last") if ck.exists() else [],
            "error": "Traceback" in txt[-4000:]}


def status(st):
    workers = []
    if LIVE.exists():
        for d in sorted(LIVE.iterdir()):
            sp = d / "status.json"
            try:
                s = json.loads(sp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            s["name"], s["age_s"] = d.name, round(time.time() - s.get("time", 0), 1)
            workers.append(s)
    plog = LOGS / f"dual_orca_pipeline{'_smoke' if st == 'smoke' else ''}.log"
    targets = {m.group(1): int(m.group(2)) for m in re.finditer(r"\[(\S+)\] attempt \d+: \d+/(\d+) episodes",
                                                                "\n".join(tail(plog, 400)))}
    jobs = []
    for d in sorted(RAW.glob(f"dual_orca_pick_{st}_*")):
        if d.name.endswith("_all"):
            continue
        job = d.name[len("dual_orca_pick_"):]
        tries = jsonl(LOGS / f"dual_orca_gen_{job}.jsonl")
        jobs.append({"job": job, "episodes": len(list(d.glob("episode_*.npz"))), "target": targets.get(job),
                     "tries": len(tries),
                     "success": sum(bool(t.get("success")) for t in tries)})
    evals = []
    for p in sorted(LOGS.glob(f"eval_act_dual_orca_pick_{st}*.episodes.jsonl")):
        tag = p.name[len("eval_"):-len(".episodes.jsonl")]
        eps = jsonl(p)
        res = None
        rp = RES / f"{tag}.json"
        if rp.exists():
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
                res = {k: r.get(k) for k in ("n_episodes", "n_success", "success_rate", "wilson95", "infer_ms_mean",
                                             "complete", "config")}
            except ValueError:
                pass
        evals.append({"tag": tag, "episodes": [{k: e.get(k) for k in ("seed", "success", "lift_m", "max_lift_m",
                                                                       "cube_to_hand_m", "infer_ms_mean")} for e in eps],
                      "result": res})
    lines = tail(plog, 60)
    stage = "idle"
    for ln in lines:
        if "pipeline start" in ln:
            stage = "data"
        elif "merged" in ln:
            stage = "convert"
        elif "lerobot_train" in ln:
            stage = "train"
        elif "eval_harness" in ln:
            stage = "eval"
        elif "pipeline finished" in ln:
            stage = "finished"
        elif "step failed" in ln or "incomplete" in ln:
            stage = "failed"
    return {"set": st, "now": time.time(), "stage": stage, "pipeline_log": lines[-25:],
            "pipeline_age_s": round(time.time() - plog.stat().st_mtime, 1) if plog.exists() else None,
            "workers": workers, "jobs": jobs, "train": train_info(st), "evals": evals, "gpu": gpu()}


GR1D = ROOT / "datasets" / "external" / "gr1"


def gr1_status():
    """GR1 (NVIDIA Mimic demo) -> YellowArm retarget work: per-demo retarget progress/errors, comparison videos,
    physics replay results."""
    demos = []
    for lg in sorted(LOGS.glob("gr1_retarget_demo*.log")):
        dn = "demo_" + lg.stem.split("demo")[-1]
        lines = tail(lg, 400)
        prog = [ln for ln in lines if ln.strip().startswith("t=")]
        summ = None
        rp = GR1D / "retarget" / f"report_{dn}.json"
        if rp.exists() and rp.stat().st_mtime >= lg.stat().st_mtime - 5:
            try:
                summ = json.loads(rp.read_text(encoding="utf-8"))["demos"].get(dn)
            except (ValueError, KeyError):
                pass
        demos.append({"demo": dn, "running": summ is None and time.time() - lg.stat().st_mtime < 900,
                      "last": prog[-1].strip() if prog else (lines[-1] if lines else ""), "summary": summ,
                      "age_s": round(time.time() - lg.stat().st_mtime, 1),
                      "error": any("Traceback" in ln for ln in lines)})
    vids = [{"name": v.stem, "mb": round(v.stat().st_size / 1e6, 1), "time": v.stat().st_mtime}
            for v in sorted((LOGS / "gr1_view").glob("*.mp4"))]
    return {"demos": demos, "videos": vids, "replays": jsonl(LOGS / "gr1_physics_replay.jsonl")[-30:]}


def episodes(st):
    out = []
    for d in sorted(RAW.glob(f"dual_orca_pick_{st}_*")):
        if d.name.endswith("_all"):
            continue
        job = d.name[len("dual_orca_pick_"):]
        for f in sorted(d.glob("episode_*.npz")):
            out.append({"job": job, "ep": int(f.stem.split("_")[1]), "size_mb": round(f.stat().st_size / 1e6, 1)})
    return out


def make_video(job, ep):
    """3 policy cameras side by side + time / grasp command overlay, H.264 (cached)."""
    out = REVIEW / job / f"ep{ep:04d}.mp4"
    if out.exists():
        return out
    import av
    import numpy as np

    src = RAW / f"dual_orca_pick_{job}" / f"episode_{ep:04d}.npz"
    with _vid_lock:
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        with np.load(src) as z:
            frames = np.concatenate([z["camera1"], z["camera2"], z["camera3"]], axis=2)
            fps = int(round(float(z["fps"])))
        tmp = out.with_suffix(".tmp.mp4")
        c = av.open(str(tmp), "w")
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": "26", "preset": "veryfast", "movflags": "+faststart"}
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
        tmp.replace(out)
    return out


def episode_meta(job, ep):
    import numpy as np

    with np.load(RAW / f"dual_orca_pick_{job}" / f"episode_{ep:04d}.npz") as z:
        m = json.loads(str(z["meta"]))
        a = z["action"]
        names = [str(n) for n in z["names"]]
    m["action_right"] = {n: [round(float(v), 3) for v in a[::5, i]] for i, n in enumerate(names) if n.startswith("right")}
    return m


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        st = (q.get("set") or [self.server.set])[0]
        try:
            if u.path == "/":
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self.send(200, json.dumps(status(st)).encode(), "application/json")
            if u.path == "/api/gr1":
                return self.send(200, json.dumps(gr1_status()).encode(), "application/json")
            m = re.fullmatch(r"/gr1video/([\w.-]+)\.mp4", u.path)
            if m:
                p = LOGS / "gr1_view" / f"{m.group(1)}.mp4"
                return self.send(200, p.read_bytes(), "video/mp4") if p.exists() else self.send(404, b"", "text/plain")
            if u.path == "/api/episodes":
                return self.send(200, json.dumps(episodes(st)).encode(), "application/json")
            m = re.fullmatch(r"/live/([\w.-]+)\.jpg", u.path)
            if m:
                p = LIVE / m.group(1) / "live.jpg"
                return self.send(200, p.read_bytes(), "image/jpeg") if p.exists() else self.send(404, b"", "text/plain")
            m = re.fullmatch(r"/video/([\w.-]+)/(\d+)\.mp4", u.path)
            if m:
                p = make_video(m.group(1), int(m.group(2)))
                return self.send(200, p.read_bytes(), "video/mp4", {"Accept-Ranges": "none"})
            m = re.fullmatch(r"/api/episode/([\w.-]+)/(\d+)", u.path)
            if m:
                return self.send(200, json.dumps(episode_meta(m.group(1), int(m.group(2)))).encode(), "application/json")
            self.send(404, b"not found", "text/plain")
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as e:  # noqa: BLE001 (show the error in the page instead of dropping the request)
            self.send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--set", default="v1", help="pipeline set: v1 (default) or smoke")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.set = a.set
    print(f"[monitor] http://localhost:{a.port}/  (set {a.set}; Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
