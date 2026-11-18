"""Unattended data generation for the humanoid parts-to-pallet task (PLAN.md 3.1), after stack_dart_pipeline.py:
2 headless Isaac workers -> per level 200 clean + 100 DART expert episodes (split train, domain randomization on)
-> merge by hard links -> convert to LeRobot v3 -> dataset checks and statistics (PLAN 3.2).

Run detached with the SmolVLA env:  env\\python.exe -u scripts\\pallet_pipeline.py   (progress: logs\\pallet_pipeline.log)
Re-running it resumes: a job with enough episodes is skipped, an unfinished one continues after its last recorded seed
(humanoid_pallet_sim.py --record resumes), a worker whose log stops changing for STALE_SEC is killed and restarted.

Jobs (seed bases are disjoint and below the eval seeds 100000+; each job needs < 10000 tries):
  worker a: L1 clean 100 (seed 0), L1 DART 50 (noise 0.02, seed 20000), L2 clean 100 (40000), L2 DART 50 (0.02, 60000)
  worker b: L1 clean 100 (10000),  L1 DART 50 (noise 0.03, seed 30000), L2 clean 100 (50000), L2 DART 50 (0.03, 70000)
Raw: datasets/raw/humanoid_pallet_L{1,2}_{clean,dart}_{a,b}; merged: datasets/raw/humanoid_pallet_{L1,L2,all};
LeRobot: datasets/lerobot/yellowarm_pallet_{L1,L2,all} (repo id local/yellowarm_pallet_*).
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
LR = ROOT / "datasets" / "lerobot"
LOG = ROOT / "logs" / "pallet_pipeline.log"
PY = ROOT / "env" / "python.exe"
from mhh_paths import ISAAC_PY as _ISAAC_PY
ISAAC_PY = str(_ISAAC_PY)
SIM = ROOT / "scripts" / "humanoid_pallet_sim.py"
STALE_SEC = 20 * 60  # an expert log unchanged this long = hung worker (one try takes ~1 min)
MAX_RESTARTS = 5
ENV = {**os.environ, "HF_HOME": str(ROOT / "models" / "hf_home"), "HF_HUB_OFFLINE": "1",
       "HF_DATASETS_CACHE": str(Path(ROOT.anchor) / "hfds")}

# (name, level, episodes, dart noise, seed base)
WORKERS = {
    "a": [("L1_clean_a", 1, 100, 0.0, 0), ("L1_dart_a", 1, 50, 0.02, 20000),
          ("L2_clean_a", 2, 100, 0.0, 40000), ("L2_dart_a", 2, 50, 0.02, 60000)],
    "b": [("L1_clean_b", 1, 100, 0.0, 10000), ("L1_dart_b", 1, 50, 0.03, 30000),
          ("L2_clean_b", 2, 100, 0.0, 50000), ("L2_dart_b", 2, 50, 0.03, 70000)],
}
_lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with _lock:
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd, logfile):
    log(f"run: {' '.join(map(str, cmd))}  (output -> {logfile})")
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    log(f"exit code {rc} ({logfile.name})")
    if rc != 0:
        sys.exit(f"step failed, see {logfile}")


def n_episodes(d):
    return len(list(d.glob("episode_*.npz"))) if d.exists() else 0


def kill_tree(proc):
    subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gen_job(name, level, episodes, noise, seed):
    out = RAW / f"humanoid_pallet_{name}"
    for attempt in range(MAX_RESTARTS + 1):
        have = n_episodes(out)
        if have >= episodes:
            log(f"[{name}] done: {have} episodes")
            return True
        glog = ROOT / "logs" / f"pallet_gen_{name}_{attempt}.log"
        cmd = ["cmd", "/c", ISAAC_PY, SIM, "--headless", "--expert", "--level", level, "--split", "train",
               "--record", out, "--episodes", episodes, "--seed", seed, "--dart-noise", noise,
               "--results-json", ROOT / "logs" / f"pallet_gen_{name}.jsonl"]
        log(f"[{name}] attempt {attempt}: {have}/{episodes} episodes, log {glog.name}")
        with open(glog, "w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
            while proc.poll() is None:
                time.sleep(30)
                if time.time() - glog.stat().st_mtime > STALE_SEC:
                    log(f"[{name}] WARNING: log unchanged for {STALE_SEC // 60} min - killing the worker")
                    kill_tree(proc)
                    proc.wait()
        log(f"[{name}] worker exited with {proc.returncode}, {n_episodes(out)}/{episodes} episodes")
    ok = n_episodes(out) >= episodes
    if not ok:
        log(f"[{name}] FAILED after {MAX_RESTARTS} restarts")
    return ok


def worker(w, status):
    status[w] = all([gen_job(*job) for job in WORKERS[w]])  # list: run every job even if one fails


def merge(name, jobs):
    dst = RAW / f"humanoid_pallet_{name}"
    if dst.exists():
        shutil.rmtree(dst)  # only hard links: the episodes themselves stay in the per-job folders
    dst.mkdir(parents=True)
    k = 0
    for j in jobs:
        for f in sorted((RAW / f"humanoid_pallet_{j}").glob("episode_*.npz")):
            os.link(f, dst / f"episode_{k:04d}.npz")
            k += 1
    log(f"merged {k} episodes into {dst.name} from {', '.join(jobs)}")
    return dst


def main():
    log("pallet pipeline start")
    status = {}
    threads = []
    for w in WORKERS:
        t = threading.Thread(target=worker, args=(w, status))
        t.start()
        threads.append(t)
        time.sleep(60)  # stagger the Isaac start-ups
    for t in threads:
        t.join()
    if not all(status.values()):
        sys.exit(f"generation incomplete: {status}")
    jobs = {L: [j[0] for w in "ab" for j in WORKERS[w] if j[1] == L] for L in (1, 2)}
    sets = {"L1": jobs[1], "L2": jobs[2], "all": jobs[1] + jobs[2]}
    for name, js in sets.items():
        raw = merge(name, js)
        run([PY, "-u", ROOT / "scripts" / "npz_to_lerobot.py", "--raw", raw, "--out", LR / f"yellowarm_pallet_{name}",
             "--repo-id", f"local/yellowarm_pallet_{name}", "--robot-type", "humanoid_yellowarm_isaac", "--overwrite"],
            ROOT / "logs" / f"pallet_convert_{name}.log")
    # PLAN 3.2: checks + statistics
    for name in sets:
        run([PY, "-u", ROOT / "scripts" / "verify_npz.py", RAW / f"humanoid_pallet_{name}", "--images", "6"],
            ROOT / "logs" / f"pallet_verify_npz_{name}.log")
        run([PY, "-u", ROOT / "scripts" / "verify_dataset.py", RAW / f"humanoid_pallet_{name}",
             "--lerobot", LR / f"yellowarm_pallet_{name}"], ROOT / "logs" / f"pallet_verify_dataset_{name}.log")
    run([PY, "-u", ROOT / "scripts" / "pallet_dataset_stats.py"], ROOT / "logs" / "pallet_dataset_stats.log")
    log("pallet pipeline finished")


if __name__ == "__main__":
    main()
