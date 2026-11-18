"""Unattended data -> train -> eval for the dual YellowArm + ORCA pick-and-lift task (dual_orca_pick_sim.py).

Run detached with the SmolVLA env:  env\\python.exe -u scripts\\dual_orca_pipeline.py [--smoke]
Progress: logs\\dual_orca_pipeline[_smoke].log. Re-running resumes every stage (episodes already recorded are kept,
a finished conversion / checkpoint / eval result is skipped).

  1. data   2 headless Isaac workers; per worker a clean job and a DART job (noise on the transit only, labels are
            the clean expert targets). Seeds: a 0 / 20000, b 10000 / 30000 (eval seeds 100000+ are never used).
  2. merge  hard links into datasets/raw/dual_orca_pick_<set>
  3. convert -> datasets/lerobot/dual_orca_pick_<set> (18-D state/action with joint names, 3 cameras, 25 fps)
  4. train  ACT from scratch (lerobot defaults, batch 8) -> outputs/act_dual_orca_pick_<set>, results/train_*.json
  5. eval   eval_harness.py on the final checkpoint, test seeds 100000.. -> results/act_dual_orca_pick_<set>_<k>k.json
--smoke: 4+2 episodes, 300 train steps, 3 eval episodes (checks the whole chain in ~15 min).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from mhh_paths import ENV_PY, HF_HOME, ISAAC_PY, LEROBOT_TRAIN, ROOT

SIM = ROOT / "scripts" / "dual_orca_pick_sim.py"
RAW, LR, OUT, RES = ROOT / "datasets" / "raw", ROOT / "datasets" / "lerobot", ROOT / "outputs", ROOT / "results"
# HF datasets names its cache lock files after the (flattened) cache path: under models\hf_home they exceed
# Windows MAX_PATH (WinError 206) since the repo moved to My-Humanoid-Hand. Keep that cache short, on the repo drive.
HF_DS_CACHE = Path(ROOT.anchor) / "hfds"
ENV = {**os.environ, "HF_HOME": str(HF_HOME), "HF_HUB_OFFLINE": "1", "HF_DATASETS_CACHE": str(HF_DS_CACHE),
       "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
STALE_SEC = 15 * 60
LIVE = ROOT / "logs" / "live"  # dual_orca_pick_sim.py --live-dir, read by dual_orca_monitor.py
MAX_RESTARTS = 4

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--clean", type=int, default=70, help="clean expert episodes per worker")
ap.add_argument("--dart", type=int, default=30, help="DART expert episodes per worker")
ap.add_argument("--train-steps", type=int, default=30000)
ap.add_argument("--save-freq", type=int, default=10000)
ap.add_argument("--eval-episodes", type=int, default=50)
ap.add_argument("--exec-steps", type=int, default=10)
ap.add_argument("--episode-seconds", type=float, default=8.0)
ap.add_argument("--skip-eval", action="store_true")
ap.add_argument("--monitor-port", type=int, default=8765, help="web monitor port (0 = do not start it)")
A = ap.parse_args()
if A.smoke:
    A.clean, A.dart, A.train_steps, A.save_freq, A.eval_episodes = 2, 1, 300, 300, 3
SET = "smoke" if A.smoke else "v1"
LOG = ROOT / "logs" / f"dual_orca_pipeline{'_smoke' if A.smoke else ''}.log"
WORKERS = {"a": [(f"{SET}_clean_a", A.clean, 0.0, 0), (f"{SET}_dart_a", A.dart, 0.02, 20000)],
           "b": [(f"{SET}_clean_b", A.clean, 0.0, 10000), (f"{SET}_dart_b", A.dart, 0.03, 30000)]}
_lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with _lock:
        print(line, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def kill_tree(p):
    subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(cmd, logfile):
    log(f"run: {' '.join(map(str, cmd))}  (-> {logfile.name})")
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    log(f"exit code {rc} ({logfile.name})")
    if rc != 0:
        sys.exit(f"step failed, see {logfile}")


def n_eps(d):
    return len(list(d.glob("episode_*.npz"))) if d.exists() else 0


def gen_job(name, episodes, noise, seed):
    out = RAW / f"dual_orca_pick_{name}"
    for attempt in range(MAX_RESTARTS + 1):
        have = n_eps(out)
        if have >= episodes:
            log(f"[{name}] done: {have} episodes")
            return True
        glog = ROOT / "logs" / f"dual_orca_gen_{name}_{attempt}.log"
        cmd = ["cmd", "/c", ISAAC_PY, SIM, "--headless", "--expert", "--record", out, "--episodes", episodes,
               "--seed", seed, "--dart-noise", noise, "--results-json", ROOT / "logs" / f"dual_orca_gen_{name}.jsonl",
               "--live-dir", LIVE / name]
        log(f"[{name}] attempt {attempt}: {have}/{episodes} episodes, log {glog.name}")
        with open(glog, "w", encoding="utf-8", errors="replace") as f:
            p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
            while p.poll() is None:
                time.sleep(20)
                if time.time() - glog.stat().st_mtime > STALE_SEC:
                    log(f"[{name}] log unchanged for {STALE_SEC // 60} min - killing the worker")
                    kill_tree(p)
                    p.wait()
        log(f"[{name}] worker exit {p.returncode}, {n_eps(out)}/{episodes} episodes")
    return n_eps(out) >= episodes


def worker(w, status):
    status[w] = all([gen_job(*j) for j in WORKERS[w]])


def merge(name, jobs):
    dst = RAW / f"dual_orca_pick_{name}"
    if dst.exists():
        shutil.rmtree(dst)  # hard links only: the episodes stay in the job folders
    dst.mkdir(parents=True)
    k = 0
    for j in jobs:
        for f in sorted((RAW / f"dual_orca_pick_{j}").glob("episode_*.npz")):
            os.link(f, dst / f"episode_{k:04d}.npz")
            k += 1
    log(f"merged {k} episodes into {dst.name}")
    return dst, k


def parse_train_log(p):
    steps, losses = [], []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"step:(\S+).*?loss:([0-9.]+)", line)
        if m:
            steps.append(m.group(1))
            losses.append(float(m.group(2)))
    return steps, losses


def train(ds, run_name):
    out = OUT / run_name
    ck = out / "checkpoints"
    final = ck / f"{A.train_steps:06d}" / "pretrained_model"
    if final.exists():
        log(f"[{run_name}] final checkpoint exists, skip")
        return final
    if (ck / "last" / "pretrained_model" / "train_config.json").exists():
        cmd = [*LEROBOT_TRAIN, f"--config_path={ck / 'last' / 'pretrained_model' / 'train_config.json'}", "--resume=true"]
    else:
        if out.exists():
            shutil.rmtree(out)  # an aborted run without a checkpoint (lerobot refuses an existing output dir)
        cmd = [*LEROBOT_TRAIN, "--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false",
               f"--dataset.repo_id=local/{ds.name}", f"--dataset.root={ds}", "--dataset.video_backend=pyav",
               "--batch_size=8", "--num_workers=4", "--log_freq=200", "--wandb.enable=false",
               f"--steps={A.train_steps}", f"--save_freq={A.save_freq}", f"--output_dir={out}", f"--job_name={run_name}"]
    tlog = ROOT / "logs" / f"train_{run_name}.log"
    t0 = time.time()
    run(cmd, tlog)
    steps, losses = parse_train_log(tlog)
    res = {"generated_by": "scripts/dual_orca_pipeline.py", "run": run_name, "model": "act", "init": "scratch",
           "dataset": str(ds.relative_to(ROOT)), "steps": A.train_steps, "batch_size": 8,
           "wall_hours": round((time.time() - t0) / 3600, 3), "finished": final.exists(),
           "loss_first_log": losses[0] if losses else None,
           "final_loss_mean_last5_logs": round(sum(losses[-5:]) / len(losses[-5:]), 4) if losses else None,
           "command": " ".join(map(str, cmd)), "log": f"logs/{tlog.name}"}
    RES.mkdir(exist_ok=True)
    (RES / f"train_{run_name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"[{run_name}] trained: {res}")
    if not final.exists():
        sys.exit(f"no final checkpoint {final}")
    return final


def start_monitor():
    """Start the read-only web monitor (detached) unless something already listens on its port."""
    import socket

    if not A.monitor_port:
        return
    try:
        socket.create_connection(("127.0.0.1", A.monitor_port), timeout=1).close()
        log(f"web monitor already running: http://localhost:{A.monitor_port}/")
        return
    except OSError:
        pass
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    with open(ROOT / "logs" / "dual_orca_monitor.log", "a", encoding="utf-8") as f:
        subprocess.Popen([str(ENV_PY), "-u", str(ROOT / "scripts" / "dual_orca_monitor.py"), "--port", str(A.monitor_port),
                          "--set", SET], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, creationflags=flags)
    log(f"web monitor: http://localhost:{A.monitor_port}/  (Isaac runs headless; open this in a browser)")


def main():
    log(f"dual ORCA pipeline start ({SET}): {A}")
    start_monitor()
    status, threads = {}, []
    for w in WORKERS:
        t = threading.Thread(target=worker, args=(w, status))
        t.start()
        threads.append(t)
        time.sleep(45)  # stagger the Isaac start-ups
    for t in threads:
        t.join()
    if not all(status.values()):
        sys.exit(f"generation incomplete: {status}")
    raw, n = merge(f"{SET}_all", [j[0] for w in WORKERS for j in WORKERS[w]])
    ds = LR / f"dual_orca_pick_{SET}_all"
    if not (ds / "meta" / "info.json").exists() or json.loads((ds / "meta" / "info.json").read_text())[
            "total_episodes"] != n:
        run([ENV_PY, "-u", ROOT / "scripts" / "npz_to_lerobot.py", "--raw", raw, "--out", ds,
             "--repo-id", f"local/{ds.name}", "--robot-type", "yellowarm_two_orca_isaac", "--overwrite"],
            ROOT / "logs" / f"dual_orca_convert_{SET}.log")
    else:
        log(f"{ds.name} already converted ({n} episodes)")
    run_name = f"act_dual_orca_pick_{SET}"
    ckpt = train(ds, run_name)
    if A.skip_eval:
        log("eval skipped")
    else:
        tag = f"{run_name}_{A.train_steps // 1000}k" if A.train_steps >= 1000 else f"{run_name}_{A.train_steps}"
        if (RES / f"{tag}.json").exists():
            log(f"{tag}.json exists, skip eval")
        else:
            run([ENV_PY, "-u", ROOT / "scripts" / "eval_harness.py", "--checkpoint", ckpt, "--sim-script", SIM,
                 "--episodes", A.eval_episodes, "--seed", 100000, "--exec-steps", A.exec_steps,
                 "--episode-seconds", A.episode_seconds, "--tag", tag, "--", "--live-dir", LIVE / f"eval_{tag}"],
                ROOT / "logs" / f"{tag}.log")
            r = json.loads((RES / f"{tag}.json").read_text(encoding="utf-8"))
            log(f"[eval] {tag}: {r['n_success']}/{r['n_episodes']} success, Wilson95 {r['wilson95']}")
    log("dual ORCA pipeline finished")


if __name__ == "__main__":
    main()
