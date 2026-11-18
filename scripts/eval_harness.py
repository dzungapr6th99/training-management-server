"""Run one policy evaluation end to end and write results/<tag>.json. Run with the env python.

  env\\python.exe scripts\\eval_harness.py --checkpoint outputs\\...\\030000\\pretrained_model --episodes 50 \\
      --seed 1000 --exec-steps 8 --tag baseline_cubebox_30k [-- extra sim args, e.g. --cube-x 0.2 0.5]

Starts scripts/policy_server.py, waits for its port, runs the sim script headless with --results-json, and if the
server (or the sim) dies mid-run, restarts what died and resumes at the first unfinished seed (the sim seeds episode i
with seed + i), at most --retries times. Episodes are counted from the per-episode JSONL, not from the sim's exit code.
Output: config, every episode, success rate with a Wilson 95% CI, wall time, mean infer latency (sim-side round trip
and server-side model time), the server's info (policy type, n_action_steps, dims) and its peak VRAM.
"""

import argparse
import datetime as dt
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wire import recv_msg, send_msg  # noqa: E402

ENV_PY = ROOT / "env" / "python.exe"
from mhh_paths import ISAAC_PY  # noqa: E402  (ISAAC_SIM_DIR env var overrides the Isaac Sim location)
HOST = "127.0.0.1"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def kill_tree(proc):
    if proc and proc.poll() is None:
        subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            pass


def port_open(port):
    try:
        socket.create_connection((HOST, port), timeout=1).close()
        return True
    except OSError:
        return False


def start_server(args, env, log_path):
    if port_open(args.port):
        sys.exit(f"[harness] port {args.port} is already in use; stop that server or pick another --port")
    log = open(log_path, "a", encoding="utf-8")
    cmd = [str(ENV_PY), "-u", str(ROOT / "scripts" / "policy_server.py"), "--checkpoint", args.checkpoint,
           "--port", str(args.port), *args.server_args.split()]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    t0 = time.time()
    while not port_open(args.port):
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited with code {proc.returncode}, see {log_path}")
        if time.time() - t0 > args.server_timeout:
            kill_tree(proc)
            raise RuntimeError(f"policy server did not open port {args.port} within {args.server_timeout}s")
        time.sleep(1)
    print(f"[harness] server up on port {args.port} after {time.time() - t0:.0f}s", flush=True)
    return proc


def server_info(port):
    try:
        with socket.create_connection((HOST, port), timeout=60) as s:
            send_msg(s, {"cmd": "info"})
            return recv_msg(s)
    except OSError as e:
        return {"error": repr(e)}


def read_episodes(path):
    if not path.exists():
        return []
    eps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            eps.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut off by a crash
            pass
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim-script", default=str(ROOT / "scripts" / "yellowarm_sim.py"))
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="episode i uses seed + i")
    ap.add_argument("--exec-steps", type=int, required=True)
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--tag", required=True, help="writes results/<tag>.json")
    ap.add_argument("--server-args", default="", help="extra policy_server.py args in one string, e.g. "
                    "\"--noise fixed --rtc-horizon 16\"")
    ap.add_argument("--retries", type=int, default=2, help="restarts after the server or the sim dies")
    ap.add_argument("--server-timeout", type=float, default=600)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing results/<tag>.json")
    ap.add_argument("--resume", action="store_true",
                    help="keep logs/eval_<tag>.episodes.jsonl from an interrupted run of the SAME config and only run "
                         "the missing seeds")
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    out = ROOT / "results" / f"{args.tag}.json"
    if out.exists() and not args.overwrite:
        sys.exit(f"[harness] {out} exists; pass --overwrite to replace it")
    out.parent.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    jsonl = ROOT / "logs" / f"eval_{args.tag}.episodes.jsonl"
    if args.resume and jsonl.exists():
        print(f"[harness] resume: {len({e['seed'] for e in read_episodes(jsonl)})} episodes kept from {jsonl}",
              flush=True)
    else:
        jsonl.unlink(missing_ok=True)
    srv_log = ROOT / "logs" / f"eval_{args.tag}_server.log"
    srv_log.unlink(missing_ok=True)

    env = dict(os.environ, HF_HOME=str(ROOT / "models" / "hf_home"), HF_HUB_OFFLINE="1",
               HF_DATASETS_CACHE=str(Path(ROOT.anchor) / "hfds"), TOKENIZERS_PARALLELISM="false")
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    server = None
    attempts = []
    info_start = info_end = None
    try:
        for attempt in range(args.retries + 1):
            done = {e["seed"] for e in read_episodes(jsonl)}
            todo = [args.seed + i for i in range(args.episodes) if args.seed + i not in done]
            if not todo:
                break
            # resume at the first unfinished seed; any finished seeds after it are simply re-run and deduplicated
            first, n = todo[0], args.seed + args.episodes - todo[0]
            if server is None or server.poll() is not None:
                server = start_server(args, env, srv_log)
                info_start = info_start or server_info(args.port)
            sim_log = ROOT / "logs" / f"eval_{args.tag}_sim{attempt}.log"
            cmd = [str(ISAAC_PY), args.sim_script, "--headless", "--port", str(args.port), "--episodes", str(n),
                   "--seed", str(first), "--exec-steps", str(args.exec_steps),
                   "--episode-seconds", str(args.episode_seconds), "--results-json", str(jsonl), *extra]
            print(f"[harness] attempt {attempt}: seeds {first}..{first + n - 1}, sim log {sim_log}", flush=True)
            with open(sim_log, "w", encoding="utf-8") as f:
                sim = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
                reason = None
                while sim.poll() is None:
                    if server.poll() is not None:
                        reason = f"server died (code {server.returncode})"
                        time.sleep(5)  # let the sim log its ConnectionError before it is killed
                        kill_tree(sim)
                        break
                    time.sleep(2)
                kill_tree(sim)  # Kit sometimes lingers after close()
            got = len({e["seed"] for e in read_episodes(jsonl)})
            reason = reason or f"sim exited with code {sim.returncode}"
            attempts.append({"attempt": attempt, "first_seed": first, "episodes_requested": n,
                             "episodes_total_after": got, "end": reason})
            print(f"[harness] attempt {attempt} ended ({reason}); {got}/{args.episodes} episodes done", flush=True)
        if server is not None and server.poll() is None:
            info_end = server_info(args.port)  # vram_peak_gib now covers the whole run
    finally:
        kill_tree(server)

    by_seed = {}
    for e in read_episodes(jsonl):
        if args.seed <= e["seed"] < args.seed + args.episodes:
            by_seed[e["seed"]] = dict(e, episode=e["seed"] - args.seed)  # a re-run seed keeps its latest result
    eps = [by_seed[s] for s in sorted(by_seed)]
    n, k = len(eps), sum(bool(e["success"]) for e in eps)
    n_inf = sum(e.get("n_infer") or 0 for e in eps)
    infer_mean = (sum((e.get("infer_ms_mean") or 0) * (e.get("n_infer") or 0) for e in eps) / n_inf) if n_inf else None
    res = {
        "tag": args.tag,
        "config": {"checkpoint": args.checkpoint, "sim_script": args.sim_script, "episodes": args.episodes,
                   "seed": args.seed, "seeds": [args.seed, args.seed + args.episodes - 1],
                   "exec_steps": args.exec_steps, "episode_seconds": args.episode_seconds, "extra_sim_args": extra,
                   "server_args": args.server_args, "host": platform.node()},
        "started": started,
        "wall_seconds": round(time.time() - t0, 1),
        "complete": n == args.episodes,
        "n_episodes": n,
        "n_success": k,
        "success_rate": round(k / n, 4) if n else None,
        "wilson95": wilson(k, n),
        "infer_ms_mean": round(infer_mean, 2) if infer_mean is not None else None,
        "infer_ms_max": max((e.get("infer_ms_max") or 0 for e in eps), default=None),
        "server_infer_ms_mean": (round(sum(e["server_infer_ms_mean"] * e["n_infer"] for e in eps) / n_inf, 2)
                                 if n_inf and all(e.get("server_infer_ms_mean") is not None for e in eps) else None),
        "server_info": info_start,
        "vram_peak_gib": (info_end or {}).get("vram_peak_gib"),
        "attempts": attempts,
        "episodes": eps,
    }
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[harness] {args.tag}: {k}/{n} success, Wilson95 {res['wilson95']}, infer {res['infer_ms_mean']} ms, "
          f"{res['wall_seconds']}s -> {out}", flush=True)
    if n < args.episodes:
        sys.exit(f"[harness] incomplete: {n}/{args.episodes} episodes after {len(attempts)} attempts")


if __name__ == "__main__":
    main()
