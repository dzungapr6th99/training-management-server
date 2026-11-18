"""Unattended G5 evaluation of one model (PLAN.md 5.1 + 5.2) on the humanoid pallet task. Run with the env python:

  env\\python.exe -u scripts\\eval_pipeline.py --name act_pallet_all \\
      --checkpoint outputs\\act_pallet_all\\checkpoints\\100000\\pretrained_model --levels 1   (progress: logs\\eval_pipeline_<name>.log)

5.1  exec-steps: every value of --exec-values at L1 / split train, 20 episodes on the validation seeds 200000-200019;
     the best success rate wins, ties go to the larger value. Choice merged into results/exec_choice.json.
5.2  for every level in --levels x every split: 50 episodes on the test seeds 100000-100049 with the chosen exec-steps.
--episode-seconds = 2 x the median expert episode of the level, read from results/dataset_stats.json (5.1 uses L1's).
Every run goes through scripts/eval_harness.py (results/<tag>.json); --workers harness runs in parallel, each with its
own policy server port. Re-running skips runs whose results/<tag>.json is complete.
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PY = ROOT / "env" / "python.exe"
SIM = ROOT / "scripts" / "humanoid_pallet_sim.py"
SPLITS = ("train", "heldout_color", "heldout_shape", "wide")
SEED_VAL, N_VAL = 200000, 20
SEED_TEST, N_TEST = 100000, 50
LOG = None


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def result(tag):
    p = ROOT / "results" / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run_jobs(jobs, args):
    """jobs: [(tag, seed, episodes, exec_steps, episode_seconds, level, split)] -> {tag: result json}."""
    ports = list(range(args.port, args.port + args.workers))

    def one(i_job):
        i, (tag, seed, n, ex, secs, level, split) = i_job
        r = result(tag)
        if r and r.get("complete"):
            log(f"[{tag}] already complete: {r['n_success']}/{r['n_episodes']}")
            return tag, r
        port = ports[i % len(ports)]  # jobs are handed out in waves of len(ports), so ports never collide
        cmd = [str(ENV_PY), "-u", str(ROOT / "scripts" / "eval_harness.py"), "--sim-script", str(SIM),
               "--checkpoint", args.checkpoint, "--episodes", str(n), "--seed", str(seed), "--exec-steps", str(ex),
               "--episode-seconds", str(secs), "--tag", tag, "--port", str(port), "--overwrite",
               *(["--resume"] if args.resume else []),
               "--", "--level", str(level), "--split", split, *args.sim_args.split()]
        log(f"[{tag}] start (port {port}): seeds {seed}..{seed + n - 1}, exec {ex}, {secs} s, L{level} {split}")
        t0 = time.time()
        with open(ROOT / "logs" / f"eval_{tag}_harness.log", "w", encoding="utf-8") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        r = result(tag)
        if r:
            log(f"[{tag}] exit {code} after {(time.time() - t0) / 60:.0f} min: {r['n_success']}/{r['n_episodes']} "
                f"Wilson95 {r['wilson95']}, infer {r['infer_ms_mean']} ms")
        else:
            log(f"[{tag}] exit {code}: no results file")
        return tag, r

    out = {}
    for w0 in range(0, len(jobs), args.workers):
        wave = list(enumerate(jobs[w0:w0 + args.workers]))
        with ThreadPoolExecutor(len(wave)) as ex:
            out.update(dict(ex.map(one, wave)))
    return out


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="model name used in the result tags")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--exec-values", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--port", type=int, default=5561)
    ap.add_argument("--resume", action="store_true",
                    help="keep the per-episode jsonl of interrupted runs (same config only) instead of starting over")
    ap.add_argument("--sim-args", default="", help="extra humanoid_pallet_sim.py args for every run, e.g. \"--smooth\"")
    args = ap.parse_args()
    LOG = ROOT / "logs" / f"eval_pipeline_{args.name}.log"
    stats = json.loads((ROOT / "results" / "dataset_stats.json").read_text(encoding="utf-8"))["sets"]
    secs = {lv: round(2 * stats[f"L{lv}"]["length_frames"]["median_seconds"], 2) for lv in (1, 2)}
    log(f"eval pipeline start: {args.name} {args.checkpoint}, episode seconds {secs}, sim args {args.sim_args!r}")

    # 5.1 exec-steps on the validation seeds
    val = run_jobs([(f"val_{args.name}_L1_exec{e}", SEED_VAL, N_VAL, e, secs[1], 1, "train") for e in args.exec_values],
                   args)
    rates = {e: val[f"val_{args.name}_L1_exec{e}"] for e in args.exec_values}
    if not all(r and r.get("complete") for r in rates.values()):
        log("exec-steps selection incomplete - stopping")
        sys.exit(1)
    best = max(args.exec_values, key=lambda e: (rates[e]["n_success"], e))
    choice_path = ROOT / "results" / "exec_choice.json"
    choice = json.loads(choice_path.read_text(encoding="utf-8")) if choice_path.exists() else {}
    choice[args.name] = {"checkpoint": args.checkpoint, "exec_steps": best, "rule": "max success, ties -> larger", "sim_args": args.sim_args,
                         "validation": {"seeds": [SEED_VAL, SEED_VAL + N_VAL - 1], "level": 1, "split": "train",
                                        "episode_seconds": secs[1]},
                         "candidates": {str(e): {"n_success": rates[e]["n_success"], "n_episodes": rates[e]["n_episodes"],
                                                 "wilson95": rates[e]["wilson95"], "tag": f"val_{args.name}_L1_exec{e}"}
                                        for e in args.exec_values},
                         "chosen": dt.datetime.now().isoformat(timespec="seconds")}
    choice_path.write_text(json.dumps(choice, indent=2), encoding="utf-8")
    log(f"exec-steps chosen: {best} ({ {e: rates[e]['n_success'] for e in args.exec_values} } / {N_VAL})")

    # 5.2 test seeds
    run_jobs([(f"{args.name}_L{lv}_{sp}", SEED_TEST, N_TEST, best, secs[lv], lv, sp)
              for lv in args.levels for sp in SPLITS], args)
    log("eval pipeline finished")


if __name__ == "__main__":
    main()
