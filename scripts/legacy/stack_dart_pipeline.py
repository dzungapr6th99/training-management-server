"""Unattended pipeline for the DART round of the YellowArm 3-cube stack task:
wait for the two DART generators -> merge with the 300 clean episodes -> convert -> fine-tune from the stack3 40k
checkpoint -> evaluate several checkpoints in Isaac Sim (headless, 20 episodes each).

Run detached with the SmolVLA env:  env\\python.exe scripts\\stack_dart_pipeline.py   (progress: logs\\stack_dart_pipeline.log)
"""
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # scripts/ (shared modules)
ROOT = Path(__file__).resolve().parents[2]
RAW_CLEAN = ROOT / "datasets" / "raw" / "yellowarm_stack3"
RAW_DART = [ROOT / "datasets" / "raw" / f"yellowarm_stack3_dart_{w}" for w in "ab"]
GEN_LOGS = [ROOT / "logs" / f"dart_gen_{w}.log" for w in "ab"]
RAW_MIX = ROOT / "datasets" / "raw" / "yellowarm_stack3_mix"
DS = ROOT / "datasets" / "lerobot" / "yellowarm_stack3_mix"
REPO_ID = "local/yellowarm_stack3_mix"
INIT = ROOT / "outputs" / "smolvla_yellowarm_stack3" / "checkpoints" / "040000" / "pretrained_model"
OUT = ROOT / "outputs" / "smolvla_yellowarm_stack3_dart"
LOG = ROOT / "logs" / "stack_dart_pipeline.log"
PY = ROOT / "env" / "python.exe"
ISAAC_PY = r"D:\Isaac\isaacsim\python.bat"
N_PER_WORKER, STEPS, PORT = 150, 40000, 5559
EVAL_STEPS = [20000, 30000, 40000]
STALE_SEC = 30 * 60  # a generator log that has not changed for this long is considered dead
ENV = {**os.environ, "HF_HOME": str(ROOT / "models" / "hf_home"), "HF_HUB_OFFLINE": "1"}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, logfile):
    log(f"run: {' '.join(map(str, cmd))}  (output -> {logfile})")
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    log(f"exit code {rc}")
    if rc != 0:
        sys.exit(f"step failed, see {logfile}")


def wait_generators():
    log("waiting for DART data generation")
    while True:
        states = []
        for raw, gl in zip(RAW_DART, GEN_LOGS):
            n = len(list(raw.glob("episode_*.npz"))) if raw.exists() else 0
            done = gl.exists() and "[expert] done" in gl.read_text(errors="ignore")
            stale = gl.exists() and time.time() - gl.stat().st_mtime > STALE_SEC
            states.append((n, done, stale))
        if all(d for _, d, _ in states):
            break
        for (n, done, stale), gl in zip(states, GEN_LOGS):
            if stale and not done:
                log(f"WARNING: {gl.name} has not changed for {STALE_SEC // 60} min ({n} episodes) - continuing with what exists")
                return
        time.sleep(60)
    log("generation finished: " + ", ".join(f"{r.name}={n}" for r, (n, _, _) in zip(RAW_DART, states)))


def merge():
    if RAW_MIX.exists():
        shutil.rmtree(RAW_MIX)
    RAW_MIX.mkdir(parents=True)
    k = 0
    for src in [RAW_CLEAN, *RAW_DART]:
        for f in sorted(src.glob("episode_*.npz")):
            os.link(f, RAW_MIX / f"episode_{k:04d}.npz")  # hard link: same drive, no extra disk space
            k += 1
    log(f"merged {k} episodes into {RAW_MIX}")


def evaluate(step):
    ckpt = OUT / "checkpoints" / f"{step:06d}" / "pretrained_model"
    srv_log = open(ROOT / "logs" / f"stack_dart_server_{step // 1000}k.log", "w", encoding="utf-8", errors="replace")
    srv = subprocess.Popen([str(PY), "-u", str(ROOT / "scripts" / "policy_server.py"), "--checkpoint", str(ckpt),
                            "--port", str(PORT)], stdout=srv_log, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    log(f"eval {step // 1000}k: server pid {srv.pid} on port {PORT}")
    time.sleep(5)
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            break
        except OSError:
            time.sleep(2)
    eval_log = ROOT / "logs" / f"stack_dart_eval_{step // 1000}k.log"
    try:
        run(["cmd", "/c", ISAAC_PY, ROOT / "scripts" / "legacy" / "yellowarm_stack_sim.py", "--headless", "--port", PORT,
             "--episodes", "20", "--episode-seconds", "16", "--seed", "777"], eval_log)
    finally:
        srv.terminate()
        srv.wait()
    result = [l for l in eval_log.read_text(errors="ignore").splitlines() if "evaluation done" in l]
    log(f"RESULT {step // 1000}k: {result[-1] if result else 'no result line'}")


def main():
    wait_generators()
    merge()
    run([PY, "-u", ROOT / "scripts" / "npz_to_lerobot.py", "--raw", RAW_MIX, "--out", DS,
         "--repo-id", REPO_ID, "--robot-type", "yellowarm_isaac", "--overwrite"],
        ROOT / "logs" / "stack_dart_convert.log")
    run([ROOT / "env" / "python.exe", "-m", "lerobot.scripts.lerobot_train", f"--policy.path={INIT}", "--policy.push_to_hub=false",
         f"--dataset.repo_id={REPO_ID}", f"--dataset.root={DS}", "--dataset.video_backend=pyav",
         "--batch_size=8", "--num_workers=4", f"--steps={STEPS}", "--log_freq=200", "--save_freq=10000",
         f"--output_dir={OUT}", "--job_name=smolvla_yellowarm_stack3_dart", "--wandb.enable=false"],
        ROOT / "logs" / "stack_dart_train.log")
    for step in EVAL_STEPS:
        evaluate(step)
    log("pipeline finished")


if __name__ == "__main__":
    main()
