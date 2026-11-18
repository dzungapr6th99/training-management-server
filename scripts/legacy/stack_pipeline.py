"""Unattended pipeline for the YellowArm 3-cube stacking task:
wait for expert data generation -> convert to LeRobot -> fine-tune SmolVLA -> evaluate in Isaac Sim.

Run detached with the SmolVLA env:  env\\python.exe scripts\\stack_pipeline.py   (progress: logs\\stack_pipeline.log)
"""

import glob
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # scripts/ (shared modules)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw" / "yellowarm_stack3"
DS = ROOT / "datasets" / "lerobot" / "yellowarm_stack3"
OUT = ROOT / "outputs" / "smolvla_yellowarm_stack3"
INIT = ROOT / "outputs" / "smolvla_yellowarm_cube_in_box" / "checkpoints" / "030000" / "pretrained_model"
GEN_LOG = ROOT / "logs" / "stack_gen2.log"
LOG = ROOT / "logs" / "stack_pipeline.log"
PY = ROOT / "env" / "python.exe"
from mhh_paths import ISAAC_PY as _ISAAC_PY
ISAAC_PY = str(_ISAAC_PY)
N_EPISODES, STEPS, PORT = 300, 40000, 5559
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


def main():
    # 1. wait for the generator to finish
    log("waiting for data generation")
    while True:
        n = len(glob.glob(str(RAW / "episode_*.npz")))
        done = GEN_LOG.exists() and "[expert] done" in GEN_LOG.read_text(errors="ignore")
        if n >= N_EPISODES and done:
            break
        time.sleep(60)
    log(f"generation finished: {n} episodes")

    # 2. convert
    run([PY, "-u", ROOT / "scripts" / "npz_to_lerobot.py", "--raw", RAW, "--out", DS,
         "--repo-id", "local/yellowarm_stack3", "--robot-type", "yellowarm_isaac", "--overwrite"],
        ROOT / "logs" / "stack_convert.log")

    # 3. fine-tune from the cube-in-box YellowArm checkpoint
    run([ROOT / "env" / "python.exe", "-m", "lerobot.scripts.lerobot_train", f"--policy.path={INIT}", "--policy.push_to_hub=false",
         "--dataset.repo_id=local/yellowarm_stack3", f"--dataset.root={DS}", "--dataset.video_backend=pyav",
         "--batch_size=8", "--num_workers=4", f"--steps={STEPS}", "--log_freq=200", "--save_freq=10000",
         f"--output_dir={OUT}", "--job_name=smolvla_yellowarm_stack3", "--wandb.enable=false"],
        ROOT / "logs" / "stack_train.log")

    # 4. evaluate: policy server + Isaac Sim (headless, 20 episodes)
    ckpt = OUT / "checkpoints" / f"{STEPS:06d}" / "pretrained_model"
    srv_log = open(ROOT / "logs" / "stack_eval_server.log", "w", encoding="utf-8", errors="replace")
    srv = subprocess.Popen([str(PY), "-u", str(ROOT / "scripts" / "policy_server.py"), "--checkpoint", str(ckpt),
                            "--port", str(PORT)], stdout=srv_log, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    log(f"eval server pid {srv.pid} on port {PORT}")
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            break
        except OSError:
            time.sleep(2)
    try:
        run(["cmd", "/c", ISAAC_PY, ROOT / "scripts" / "legacy" / "yellowarm_stack_sim.py", "--headless", "--port", PORT,
             "--episodes", "20", "--episode-seconds", "16", "--seed", "777"], ROOT / "logs" / "stack_eval.log")
    finally:
        srv.terminate()
    result = [l for l in (ROOT / "logs" / "stack_eval.log").read_text(errors="ignore").splitlines() if "evaluation done" in l]
    log(f"RESULT: {result[-1] if result else 'no result line'}")
    log("pipeline finished")


if __name__ == "__main__":
    main()
