"""Unattended training of the G4 model matrix (PLAN.md 4.1 ACT, 4.2 SmolVLA) on the humanoid pallet dataset _all.

Run detached with the SmolVLA env:  env\\python.exe -u scripts\\train_pipeline.py   (progress: logs\\train_pipeline.log)

- Waits for scripts/pallet_pipeline.py to log "pallet pipeline finished" (its dataset checks exit != 0 on any problem, so
  a failed check stops the data pipeline and training never starts on a bad dataset).
- Trains one model after the other (never in parallel, never alongside an Isaac eval):
    act_pallet_all     ACT from scratch, lerobot defaults, batch 8, 100k steps (checkpoint every 20k)
    smolvla_pallet_all SmolVLA from models/smolvla_base_humanoid, batch 8, 40k steps (checkpoint every 10k), same
                       recipe as the earlier SmolVLA runs
- Re-running resumes: a run with its final checkpoint is skipped, an unfinished one continues from checkpoints/last.
- Per run writes results/train_<run>.json: steps, wall time, steps/s, peak GPU memory and final loss as logged by
  lerobot-train, checkpoints, command line.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "train_pipeline.log"
DATA_LOG = ROOT / "logs" / "pallet_pipeline.log"
# lerobot-train.exe has an absolute python path baked in -> always go through the interpreter
TRAIN = [str(ROOT / "env" / "python.exe"), "-m", "lerobot.scripts.lerobot_train"]
DS_NAME = "yellowarm_pallet_all"
DS = ROOT / "datasets" / "lerobot" / DS_NAME
ENV = {**__import__("os").environ, "HF_HOME": str(ROOT / "models" / "hf_home"), "HF_HUB_OFFLINE": "1",
       "HF_DATASETS_CACHE": str(Path(ROOT.anchor) / "hfds"), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # pi05 prints a non-cp1252 char after loading; a crash there drops the weights
WAIT_MAX_SEC = 6 * 3600

def hf_snapshot(repo):
    """Local snapshot dir of a cached hub repo: on Windows lerobot turns 'org/name' into a Path ('org\name') and then
    silently builds the policy WITHOUT pretrained weights, so the large VLAs are always given a local directory."""
    snaps = sorted((ROOT / "models" / "hf_home" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())
    return snaps[-1]


COMMON = ["--policy.push_to_hub=false", f"--dataset.repo_id=local/{DS_NAME}", f"--dataset.root={DS}",
          "--dataset.video_backend=pyav", "--batch_size=8", "--num_workers=4", "--log_freq=200", "--wandb.enable=false"]
RUNS = [
    {"run": "act_pallet_all", "model": "act", "init": "scratch (lerobot defaults)", "steps": 100000, "save_freq": 20000,
     "args": ["--policy.type=act", "--policy.device=cuda"]},
    {"run": "smolvla_pallet_all", "model": "smolvla", "init": "models/smolvla_base_humanoid", "steps": 40000,
     "save_freq": 10000, "args": [f"--policy.path={ROOT / 'models' / 'smolvla_base_humanoid'}"]},
    # added 2026-10-10 (user request): the two large VLAs, sized for 16 GB (frozen VLM, bf16, gradient checkpointing)
    {"run": "pi05_pallet_all", "model": "pi05", "init": "lerobot/pi05_base (expert-only fine-tune)", "steps": 20000,
     "save_freq": 5000, "args": ["--policy.type=pi05", f"--policy.pretrained_path={hf_snapshot('lerobot/pi05_base')}",
                                 "--policy.device=cuda", "--policy.dtype=bfloat16", "--policy.gradient_checkpointing=true",
                                 "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true",
                                 "--policy.scheduler_decay_steps=20000",
                                 '--rename_map={"observation.images.camera1": "observation.images.base_0_rgb", '
                                 '"observation.images.camera2": "observation.images.left_wrist_0_rgb", '
                                 '"observation.images.camera3": "observation.images.right_wrist_0_rgb"}']},
    {"run": "groot_pallet_all", "model": "groot", "init": "nvidia/GR00T-N1.7-3B (LLM + vision + DiT frozen, bf16)", "steps": 20000,
     "save_freq": 5000, "args": ["--policy.type=groot", f"--policy.base_model_path={hf_snapshot('nvidia/GR00T-N1.7-3B')}",
                                 "--policy.embodiment_tag=new_embodiment", "--policy.use_relative_actions=false",
                                 "--policy.use_bf16=true", "--policy.device=cuda",
                                 # 16 GB: with the DiT trainable it is 1.62B learnable params in fp32 (+AdamW) ->
                                 # spills to shared memory at ~160 s/step; Isaac-GR00T's consumer-GPU recipe
                                 "--policy.tune_diffusion_model=false", "--policy.model_params_fp32=false"]},
]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_data():
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_SEC:
        text = DATA_LOG.read_text(encoding="utf-8", errors="replace")
        text = text[text.rfind("pallet pipeline start"):]  # the latest run only
        if "pallet pipeline finished" in text:
            return True
        if "step failed" in text or "generation incomplete" in text:
            return False
        time.sleep(60)
    return False


def data_pipeline_alive():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object CommandLine"],
                         capture_output=True, text=True).stdout
    return "pallet_pipeline.py" in out


def parse_train_log(path):
    steps, losses, mem = [], [], []
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"step:(\S+) smpl.*?loss:([\d.]+).*?mem_gb:([\d.]+)", text, re.S):
        steps.append(m.group(1))
        losses.append(float(m.group(2)))
        mem.append(float(m.group(3)))
    return steps, losses, mem


def train(cfg):
    out = ROOT / "outputs" / cfg["run"]
    ck = out / "checkpoints"
    final = ck / f"{cfg['steps']:06d}" / "pretrained_model"
    res_path = ROOT / "results" / f"train_{cfg['run']}.json"
    if final.exists():
        log(f"[{cfg['run']}] final checkpoint exists, skip")
        return True
    if (ck / "last" / "pretrained_model" / "train_config.json").exists():
        cmd = [*TRAIN, f"--config_path={ck / 'last' / 'pretrained_model' / 'train_config.json'}", "--resume=true"]
        mode = "resume"
    else:
        cmd = [*TRAIN, *cfg["args"], *COMMON, f"--steps={cfg['steps']}", f"--save_freq={cfg['save_freq']}",
               f"--output_dir={out}", f"--job_name={cfg['run']}"]
        mode = "fresh"
    tlog = ROOT / "logs" / f"train_{cfg['run']}{'_resume' if mode == 'resume' else ''}.log"
    log(f"[{cfg['run']}] {mode}: {' '.join(map(str, cmd))}  (output -> {tlog.name})")
    t0 = time.time()
    with open(tlog, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
        while p.poll() is None:  # never train a large VLA from random init by accident
            if "without loading pretrained weights" in tlog.read_text(encoding="utf-8", errors="replace"):
                log(f"[{cfg['run']}] pretrained weights NOT loaded - killing the run")
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL)
                p.wait()
                break
            time.sleep(10)
        rc = p.returncode if "without loading pretrained weights" not in tlog.read_text(
            encoding="utf-8", errors="replace") else 99
    wall = time.time() - t0
    log(f"[{cfg['run']}] exit code {rc} after {wall / 3600:.2f} h")
    steps, losses, mem = parse_train_log(tlog)
    res = {"generated_by": "scripts/train_pipeline.py", "run": cfg["run"], "model": cfg["model"], "init": cfg["init"],
           "dataset": f"datasets/lerobot/{DS_NAME}", "steps": cfg["steps"], "batch_size": 8, "mode": mode,
           "exit_code": rc, "finished": final.exists(), "wall_hours": round(wall / 3600, 3),
           "steps_per_s": round(cfg["steps"] / wall, 2) if mode == "fresh" else None,
           "peak_mem_gb_logged": max(mem) if mem else None, "last_logged_step": steps[-1] if steps else None,
           "final_loss_mean_last5_logs": round(sum(losses[-5:]) / len(losses[-5:]), 4) if losses else None,
           "loss_first_log": losses[0] if losses else None,
           "checkpoints": sorted(p.name for p in ck.iterdir() if p.name != "last") if ck.exists() else [],
           "command": " ".join(map(str, cmd)), "log": f"logs/{tlog.name}"}
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"[{cfg['run']}] -> {res_path.name}: {res}")
    return rc == 0 and final.exists()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="train only these runs (skips waiting for the data pipeline)")
    ap.add_argument("--steps", type=int, help="override steps/save_freq (smoke test; results tagged _smoke)")
    a = ap.parse_args()
    if a.runs:
        todo = [dict(c) for c in RUNS if c["run"] in a.runs]
        assert len(todo) == len(a.runs), f"unknown run in {a.runs}"
        for c in todo:
            if a.steps:
                c.update(run=c["run"] + "_smoke", steps=a.steps, save_freq=a.steps)
        log(f"train pipeline start (runs {[c['run'] for c in todo]})")
        for cfg in todo:
            if not train(cfg):
                log(f"[{cfg['run']}] FAILED - continuing with the next model")
        log("train pipeline finished")
        return
    log("train pipeline start; waiting for the data pipeline")
    if not wait_for_data():
        log("data pipeline did not finish cleanly - no training started")
        sys.exit(1)
    while data_pipeline_alive():  # let it exit fully before the GPU is taken
        time.sleep(15)
    log("data pipeline finished")
    for cfg in RUNS:
        ok = train(cfg)
        if not ok:
            log(f"[{cfg['run']}] FAILED - continuing with the next model")
    log("train pipeline finished")


if __name__ == "__main__":
    main()
