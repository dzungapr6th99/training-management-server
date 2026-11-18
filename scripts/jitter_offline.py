"""Offline jitter check of a chunked flow-matching policy (SmolVLA) on dataset observations, no sim. Run with the env
python (HF_HOME / HF_HUB_OFFLINE set):

  env\\python.exe scripts\\jitter_offline.py --checkpoint outputs\\smolvla_pallet_all\\checkpoints\\040000\\pretrained_model
      --out results\\jitter_offline_smolvla.json

Open loop: every --exec-steps frames of an expert episode the policy predicts a chunk from that frame's observation and
its first --exec-steps actions are "executed" (concatenated). Variants of how a chunk is sampled:
  random    fresh N(0, 1) noise every chunk (what the server does now)
  fixed     one noise tensor per episode, reused for every chunk
  rtc<H>    real-time chunking guidance (lerobot RTCProcessor, guided mode) towards the previous chunk's unexecuted tail,
            inference_delay 0, execution horizon H; fresh noise
  rtc<H>f   rtc<H> with the fixed noise
each without and with scripts/action_smoother.py (the sim's --smooth), and with --smooth plus an in-chunk
Savitzky-Golay filter for every window in --filters (key "+sg<W>"). Metrics on the 14 joint channels (mrad):
  jerk_p99        p99 of |a[t+1] - 2 a[t] + a[t-1]|, all steps (the expert's own actions as reference)
  jerk_p99_join   same, only the steps touching a chunk join; jerk_p99_inner the others
  switch_mean     mean |new_chunk[0:8] - previous_chunk[exec:exec+8]| = how much the plan changes between chunks
  jerk_mean       mean of the same (p99 saturates at the smoother's acceleration cap, the mean does not)
  rev_per_s       joint velocity sign flips per second per joint, steps moving > 0.5 mrad (visible shaking)
  err_mean        mean |executed - expert action| at the same frame (accuracy proxy, open loop)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from action_smoother import ActionSmoother, savgol  # noqa: E402
import humanoid_pallet_plan as HP  # noqa: E402
from policy_server import load  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JI = list(range(7)) + list(range(8, 15))


def episodes(ds, n_per_level):
    """Clean (DART 0) episodes, n per level, from the expert review metrics (same order as the dataset)."""
    m = json.loads((ROOT / "logs" / "expert_review" / "metrics.json").read_text(encoding="utf-8"))
    out = []
    for lv in (1, 2):
        out += [e["ep"] for e in m if e["level"] == lv and e["dart"] == 0][:: 200 // n_per_level][:n_per_level]
    return out


def run_episode(policy, pre, post, ds, ep, variant, a):
    meta = ds.meta.episodes[ep]
    lo, hi = meta["dataset_from_index"], meta["dataset_to_index"]
    rtc_h = int(variant[3:].rstrip("f")) if variant.startswith("rtc") else 0
    policy.config.rtc_config = RTCConfig(execution_horizon=rtc_h, max_guidance_weight=a.guidance) if rtc_h else None
    policy.init_rtc_processor()
    policy.reset()
    g = torch.Generator(device="cpu").manual_seed(1000 + ep)
    cfg = policy.config
    shape = (1, cfg.chunk_size, cfg.max_action_dim)
    fixed = torch.randn(shape, generator=g).to(cfg.device) if variant == "fixed" or variant.endswith("f") else None
    chunks, prev, ms = [], None, []
    for t in range(lo, hi, a.exec_steps):
        it = ds[t]
        obs = {k: v for k, v in it.items() if k.startswith("observation.")}
        obs["task"] = it["task"]
        batch = pre(obs)
        noise = fixed if fixed is not None else torch.randn(shape, generator=g).to(cfg.device)
        kw = {}
        if rtc_h and prev is not None:
            kw = {"inference_delay": 0, "prev_chunk_left_over": prev[:, a.exec_steps:], "execution_horizon": rtc_h}
        t0 = time.perf_counter()
        norm = policy.predict_action_chunk(batch, noise=noise, **kw)
        ms.append(1000 * (time.perf_counter() - t0))
        prev = norm.detach()
        chunks.append(post(norm.clone())[0].float().cpu().numpy())
    expert = np.stack([np.asarray(ds.hf_dataset[i]["action"], dtype=np.float32) for i in range(lo, hi)])
    return chunks, expert, ms


def execute(chunks, exec_steps, smooth, n, fw=0):
    if fw:
        chunks = [c.copy() for c in chunks]
        for c in chunks:
            c[:, JI] = savgol(c[:, JI], fw)
    if not smooth:
        return np.concatenate([c[:exec_steps] for c in chunks])[:n]
    sm = ActionSmoother(JI, np.concatenate([W.vmax, W.vmax]), 1.0 / 25, accel_scale=6.0, blend_steps=8)
    sm.reset(chunks[0][0][JI])
    return np.stack([sm.step(x) for c in chunks for x in sm.new_chunk(c, exec_steps)])[:n]


def jerk(a):
    return np.abs(a[2:] - 2 * a[1:-1] + a[:-2])[:, JI] * 1000  # row k = step k+1


def reversals(a, thr=0.5):
    """Velocity sign flips per second per joint, counting only steps moving > thr mrad (visible back-and-forth)."""
    v = np.diff(a[:, JI], axis=0) * 1000
    flip = (np.sign(v[1:]) != np.sign(v[:-1])) & (np.abs(v[1:]) > thr) & (np.abs(v[:-1]) > thr)
    return np.array([flip.sum() / len(JI) / (len(a) / 25.0)])


def metrics(seq, chunks, expert, exec_steps):
    j = jerk(seq)
    t = np.arange(1, len(seq) - 1)
    join = (t % exec_steps == 0) | (t % exec_steps == exec_steps - 1)
    sw = [np.abs(chunks[k][:8][:, JI] - chunks[k - 1][exec_steps:exec_steps + 8][:, JI]).mean() * 1000
          for k in range(1, len(chunks))]
    return {"jerk": j, "join": j[join], "inner": j[~join], "switch": sw, "rev": reversals(seq),
            "err": np.abs(seq[:, JI] - expert[: len(seq), JI]).mean(1) * 1000}


def main():
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs/smolvla_pallet_all/checkpoints/040000/pretrained_model"))
    ap.add_argument("--dataset", default=str(ROOT / "datasets/lerobot/yellowarm_pallet_all"))
    ap.add_argument("--exec-steps", type=int, default=16)
    ap.add_argument("--n-per-level", type=int, default=5)
    ap.add_argument("--variants", default="random,fixed,rtc16,rtc16f,rtc32,rtc32f")
    ap.add_argument("--filters", default="", help="Savitzky-Golay windows (order 2) to try on top of --smooth, e.g. 7,11,15")
    ap.add_argument("--guidance", type=float, default=10.0, help="RTC max_guidance_weight")
    ap.add_argument("--out", default=str(ROOT / "results/jitter_offline_smolvla.json"))
    a = ap.parse_args()
    W = HP.World(ctrl_hz=25)
    policy, pre, post, _, _, _ = load(argparse.Namespace(checkpoint=a.checkpoint, n_action_steps=0, state_dim=0,
                                                          action_dim=0))
    ds = LeRobotDataset("local/yellowarm_pallet_all", root=a.dataset)
    eps = episodes(ds, a.n_per_level)
    print(f"episodes {eps}", flush=True)
    acc = {}
    ref = []
    for v in a.variants.split(","):
        for ep in eps:
            chunks, expert, ms = run_episode(policy, pre, post, ds, ep, v, a)
            if v == a.variants.split(",")[0]:
                ref.append((jerk(expert), reversals(expert)))
            for fw, smooth in [(0, False), (0, True)] + [(int(f), True) for f in a.filters.split(",") if int(f)]:
                key = f"{v}{f'+sg{fw}' if fw else ''}{'+smooth' if smooth else ''}"
                m = metrics(execute(chunks, a.exec_steps, smooth, len(expert), fw), chunks, expert, a.exec_steps)
                d = acc.setdefault(key, {k: [] for k in (*m, "ms")})
                for k in m:
                    d[k].append(np.ravel(m[k]))
                d["ms"] += ms
            print(f"{v} ep {ep}: {len(chunks)} chunks, infer {np.mean(ms):.0f} ms", flush=True)
    rows = {"expert": {"jerk_p99": round(float(np.percentile(np.concatenate([r[0] for r in ref]), 99)), 1),
                       "jerk_mean": round(float(np.concatenate([r[0] for r in ref]).mean()), 2),
                       "rev_per_s": round(float(np.mean([r[1] for r in ref])), 2)}}
    for key, d in acc.items():
        c = {k: np.concatenate(v) if k != "ms" else np.asarray(v) for k, v in d.items()}
        rows[key] = {"jerk_p99": round(float(np.percentile(c["jerk"], 99)), 1),
                     "jerk_mean": round(float(c["jerk"].mean()), 2),
                     "rev_per_s": round(float(c["rev"].mean()), 2),
                     "jerk_p99_join": round(float(np.percentile(c["join"], 99)), 1),
                     "jerk_p99_inner": round(float(np.percentile(c["inner"], 99)), 1),
                     "switch_mean": round(float(c["switch"].mean()), 1),
                     "err_mean": round(float(c["err"].mean()), 1),
                     "infer_ms_mean": round(float(c["ms"].mean()), 0)}
    for k, r in rows.items():
        print(f"{k:16s} {r}")
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "episodes": eps, "exec_steps": a.exec_steps,
                                       "guidance": a.guidance, "units": "mrad", "rows": rows}, indent=1),
                           encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
