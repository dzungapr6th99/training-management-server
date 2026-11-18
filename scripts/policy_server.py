"""Policy server for any lerobot checkpoint. Run with the env python (env\\python.exe).

The policy type (smolvla / act / diffusion / pi0 / pi05 / ...) is read from the checkpoint's config.json and the
class comes from lerobot's factory; the pre/post-processors saved with the checkpoint are used as is. Policies
without a language tokenizer (ACT, Diffusion) never see the request's "task".
Diffusion keeps an observation history of n_obs_steps; here a history entry is one infer request, i.e. entries
are --exec-steps control steps apart, not one step apart as in training.

Request  : {"cmd": "infer", "state": float32[S], "images": [uint8[H,W,3], ...], "task": str, "exec_steps": int (opt.)}
           {"cmd": "reset", "seed": int (optional)} | {"cmd": "info"}
Response : infer -> {"actions": float32[T, A], "infer_ms": float}  (a full action chunk; "ms" = same, legacy)
           reset -> {"ok": True}
           info  -> {"ok": True, "policy_type", "image_keys", "n_action_steps", "state_dim", "action_dim",
                     "uses_task", "vram_gib" (allocated now), "vram_peak_gib"}

If the checkpoint's state/action dims differ from --state-dim/--action-dim (e.g. smolvla_base was
pretrained on 6-DoF SO100 while YellowArm is 7 joints + gripper = 8), the input/output features are
rewritten to the robot's dims and normalization falls back to identity. This only works for policies that
pad state/action internally (SmolVLA, pi0/pi05: max_state_dim/max_action_dim = 32), so the weights load
unchanged. Expect zero-shot behaviour from a base model to be poor; fine-tune on robot data for real tasks.

Flow-matching policies (SmolVLA, pi0/pi05) sample every chunk from noise, so consecutive chunks may pick different
plans (visible shaking at the chunk rate). Two inference-only options against that (no retraining):
  --noise fixed       one noise tensor per episode (drawn at reset from its "seed"), reused for every chunk
  --rtc-horizon H     real-time chunking guidance (lerobot RTCProcessor, guided mode): each chunk is pulled towards the
                      previous chunk's unexecuted tail over its first H steps (inference_delay 0: the sim waits for the
                      chunk). The tail starts at the request's "exec_steps" (the actions the client executed).
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from wire import HOST, PORT, recv_msg, send_msg  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_type = PreTrainedConfig.from_pretrained(args.checkpoint).type
    policy = get_policy_class(policy_type).from_pretrained(args.checkpoint)
    cfg = policy.config
    cfg.device = device
    if args.n_action_steps:
        cfg.n_action_steps = args.n_action_steps

    ckpt_state = cfg.input_features["observation.state"].shape[0]
    ckpt_action = cfg.output_features["action"].shape[0]
    args.state_dim = args.state_dim or ckpt_state
    args.action_dim = args.action_dim or ckpt_action
    adapted = (ckpt_state, ckpt_action) != (args.state_dim, args.action_dim)
    if adapted:
        print(f"[server] adapting features: state {ckpt_state}->{args.state_dim}, action {ckpt_action}->{args.action_dim}")
        cfg.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (args.state_dim,))
        cfg.output_features["action"] = PolicyFeature(FeatureType.ACTION, (args.action_dim,))
        # Checkpoint stats belong to another robot/unit system; use identity (raw radians / metres).
        pre, post = make_pre_post_processors(cfg, dataset_stats=None)
    else:
        pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint,
                                             preprocessor_overrides={"device_processor": {"device": device}})
    # GR00T packs the instruction in its own pack/encode steps (no "Tokenizer" step) but does read "task"
    uses_task = any("Tokenizer" in type(step).__name__ for step in pre.steps) or policy_type == "groot"

    args.noise, args.rtc_horizon = getattr(args, "noise", "random"), getattr(args, "rtc_horizon", 0)
    if args.noise != "random" or args.rtc_horizon:
        if not hasattr(policy, "init_rtc_processor"):
            sys.exit(f"[server] --noise/--rtc-horizon need a flow-matching policy, not {policy_type}")
    if args.rtc_horizon:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        cfg.rtc_config = RTCConfig(execution_horizon=args.rtc_horizon, max_guidance_weight=getattr(args, "rtc_guidance", 10.0))
        policy.init_rtc_processor()
    policy.to(device).eval()
    image_keys = list(cfg.image_features.keys())
    chunk = getattr(cfg, "chunk_size", None) or getattr(cfg, "horizon", None)
    print(f"[server] checkpoint={args.checkpoint} type={policy_type} device={device} image_keys={image_keys} "
          f"chunk={chunk} n_action_steps={cfg.n_action_steps} uses_task={uses_task} adapted={adapted} "
          f"noise={args.noise} rtc_horizon={args.rtc_horizon}")
    return policy, pre, post, image_keys, device, uses_task


class ChunkState:
    """Per-episode sampling state for --noise fixed / --rtc-horizon (reset by the client's "reset")."""

    def __init__(self, args):
        self.args, self.noise, self.prev = args, None, None

    def reset(self, policy, seed=None):
        self.prev = None
        self.noise = None
        if self.args.noise == "fixed":
            cfg = policy.config
            g = torch.Generator().manual_seed(int(seed) if seed is not None else int(time.time()))
            self.noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g).to(cfg.device)


def infer(policy, pre, post, image_keys, uses_task, req, cs=None):
    obs = {"observation.state": torch.from_numpy(np.asarray(req["state"], dtype=np.float32))}
    if uses_task:
        obs["task"] = req["task"]
    # Match cameras to the checkpoint's image keys by name (observation.images.<name>) when the client's
    # cam_names cover them all; otherwise fall back to order. Extra cameras are dropped.
    names = req.get("cam_names") or []
    by_name = {f"observation.images.{n}": img for n, img in zip(names, req["images"])}
    pairs = [(k, by_name[k]) for k in image_keys] if all(k in by_name for k in image_keys) \
        else list(zip(image_keys, req["images"]))
    for key, img in pairs:
        obs[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC uint8 -> CHW [0,1]
    batch = pre(obs)
    with torch.inference_mode():
        if policy.config.type == "diffusion":  # mirrors DiffusionPolicy.select_action: stack cams, push history
            batch = {k: v for k, v in batch.items() if k != ACTION}  # pre() leaves action=None in the batch
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
            policy._queues = populate_queues(policy._queues, batch)
        if cs is None or (cs.noise is None and not cs.args.rtc_horizon):
            chunk = policy.predict_action_chunk(batch)  # (1, T, A)
    if cs is not None and (cs.noise is not None or cs.args.rtc_horizon):
        # not under inference_mode: RTC guidance takes a gradient (predict_action_chunk itself runs under no_grad)
        kw = {}
        if cs.args.rtc_horizon and cs.prev is not None:
            ex = int(req.get("exec_steps") or cs.args.exec_steps)
            kw = {"inference_delay": 0, "prev_chunk_left_over": cs.prev[:, ex:], "execution_horizon": cs.args.rtc_horizon}
        chunk = policy.predict_action_chunk(batch, noise=cs.noise, **kw)
        cs.prev = chunk.detach().clone()
    chunk = post(chunk)
    return chunk[0, : policy.config.n_action_steps].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "smolvla_base"),
                    help="local dir or HF repo id of any lerobot policy checkpoint")
    ap.add_argument("--state-dim", type=int, default=0, help="robot state dim (0 = use the checkpoint's)")
    ap.add_argument("--action-dim", type=int, default=0, help="robot action dim (0 = use the checkpoint's)")
    ap.add_argument("--n-action-steps", type=int, default=0, help="actions executed per chunk (0 = checkpoint value)")
    ap.add_argument("--noise", default="random", choices=["random", "fixed"],
                    help="flow-matching policies: fresh noise per chunk, or one per episode (seeded by reset's seed)")
    ap.add_argument("--rtc-horizon", type=int, default=0, help="flow-matching policies: RTC guidance horizon (0 = off)")
    ap.add_argument("--rtc-guidance", type=float, default=10.0, help="RTC max_guidance_weight")
    ap.add_argument("--exec-steps", type=int, default=16, help="RTC: actions executed per chunk when the request has none")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--selftest", action="store_true", help="run one dummy inference and exit")
    args = ap.parse_args()

    policy, pre, post, image_keys, device, uses_task = load(args)
    cs = ChunkState(args) if (args.noise != "random" or args.rtc_horizon) else None
    if cs:
        cs.reset(policy, 0)

    if args.selftest:
        req = {"state": np.zeros(args.state_dim, np.float32),
               "images": [np.random.randint(0, 255, (256, 256, 3), np.uint8) for _ in image_keys],
               "task": "pick up the red cube"}
        for i in range(3):
            t = time.perf_counter()
            a = infer(policy, pre, post, image_keys, uses_task, req)
            print(f"[selftest] run {i}: actions {a.shape} in {1000 * (time.perf_counter() - t):.0f} ms; "
                  f"first={np.round(a[0], 3)}")
        if device == "cuda":
            print(f"[selftest] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, args.port))
    srv.listen(1)
    print(f"[server] listening on {HOST}:{args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client {addr} connected", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                cmd = req.get("cmd", "infer")
                if cmd == "infer":
                    t = time.perf_counter()
                    actions = infer(policy, pre, post, image_keys, uses_task, req, cs)
                    ms = 1000 * (time.perf_counter() - t)  # actions are already on the CPU, so the GPU is synced
                    send_msg(conn, {"actions": actions, "infer_ms": ms, "ms": ms})
                elif cmd == "reset":
                    policy.reset()
                    if cs:
                        cs.reset(policy, req.get("seed"))
                    send_msg(conn, {"ok": True})
                elif cmd == "info":
                    cuda = device == "cuda"
                    send_msg(conn, {"ok": True, "policy_type": policy.config.type, "image_keys": image_keys,
                                    "n_action_steps": policy.config.n_action_steps, "state_dim": args.state_dim,
                                    "action_dim": args.action_dim, "uses_task": uses_task,
                                    "noise": args.noise, "rtc_horizon": args.rtc_horizon,
                                    "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 3) if cuda else 0.0,
                                    "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else 0.0})
        except (ConnectionError, OSError) as e:
            print(f"[server] client disconnected: {e}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
