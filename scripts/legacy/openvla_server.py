"""OpenVLA-7B policy server. Run with env_openvla\\python.exe (transformers 4.40.1).

Same wire protocol as policy_server.py, but OpenVLA predicts ONE end-effector delta per call:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]   (robot base frame, gripper: 1 = open, 0 = close)
so `info` reports action_type="eef_delta" and the sim client turns it into joint targets with IK.
It uses a single camera (--camera, picked by name from the client's cam_names).
"""

import argparse
import sys
import socket
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wire import HOST, PORT, recv_msg, send_msg  # noqa: E402


def load(args):
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager")
    if args.quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_quant_type="nf4")
    elif args.quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForVision2Seq.from_pretrained(args.checkpoint, **kw)
    if args.quant == "none":
        model = model.to("cuda")
    model.eval()
    if args.unnorm_key not in model.norm_stats:
        raise SystemExit(f"unnorm_key {args.unnorm_key!r} not in {sorted(model.norm_stats)}")
    print(f"[openvla] loaded {args.checkpoint} quant={args.quant} unnorm_key={args.unnorm_key} "
          f"VRAM={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return processor, model


def infer(processor, model, args, img: np.ndarray, task: str) -> np.ndarray:
    prompt = f"In: What action should the robot take to {task.lower().strip().rstrip('.')}?\nOut:"
    inputs = processor(prompt, Image.fromarray(img).convert("RGB")).to("cuda", dtype=torch.bfloat16)
    # predict_action() appends the empty token 29871 to input_ids but not to attention_mask, which breaks the
    # causal mask (off by one). Append it to both here so predict_action's own check skips its append.
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        inputs["input_ids"] = torch.cat([ids, torch.full_like(ids[:, :1], 29871)], dim=1)
        inputs["attention_mask"] = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float32)[None]  # (1, 7): a chunk of length 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path(__file__).resolve().parents[2] / "models" / "openvla-7b"))
    ap.add_argument("--unnorm-key", default="bridge_orig", help="dataset whose action stats to un-normalize with")
    ap.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit",
                    help="bf16 (none) needs ~15 GB VRAM, which won't fit next to Isaac Sim on a 16 GB GPU")
    ap.add_argument("--camera", default="front", help="which sim camera OpenVLA sees")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    processor, model = load(args)

    if args.selftest:
        img = np.random.randint(0, 255, (256, 256, 3), np.uint8)
        for i in range(3):
            t = time.perf_counter()
            a = infer(processor, model, args, img, "pick up the red cube")
            print(f"[selftest] run {i}: {np.round(a[0], 4)} in {1000 * (time.perf_counter() - t):.0f} ms")
        print(f"[selftest] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, args.port))
    srv.listen(1)
    print(f"[openvla] listening on {HOST}:{args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[openvla] client {addr} connected", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                cmd = req.get("cmd", "infer")
                if cmd == "infer":
                    names = req.get("cam_names", [])
                    img = req["images"][names.index(args.camera)] if args.camera in names else req["images"][0]
                    t = time.perf_counter()
                    actions = infer(processor, model, args, img, req["task"])
                    send_msg(conn, {"actions": actions, "ms": 1000 * (time.perf_counter() - t)})
                elif cmd == "reset":
                    send_msg(conn, {"ok": True})
                elif cmd == "info":
                    send_msg(conn, {"ok": True, "model": "openvla-7b", "action_type": "eef_delta",
                                    "camera": args.camera, "unnorm_key": args.unnorm_key})
        except (ConnectionError, OSError) as e:
            print(f"[openvla] client disconnected: {e}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
