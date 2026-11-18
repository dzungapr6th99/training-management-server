"""Turn the per-episode JPEG frames of humanoid_pallet_sim.py --save-video DIR into mp4 files. Run with the env python:

  env\\python.exe scripts\\frames_to_mp4.py DIR      -> DIR\\seed_<seed>_<SUCCESS|FAIL>.mp4 (H.264, fps from info.json)

Episode folders without info.json (an episode still running or cut off) are skipped. --keep-frames keeps the JPEGs.
"""
import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
from PIL import Image


def encode(frames, out, fps):
    first = np.asarray(Image.open(frames[0]).convert("RGB"))
    h, w = first.shape[:2]
    for codec in ("libx264", "h264_nvenc", "mpeg4"):
        try:
            with av.open(str(out), "w") as c:
                s = c.add_stream(codec, rate=int(round(fps)))
                s.width, s.height, s.pix_fmt = w, h, "yuv420p"
                if codec == "libx264":
                    s.options = {"crf": "20", "preset": "medium"}
                for f in frames:
                    img = np.asarray(Image.open(f).convert("RGB"))
                    for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                        c.mux(pkt)
                for pkt in s.encode():
                    c.mux(pkt)
            return codec
        except Exception as e:  # codec not in this PyAV build: try the next one
            print(f"  {codec} failed: {e!r}")
    raise RuntimeError(f"no usable video codec for {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()
    root = Path(a.dir)
    for d in sorted(p for p in root.glob("seed_*") if p.is_dir()):
        info_p = d / "info.json"
        if not info_p.exists():
            print(f"skip {d.name}: no info.json")
            continue
        info = json.loads(info_p.read_text(encoding="utf-8"))
        frames = sorted(d.glob("*.jpg"))
        out = root / f"{d.name}_{'SUCCESS' if info['success'] else 'FAIL'}.mp4"
        codec = encode(frames, out, info["fps"])
        (root / f"{d.name}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"{out.name}: {len(frames)} frames, {codec}, {out.stat().st_size / 1e6:.1f} MB")
        if not a.keep_frames:
            shutil.rmtree(d)


if __name__ == "__main__":
    main()
