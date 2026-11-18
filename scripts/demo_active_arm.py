"""Re-cut demo videos for the judges: drop the wrist-camera panel of the arm that does nothing, keep head + active wrist.
Run with the env python:

  env\\python.exe scripts\\demo_active_arm.py SRC.mp4 OUT.mp4 [--arm L|R|auto]
  env\\python.exe scripts\\demo_active_arm.py --concat OUT.mp4 A.mp4 B.mp4 ... [--titles "t1" "t2" ...]

Input layouts (from humanoid_pallet_sim.py --save-video / frames_to_mp4.py, and from expert_review.py):
  policy 768 x 1024 = overview 768 x 768 over [head | left wrist | right wrist] at 256 each
  expert 768 x 256  = [head | left wrist | right wrist]
Output: policy 768 x 1152 = overview over [head | active wrist] at 384 each; expert 512 x 256 = [head | active wrist].
--arm auto picks the wrist panel whose image changes more over the episode (the moving arm's camera moves).
"""
import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = 256  # policy camera panel size in the source videos


def read_frames(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = int(round(float(s.average_rate)))
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)], fps


def strip_y(frame):
    return frame.shape[0] - P  # the camera strip is always the bottom 256 rows


def active_arm(frames):
    y = strip_y(frames[0])
    motion = {}
    for arm, x in (("L", P), ("R", 2 * P)):
        pan = np.stack([f[y:y + P, x:x + P].astype(np.int16) for f in frames[::5]])
        motion[arm] = float(np.abs(np.diff(pan, axis=0)).mean())
    return max(motion, key=motion.get), motion


def recut(frames, arm):
    h, w = frames[0].shape[:2]
    y, xw = strip_y(frames[0]), (P if arm == "L" else 2 * P)
    out = []
    for f in frames:
        head, wrist = f[y:y + P, 0:P], f[y:y + P, xw:xw + P]
        if h == P:  # expert: head | active wrist, native size
            out.append(np.concatenate([head, wrist], axis=1))
            continue
        q = w // 2
        bot = np.concatenate([np.asarray(Image.fromarray(p).resize((q, q), Image.LANCZOS)) for p in (head, wrist)], axis=1)
        out.append(np.concatenate([f[:y], bot], axis=0))
    return out


def encode(frames, out, fps):
    h, w = frames[0].shape[:2]
    with av.open(str(out), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "20", "preset": "medium"}
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def title_card(w, h, text, n):
    img = Image.new("RGB", (w, h), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    lines = text.split("|")
    y = h // 2 - 25 * len(lines)
    for ln in lines:
        tw = dr.textlength(ln.strip(), font=font)
        dr.text(((w - tw) / 2, y), ln.strip(), fill=(255, 255, 255), font=font)
        y += 50
    return [np.asarray(img)] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--arm", default="auto", choices=["auto", "L", "R"])
    ap.add_argument("--concat", action="store_true", help="paths = OUT then inputs (already re-cut), joined in order")
    ap.add_argument("--titles", nargs="*", default=None, help="one 1.5 s title card before each input; '|' = new line")
    a = ap.parse_args()
    if a.concat:
        out, ins = a.paths[0], a.paths[1:]
        allf, fps = [], 25
        for i, p in enumerate(ins):
            fr, fps = read_frames(p)
            if a.titles:
                allf += title_card(fr[0].shape[1], fr[0].shape[0], a.titles[i], int(1.5 * fps))
            allf += fr
        encode(allf, out, fps)
        print(f"{out}: {len(ins)} clips, {len(allf)} frames")
        return
    src, out = a.paths
    frames, fps = read_frames(src)
    arm, motion = active_arm(frames)
    if a.arm != "auto":
        arm = a.arm
    encode(recut(frames, arm), out, fps)
    print(f"{Path(out).name}: arm {arm} (wrist motion L {motion['L']:.1f} R {motion['R']:.1f}), {len(frames)} frames")


if __name__ == "__main__":
    main()
