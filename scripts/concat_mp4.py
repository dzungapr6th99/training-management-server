"""Join mp4 clips (e.g. the *_SUCCESS.mp4 of frames_to_mp4.py) into one video, each clip preceded by a 1 s title card
with its file name. Run with the env python:

  env\\python.exe scripts\\concat_mp4.py OUT.mp4 CLIP1.mp4 CLIP2.mp4 ...
"""
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    out, clips = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    assert clips, "no clips"
    first = av.open(str(clips[0]))
    vs = first.streams.video[0]
    w, h, fps = vs.codec_context.width, vs.codec_context.height, int(round(float(vs.average_rate or 25)))
    first.close()
    with av.open(str(out), "w") as o:
        s = o.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "20", "preset": "medium"}

        def put(img):
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.asarray(img.convert("RGB")), format="rgb24")):
                o.mux(pkt)

        for i, c in enumerate(clips):
            card = Image.new("RGB", (w, h), (20, 20, 20))
            ImageDraw.Draw(card).text((40, h // 2 - 10), f"{i + 1}/{len(clips)}  {c.stem}", fill=(255, 255, 255))
            for _ in range(fps):
                put(card)
            with av.open(str(c)) as src:
                for f in src.decode(video=0):
                    img = f.to_image()
                    put(img if img.size == (w, h) else img.resize((w, h)))
        for pkt in s.encode():
            o.mux(pkt)
    print(f"{out}: {len(clips)} clips, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
