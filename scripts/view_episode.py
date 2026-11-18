#!/usr/bin/env python3
"""Xem lai episode expert (.npz) cua yellowarm/openarm.

Hien 3 camera canh nhau + so step, state/action, gripper, task.

Phim:  SPACE pause/play | d / a  tien / lui 1 frame (tu dong pause)
       n / p  episode ke / truoc | + / -  nhanh / cham | s  luu mp4 | q / ESC thoat

Vi du:
  env\\python.exe scripts\\view_episode.py datasets\\raw\\yellowarm_cube_in_box
  env\\python.exe scripts\\view_episode.py datasets\\raw\\yellowarm_cube_in_box --ep 5
  env\\python.exe scripts\\view_episode.py datasets\\raw\\yellowarm_cube_in_box\\episode_0003.npz --save out.mp4
"""
import argparse, glob, os, sys
import numpy as np
import cv2

def load(path):
    d = np.load(path, allow_pickle=True)
    cams = sorted(k for k in d.files if d[k].ndim == 4)
    ep = {k: d[k] for k in d.files}
    ep["_cams"] = cams
    ep["_name"] = os.path.basename(path)
    return ep

def fmt(v):
    return " ".join(f"{x:+.2f}" for x in v)

def render(ep, t, scale=2):
    imgs = [ep[c][t] for c in ep["_cams"]]
    row = np.concatenate(imgs, axis=1)                # RGB
    row = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    T = len(ep["state"])
    W = row.shape[1]
    panel = np.full((110, W, 3), 30, np.uint8)
    task = str(ep.get("task", ""))
    lines = [
        f"{ep['_name']}   step {t+1}/{T}   task: {task}",
        f"state : {fmt(ep['state'][t])}",
        f"action: {fmt(ep['action'][t])}",
    ]
    for i, s in enumerate(lines):
        cv2.putText(panel, s, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    # thanh tien do + gripper (cot cuoi cua action)
    cv2.rectangle(panel, (10, 98), (10 + int((W - 20) * (t + 1) / T), 104), (0, 200, 255), -1)
    g = float(ep["action"][t][-1])
    cv2.putText(panel, f"gripper cmd {g:+.2f}", (W - 220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if g > 0.5 else (0, 120, 255), 2, cv2.LINE_AA)
    for i, c in enumerate(ep["_cams"]):
        cv2.putText(row, c, (8 + i * imgs[0].shape[1] * scale, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    return np.concatenate([row, panel], axis=0)

def save_mp4(ep, out, fps, scale):
    f0 = render(ep, 0, scale)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f0.shape[1], f0.shape[0]))
    for t in range(len(ep["state"])):
        vw.write(render(ep, t, scale))
    vw.release()
    print("Da luu", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="thu muc chua episode_*.npz hoac 1 file .npz")
    ap.add_argument("--ep", type=int, default=0, help="chi so episode bat dau")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--save", help="chi xuat mp4 (khong mo cua so)")
    a = ap.parse_args()

    files = [a.path] if a.path.endswith(".npz") else sorted(glob.glob(os.path.join(a.path, "*.npz")))
    if not files:
        sys.exit(f"Khong thay .npz trong {a.path}")
    i = min(a.ep, len(files) - 1)
    ep = load(files[i])
    fps = float(ep.get("fps", 25))

    if a.save:
        save_mp4(ep, a.save, fps, a.scale); return

    run_viewer(files, i, fps, a.scale)

def run_viewer(files, i, fps, scale):
    # Dung tkinter vi env cai opencv-python-headless (khong co cv2.imshow)
    import tkinter as tk
    from PIL import Image, ImageTk
    st = {"i": i, "ep": load(files[i]), "t": 0, "paused": False, "speed": 1.0}
    root = tk.Tk(); root.title("expert episode viewer")
    lbl = tk.Label(root); lbl.pack()

    def show():
        frame = cv2.cvtColor(render(st["ep"], st["t"], scale), cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        lbl.configure(image=img); lbl.image = img

    def tick():
        if not st["paused"]:
            st["t"] = (st["t"] + 1) % len(st["ep"]["state"])
            show()
        root.after(max(1, int(1000 / (fps * st["speed"]))), tick)

    def key(e):
        k, T = e.keysym, len(st["ep"]["state"])
        if k in ("q", "Escape"): root.destroy(); return
        if k == "space": st["paused"] = not st["paused"]
        elif k == "d": st["paused"] = True; st["t"] = min(st["t"] + 1, T - 1)
        elif k == "a": st["paused"] = True; st["t"] = max(st["t"] - 1, 0)
        elif k in ("plus", "equal", "KP_Add"): st["speed"] *= 2
        elif k in ("minus", "KP_Subtract"): st["speed"] /= 2
        elif k in ("n", "p"):
            st["i"] = (st["i"] + (1 if k == "n" else -1)) % len(files)
            st["ep"], st["t"] = load(files[st["i"]]), 0
        elif k == "s":
            save_mp4(st["ep"], st["ep"]["_name"].replace(".npz", ".mp4"), fps, scale)
        show()

    root.bind("<Key>", key)
    show(); tick(); root.mainloop()

if __name__ == "__main__":
    main()
