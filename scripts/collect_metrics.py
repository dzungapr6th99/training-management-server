"""Per-episode health metrics for the raw .npz demonstrations.
Appends one JSON object per line to logs/episode_metrics.jsonl (resumable)."""
import numpy as np, glob, os, json, sys

D = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/openarm_cube_in_box_isaac"
OUT = "logs/episode_metrics.jsonl"
os.makedirs("logs", exist_ok=True)
done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try: done.add(json.loads(ln)["ep"])
        except Exception: pass

LIM = [(-3.4907,1.3963),(-3.3161,0.17453),(-1.5708,1.5708),(0.0,2.4435),
       (-1.5708,1.5708),(-0.7854,0.7854),(-1.5708,1.5708),(0.0,0.7854),(0.0,0.7854)]

def yellow(im):
    R,G,B = im[...,0].astype(np.int16), im[...,1].astype(np.int16), im[...,2].astype(np.int16)
    return ((R>120)&(G>90)&(R-B>60)&(G-B>40)).reshape(len(im),-1).sum(1)

def box_c(f0):
    R,G,B = f0[...,0].astype(np.int16), f0[...,1].astype(np.int16), f0[...,2].astype(np.int16)
    m = (B>110)&(B-R>70)&(B-G>45); m[:280,:]=False; m[:,:250]=False
    if m.sum()<200: return None
    ys,xs = np.nonzero(m); return [round(float(xs.mean()),2), round(float(ys.mean()),2), int(m.sum())]

files = sorted(glob.glob(os.path.join(D, "*.npz")))
print(f"{len(files)} episodes, {len(done)} already done", flush=True)
with open(OUT, "a") as out:
    for f in files:
        ep = os.path.basename(f)[:-4]
        if ep in done: continue
        try:
            with np.load(f) as z:
                st = z["state"].astype(np.float32); ac = z["action"].astype(np.float32)
                fps = float(z["fps"]) if "fps" in z else 25.0
                T = len(st)
                r = {"ep": ep, "T": T, "fps": fps, "dur": round(T/fps, 2),
                     "start": [round(float(v),4) for v in st[0]],
                     "end":   [round(float(v),4) for v in st[-1]],
                     "finger_min": round(float(st[:,7].min()), 4)}
                g = ac[:,7]; ch = np.where(np.diff(g) != 0)[0] + 1
                r["grip_close_f"] = int(ch[0]) if len(ch) else None
                r["grip_open_f"]  = int(ch[-1]) if len(ch) > 1 else None
                pin = {}
                for j,(lo,hi) in enumerate(LIM):
                    tol = 1e-3*max(1.0, hi-lo)
                    pin[f"j{j+1}"] = round(float(((st[:,j] <= lo+tol) | (st[:,j] >= hi-tol)).mean()*100), 1)
                r["pinned_pct"] = pin
                r["track_err"] = round(float(np.abs(ac[:-1,:7] - st[1:,:7]).mean()), 5)

                im = z["base"]
                bright = im.reshape(T,-1).mean(1)
                blk = np.where(bright < 5)[0]
                r["base_black_n"] = int(len(blk)); r["base_black_idx"] = blk.tolist()[:20]
                r["base_bright"] = round(float(bright[bright>=5].mean()), 1)
                yb = yellow(im)
                r["cube_px_f0"] = int(yb[0]); r["cube_seen_frames"] = int((yb>30).sum())
                r["cube_first_seen"] = int(np.argmax(yb>30)) if (yb>30).any() else -1
                r["box"] = box_c(im[0])
                vis = np.where(yb>30)[0]
                if len(vis):
                    fr = im[vis[0]]
                    R,G,B = fr[...,0].astype(np.int16), fr[...,1].astype(np.int16), fr[...,2].astype(np.int16)
                    m = (R>120)&(G>90)&(R-B>60)&(G-B>40)
                    ys,xs = np.nonzero(m)
                    r["cube"] = [round(float(xs.mean()),2), round(float(ys.mean()),2), int(m.sum())]
                else:
                    r["cube"] = None
                del im
                for cam in ("wrist_left","wrist_right"):
                    a = z[cam]
                    d = np.abs(np.diff(a[::4].reshape(-1, a.shape[1]*a.shape[2]*3).astype(np.int16), axis=0)).mean()
                    r[cam+"_activity"] = round(float(d), 3)
                    del a
            out.write(json.dumps(r) + "\n"); out.flush()
            print("ok", ep, flush=True)
        except Exception as e:
            print("FAIL", ep, e, flush=True)
print("ALL DONE", flush=True)
