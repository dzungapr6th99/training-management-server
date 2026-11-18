"""Coverage statistics of the generated pallet datasets (PLAN.md 3.2) -> results/dataset_stats.json and
logs/pallet_probe/dataset_positions.png.

  env\\python.exe scripts\\pallet_dataset_stats.py

Per merged set (datasets/raw/humanoid_pallet_{L1,L2,all}): episodes, frames, episode length, target shape x colour and
all-parts shape x colour counts, sizes, arm, target cell, prefilled parts, target position (2.5 cm grid) and yaw,
DART share, domain-randomization ranges actually drawn; per generation job (logs/pallet_gen_<job>.jsonl): expert
success rate over all tries with its Wilson 95% interval, and how many successes were not recorded
(the part slipped out of the gripper during the carry).
"""
import collections
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
SETS = ("L1", "L2", "all")
OUT = ROOT / "results" / "dataset_stats.json"
PNG = ROOT / "logs" / "pallet_probe" / "dataset_positions.png"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def counter(it):
    return dict(sorted(collections.Counter(it).items(), key=lambda kv: str(kv[0])))


def rng_of(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return [round(float(v.min()), 3), round(float(v.max()), 3)] if len(v) else None


def load(set_name):
    eps = []
    for f in sorted((RAW / f"humanoid_pallet_{set_name}").glob("episode_*.npz")):
        with np.load(f) as z:
            m = json.loads(str(z["meta"]))
            m["_frames"] = int(z["action"].shape[0])
            m["_fps"] = float(z["fps"])
            m["_file"] = f.name
        eps.append(m)
    return eps


def stats(eps):
    if not eps:
        return {"episodes": 0}
    fr = np.array([e["_frames"] for e in eps])
    fps = eps[0]["_fps"]
    tg = [e["parts"][e["target"]] for e in eps]
    allp = [p for e in eps for p in e["parts"]]
    xy = np.array([p["xy"] for p in tg])
    dr = [e.get("dr") or {} for e in eps]
    seeds = [e["seed"] for e in eps]
    return {
        "episodes": len(eps),
        "frames": int(fr.sum()),
        "length_frames": {"min": int(fr.min()), "median": float(np.median(fr)), "max": int(fr.max()),
                          "median_seconds": round(float(np.median(fr)) / fps, 2)},
        "fps": fps,
        "dart_noise": counter(e.get("dart_noise", 0.0) for e in eps),
        "level": counter(e["level"] for e in eps),
        "split": counter(e["split"] for e in eps),
        "n_table_parts": counter(len(e["parts"]) for e in eps),
        "target_shape_color": counter(f"{p['shape']}/{p['color']}" for p in tg),
        "target_shape": counter(p["shape"] for p in tg),
        "target_color": counter(p["color"] for p in tg),
        "target_scale": counter(p["scale"] for p in tg),
        "all_table_parts_shape_color": counter(f"{p['shape']}/{p['color']}" for p in allp),
        "arm": counter(e["arm"] for e in eps),
        "target_cell": counter(e["target_cell"] for e in eps),
        "n_prefilled": counter(len(e.get("prefill", [])) for e in eps),
        "target_x_range": [round(float(xy[:, 0].min()), 3), round(float(xy[:, 0].max()), 3)],
        "target_y_range": [round(float(xy[:, 1].min()), 3), round(float(xy[:, 1].max()), 3)],
        "target_xy_grid_2p5cm": counter(f"{round(x / 0.025) * 0.025:.3f},{round(y / 0.025) * 0.025:.3f}" for x, y in xy),
        "target_yaw_deg_hist_30": counter(int(((p["yaw_deg"] + 180) % 360) // 30) * 30 - 180 for p in tg),
        "replanned_place": sum(bool(e.get("replanned")) for e in eps),
        "dr": {k: rng_of(d.get(k) for d in dr) for k in ("dome_intensity", "sun_intensity", "sun_elevation_deg",
                                                          "color_temperature_k", "table_roughness")}
              | {"table_texture": counter(d.get("table_texture") or "plain" for d in dr)},
        "unique_seeds": len(set(seeds)),
    }


def jobs():
    out = {}
    for f in sorted((ROOT / "logs").glob("pallet_gen_*.jsonl")):
        rs = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rs[r["seed"]] = r  # a restarted worker may re-run a seed after a crash: keep the latest
        k, n = sum(r["success"] for r in rs.values()), len(rs)
        kept = sum(r.get("kept", r["success"]) for r in rs.values())
        out[f.stem.replace("pallet_gen_", "")] = {"tries": n, "successes": k, "rate": round(k / max(n, 1), 3),
                                                  "wilson95": wilson(k, n), "recorded": kept,
                                                  "success_but_part_slipped": k - kept}
    return out


def plot(data):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: no position plot")
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, name in zip(axs, ("L1", "L2")):
        for e in data[name]:
            p = e["parts"][e["target"]]
            ax.plot(p["xy"][1], p["xy"][0], ".", ms=4, color="tab:red" if e["arm"] == "R" else "tab:blue")
        ax.set_title(f"{name}: target part positions ({len(data[name])} ep; blue left arm, red right)")
        ax.set_xlabel("y (m, left +)")
        ax.set_ylabel("x (m, forward)")
        ax.invert_xaxis()
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"saved {PNG}")


def main():
    data = {s: load(s) for s in SETS}
    res = {"generated_by": "scripts/pallet_dataset_stats.py", "sets": {s: stats(e) for s, e in data.items()},
           "generation_jobs": jobs()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for s, st in res["sets"].items():
        print(s, {k: st.get(k) for k in ("episodes", "frames", "length_frames", "arm", "target_cell", "target_shape",
                                           "target_color", "dart_noise")})
    for j, st in res["generation_jobs"].items():
        print(j, st)
    plot(data)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
