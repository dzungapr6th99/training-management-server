"""Procedural table-top textures for the domain randomization of humanoid_pallet_sim.py (PLAN.md 2.5).

  env\\python.exe scripts\\make_table_textures.py      -> scripts/data/textures/table_*.png (512 x 512, tileable-ish)

Grey-scale-ish patterns (wood grain, granite, concrete, fabric, laminate, plywood); the sim multiplies them by a random
tint, so a few files give many table looks. Deterministic (fixed seed): re-running gives the same files.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "textures")
N = 512


def smooth_noise(rng, cells, n=N):
    """Value noise: random grid of `cells` x `cells`, bicubic-upsampled and wrapped (tileable)."""
    g = rng.random((cells, cells))
    img = Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
    return np.asarray(img, np.float32) / 255.0


def fbm(rng, octaves=(4, 8, 16, 32, 64), n=N):
    out, amp, tot = np.zeros((n, n), np.float32), 1.0, 0.0
    for c in octaves:
        out += amp * smooth_noise(rng, c, n)
        tot += amp
        amp *= 0.55
    return out / tot


def norm(a):
    return (a - a.min()) / max(1e-6, a.max() - a.min())


def to_rgb(gray, tint):
    return np.clip(gray[..., None] * np.asarray(tint, np.float32)[None, None], 0, 1)


def wood(rng):
    y, x = np.mgrid[0:N, 0:N].astype(np.float32) / N
    warp = fbm(rng, (3, 6, 12)) * 3.0
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (x * 14 + warp))
    fine = fbm(rng, (32, 64, 128))
    g = 0.55 + 0.25 * rings ** 3 + 0.2 * fine
    return to_rgb(norm(g) * 0.5 + 0.5, (1.0, 0.78, 0.55))


def granite(rng):
    base = fbm(rng, (8, 16, 32, 64, 128))
    spk = rng.random((N, N)).astype(np.float32)
    g = 0.55 + 0.25 * norm(base) - 0.35 * (spk > 0.985) + 0.25 * (spk < 0.01)
    return to_rgb(np.clip(g, 0, 1), (0.95, 0.95, 0.97))


def concrete(rng):
    g = 0.6 + 0.3 * norm(fbm(rng, (4, 8, 16, 32, 64, 128))) + 0.08 * (rng.random((N, N)) - 0.5)
    return to_rgb(np.clip(g, 0, 1), (0.92, 0.92, 0.9))


def fabric(rng):
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(x * 2 * np.pi / 6) * np.sin(y * 2 * np.pi / 6)
    g = 0.55 + 0.3 * weave + 0.15 * fbm(rng, (16, 32, 64))
    return to_rgb(np.clip(g, 0, 1), (0.85, 0.9, 1.0))


def laminate(rng):
    g = 0.85 + 0.1 * fbm(rng, (2, 4, 8)) + 0.03 * (rng.random((N, N)) - 0.5)
    return to_rgb(np.clip(g, 0, 1), (1.0, 1.0, 1.0))


def plywood(rng):
    y, x = np.mgrid[0:N, 0:N].astype(np.float32) / N
    bands = 0.5 + 0.5 * np.sin(2 * np.pi * (y * 40 + fbm(rng, (4, 8)) * 1.5))
    g = 0.6 + 0.2 * bands + 0.2 * fbm(rng, (32, 64))
    return to_rgb(norm(g) * 0.45 + 0.55, (1.0, 0.85, 0.65))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    for name, fn in (("wood", wood), ("granite", granite), ("concrete", concrete), ("fabric", fabric),
                     ("laminate", laminate), ("plywood", plywood)):
        img = (fn(rng) * 255).astype(np.uint8)
        path = os.path.join(OUT, f"table_{name}.png")
        Image.fromarray(img).save(path)
        print(path)


if __name__ == "__main__":
    main()
