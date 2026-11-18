"""Rewrite the old repo location inside saved train_config.json files (needed only for `--resume=true`).

lerobot stores absolute `root` / `output_dir` in each checkpoint's train_config.json; after the folder move they still
say D:\\Isaac\\smolvla-openarm. Default is a DRY RUN. With --apply, each changed file is first copied to
train_config.json.bak. Inference does not read these paths, so skip this if you never resume old runs.

    env\\python.exe scripts\\fix_moved_paths.py            # list what would change
    env\\python.exe scripts\\fix_moved_paths.py --apply
"""
import argparse
import shutil
from pathlib import Path

from mhh_paths import ROOT

OLD = [r"D:\Isaac\smolvla-openarm", r"D:/Isaac/smolvla-openarm",
       r"D:\AI-Agent-Project\My-Humanoid-Hand\smolvla-openarm", r"D:/AI-Agent-Project/My-Humanoid-Hand/smolvla-openarm"]


def variants(p):
    return [(p, str(ROOT)), (p.replace("\\", "\\\\"), str(ROOT).replace("\\", "\\\\")),
            (p.replace("\\", "/"), ROOT.as_posix())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    seen, changed = set(), 0
    for f in sorted((ROOT / "outputs").glob("*/checkpoints/*/pretrained_model/train_config.json")):
        real = f.resolve()
        if real in seen:  # 'last' resolves to a numbered checkpoint
            continue
        seen.add(real)
        text = f.read_text(encoding="utf-8")
        new = text
        for old in OLD:
            for a, b in variants(old):
                new = new.replace(a, b)
        if new == text:
            continue
        changed += 1
        print(("rewrite " if args.apply else "would rewrite ") + str(f.relative_to(ROOT)))
        if args.apply:
            shutil.copy2(f, f.with_suffix(".json.bak"))
            f.write_text(new, encoding="utf-8")
    print(f"{changed} file(s) {'rewritten' if args.apply else 'to rewrite (dry run, pass --apply)'}")


if __name__ == "__main__":
    main()
