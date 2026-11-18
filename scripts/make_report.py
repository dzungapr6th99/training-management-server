"""G6.1 — gather results/*.json into tables and charts for the Parts-to-Pallet benchmark.

Stdlib only (env python or any Python 3.10+). Reads:
  results/<run>_L<level>_<split>.json   official evals from eval_harness.py (5.2, seed test 100000-100049)
  results/val_<run>_L1_exec<N>.json     exec-steps selection (5.1, seed validation 200000+)
  results/exec_choice.json              chosen exec-steps per run
  results/train_*.json                  training cost (train_pipeline.py)
  results/expert_*.json                 scripted expert reference (JSONL, one episode per line)
  results/dataset_stats.json            dataset size / arm split
Writes:
  results/summary.csv          one row per (model, level, split) eval cell
  results/summary_train.csv    one row per training run
  results/summary_tables.md    markdown tables (numbers come only from here)
  results/REPORT.md            the same tables spliced between its BEGIN/END:tables markers
  results/figures/*.svg        success_rates, generalization, cost

Usage: env\\python.exe scripts\\make_report.py [--results results]
"""
import argparse
import csv
import glob
import html
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["train", "heldout_color", "heldout_shape", "wide"]
SPLIT_LABEL = {"train": "in-dist", "heldout_color": "heldout màu", "heldout_shape": "heldout hình", "wide": "wide"}
LEVELS = [1, 2]
CTRL_HZ = 25.0
TAG_RE = re.compile(r"^(?P<run>.+)_L(?P<level>\d)_(?P<split>" + "|".join(SPLITS) + r")$")
SKIP_PREFIX = ("val_", "demo_", "smoke_", "test_", "expert_", "baseline_", "train_")
# entity -> fixed categorical slot (color follows the model, never its rank)
MODEL_ORDER = ["act", "smolvla", "groot", "pi05"]
MODEL_LABEL = {"act": "ACT", "smolvla": "SmolVLA", "groot": "GR00T N1.7", "pi05": "π0.5"}
FAIL_ORDER = ["success", "wrong_part", "no_lift", "lifted_not_in_cell", "in_cell_bad_place"]
FAIL_LABEL = {"success": "thành công", "wrong_part": "vật nhiễu lên pallet", "no_lift": "không nhấc được vật đích",
              "lifted_not_in_cell": "nhấc nhưng không vào ô", "in_cell_bad_place": "vào ô nhưng đặt hỏng"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fail_stage(ep):
    """Mutually exclusive outcome of one eval episode (first matching cause wins)."""
    if ep.get("success"):
        return "success"
    if ep.get("others_on_pallet"):
        return "wrong_part"
    if not ep.get("lifted"):
        return "no_lift"
    if ep.get("first_in_cell_step") is None:
        return "lifted_not_in_cell"
    return "in_cell_bad_place"


def tqdm_segment(log_path):
    """(last_step, elapsed_hours) of the last tqdm progress entry in a lerobot-train log, or None."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = None
    for m in re.finditer(r"\|\s*(\d+)/(\d+) \[([\d:]+)<", text):
        pass
    if m is None:
        return None
    secs = 0
    for part in m.group(3).split(":"):
        secs = secs * 60 + int(part)
    return int(m.group(1)), secs / 3600.0


# ---------------------------------------------------------------- gather

def gather_evals(res_dir):
    cells = []
    for path in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith(SKIP_PREFIX):
            continue
        m = TAG_RE.match(name)
        if not m:
            continue
        d = load_json(path)
        if not isinstance(d, dict) or "episodes" not in d:
            continue
        eps = d["episodes"]
        n = len(eps)
        k = sum(1 for e in eps if e.get("success"))
        lo, hi = wilson(k, n)
        cfg = d.get("config", {})
        info = d.get("server_info", {})
        stages = {s: 0 for s in FAIL_ORDER}
        for e in eps:
            stages[fail_stage(e)] += 1
        arm = {a: [0, 0] for a in ("L", "R")}
        for e in eps:
            if e.get("arm") in arm:
                arm[e["arm"]][1] += 1
                arm[e["arm"]][0] += bool(e.get("success"))
        seeds = sorted(e["seed"] for e in eps)
        exec_steps = cfg.get("exec_steps")
        cells.append({
            "run": m.group("run"), "model": info.get("policy_type", "?"), "level": int(m.group("level")),
            "split": m.group("split"), "n": n, "k": k, "rate": k / n if n else 0.0, "lo": lo, "hi": hi,
            "complete": d.get("complete"), "seed_first": seeds[0] if seeds else None,
            "seed_last": seeds[-1] if seeds else None,
            "checkpoint": cfg.get("checkpoint", "").replace("\\", "/"), "exec_steps": exec_steps,
            "episode_seconds": cfg.get("episode_seconds"), "smooth": all(e.get("smooth") for e in eps),
            "chunk": info.get("n_action_steps"), "uses_task": info.get("uses_task"),
            "infer_ms_sim": d.get("infer_ms_mean"), "infer_ms_server": d.get("server_infer_ms_mean"),
            "infer_ms_max": d.get("infer_ms_max"),
            "budget_ms": 1000.0 * exec_steps / CTRL_HZ if exec_steps else None,
            "vram_infer_gib": d.get("vram_peak_gib"), "lifted": sum(1 for e in eps if e.get("lifted")),
            "arm_L": arm["L"], "arm_R": arm["R"], "stages": stages, "source": os.path.basename(path),
        })
    return cells


def gather_train(res_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(res_dir, "train_*.json"))):
        d = load_json(path)
        hours, steps_note = d.get("wall_hours"), ""
        if d.get("mode") == "resume":
            # the json only covers the resumed segment; add the first segment from the original log
            first = tqdm_segment(os.path.join(ROOT, "logs", f"train_{d['run']}.log"))
            if first and hours is not None:
                hours += first[1]
                steps_note = f"2 đoạn: 0–{first[0]} ({first[1]:.2f} h) + resume ({d['wall_hours']:.2f} h)"
        sps = d.get("steps_per_s")
        if sps is None and hours and d.get("finished"):
            sps = d["steps"] / (hours * 3600)
        rows.append({
            "run": d["run"], "model": d.get("model"), "init": d.get("init"), "steps": d.get("steps"),
            "batch_size": d.get("batch_size"), "finished": d.get("finished"), "exit_code": d.get("exit_code"),
            "wall_hours": hours, "steps_per_s": sps, "peak_mem_gb_logged": d.get("peak_mem_gb_logged"),
            "loss_first": d.get("loss_first_log"), "loss_final": d.get("final_loss_mean_last5_logs"),
            "note": steps_note, "source": os.path.basename(path),
        })
    return rows


def gather_exec(res_dir):
    path = os.path.join(res_dir, "exec_choice.json")
    out = {}
    if not os.path.exists(path):
        return out
    for run, c in load_json(path).items():
        cands = {int(e): (v["n_success"], v["n_episodes"]) for e, v in c.get("candidates", {}).items()}
        out[run] = {"chosen": c.get("exec_steps"), "candidates": cands, "sim_args": c.get("sim_args", ""),
                    "seeds": c.get("validation", {}).get("seeds")}
    return out


def gather_expert(res_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(res_dir, "expert_*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith("_v1"):
            continue
        eps = load_jsonl(path)
        n, k = len(eps), sum(1 for e in eps if e.get("success"))
        lo, hi = wilson(k, n)
        seeds = sorted(e["seed"] for e in eps)
        dr = all(e.get("dr") for e in eps)
        out.append({"file": name, "level": eps[0].get("level") if eps else None,
                    "split": eps[0].get("split") if eps else None, "n": n, "k": k, "lo": lo, "hi": hi,
                    "seeds": f"{seeds[0]}–{seeds[-1]}" if seeds else "", "dr": dr})
    return out


# ---------------------------------------------------------------- outputs

def fmt_cell(c):
    return f"{c['k']}/{c['n']} ({100 * c['rate']:.0f}%) [{c['lo']:.2f}, {c['hi']:.2f}]"


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def by_key(cells):
    return {(c["model"], c["level"], c["split"]): c for c in cells}


def models_in(cells):
    present = {c["model"] for c in cells}
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def ratio(a, b):
    return a / b if b else None


def markdown(cells, train, execs, experts, dstats):
    idx = by_key(cells)
    models = models_in(cells)
    L = []
    L.append("<!-- sinh tự động bởi scripts/make_report.py — không sửa tay -->")
    L.append("## Bảng 1 — Tỉ lệ thành công (seed test, Wilson 95%)\n")
    hdr = ["model", "exec", "chunk"] + [f"L{l} {SPLIT_LABEL[s]}" for l in LEVELS for s in SPLITS]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for m in models:
        any_c = next(c for c in cells if c["model"] == m)
        row = [MODEL_LABEL.get(m, m), str(any_c["exec_steps"]), str(any_c["chunk"])]
        for l in LEVELS:
            for s in SPLITS:
                c = idx.get((m, l, s))
                row.append(fmt_cell(c) if c else "N/A")
        L.append("| " + " | ".join(row) + " |")
    seeds = sorted({(c["seed_first"], c["seed_last"]) for c in cells})
    esec = sorted({(c["level"], c["episode_seconds"]) for c in cells})
    L.append("")
    L.append(f"Seed: {', '.join(f'{a}–{b}' for a, b in seeds)}; episode-seconds: "
             + ", ".join(f"L{l} {s} s" for l, s in esec)
             + f"; chống rung (`--smooth`): {'có' if all(c['smooth'] for c in cells) else 'không đồng nhất'}; "
             + f"mọi ô hoàn tất: {'có' if all(c['complete'] for c in cells) else 'KHÔNG'}.")
    L.append("N/A = không chạy (ACT không nhận câu lệnh nên không có L2).\n")

    L.append("## Bảng 2 — Tổng quát hoá: tỉ lệ split / tỉ lệ in-dist cùng mức\n")
    L.append("| model | mức | in-dist | heldout màu | heldout hình | wide |")
    L.append("|---|---|---|---|---|---|")
    for m in models:
        for l in LEVELS:
            base = idx.get((m, l, "train"))
            if not base:
                continue
            row = [MODEL_LABEL.get(m, m), f"L{l}", f"{100 * base['rate']:.0f}%"]
            for s in SPLITS[1:]:
                c = idx.get((m, l, s))
                r = ratio(c["rate"], base["rate"]) if c else None
                row.append("N/A" if c is None else (f"{100 * c['rate']:.0f}% (×{r:.2f})" if r is not None else f"{100 * c['rate']:.0f}%"))
            L.append("| " + " | ".join(row) + " |")
    L.append("\nTỉ số tính trên điểm ước lượng; với k nhỏ (≤ 15/50) khoảng tin cậy của tỉ số rất rộng — đọc kèm Bảng 1.\n")

    L.append("## Bảng 3 — Phân loại kết cục (mỗi episode đúng 1 loại)\n")
    L.append("| model | mức | split | " + " | ".join(FAIL_LABEL[s] for s in FAIL_ORDER) + " | nhấc được vật đích | thành công tay trái | tay phải |")
    L.append("|---|---|---|" + "---|" * (len(FAIL_ORDER) + 3))
    for m in models:
        for l in LEVELS:
            for s in SPLITS:
                c = idx.get((m, l, s))
                if not c:
                    continue
                L.append("| " + " | ".join([MODEL_LABEL.get(m, m), f"L{l}", SPLIT_LABEL[s]]
                                           + [str(c["stages"][f]) for f in FAIL_ORDER]
                                           + [f"{c['lifted']}/{c['n']}", f"{c['arm_L'][0]}/{c['arm_L'][1]}",
                                              f"{c['arm_R'][0]}/{c['arm_R'][1]}"]) + " |")
    L.append("\nThứ tự xét: thành công → có vật khác trên pallet → không nhấc vật đích (> 3 cm) → chưa từng vào ô → "
             "còn lại (vào ô nhưng nghiêng/chưa yên/ô đã có vật/không còn trong ô lúc kết thúc).\n")

    L.append("## Bảng 4 — Chi phí và độ trễ\n")
    L.append("| run | model | khởi tạo | step | batch | giờ train | step/s | VRAM train (log, GB) | loss đầu → cuối | ghi chú |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(train, key=lambda t: (MODEL_ORDER.index(t["model"]) if t["model"] in MODEL_ORDER else 99, t["run"])):
        L.append("| " + " | ".join([
            t["run"], MODEL_LABEL.get(t["model"], t["model"]), str(t["init"]), str(t["steps"]), str(t["batch_size"]),
            f"{t['wall_hours']:.2f}" if t["wall_hours"] is not None else "–",
            f"{t['steps_per_s']:.2f}" if t["steps_per_s"] else "–",
            f"{t['peak_mem_gb_logged']:.2f}" if t["peak_mem_gb_logged"] is not None else "–",
            (f"{t['loss_first']} → {t['loss_final']}" if t["loss_final"] is not None else "–"),
            t["note"] or ("không hoàn tất (exit %s)" % t["exit_code"] if not t["finished"] else ""),
        ]) + " |")
    L.append("")
    L.append("| model | exec | ngân sách exec/25 Hz | infer server TB (ms) | infer phía sim TB (ms) | infer phía sim max (ms) | đạt ngân sách (TB) | VRAM infer đỉnh (GiB) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        cs = [c for c in cells if c["model"] == m]
        n = sum(c["n"] for c in cs)
        srv = sum(c["infer_ms_server"] * c["n"] for c in cs) / n
        sim = sum(c["infer_ms_sim"] * c["n"] for c in cs) / n
        mx = max(c["infer_ms_max"] for c in cs)
        bud = cs[0]["budget_ms"]
        vr = max(c["vram_infer_gib"] or 0 for c in cs)
        L.append(f"| {MODEL_LABEL.get(m, m)} | {cs[0]['exec_steps']} | {bud:.0f} ms | {srv:.0f} | {sim:.0f} | {mx:.0f} | "
                 f"{'có' if sim <= bud else 'KHÔNG'}{' (max vượt)' if mx > bud else ''} | {vr:.2f} |")
    L.append("\nĐộ trễ là trung bình có trọng số theo episode trên mọi ô của Bảng 1; đo khi 2 worker eval dùng chung GPU "
             "(RTX 4070 Ti SUPER) nên cao hơn lúc GPU rảnh. VRAM train là số `mem` lerobot in ra trong log, không phải "
             "đỉnh `nvidia-smi`.\n")

    L.append("## Bảng 5 — Chọn exec-steps (seed validation, L1 in-dist, 20 ep)\n")
    L.append("| run | " + " | ".join(f"exec {e}" for e in (4, 8, 16, 32)) + " | chọn |")
    L.append("|---|---|---|---|---|---|")
    for run, e in execs.items():
        row = [run + (f" ({e['sim_args']})" if e["sim_args"] else "")]
        for x in (4, 8, 16, 32):
            kn = e["candidates"].get(x)
            row.append(f"{kn[0]}/{kn[1]}" if kn else "–")
        row.append(str(e["chosen"]))
        L.append("| " + " | ".join(row) + " |")
    L.append("\nRun không có `--smooth` chỉ để tham khảo (5.2 tương ứng đã dừng giữa chừng, không có kết quả).\n")

    L.append("## Bảng 6 — Expert lập trình sẵn (tham chiếu trần, KHÔNG cùng bộ seed với Bảng 1)\n")
    L.append("| file | mức | split | DR | seed | thành công |")
    L.append("|---|---|---|---|---|---|")
    for e in experts:
        L.append(f"| {e['file']} | L{e['level']} | {e['split']} | {'có' if e['dr'] else 'không'} | {e['seeds']} | "
                 f"{e['k']}/{e['n']} [{e['lo']:.2f}, {e['hi']:.2f}] |")
    if dstats:
        a = dstats["sets"]["all"]
        L.append(f"\nDataset `_all`: {a['episodes']} episode, {a['frames']} frame; tay trái/phải "
                 f"{a['arm'].get('L')}/{a['arm'].get('R')}.\n")

    L.append("## Bảng 7 — Ngưỡng G7 (đề xuất trong PLAN.md)\n")
    L.append("| model | L1 in-dist ≥ 80% | L2 in-dist ≥ 60% | wide ≥ ½ in-dist (L1) | infer TB ≤ ngân sách |")
    L.append("|---|---|---|---|---|")
    for m in models:
        l1, l2, w = idx.get((m, 1, "train")), idx.get((m, 2, "train")), idx.get((m, 1, "wide"))
        cs = [c for c in cells if c["model"] == m]
        sim = sum(c["infer_ms_sim"] * c["n"] for c in cs) / sum(c["n"] for c in cs)

        def yn(ok, txt):
            return f"{'ĐẠT' if ok else 'không'} ({txt})"
        L.append("| " + " | ".join([
            MODEL_LABEL.get(m, m),
            yn(l1 and l1["rate"] >= 0.8, f"{100 * l1['rate']:.0f}%") if l1 else "N/A",
            yn(l2 and l2["rate"] >= 0.6, f"{100 * l2['rate']:.0f}%") if l2 else "N/A",
            yn(l1 and w and w["rate"] >= 0.5 * l1["rate"] and l1["rate"] > 0, f"{100 * w['rate']:.0f}% vs {100 * l1['rate']:.0f}%") if (l1 and w) else "N/A",
            yn(sim <= cs[0]["budget_ms"], f"{sim:.0f} ≤ {cs[0]['budget_ms']:.0f} ms"),
        ]) + " |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- SVG charts

SVG_STYLE = """
<style>
svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e0;--axis:#c9c8c2;
--act:#2a78d6;--smolvla:#eb6834;--groot:#1baf7a;--pi05:#eda100;--ref:#52514e;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--grid:#2c2c2a;
--axis:#45443f;--act:#3987e5;--smolvla:#d95926;--groot:#199e70;--pi05:#c98500;--ref:#c3c2b7;}}
.bg{fill:var(--surface)} .t{fill:var(--ink);font-size:15px;font-weight:600} .st{fill:var(--ink2);font-size:12px}
.lab{fill:var(--ink2);font-size:12px} .val{fill:var(--ink);font-size:11px} .na{fill:var(--muted);font-size:11px;font-style:italic}
.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}
.ci{stroke:var(--ink);stroke-width:1.5} .ref{stroke:var(--ref);stroke-width:1.5;stroke-dasharray:4 3}
.act{fill:var(--act)} .smolvla{fill:var(--smolvla)} .groot{fill:var(--groot)} .pi05{fill:var(--pi05)}
.ref-mark{fill:var(--ref)} .on-bar{fill:#fff}
.dot-act{fill:var(--act);stroke:var(--surface);stroke-width:2} .dot-smolvla{fill:var(--smolvla);stroke:var(--surface);stroke-width:2}
</style>"""


def esc(s):
    return html.escape(str(s), quote=True)


def rbar(x, y0, w, h, cls, title):
    """Vertical bar anchored at baseline y0 growing up h px, 4px rounded top only."""
    if h <= 0.5:
        return f'<rect x="{x:.1f}" y="{y0 - 1:.1f}" width="{w:.1f}" height="1" class="{cls}"><title>{esc(title)}</title></rect>'
    r = min(4.0, w / 2, h)
    top = y0 - h
    d = (f"M{x:.1f},{y0:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} H{x + w - r:.1f} "
         f"Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{y0:.1f} Z")
    return f'<path d="{d}" class="{cls}"><title>{esc(title)}</title></path>'


def hbar(x0, y, w, h, cls, title):
    """Horizontal bar anchored at x0, 4px rounded right end."""
    if w <= 0.5:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="1" height="{h:.1f}" class="{cls}"><title>{esc(title)}</title></rect>'
    r = min(4.0, h / 2, w)
    d = (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} V{y + h - r:.1f} "
         f"Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return f'<path d="{d}" class="{cls}"><title>{esc(title)}</title></path>'


def svg_doc(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="ttl dsc"><title id="ttl">{esc(title)}</title><desc id="dsc">{esc(desc)}</desc>'
            f'{SVG_STYLE}<rect class="bg" width="{w}" height="{h}"/>{body}</svg>\n')


def legend(models, x, y):
    out, cx = [], x
    for m in models:
        out.append(f'<rect x="{cx}" y="{y - 10}" width="12" height="12" rx="3" class="{m}"/>'
                   f'<text x="{cx + 18}" y="{y}" class="lab">{esc(MODEL_LABEL.get(m, m))}</text>')
        cx += 30 + 8 * len(MODEL_LABEL.get(m, m))
    return "".join(out)


def chart_success(cells, path):
    idx = by_key(cells)
    models = models_in(cells)
    W, H = 960, 436
    ml, mr, mt, mb = 56, 20, 108, 56
    gap = 36
    pw = (W - ml - mr - gap) / 2
    ph = H - mt - mb
    b = []
    b.append('<text x="20" y="30" class="t">Tỉ lệ thành công theo model, mức và split</text>')
    n = cells[0]["n"]
    b.append(f'<text x="20" y="50" class="st">{n} episode/ô, seed test {cells[0]["seed_first"]}–{cells[0]["seed_last"]}; '
             'thanh = điểm ước lượng, vạch = Wilson 95%</text>')
    b.append(legend(models, 20, 76))
    for li, lvl in enumerate(LEVELS):
        x0 = ml + li * (pw + gap)
        y0 = mt + ph
        b.append(f'<text x="{x0 + pw / 2:.0f}" y="{mt - 8}" text-anchor="middle" class="lab" font-weight="600">L{lvl}</text>')
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 - t * ph
            b.append(f'<line x1="{x0}" x2="{x0 + pw:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" class="{"axis" if t == 0 else "grid"}"/>')
            if li == 0:
                b.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="lab">{int(t * 100)}%</text>')
        gw = pw / len(SPLITS)
        bw = min(34.0, (gw - 24) / len(models))
        for si, s in enumerate(SPLITS):
            gx = x0 + si * gw + (gw - bw * len(models) - 2 * (len(models) - 1)) / 2
            b.append(f'<text x="{x0 + si * gw + gw / 2:.1f}" y="{y0 + 20}" text-anchor="middle" class="lab">{esc(SPLIT_LABEL[s])}</text>')
            for mi, m in enumerate(models):
                bx = gx + mi * (bw + 2)
                c = idx.get((m, lvl, s))
                if c is None:
                    b.append(f'<text x="{bx + bw / 2:.1f}" y="{y0 - 6}" text-anchor="middle" class="na">N/A</text>')
                    continue
                tip = (f"{MODEL_LABEL.get(m, m)} · L{lvl} {SPLIT_LABEL[s]}: {c['k']}/{c['n']} = {100 * c['rate']:.0f}% "
                       f"(Wilson 95% {100 * c['lo']:.0f}–{100 * c['hi']:.0f}%)")
                b.append(rbar(bx, y0, bw, c["rate"] * ph, m, tip))
                cx = bx + bw / 2
                ylo, yhi = y0 - c["lo"] * ph, y0 - c["hi"] * ph
                b.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{ylo:.1f}" y2="{yhi:.1f}" class="ci"/>'
                         f'<line x1="{cx - 4:.1f}" x2="{cx + 4:.1f}" y1="{yhi:.1f}" y2="{yhi:.1f}" class="ci"/>'
                         f'<line x1="{cx - 4:.1f}" x2="{cx + 4:.1f}" y1="{ylo:.1f}" y2="{ylo:.1f}" class="ci"/>')
                b.append(f'<text x="{cx:.1f}" y="{yhi - 5:.1f}" text-anchor="middle" class="val">{c["k"]}</text>')
    b.append(f'<text x="20" y="{H - 12}" class="st">Số trên vạch = số episode thành công. ACT không nhận câu lệnh nên không chạy L2.</text>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_doc(W, H, "".join(b), "Tỉ lệ thành công", "Grouped bars with Wilson 95% intervals"))


def chart_generalization(cells, path):
    """Dot plot: success rate of each split divided by the in-dist rate of the same model and level."""
    idx = by_key(cells)
    models = models_in(cells)
    rows = []
    for m in models:
        for lvl in LEVELS:
            base = idx.get((m, lvl, "train"))
            if base and base["rate"] > 0:
                rows.append((m, lvl, base))
    W = 760
    ml, mr, mt = 150, 30, 92
    rh = 34
    H = mt + rh * len(rows) * 1 + 70
    vmax = 1.5
    for m, lvl, base in rows:
        for s in SPLITS[1:]:
            c = idx.get((m, lvl, s))
            if c:
                vmax = max(vmax, c["rate"] / base["rate"] + 0.1)
    pw = W - ml - mr
    xs = lambda v: ml + v / vmax * pw  # noqa: E731
    b = ['<text x="20" y="30" class="t">Tỉ lệ thành công ngoài phân phối so với in-dist</text>',
         '<text x="20" y="50" class="st">tỉ số = rate(split) / rate(in-dist) cùng model và mức; 1.0 = không tụt; 0.5 = ngưỡng G7 cho wide</text>']
    shapes = {"heldout_color": "circle", "heldout_shape": "square", "wide": "diamond"}
    lx = 20
    for s in SPLITS[1:]:
        b.append(marker(shapes[s], lx + 6, 72, "ref-mark") + f'<text x="{lx + 16}" y="76" class="lab">{esc(SPLIT_LABEL[s])}</text>')
        lx += 130
    yb = mt + rh * len(rows)
    for t in [x / 4 for x in range(0, int(vmax * 4) + 1)]:
        b.append(f'<line x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{mt - 6}" y2="{yb}" class="grid"/>'
                 f'<text x="{xs(t):.1f}" y="{yb + 18}" text-anchor="middle" class="lab">{t:.2f}</text>')
    b.append(f'<line x1="{xs(1):.1f}" x2="{xs(1):.1f}" y1="{mt - 6}" y2="{yb}" class="axis" stroke-width="2"/>')
    b.append(f'<line x1="{xs(0.5):.1f}" x2="{xs(0.5):.1f}" y1="{mt - 6}" y2="{yb}" class="ref"/>')
    for i, (m, lvl, base) in enumerate(rows):
        y = mt + i * rh + rh / 2
        b.append(f'<text x="{ml - 12}" y="{y + 4:.1f}" text-anchor="end" class="lab">{esc(MODEL_LABEL.get(m, m))} L{lvl} '
                 f'({100 * base["rate"]:.0f}%)</text>')
        for s in SPLITS[1:]:
            c = idx.get((m, lvl, s))
            if not c:
                continue
            r = c["rate"] / base["rate"]
            tip = (f"{MODEL_LABEL.get(m, m)} L{lvl} {SPLIT_LABEL[s]}: {c['k']}/{c['n']} vs in-dist {base['k']}/{base['n']} "
                   f"→ ×{r:.2f}")
            b.append(f'<g>{marker(shapes[s], xs(r), y, "dot-" + m)}<title>{esc(tip)}</title></g>')
    b.append(f'<text x="20" y="{H - 14}" class="st">Mẫu nhỏ (k ≤ 15/50 ở in-dist): tỉ số dao động mạnh, chỉ đọc xu hướng; '
             'số tuyệt đối ở biểu đồ tỉ lệ thành công.</text>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_doc(W, H, "".join(b), "Tổng quát hoá", "Dot plot of split/in-dist success ratio"))


def marker(shape, x, y, cls):
    r = 6
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}"/>'
    if shape == "square":
        return f'<rect x="{x - r + 0.5:.1f}" y="{y - r + 0.5:.1f}" width="{2 * r - 1}" height="{2 * r - 1}" rx="2" class="{cls}"/>'
    return f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y:.1f} L{x:.1f},{y + r + 1:.1f} L{x - r - 1:.1f},{y:.1f} Z" class="{cls}"/>'


def chart_cost(cells, train, path):
    """Three small multiples (one measure each, own axis): train hours, train memory, inference latency."""
    done = [t for t in train if t["finished"] and t["model"] in MODEL_ORDER]
    done.sort(key=lambda t: MODEL_ORDER.index(t["model"]))
    lat = []
    for m in models_in(cells):
        cs = [c for c in cells if c["model"] == m]
        n = sum(c["n"] for c in cs)
        lat.append((m, sum(c["infer_ms_sim"] * c["n"] for c in cs) / n, max(c["infer_ms_max"] for c in cs), cs[0]["budget_ms"]))
    panels = [
        ("Giờ train (GPU, wall)", [(t["model"], t["wall_hours"], f"{t['wall_hours']:.2f} h · {t['steps']} step") for t in done], "h", None),
        ("Bộ nhớ train (log lerobot)", [(t["model"], t["peak_mem_gb_logged"], f"{t['peak_mem_gb_logged']:.1f} GB · batch {t['batch_size']}") for t in done], "GB", 16.0),
        ("Độ trễ infer TB phía sim", [(m, v, f"{v:.0f} ms") for m, v, mx, _ in lat], "ms", None),
    ]
    budget = lat[0][3] if lat else None
    W, ml, mr = 960, 20, 20
    pw = (W - ml - mr - 2 * 30) / 3
    rowh, mt = 30, 124
    nrows = max(len(p[1]) for p in panels)
    H = mt + nrows * rowh + 74
    b = ['<text x="20" y="30" class="t">Chi phí train và độ trễ suy luận</text>',
         '<text x="20" y="50" class="st">mỗi ô một đại lượng, trục riêng; RTX 4070 Ti SUPER 16 GB</text>']
    b.append(legend([t["model"] for t in done], 20, 76))
    for pi, (title, rows, unit, cap) in enumerate(panels):
        x0 = ml + pi * (pw + 30)
        lw = 0
        vmax = max([v for _, v, _ in rows] + ([cap] if cap else []) + ([budget] if unit == "ms" and budget else []))
        vmax *= 1.08
        bar_x0 = x0 + lw
        bw = pw - lw - 10
        b.append(f'<text x="{x0}" y="{mt - 14}" class="lab" font-weight="600">{esc(title)}</text>')
        for i, (m, v, lab) in enumerate(rows):
            y = mt + i * rowh
            w = v / vmax * bw
            b.append(hbar(bar_x0, y, w, 16, m, f"{MODEL_LABEL.get(m, m)}: {lab}"))
            inside = w > bw * 0.55
            tx = bar_x0 + (w - 6 if inside else w + 6)
            cls = "val on-bar" if inside else "val"
            b.append(f'<text x="{tx:.1f}" y="{y + 12.5}" text-anchor="{"end" if inside else "start"}" class="{cls}">'
                     f'{esc(MODEL_LABEL.get(m, m))} {esc(lab)}</text>')
        yb = mt + nrows * rowh
        b.append(f'<line x1="{bar_x0}" x2="{bar_x0}" y1="{mt - 6}" y2="{yb}" class="axis"/>')
        refv, reft = (cap, "16 GB GPU") if cap else ((budget, f"ngân sách {budget:.0f} ms") if unit == "ms" and budget else (None, ""))
        if refv:
            rx = bar_x0 + refv / vmax * bw
            b.append(f'<line x1="{rx:.1f}" x2="{rx:.1f}" y1="{mt - 6}" y2="{yb}" class="ref"/>'
                     f'<text x="{rx:.1f}" y="{yb + 16}" text-anchor="end" class="st">{esc(reft)}</text>')
    b.append(f'<text x="20" y="{H - 14}" class="st">ACT gồm 2 đoạn train (0–20k + resume). Độ trễ đo khi 2 worker eval dùng chung GPU; '
             'ngân sách = exec-steps / 25 Hz. GR00T chưa eval nên không có độ trễ.</text>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_doc(W, H, "".join(b), "Chi phí", "Small multiples: train hours, train memory, inference latency"))


def splice_report(path, tables):
    """Replace the text between the BEGIN/END:tables markers of REPORT.md (headings demoted one level)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    begin, end = "<!-- BEGIN:tables -->", "<!-- END:tables -->"
    if begin not in text or end not in text:
        print(f"warning: {path} has no {begin}/{end} markers, not updated")
        return
    body = re.sub(r"^## ", "### ", tables, flags=re.M)
    head, rest = text.split(begin, 1)
    tail = rest.split(end, 1)[1]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{head}{begin}\n{body.strip()}\n{end}{tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    res = args.results
    cells = gather_evals(res)
    if not cells:
        raise SystemExit(f"no eval results found in {res}")
    cells.sort(key=lambda c: (MODEL_ORDER.index(c["model"]) if c["model"] in MODEL_ORDER else 99, c["level"],
                              SPLITS.index(c["split"])))
    train = gather_train(res)
    execs = gather_exec(res)
    experts = gather_expert(res)
    ds_path = os.path.join(res, "dataset_stats.json")
    dstats = load_json(ds_path) if os.path.exists(ds_path) else None

    flat = []
    for c in cells:
        r = {k: v for k, v in c.items() if k not in ("stages", "arm_L", "arm_R")}
        r.update({f"n_{s}": c["stages"][s] for s in FAIL_ORDER})
        r.update({"arm_L_success": c["arm_L"][0], "arm_L_n": c["arm_L"][1],
                  "arm_R_success": c["arm_R"][0], "arm_R_n": c["arm_R"][1]})
        r["budget_ok"] = (c["infer_ms_sim"] <= c["budget_ms"]) if c["budget_ms"] else None
        flat.append(r)
    fields = ["model", "run", "level", "split", "n", "k", "rate", "lo", "hi", "exec_steps", "chunk", "episode_seconds",
              "smooth", "seed_first", "seed_last", "complete", "uses_task", "infer_ms_server", "infer_ms_sim",
              "infer_ms_max", "budget_ms", "budget_ok", "vram_infer_gib", "lifted"] + [f"n_{s}" for s in FAIL_ORDER] + \
             ["arm_L_success", "arm_L_n", "arm_R_success", "arm_R_n", "checkpoint", "source"]
    write_csv(os.path.join(res, "summary.csv"), flat, fields)
    write_csv(os.path.join(res, "summary_train.csv"), train,
              ["run", "model", "init", "steps", "batch_size", "finished", "exit_code", "wall_hours", "steps_per_s",
               "peak_mem_gb_logged", "loss_first", "loss_final", "note", "source"])
    tables = markdown(cells, train, execs, experts, dstats)
    with open(os.path.join(res, "summary_tables.md"), "w", encoding="utf-8") as f:
        f.write(tables)
    splice_report(os.path.join(res, "REPORT.md"), tables)
    fig = os.path.join(res, "figures")
    os.makedirs(fig, exist_ok=True)
    chart_success(cells, os.path.join(fig, "success_rates.svg"))
    chart_generalization(cells, os.path.join(fig, "generalization.svg"))
    chart_cost(cells, train, os.path.join(fig, "cost.svg"))
    print(f"{len(cells)} eval cells, {len(train)} train runs, {len(experts)} expert sets")
    print("wrote results/summary.csv, summary_train.csv, summary_tables.md, REPORT.md tables, figures/*.svg")


if __name__ == "__main__":
    main()
