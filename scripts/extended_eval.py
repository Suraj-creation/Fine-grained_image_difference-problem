"""Run pretrained TinyCD (levir_best.pth) on every pair in extended_test/pairs_manifest.csv.

Evidence tiers (kept separate everywhere):
  A  official GT, in-domain (LEVIR-CD)               -> valid F1/IoU
  B  official GT, cross-dataset (SYSU-CD)            -> valid F1/IoU, but a domain-shift test with coarse labels
  C  proxy GT = region the annotator was allowed to edit (MagicBrush)
                                                      -> F1/IoU indicative only; outside-region pixels are trustworthy negatives
  D  no GT (Wikimedia before/after)                  -> qualitative only, no F1/IoU is computed

Preprocessing = official (RGB/255, ImageNet mean/std, threshold 0.5). Deviation for non-256 inputs: photos are resized so the
long side is 512 px (H and W multiples of 8) because the model was only ever trained on 256x256 aerial tiles.
"""
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import cdcommon as C

ROOT = C.ROOT
EXT = ROOT / "extended_test"
TIER = {"aerial_levir": "A: official GT, in-domain", "aerial_sysu": "B: official GT, cross-dataset",
        "photo_edit": "C: proxy GT (indicative)", "photo_real": "D: no GT (qualitative)"}
GROUP_TITLE = {"aerial_levir": "Aerial building change - LEVIR-CD test tiles (in-domain, official GT)",
               "aerial_sysu": "Aerial - SYSU-CD test tiles (cross-dataset, official coarse GT)",
               "photo_edit": "Ordinary photos - MagicBrush edits (proxy GT = editable region)",
               "photo_real": "Real before/after photos - Wikimedia Commons (no ground truth)"}
BASE_THR = None


def _round8(x):
    return max(8, int(round(x / 8)) * 8)


def read_manifest():
    return list(csv.DictReader(open(EXT / "pairs_manifest.csv")))


def load_pair(row, long_side=512):
    d = EXT / "inputs" / row["group"] / row["category"]
    a = np.array(Image.open(d / f"{row['pair_id']}_A.png").convert("RGB"))
    b = np.array(Image.open(d / f"{row['pair_id']}_B.png").convert("RGB"))
    gtp = d / f"{row['pair_id']}_GT.png"
    gt = (np.array(Image.open(gtp)) > 0) if gtp.exists() else None
    orig = a.shape[:2]
    if row["group"] in ("photo_edit", "photo_real"):  # not 256x256 aerial tiles -> resize (documented deviation)
        h, w = a.shape[:2]
        s = long_side / max(h, w)
        nh, nw = _round8(h * s), _round8(w * s)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if gt is not None:
            gt = cv2.resize(gt.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    return a, b, gt, orig


def run_all(device=None, thr=0.5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = __import__("json").loads((ROOT / "configs" / "baseline.json").read_text())["rgb_fixed_threshold"]
    model = C.load_model(C.CKPT["levir"], device)
    rows = read_manifest()
    # warm-up so timing is not dominated by CUDA init
    a, b, _, _ = load_pair(rows[0])
    for _ in range(3):
        C.predict_prob(model, a, b, device)
    results = []
    for r in rows:
        a, b, gt, orig = load_pair(r)
        t0 = time.perf_counter()
        prob = C.predict_prob(model, a, b, device)
        C.sync()
        ms = (time.perf_counter() - t0) * 1000
        pred = prob > thr
        otsu, fixed = C.pred_otsu(a, b), C.pred_fixed(a, b, base)
        out = EXT / "outputs" / r["group"] / r["category"]
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(prob * 255).astype(np.uint8)).save(out / f"{r['pair_id']}_prob.png")
        Image.fromarray((pred * 255).astype(np.uint8)).save(out / f"{r['pair_id']}_mask.png")
        Image.fromarray((otsu * 255).astype(np.uint8)).save(out / f"{r['pair_id']}_baseline_otsu.png")
        Image.fromarray((fixed * 255).astype(np.uint8)).save(out / f"{r['pair_id']}_baseline_fixed{base}.png")
        rec = dict(r, tier=TIER[r["group"]], infer_size=f"{a.shape[1]}x{a.shape[0]}", latency_ms=round(ms, 1),
                   pred_change_pct=round(pred.mean() * 100, 3), max_prob=round(float(prob.max()), 3))
        if gt is not None:
            for name, p in (("tinycd", pred), ("otsu", otsu), ("fixed", fixed)):
                tp, fp, fn, tn = C.confusion(p, gt)
                s = C.scores(tp, fp, fn, tn)
                rec.update({f"{name}_{k}": round(v, 4) for k, v in s.items() if k != "oa"})
                rec.update({f"{name}_TP": tp, f"{name}_FP": fp, f"{name}_FN": fn, f"{name}_TN": tn})
            rec["gt_pct"] = round(gt.mean() * 100, 3)
            if r["group"] == "photo_edit":  # outside the editable region nothing was changed -> trustworthy negatives
                rec["tinycd_false_alarm_outside_region_pct"] = round(100 * (pred & ~gt).sum() / max(1, (~gt).sum()), 3)
                rec["otsu_false_alarm_outside_region_pct"] = round(100 * (otsu & ~gt).sum() / max(1, (~gt).sum()), 3)
                rec["tinycd_share_of_prediction_inside_region_pct"] = round(100 * (pred & gt).sum() / max(1, pred.sum()), 1) if pred.any() else ""
        results.append((rec, a, b, gt, prob, pred, otsu, fixed))
    return results


def write_tables(results):
    recs = [x[0] for x in results]
    keys = []
    for r in recs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(EXT / "results_per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    cat_rows = []
    cats = []
    for r in recs:
        if (r["group"], r["category"]) not in cats:
            cats.append((r["group"], r["category"]))
    for g, c in cats:
        sel = [r for r in recs if r["group"] == g and r["category"] == c]
        row = dict(group=g, category=c, tier=TIER[g], valid_f1_iou={"aerial_levir": "yes", "aerial_sysu": "yes (cross-dataset)", "photo_edit": "no - proxy only", "photo_real": "no - no GT"}[g], n_pairs=len(sel),
                   mean_pred_change_pct=round(float(np.mean([r["pred_change_pct"] for r in sel])), 2),
                   mean_latency_ms=round(float(np.mean([r["latency_ms"] for r in sel])), 1))
        if g != "photo_real":
            for name in ("tinycd", "otsu", "fixed"):
                tp, fp, fn, tn = (sum(r[f"{name}_{k}"] for r in sel) for k in ("TP", "FP", "FN", "TN"))
                s = C.scores(tp, fp, fn, tn)
                if tp + fn == 0:  # no labelled change pixels at all -> precision/recall/F1/IoU are undefined, report false alarms instead
                    row.update({f"{name}_precision": np.nan, f"{name}_recall": np.nan, f"{name}_f1": np.nan, f"{name}_iou": np.nan,
                                f"{name}_false_alarm_pixels": int(fp)})
                else:
                    row.update({f"{name}_precision": round(s["precision"], 4), f"{name}_recall": round(s["recall"], 4),
                                f"{name}_f1": round(s["f1"], 4), f"{name}_iou": round(s["iou"], 4)})
            if g == "photo_edit":
                row["tinycd_false_alarm_outside_region_pct"] = round(float(np.mean([r["tinycd_false_alarm_outside_region_pct"] for r in sel])), 3)
                row["otsu_false_alarm_outside_region_pct"] = round(float(np.mean([r["otsu_false_alarm_outside_region_pct"] for r in sel])), 3)
        cat_rows.append(row)
    ck = []
    for r in cat_rows:
        for k in r:
            if k not in ck:
                ck.append(k)
    with open(EXT / "results_per_category.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ck)
        w.writeheader()
        w.writerows(cat_rows)
    return recs, cat_rows


def bootstrap_group_f1(recs, group, name="tinycd", n_boot=2000, seed=0):
    sel = [r for r in recs if r["group"] == group]
    cm = np.array([[r[f"{name}_{k}"] for k in ("TP", "FP", "FN", "TN")] for r in sel])
    return C.bootstrap_ci(cm, n_boot=n_boot, seed=seed)


def figure(results, group, ids=None, save_dir=None, dpi=95):
    """Same 6-column format as the first notebook: A | B | ground truth | TinyCD prob | TinyCD mask | RGB-diff Otsu."""
    import matplotlib.pyplot as plt
    sel = [x for x in results if x[0]["group"] == group and (ids is None or x[0]["pair_id"] in ids)]
    cols = ["Image A (before)", "Image B (after)", "Ground truth" if group != "photo_real" else "Ground truth\n(none available)",
            "TinyCD probability", "TinyCD mask (>0.5)", "RGB-diff Otsu"]
    fig, ax = plt.subplots(len(sel), 6, figsize=(13, 2.15 * len(sel) + 0.3), squeeze=False)
    for i, (rec, a, b, gt, prob, pred, otsu, fixed) in enumerate(sel):
        ims = [a, b, gt if gt is not None else np.full(a.shape[:2], 0.5), prob, pred, otsu]
        for j, im in enumerate(ims):
            ax[i, j].imshow(im, cmap=None if j < 2 else "gray", vmin=0, vmax=1 if j >= 2 else None)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_visible(False)
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=9)
        if gt is None:
            ax[i, 2].text(0.5, 0.5, "n/a", transform=ax[i, 2].transAxes, ha="center", va="center", color="white", fontsize=9)
        lab = rec["pair_id"] + (f"\nF1 {rec['tinycd_f1']:.2f}" if "tinycd_f1" in rec else "")
        ax[i, 0].set_ylabel(lab, fontsize=7, rotation=90, labelpad=4)
    fig.suptitle(GROUP_TITLE[group], y=1.0, fontsize=10)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    return fig


def save_pair_figures(results, dpi=90):
    """One compare PNG per pair next to its other outputs."""
    import matplotlib.pyplot as plt
    for x in results:
        rec = x[0]
        fig = figure([x], rec["group"])
        fig.savefig(EXT / "outputs" / rec["group"] / rec["category"] / f"{rec['pair_id']}_compare.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
