#!/usr/bin/env python
"""Evaluate a released TinyCD checkpoint on a dataset laid out as A/ B/ label/ list/.

Protocol = official `test_ondata.py`: official normalisation, sigmoid output, threshold 0.5,
change class = positive, metrics pooled over all pixels of the split.
Additions (not in the official script): per-image counts, bootstrap CIs, size-bucketed results,
object-level recall, latency and peak GPU memory.

example:
  python scripts/evaluate.py --dataset levir --split test --ckpt levir
  python scripts/evaluate.py --dataset sysu  --split test --ckpt levir   # cross-dataset (generalisation)
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import cdcommon as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="levir", help="'levir', 'whu' or a path")
    ap.add_argument("--data-root", default=str(C.ROOT / "data"))
    ap.add_argument("--out", default=str(C.ROOT / "results"))
    ap.add_argument("--pred-dir", default=str(C.ROOT / "outputs" / "pred_prob"))
    ap.add_argument("--threshold", type=float, default=0.5, help="official value; do not tune on test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-save-pred", action="store_true")
    args = ap.parse_args()

    ckpt = C.CKPT.get(args.ckpt, Path(args.ckpt))
    ckpt_tag = args.ckpt if args.ckpt in C.CKPT else Path(args.ckpt).stem
    tag = f"tinycd-{ckpt_tag}_on_{args.dataset}-{args.split}"
    if args.threshold != 0.5:
        tag += f"_thr{args.threshold:g}"
    out = Path(args.out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    root, names = C.list_split(Path(args.data_root) / args.dataset, args.split)
    pred_dir = Path(args.pred_dir) / args.dataset
    if not args.no_save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    dev = args.device
    model = C.load_model(ckpt, dev)
    pc = C.param_counts(model)
    print(f"[{tag}] {len(names)} pairs | params={pc['parameters']:,} | ckpt={ckpt} "
          f"({Path(ckpt).stat().st_size/1e6:.2f} MB)")

    rows, objs = [], []
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in tqdm(range(0, len(names), args.batch), desc=tag):
        chunk = names[i:i + args.batch]
        trip = [C.load_triplet(root, n) for n in chunk]
        a = torch.cat([C.to_tensor(t[0], dev) for t in trip])
        b = torch.cat([C.to_tensor(t[1], dev) for t in trip])
        with torch.no_grad():
            prob = model(a, b)[:, 0].float().cpu().numpy()
        for n, (_, _, gt), p in zip(chunk, trip, prob):
            pred = p > args.threshold
            tp, fp, fn, tn = C.confusion(pred, gt)
            frac = float(gt.mean())
            rows.append([n, tp, fp, fn, tn, frac, C.bucket_of(frac), int(pred.any())])
            objs += [(n, ar, cov) for ar, cov in C.object_table(pred, gt)]
            if not args.no_save_pred:
                Image.fromarray(np.rint(p * 255).astype(np.uint8)).save(pred_dir / n)
    wall = time.time() - t0
    peak_batch_mb = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None

    counts = np.array([r[1:5] for r in rows], dtype=np.int64)
    tp, fp, fn, tn = counts.sum(0)
    res = {"tag": tag, "dataset": args.dataset, "split": args.split, "n_pairs": len(rows),
           "checkpoint": str(ckpt.relative_to(C.ROOT)) if Path(ckpt).is_relative_to(C.ROOT) else str(ckpt),
           "threshold": args.threshold, **C.scores(tp, fp, fn, tn),
           "confusion_pixels": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
           "ci95_bootstrap_over_images": C.bootstrap_ci(counts)}

    # no-change pairs
    nc = [r for r in rows if r[5] == 0]
    res["no_change_pairs"] = {
        "n": len(nc),
        "pairs_with_any_false_alarm": int(sum(r[7] for r in nc)),
        "pixel_fpr": float(sum(r[2] for r in nc) / max(1, sum(r[2] + r[4] for r in nc))) if nc else None,
    }
    # by changed-area bucket
    res["by_changed_area_bucket"] = {}
    for bk in C.BUCKETS:
        sel = np.array([r[1:5] for r in rows if r[6] == bk], dtype=np.int64)
        if len(sel):
            s = sel.sum(0)
            res["by_changed_area_bucket"][bk] = {"n_pairs": int(len(sel)), **C.scores(*s)}
    # object recall by size
    res["object_recall_by_gt_component_size"] = {
        k: {"n_objects": n, "recall_at_50pct_overlap": r}
        for k, (n, r) in C.object_recall_by_size([(a, c) for _, a, c in objs]).items()}

    # latency / memory (batch 1)
    sample = [C.load_triplet(root, n) for n in names[:args.bench_n]]
    res["inference"] = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
                        "eval_wall_clock_s": wall, "eval_batch": args.batch,
                        "eval_peak_gpu_mem_MiB_batch": peak_batch_mb, **pc,
                        "checkpoint_bytes": Path(ckpt).stat().st_size}
    ta = [C.to_tensor(s[0], dev) for s in sample]
    tb = [C.to_tensor(s[1], dev) for s in sample]
    with torch.no_grad():
        for k in range(10):  # warm-up
            model(ta[k % len(ta)], tb[k % len(tb)])
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        it = iter(range(len(ta)))
        lat = C.timeit(lambda: (lambda j: model(ta[j], tb[j]))(next(it)), len(sample))
        if dev == "cuda":
            res["inference"]["peak_gpu_mem_MiB_batch1_256x256"] = torch.cuda.max_memory_allocated() / 2**20
    res["inference"]["latency_ms_batch1_model_only"] = {
        "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "n": len(lat)}
    e2e = C.timeit(lambda: C.predict_prob(model, sample[0][0], sample[0][1], dev), min(50, len(sample)))
    res["inference"]["latency_ms_batch1_with_preprocess"] = {"median": float(np.median(e2e))}
    res["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                          "cuda_runtime": torch.version.cuda, "platform": platform.platform()}

    (out / "metrics" / f"{tag}.json").write_text(json.dumps(res, indent=2))
    with open(out / "metrics" / f"{tag}.per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "TP", "FP", "FN", "TN", "gt_change_frac", "bucket", "pred_any"])
        w.writerows(rows)
    with open(out / "metrics" / f"{tag}.objects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "gt_component_area_px", "covered_fraction"])
        w.writerows(objs)
    print(json.dumps({k: res[k] for k in ("precision", "recall", "f1", "iou", "oa")}, indent=1))
    print("CI95 F1:", res["ci95_bootstrap_over_images"]["f1"], "| latency ms:",
          res["inference"]["latency_ms_batch1_model_only"])


if __name__ == "__main__":
    main()
