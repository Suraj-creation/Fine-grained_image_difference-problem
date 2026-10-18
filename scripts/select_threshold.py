#!/usr/bin/env python
"""Sweep TinyCD's probability threshold on a VALIDATION split (never the test split).

Purpose: separate "ranking quality" from "calibration" under domain shift. The official protocol
(threshold 0.5) stays the headline number; a val-selected threshold is reported separately and is
labelled as a new experiment, not a reproduction.

  python scripts/select_threshold.py --dataset sysu --split val --ckpt levir
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import cdcommon as C

THRS = np.round(np.arange(0.02, 0.96, 0.02), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", default="levir")
    ap.add_argument("--data-root", default=str(C.ROOT / "data"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to tune a threshold on the test split")
    ckpt = C.CKPT.get(args.ckpt, Path(args.ckpt))
    root, names = C.list_split(Path(args.data_root) / args.dataset, args.split)
    model = C.load_model(ckpt, args.device)
    cm = np.zeros((len(THRS), 4), np.int64)
    for i in tqdm(range(0, len(names), args.batch), desc="val sweep"):
        trip = [C.load_triplet(root, n) for n in names[i:i + args.batch]]
        a = torch.cat([C.to_tensor(t[0], args.device) for t in trip])
        b = torch.cat([C.to_tensor(t[1], args.device) for t in trip])
        with torch.no_grad():
            prob = model(a, b)[:, 0].float().cpu().numpy()
        for p, (_, _, gt) in zip(prob, trip):
            for k, t in enumerate(THRS):
                cm[k] += C.confusion(p > t, gt)
    sc = [C.scores(*c) for c in cm]
    best = int(np.argmax([s["f1"] for s in sc]))
    res = {"dataset": args.dataset, "split": args.split, "n_pairs": len(names), "checkpoint": args.ckpt,
           "best_threshold": float(THRS[best]), "best_val_f1": sc[best]["f1"],
           "val_f1_at_0.5": sc[int(np.argmin(np.abs(THRS - 0.5)))]["f1"],
           "sweep": [{"threshold": float(t), **s} for t, s in zip(THRS, sc)]}
    out = C.ROOT / "results" / "metrics" / f"threshold_sweep_tinycd-{args.ckpt}_on_{args.dataset}-{args.split}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"best val threshold {res['best_threshold']} (F1 {res['best_val_f1']:.4f}); F1 at 0.5 = {res['val_f1_at_0.5']:.4f}")


if __name__ == "__main__":
    main()
