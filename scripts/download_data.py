#!/usr/bin/env python
"""Acquire evaluation data and convert it to the TinyCD folder layout.

Target layout (what the official `dataset/dataset.py` expects):
    data/<name>/{A,B,label}/<file>.png and data/<name>/list/<split>.txt

Sources actually used (see reports/reproduction_report.md for provenance and caveats):
  levir : HF mirror `ericyu/LEVIRCD_Cropped256` (256x256 crops; 7120/1024/2048 train/val/test)
          Only val+test are downloaded - we never train.
  sysu  : HF mirror `ericyu/SYSU_CD` (256x256; 12000/4000/4000). test (3 shards) + val shard 0.
  whu   : NO automatic source. The official page (study.rsgis.whu.edu.cn) rejects scripted
          requests (WAF) and the TinyCD authors' preprocessed copy is on a Google Drive folder that
          returns HTTP 401. Place WHU-CD-256 manually at data/whu/{A,B,label,list}.

These are third-party mirrors, not the original authors' hosting. LEVIR-CD and SYSU-CD are
academic-use-only datasets (see MODEL_CARD.md).
"""
import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "levir": {
        "repo": "ericyu/LEVIRCD_Cropped256",
        "files": {
            "test": ["data/test-00000-of-00001-31d7c3e3444e5b5d.parquet"],
            "val": ["data/val-00000-of-00001-d09d88a7419f2427.parquet"],
        },
        "cols": ("imageA", "imageB", "label"),
        "expected": {"test": 2048, "val": 1024},
    },
    "sysu": {
        "repo": "ericyu/SYSU_CD",
        "files": {
            "test": [
                "data/test-00000-of-00003-64a36a60d556358d.parquet",
                "data/test-00001-of-00003-8fa8df86ea2b8fcc.parquet",
                "data/test-00002-of-00003-54d45d93442e482d.parquet",
            ],
            # one val shard only: used for baseline threshold selection, not for TinyCD
            "val": ["data/val-00000-of-00003-d66043eece2e3354.parquet"],
        },
        "cols": ("imageA", "imageB", "label"),
        "expected": {"test": 4000},  # val is a partial shard
    },
}


def _png_bytes(cell, mode=None):
    """Return PNG bytes for an HF image cell, re-encoding only if it is not already PNG."""
    raw = cell["bytes"]
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and mode is None:
        return raw
    im = Image.open(io.BytesIO(raw))
    if mode:
        im = im.convert(mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def convert(name, split, parquet_paths, cols, out):
    for d in ("A", "B", "label", "list"):
        (out / d).mkdir(parents=True, exist_ok=True)
    names, idx = [], 0
    for pp in parquet_paths:
        pf = pq.ParquetFile(pp)
        for batch in pf.iter_batches(batch_size=64, columns=list(cols)):
            for row in batch.to_pylist():
                fn = f"{split}_{idx:05d}.png"
                (out / "A" / fn).write_bytes(_png_bytes(row[cols[0]]))
                (out / "B" / fn).write_bytes(_png_bytes(row[cols[1]]))
                (out / "label" / fn).write_bytes(_png_bytes(row[cols[2]], mode="L"))
                names.append(fn)
                idx += 1
    (out / "list" / f"{split}.txt").write_text("\n".join(names) + "\n")
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["levir", "sysu"], choices=["levir", "sysu", "whu"])
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "_hf_cache"))
    args = ap.parse_args()
    root = Path(args.data_root)
    summary = {}
    for name in args.datasets:
        if name == "whu":
            print("[whu] no scripted source available - see docstring. Expected at", root / "whu")
            continue
        src = SOURCES[name]
        out = root / name
        for split, files in src["files"].items():
            paths = [hf_hub_download(src["repo"], f, repo_type="dataset", cache_dir=args.cache) for f in files]
            n = convert(name, split, paths, src["cols"], out)
            exp = src["expected"].get(split)
            flag = "" if exp is None or exp == n else f"  !! expected {exp}"
            print(f"[{name}/{split}] {n} pairs{flag}")
            summary[f"{name}/{split}"] = n
            if exp is not None and exp != n:
                sys.exit(f"split size mismatch for {name}/{split}")
    (root / "download_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
