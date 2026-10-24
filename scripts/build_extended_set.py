#!/usr/bin/env python
"""Assemble the extended test set under extended_test/inputs/ and write pairs_manifest.csv.

Groups (see notebooks/tinycd_demo_extended.ipynb for how each is evaluated):
  A  aerial_levir      LEVIR-CD test pairs not used in the first notebook      official GT  -> valid F1/IoU
  B  aerial_sysu       SYSU-CD test pairs (already in samples/)                official GT  -> valid but cross-dataset
  C  photo_edit        MagicBrush dev set (real COCO photos + human-directed    proxy GT     -> indicative only
                       AI edits; mask = region the annotator asked to edit)
  D  photo_real        Wikimedia Commons before/after collages, split in half   no GT        -> qualitative only

Idempotent: re-running skips files that already exist. Data licences/attribution go to the CSV.
"""
import csv
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_test"
INP = OUT / "inputs"
UA = {"User-Agent": "tinycd-demo/0.1 (research notebook; GitHub Suraj-creation)"}
MANIFEST = []


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def add(pair_id, group, category, source, source_ref, licence, attribution, gt_type, size, notes=""):
    MANIFEST.append(dict(pair_id=pair_id, group=group, category=category, source_dataset=source, source_reference=source_ref,
                         licence=licence, attribution=attribution, gt_type=gt_type, width=size[0], height=size[1], notes=notes))


# ----------------------------------------------------------------------------- A: LEVIR
def build_levir(n_changed=10, n_nochange=2, seed=7):
    used = {l.strip() for l in (ROOT / "samples/levir/list.txt").read_text().splitlines()}
    p = hf_hub_download("ericyu/LEVIRCD_Cropped256", "data/test-00000-of-00001-31d7c3e3444e5b5d.parquet",
                        repo_type="dataset", cache_dir=str(ROOT / "data" / "_hf_cache"))
    t = pq.read_table(p, columns=["imageA", "imageB", "label"]).to_pylist()
    names = [f"test_{i:05d}.png" for i in range(len(t))]
    frac = np.array([(np.array(Image.open(io.BytesIO(r["label"]["bytes"]))) > 0).mean() for r in t])
    rng = np.random.default_rng(seed)
    cand = [i for i in range(len(t)) if names[i] not in used]
    pick = []
    for lo, hi, k in [(0.0005, 0.01, 3), (0.01, 0.05, 3), (0.05, 1.0, n_changed - 6)]:
        pool = [i for i in cand if lo < frac[i] <= hi and i not in pick]
        pick += list(rng.choice(pool, k, replace=False))
    pick += list(rng.choice([i for i in cand if frac[i] == 0], n_nochange, replace=False))
    for j, i in enumerate(pick, 1):
        pid = f"levir_{j:02d}"
        r = t[i]
        cat = "aerial-building-change" if frac[i] > 0 else "aerial-no-change"
        for k, key in (("A", "imageA"), ("B", "imageB")):
            save(Image.open(io.BytesIO(r[key]["bytes"])).convert("RGB"), INP / "aerial_levir" / cat / f"{pid}_{k}.png")
        save(Image.open(io.BytesIO(r["label"]["bytes"])).convert("L"), INP / "aerial_levir" / cat / f"{pid}_GT.png")
        add(pid, "aerial_levir", cat, "LEVIR-CD (256x256 test tile, HF mirror ericyu/LEVIRCD_Cropped256)",
            f"row {i} of the mirror's test split ({names[i]})", "academic use only (LEVIR-CD terms)", "Chen & Shi 2020, doi:10.3390/rs12101662",
            "official", (256, 256), f"changed-pixel share {frac[i]*100:.2f}%")


# ----------------------------------------------------------------------------- B: SYSU
def build_sysu(n=4):
    names = [l.strip() for l in (ROOT / "samples/sysu/list.txt").read_text().splitlines()][:n]
    for j, nm in enumerate(names, 1):
        pid = f"sysu_{j:02d}"
        for k, d in (("A", "A"), ("B", "B"), ("GT", "label")):
            img = Image.open(ROOT / "samples/sysu" / d / nm).convert("RGB" if k != "GT" else "L")
            save(img, INP / "aerial_sysu" / "aerial-cross-dataset" / f"{pid}_{k}.png")
        add(pid, "aerial_sysu", "aerial-cross-dataset", "SYSU-CD (256x256 test tile, HF mirror ericyu/SYSU_CD)", f"samples/sysu/{nm}",
            "not stated by authors", "Shi et al. 2021, doi:10.1109/TGRS.2021.3085870", "official (coarse region polygons)", (256, 256),
            "labels are rough regions, not building outlines")


# ----------------------------------------------------------------------------- C: MagicBrush
MB_SHARD = "datasets/osunlp/MagicBrush/data/dev-00000-of-00004-f147d414270a90e1.parquet"
MB_PICK = [(0, 20, "vehicles"), (0, 27, "vehicles"), (2, 6, "vehicles"),
           (0, 4, "people"), (2, 8, "people"), (0, 17, "people"),
           (0, 14, "everyday-objects"), (2, 2, "everyday-objects"), (2, 25, "everyday-objects"), (2, 12, "everyday-objects"),
           (0, 24, "outdoor-scenes"), (0, 0, "outdoor-scenes")]


def build_magicbrush():
    fs = HfFileSystem()
    groups = {}
    with fs.open(MB_SHARD, "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in sorted({g for g, _, _ in MB_PICK}):
            groups[rg] = pf.read_row_group(rg).to_pylist()
    for j, (rg, i, cat) in enumerate(MB_PICK, 1):
        r = groups[rg][i]
        a = Image.open(io.BytesIO(r["source_img"]["bytes"])).convert("RGB")
        b = Image.open(io.BytesIO(r["target_img"]["bytes"])).convert("RGB")
        m = Image.open(io.BytesIO(r["mask_img"]["bytes"])).convert("L")
        if b.size != a.size:
            b = b.resize(a.size, Image.BICUBIC)
        if m.size != a.size:
            m = m.resize(a.size, Image.NEAREST)
        # polarity verified numerically (>=78-99% of strongly changed pixels fall in the BLACK part): stored white = preserved,
        # so the edit region is the inverse of the stored mask
        m = Image.fromarray(((np.array(m) <= 127) * 255).astype(np.uint8))
        pid = f"photo_{cat.split('-')[0]}_{j:02d}"
        for k, im in (("A", a), ("B", b), ("GT", m)):
            save(im, INP / "photo_edit" / cat / f"{pid}_{k}.png")
        add(pid, "photo_edit", cat, "MagicBrush dev (osunlp/MagicBrush, source=real MS-COCO photo, target=AI-inpainted edit)",
            f"img_id {r['img_id']} turn {r['turn_index']}: \"{r['instruction']}\"", "CC BY 4.0", "Zhang et al. 2023 (MagicBrush); MS-COCO photo authors",
            "proxy (editor's mask)", a.size, "GT = region the annotator allowed to edit (inverse of stored mask); superset of true changes, not pixel-exact")


# ----------------------------------------------------------------------------- D: Wikimedia
API = "https://commons.wikimedia.org/w/api.php"
WM = [  # title, swap (True if collage is after|before), category, id, note
    ("File:Notre Dame-before-the-fire-and-after-the-fire-2019.jpg", False, "landmarks", "wm_landmark_notredame", "different camera position & weather; NOT registered"),
    ("File:Views before and after demolition of Fawley Power Station chimney, Hampshire, England.jpg", False, "landmarks", "wm_landmark_fawley", "same viewpoint; chimney removed; roughly registered"),
    ("File:Sydenham Heritage Church Before and After.jpg", False, "buildings", "wm_building_sydenham", "similar viewpoint; church demolished; lighting differs"),
    ("File:Big Mill Before and After Demolition.png", True, "buildings-aerial", "wm_building_bigmill", "drone view; collage order is after|before so halves are swapped (building present = A); different height/angle; NOT registered"),
]


def _get(url, **kw):
    for k in range(7):
        r = requests.get(url, headers=UA, timeout=90, **kw)
        if r.status_code == 200 and r.content:
            return r
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url} -> {r.status_code}")


def build_wikimedia(max_side=1024):
    for title, swap, cat, pid, note in WM:
        info = list(_get(API, params=dict(action="query", titles=title, prop="imageinfo", iiprop="url|extmetadata",
                                          iiurlwidth=2400, format="json")).json()["query"]["pages"].values())[0]["imageinfo"][0]
        time.sleep(1.5)
        md = info["extmetadata"]
        strip = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        im = Image.open(io.BytesIO(_get(info["thumburl"]).content)).convert("RGB")
        w, h = im.size
        half = w // 2
        pad = max(2, w // 400)  # trim the collage divider
        a, b = im.crop((0, 0, half - pad, h)), im.crop((half + pad, 0, w, h))
        b = b.crop((0, 0, a.width, a.height)) if b.width >= a.width else b.resize(a.size)
        if swap:
            a, b = b, a
        s = max_side / max(a.size)
        if s < 1:
            a, b = a.resize((round(a.width * s), round(a.height * s)), Image.LANCZOS), b.resize((round(b.width * s), round(b.height * s)), Image.LANCZOS)
        b = b.resize(a.size, Image.LANCZOS)
        for k, x in (("A", a), ("B", b)):
            save(x, INP / "photo_real" / cat / f"{pid}_{k}.png")
        add(pid, "photo_real", cat, "Wikimedia Commons (before/after collage split into left=before, right=after)",
            info["descriptionurl"], strip(md.get("LicenseShortName", {}).get("value", "see source")),
            strip(md.get("Artist", {}).get("value", "see source"))[:160], "none", a.size, note)
        time.sleep(1.5)


def main():
    build_levir(); build_sysu(); build_magicbrush(); build_wikimedia()
    OUT.mkdir(exist_ok=True)
    with open(OUT / "pairs_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST[0].keys()))
        w.writeheader(); w.writerows(MANIFEST)
    print(len(MANIFEST), "pairs ->", OUT / "pairs_manifest.csv")
    from collections import Counter
    print(Counter((m["group"], m["category"]) for m in MANIFEST))


if __name__ == "__main__":
    sys.exit(main())
