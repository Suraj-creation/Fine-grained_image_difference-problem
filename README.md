# Image difference with TinyCD (pretrained, sample-pair demo)

**Model:** [TinyCD](https://github.com/AndreaCodegoni/Tiny_model_4_CD) – 285,128-parameter Siamese U-Net, 1.28 MB checkpoint, outputs a per-pixel change probability map (binary mask = prob > 0.5).
**Deliverable:** `notebooks/tinycd_demo.ipynb` (already executed, outputs included). It clones the official repo if missing, verifies the checkpoint hash, loads the pretrained weights and tests 20 LEVIR-CD + 10 SYSU-CD pairs from `samples/` (8 MB) against two RGB-difference baselines. No training, no dataset download needed.

## Run
```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install numpy scipy scikit-learn opencv-python-headless matplotlib pillow pandas jupyter ipykernel
jupyter lab notebooks/tinycd_demo.ipynb          # exact versions used: requirements.lock.txt
```
Python 3.10 note: the official repo does `from collections import Sized` (removed in 3.10); `scripts/cdcommon.py` restores the alias, no official file is edited.

## What was verified
| Item | Result |
|---|---|
| Official `test_ondata.py`, `levir_best.pth`, LEVIR-CD-256 test (2048 pairs) | P 92.65 / R 89.54 / **F1 91.07** / IoU 83.60 (paper: 91.05 / 83.57) → reproduced. Log: `results/logs/official_test_levir_on_levir.log` |
| Same, my `scripts/evaluate.py` | identical to 16 digits; F1 95% CI (bootstrap over images) 90.63–91.47 |
| 20-pair sample, TinyCD vs RGB-diff Otsu (F1) | 0.894 vs 0.107; no-change pairs with false alarms 0/6 vs 6/6 |
| Speed / memory (GTX 1080 Ti, 256×256, batch 1) | ~18–22 ms/pair, ~55 MiB peak |
| Cross-dataset: LEVIR checkpoint → SYSU-CD test (4000 pairs) | F1 0.003 (recall 0.16%); on the 10-pair sample Otsu RGB-diff scores *higher* |

**Why SYSU fails:** model outputs are weak there (mean max-prob ≈ 0.14) *and* SYSU labels are coarse region polygons (~24 % of pixels), not building outlines, so the annotation style does not match what TinyCD learned. Treat it as a domain-shift warning, not a fair accuracy figure. A val-tuned threshold was not run.

## Extended test (32 pairs, mixed domains)
`notebooks/tinycd_demo_extended.ipynb` (executed; the first notebook is unchanged) runs the same pretrained model on 32 pairs and saves every input/output under `extended_test/`; see `extended_test/README.md` for the folder map and `pairs_manifest.csv` for sources and licences.

| Group | Pairs | Ground truth | Result (official 0.5 threshold, no tuning) |
|---|---|---|---|
| A - LEVIR-CD tiles (home domain) | 12 | official | pooled F1 **0.875** (95% CI 0.69-0.93, only 12 tiles; full 2,048-pair test: 0.911); 2/2 no-change tiles clean; misses tiny changes (0.1-0.3% of tile), 1 tile with many false alarms |
| B - SYSU-CD tiles | 4 | official, coarse | F1 **0.00**: outputs nothing (cross-dataset collapse); labels are rough regions |
| C - MagicBrush photos (vehicles/people/everyday/outdoor) | 12 | proxy (editable region) | **not a valid F1**; 8 of 12 edits flagged <1% of the image; partial hits only on the pizza and the added mountains, large wrong areas on the cruise-ship pair |
| D - Wikimedia real before/after (landmarks, buildings) | 4 | none | qualitative only; Fawley chimney removal and Notre-Dame **not detected** |

Only A and B are valid F1/IoU evaluations. Environment rebuilt from `requirements.lock.txt` (Python 3.10, torch 2.7.1+cu118).

## Not done (and why)
- **WHU-CD reproduction:** the official page blocks scripted access (WAF), the authors' pre-processed copy is on a Google Drive folder returning HTTP 401, and their split was *random* (5947/743/744), so a self-made split would leak training patches. `whu_best.pth` is in the repo but untested.
- Large-scale deliverables from the original brief (full reports, robustness tests) were dropped after the request to keep this to a sample-pair notebook. Extra scripts in `scripts/` (`evaluate.py`, `select_threshold.py`, `download_data.py`) and full-test results in `results/` are from the earlier run and are optional.
- Official checkpoint provenance: the repo does not say how "best" was selected (val vs test).

## Licence – read before client use
- TinyCD code (and, by implication, the weights in the same repo): *"non-commercial and research purposes only; contact the authors for commercial use"*. No LICENSE file exists.
- LEVIR-CD: academic use only. SYSU-CD: no licence stated. Both datasets here come from Hugging Face mirrors (`ericyu/…`), not the authors' hosting.
- The backbone was initialised from ImageNet weights during training (ImageNet terms are research-oriented).
→ **Not cleared for a commercial client prototype** without written permission from the authors.

## Scope
Designed and trained for **aerial building change between registered images**. Not validated for product photos, text edits, UI screenshots, lighting/shadow robustness or misaligned pairs (the paper itself flags registration as an assumption). If the client domain is different, the smallest next step is a small labelled set from that domain and a fine-tune/eval of this same model, or a permissively licensed model.
