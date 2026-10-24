# extended_test – 32 before/after pairs run through pretrained TinyCD

Produced by `notebooks/tinycd_demo_extended.ipynb` (inputs built by `scripts/build_extended_set.py`, evaluation in `scripts/extended_eval.py`).

```
extended_test/
  pairs_manifest.csv        one row per pair: id, group, category, source, source reference, licence, attribution, GT type, size, notes
  inputs/<group>/<category>/<id>_A.png, _B.png, _GT.png (only where a ground truth exists)
  outputs/<group>/<category>/<id>_prob.png, _mask.png, _baseline_otsu.png, _baseline_fixed103.png, _compare.png
  results_per_pair.csv      metrics per pair
  results_per_category.csv  pooled metrics per category, with tier + "valid F1/IoU?" column
  failure_log_auto.csv      automatically flagged misses / false alarms / silent pairs
```

| Tier | Group (folder) | Pairs | Ground truth | F1/IoU valid? |
|---|---|---|---|---|
| A | `aerial_levir` – LEVIR-CD test tiles | 12 | official | yes (small sample) |
| B | `aerial_sysu` – SYSU-CD test tiles | 4 | official, coarse regions | yes, cross-dataset |
| C | `photo_edit` – MagicBrush: vehicles 3, people 3, everyday objects 4, outdoor scenes 2 | 12 | editable region only (superset of the change) | **no – indicative only** |
| D | `photo_real` – Wikimedia: landmarks 2, buildings 1, aerial drone 1 | 4 | none | **no – qualitative** |

Notes
* Tier C mask polarity: MagicBrush stores white = preserved; the edit region is its inverse (checked: 78–99 % of strongly changed pixels fall inside it).
* Photos are resized to a 512 px long side for TinyCD (documented deviation; the model only ever saw 256×256 aerial tiles). Aerial tiles run at native size.
* Wikimedia pairs are halves of before/after collages (for `wm_building_bigmill` the collage order is after|before, so the halves are swapped).
* **Licences:** LEVIR-CD academic use only; SYSU-CD no licence stated; MagicBrush CC BY 4.0; Wikimedia items CC BY 2.0 / CC BY-SA 4.0 – attribution is in the manifest, and the CC BY-SA crops/outputs remain under CC BY-SA 4.0. TinyCD code/weights are non-commercial/research only.
