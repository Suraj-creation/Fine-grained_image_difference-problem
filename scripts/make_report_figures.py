"""Charts for reports/TinyCD_reproduction_report.pdf. Numbers are read from results/metrics/*.json
(full LEVIR test) or are the 20-pair notebook table (hard-coded below, copied from notebook cell 10)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cdcommon as C

OUT = C.ROOT / "reports" / "figures"
BLUE, ORANGE, GREY, GREEN, RED = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f9e44", "#e03131"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
res = json.load(open(C.ROOT / "results/metrics/tinycd-levir_on_levir-test.json"))

# 1. real example: A, B -> probability -> mask
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = C.load_model(C.CKPT["levir"], dev)
root = C.ROOT / "samples" / "levir"
n = "test_01777.png"
a, b, gt = C.load_triplet(root, n)
p = C.predict_prob(model, a, b, dev)
fig, ax = plt.subplots(1, 5, figsize=(13, 2.9))
for x, im, t, cm in zip(ax, [a, b, p, p > 0.5, gt], ["Image A (before)", "Image B (after)", "Model output:\nchange probability", "Mask (probability > 0.5)", "Human-drawn answer\n(ground truth)"],
                        [None, None, "magma", "gray", "gray"]):
    x.imshow(im, cmap=cm); x.set_title(t, fontsize=10); x.axis("off")
plt.tight_layout(); plt.savefig(OUT / "fig_example_pipeline.png", dpi=150); plt.close()

# 2. error map for the same pair
pred = p > 0.5
err = np.zeros((*gt.shape, 3), np.uint8) + 255
err[pred & gt] = (47, 158, 68); err[pred & ~gt] = (224, 49, 49); err[~pred & gt] = (43, 108, 176)
tp, fp, fn, tn = C.confusion(pred, gt)
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6), gridspec_kw={"width_ratios": [1, 1]})
ax[0].imshow(b); ax[0].set_title("Image B"); ax[0].axis("off")
ax[1].imshow(err); ax[1].axis("off"); ax[1].set_title("How the model's mask compares", fontsize=10)
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(color=GREEN, label=f"Correct change ({tp:,} px)"), Patch(color=RED, label=f"False alarm ({fp:,} px)"), Patch(color=BLUE, label=f"Missed change ({fn:,} px)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "fig_error_map.png", dpi=150); plt.close()

# 3. paper vs ours
paper = {"Precision": 92.68, "Recall": 89.47, "F1": 91.05, "IoU": 83.57}
ours = {"Precision": res["precision"] * 100, "Recall": res["recall"] * 100, "F1": res["f1"] * 100, "IoU": res["iou"] * 100}
ci = res["ci95_bootstrap_over_images"]; keys = {"Precision": "precision", "Recall": "recall", "F1": "f1", "IoU": "iou"}
fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(4); w = 0.36
ax.bar(x - w / 2, list(paper.values()), w, color=GREY, label="Paper (published)")
err_lo = [ours[k] - ci[keys[k]][0] * 100 for k in paper]; err_hi = [ci[keys[k]][1] * 100 - ours[k] for k in paper]
ax.bar(x + w / 2, list(ours.values()), w, color=BLUE, label="Our run (error bar = 95% range)", yerr=[err_lo, err_hi], capsize=4)
for i, k in enumerate(paper):
    ax.text(i - w / 2, paper[k] + 1.0, f"{paper[k]:.2f}", ha="center", fontsize=9)
    ax.text(i + w / 2 + 0.02, ours[k] + 1.0, f"{ours[k]:.2f}", ha="center", fontsize=9, color=BLUE, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(paper.keys()); ax.set_ylim(0, 112); ax.set_ylabel("Score (%)  - axis starts at 0")
ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9); ax.set_title("LEVIR-CD test set (2,048 image pairs)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "fig_paper_vs_ours.png", dpi=150); plt.close()

# 4. size analysis
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ob = res["object_recall_by_gt_component_size"]
labs = ["tiny\n(<50 px)", "small\n(50-200)", "medium\n(200-1000)", "large\n(>1000)"]
vals = [v["recall_at_50pct_overlap"] * 100 for v in ob.values()]
bars = ax[0].bar(labs, vals, color=[RED, ORANGE, BLUE, BLUE])
for bt, v, nn in zip(bars, vals, [v["n_objects"] for v in ob.values()]):
    ax[0].text(bt.get_x() + bt.get_width() / 2, v + 1.5, f"{v:.0f}%\n(n={nn:,})", ha="center", fontsize=8.5)
ax[0].set_ylim(0, 118); ax[0].set_ylabel("% of changed objects found"); ax[0].set_title("Does it find small changes?", fontsize=11)
bk = res["by_changed_area_bucket"]; names = ["0-1%", "1-5%", "5-20%", ">20%"]
ax[1].bar([f"{k}" for k in names], [bk[k]["f1"] * 100 for k in names], color=BLUE)
for i, k in enumerate(names): ax[1].text(i, bk[k]["f1"] * 100 + 1.5, f"{bk[k]['f1']*100:.0f}", ha="center", fontsize=9)
ax[1].set_ylim(0, 105); ax[1].set_ylabel("F1 score"); ax[1].set_xlabel("share of the picture that changed"); ax[1].set_title("Score vs. size of change", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "fig_by_size.png", dpi=150); plt.close()

# 5. baselines (20-pair notebook table)
fig, ax = plt.subplots(figsize=(7.4, 3.4))
m = ["TinyCD\n(pretrained)", "Simple colour\ndifference (fixed)", "Simple colour\ndifference (Otsu)"]
ax.bar(m, [89.4, 9.9, 10.7], color=[BLUE, GREY, GREY])
for i, v in enumerate([89.4, 9.9, 10.7]): ax.text(i, v + 2, f"{v:.1f}", ha="center", fontweight="bold")
ax.set_ylim(0, 105); ax.set_ylabel("F1 score"); ax.set_title("Same 20 LEVIR-CD test pairs", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "fig_baselines.png", dpi=150); plt.close()
print("ok")
