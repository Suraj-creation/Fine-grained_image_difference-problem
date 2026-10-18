"""Shared helpers: TinyCD loading/preprocessing, per-image metrics, bootstrap CIs, baselines.

Nothing in here changes the official TinyCD preprocessing: inputs are RGB scaled to [0,1] and
normalised with the ImageNet mean/std (identical to the official `MyDataset`), and the binary
mask is `prob > 0.5` (official `test_ondata.py`).
"""
import collections
import collections.abc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Tiny_model_4_CD"
CKPT = {
    "levir": REPO / "pretrained_models" / "levir_best.pth",
    "whu": REPO / "pretrained_models" / "whu_best.pth",
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- model
def _import_official():
    collections.Sized = collections.abc.Sized  # py>=3.10 shim, see run_official_test.py
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load_model(ckpt_path, device="cuda"):
    """Official ChangeClassifier + released weights.

    pretrained=False only skips downloading the ImageNet EfficientNet-B4 weights: the checkpoint
    contains every parameter (incl. the backbone slice) and overwrites them anyway (strict load).
    """
    _import_official()
    from models.change_classifier import ChangeClassifier

    model = ChangeClassifier(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model.eval().to(device)


def read_rgb(path):
    """uint8 HxWx3 RGB (drops alpha, expands gray)."""
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    """uint8 HxWx3 -> normalised 1x3xHxW float tensor (== official MyDataset for 8-bit PNG)."""
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def predict_prob(model, a_u8, b_u8, device):
    """Change probability map HxW float32 in [0,1] for one pair (H,W must be multiples of 8)."""
    return model(to_tensor(a_u8, device), to_tensor(b_u8, device))[0, 0].float().cpu().numpy()


def param_counts(model):
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_elements": sum(v.numel() for v in model.state_dict().values()),
    }


# --------------------------------------------------------------------------- metrics
def confusion(pred, gt):
    """pred, gt: bool HxW -> (TP, FP, FN, TN) as ints (change class = positive)."""
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn):
    eps = np.finfo(np.float32).eps  # same epsilon as the official metric tool
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return {
        "precision": p, "recall": r, "f1": f1,
        "iou": tp / (tp + fp + fn + eps),
        "oa": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def bootstrap_ci(counts, n_boot=1000, seed=0):
    """Percentile 95% CIs from resampling IMAGES with replacement. counts: (N,4) TP,FP,FN,TN."""
    counts = np.asarray(counts, dtype=np.float64)
    n = len(counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    s = counts[idx].sum(axis=1)  # (B,4)
    tp, fp, fn, tn = s.T
    eps = np.finfo(np.float32).eps
    p, r = tp / (tp + fp + eps), tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    iou = tp / (tp + fp + fn + eps)
    out = {}
    for k, v in {"precision": p, "recall": r, "f1": f1, "iou": iou}.items():
        out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def object_table(pred, gt, min_cover=0.5):
    """Per ground-truth connected component (8-conn): (area_px, covered_fraction)."""
    lab, n = ndimage.label(gt, structure=np.ones((3, 3)))
    rows = []
    for i in range(1, n + 1):
        m = lab == i
        rows.append((int(m.sum()), float(pred[m].mean())))
    return rows


AREA_BINS = [(0, 50), (50, 200), (200, 1000), (1000, 10**9)]


def object_recall_by_size(objs, min_cover=0.5):
    """objs: list of (area, covered). Returns {bin_label: (n_objects, recall)}."""
    out = {}
    for lo, hi in AREA_BINS:
        sel = [c for a, c in objs if lo <= a < hi]
        label = f"{lo}-{hi}px" if hi < 10**9 else f">={lo}px"
        out[label] = (len(sel), float(np.mean([c >= min_cover for c in sel])) if sel else float("nan"))
    return out


BUCKETS = ["no-change", "0-1%", "1-5%", "5-20%", ">20%"]  # share of pixels labelled "changed"


def bucket_of(frac):
    if frac == 0:
        return BUCKETS[0]
    for name, hi in zip(BUCKETS[1:4], (0.01, 0.05, 0.20)):
        if frac <= hi:
            return name
    return BUCKETS[4]


# --------------------------------------------------------------------------- baselines
def abs_rgb_diff(a_u8, b_u8):
    """Absolute RGB difference, averaged over channels, uint8 0..255 (no blur, no registration)."""
    d = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).mean(axis=2)
    return np.rint(d).astype(np.uint8)


def pred_fixed(a_u8, b_u8, thr):
    return abs_rgb_diff(a_u8, b_u8) > thr


def pred_otsu(a_u8, b_u8):
    """Per-image Otsu on the diff map (binary = diff > otsu_threshold)."""
    d = abs_rgb_diff(a_u8, b_u8)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def select_fixed_threshold(pairs):
    """Pick the diff threshold (0..254) maximising F1 on VALIDATION pairs [(a,b,gt_bool), ...]."""
    h_pos = np.zeros(256, np.int64)
    h_neg = np.zeros(256, np.int64)
    for a, b, gt in pairs:
        d = abs_rgb_diff(a, b)
        h_pos += np.bincount(d[gt].ravel(), minlength=256)
        h_neg += np.bincount(d[~gt].ravel(), minlength=256)
    best = (-1.0, 0)
    P, N = h_pos.sum(), h_neg.sum()
    for t in range(255):
        tp = h_pos[t + 1:].sum()
        fp = h_neg[t + 1:].sum()
        f1 = scores(tp, fp, P - tp, N - fp)["f1"]
        if f1 > best[0]:
            best = (f1, t)
    return best[1], best[0]


# --------------------------------------------------------------------------- data listing
def list_split(data_root, split):
    root = Path(data_root)
    names = [l.strip() for l in (root / "list" / f"{split}.txt").read_text().splitlines() if l.strip()]
    return root, names


def load_triplet(root, name):
    a = read_rgb(root / "A" / name)
    b = read_rgb(root / "B" / name)
    gt = np.array(Image.open(root / "label" / name)) > 0
    if gt.ndim == 3:
        gt = gt.any(axis=2)
    return a, b, gt


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, n):
    ts = []
    for _ in range(n):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    return ts
