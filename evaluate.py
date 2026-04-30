"""
evaluate.py
-----------
Evaluate pipeline outputs against ICDAR 2021 MapSeg ground truth.

Wraps icdar21-mapseg-eval and also computes:
  - mIoU per class
  - F1 score for building detection
  - Character Error Rate (CER) for OCR

Usage:
    python evaluate.py --pred outputs/ --gt data/ground_truth/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HistoMap-Extract evaluation")
    p.add_argument("--pred", required=True, help="Directory with pipeline outputs")
    p.add_argument("--gt",   required=True, help="Directory with ICDAR ground truth")
    p.add_argument("--out",  default="eval_report.json", help="Output JSON report path")
    return p.parse_args()


# ── Segmentation metrics ──────────────────────────────────────────────────────

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 4) -> dict:
    """
    Compute per-class IoU and mean IoU.

    Args:
        pred_mask:   Predicted class mask (H, W).
        gt_mask:     Ground truth class mask (H, W).
        num_classes: Number of semantic classes.

    Returns:
        Dict with per-class IoU and mIoU.
    """
    class_names = {0: "background", 1: "building", 2: "road", 3: "map_area"}
    ious        = {}

    for cls in range(num_classes):
        pred_c = (pred_mask == cls)
        gt_c   = (gt_mask   == cls)
        intersection = (pred_c & gt_c).sum()
        union        = (pred_c | gt_c).sum()
        iou          = float(intersection) / float(union + 1e-8)
        ious[class_names.get(cls, str(cls))] = round(iou, 4)

    ious["mIoU"] = round(np.mean(list(ious.values())), 4)
    return ious


def compute_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, cls: int = 1) -> dict:
    """
    Compute binary F1 for a single class (default: building = 1).
    """
    pred_c = (pred_mask == cls).flatten()
    gt_c   = (gt_mask   == cls).flatten()

    tp = int((pred_c & gt_c).sum())
    fp = int((pred_c & ~gt_c).sum())
    fn = int((~pred_c & gt_c).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── OCR metrics ───────────────────────────────────────────────────────────────

def character_error_rate(pred_texts: list, gt_texts: list) -> float:
    """
    Compute mean Character Error Rate (CER) using edit distance.
    CER = edit_distance(pred, gt) / len(gt)

    Lower is better. Publishable threshold for historical docs: CER < 0.15.
    """
    import Levenshtein  # pip install python-Levenshtein

    cers = []
    for pred, gt in zip(pred_texts, gt_texts):
        if len(gt) == 0:
            continue
        dist = Levenshtein.distance(pred, gt)
        cers.append(dist / len(gt))

    return round(float(np.mean(cers)) if cers else 1.0, 4)


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_mask(path: str) -> np.ndarray:
    """Load a PNG segmentation mask as a numpy array."""
    return np.array(Image.open(path).convert("L"))


def load_csv_texts(path: str) -> list:
    """Load OCR predictions from place_names.csv."""
    import pandas as pd
    if not Path(path).exists():
        return []
    df = pd.read_csv(path)
    return df["text"].tolist() if "text" in df.columns else []


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(pred_dir: str, gt_dir: str, out_path: str) -> dict:
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)
    report   = {}

    # ── Segmentation ──────────────────────────────────────────────────────
    # Real ICDAR structure: 2-segmaparea/validation/201-OUTPUT-GT.png
    gt_masks   = sorted(Path(gt_dir).glob("2-segmaparea/validation/*-OUTPUT-GT.png"))
    pred_masks = sorted(Path(pred_dir).glob("*.png"))

    if gt_masks and pred_masks:
        print("[Eval] Computing segmentation metrics …")
        pred_mask = load_mask(str(pred_masks[0]))
        gt_mask   = load_mask(str(gt_masks[0]))

        # Binarise GT (ICDAR GT is 0/255, convert to 0/1)
        gt_mask   = (gt_mask > 128).astype(np.uint8)
        pred_mask = (pred_mask > 128).astype(np.uint8)

        # Align sizes if necessary
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                )
            )

        report["segmentation"] = {
            "iou":         compute_iou(pred_mask, gt_mask, num_classes=2),
            "building_f1": compute_f1(pred_mask, gt_mask, cls=1),
        }
        print(f"  mIoU:        {report['segmentation']['iou']['mIoU']}")
        print(f"  Building F1: {report['segmentation']['building_f1']['f1']}")
    else:
        print("[Eval] Segmentation masks not found — skipping.")
        report["segmentation"] = "masks not found"

    # ── OCR ───────────────────────────────────────────────────────────────
    pred_csv = pred_dir / "place_names.csv"
    gt_csv   = gt_dir   / "place_names_gt.csv"

    if pred_csv.exists() and gt_csv.exists():
        print("[Eval] Computing OCR metrics …")
        pred_texts = load_csv_texts(str(pred_csv))
        gt_texts   = load_csv_texts(str(gt_csv))
        n          = min(len(pred_texts), len(gt_texts))

        try:
            cer = character_error_rate(pred_texts[:n], gt_texts[:n])
            report["ocr"] = {"cer": cer, "n_compared": n}
            print(f"  CER: {cer}  (n={n})")
        except ImportError:
            print("  [Eval] python-Levenshtein not installed — skipping CER.")
            report["ocr"] = "python-Levenshtein not installed"
    else:
        print("[Eval] OCR ground truth not found — skipping.")
        report["ocr"] = "ground truth CSV not found"

    # ── Save report ───────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Eval] Report saved → {out_path}")

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.pred, args.gt, args.out)
