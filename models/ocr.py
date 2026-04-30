"""
ocr.py
------
Extract textual annotations (place names, labels, road names) from
historical map images using Microsoft's TrOCR transformer model.

TrOCR is well-suited to historical documents: it handles degraded,
handwritten, and printed text without requiring layout preprocessing.
"""

import re
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ── Config ───────────────────────────────────────────────────────────────────

# trocr-base-printed works well for typeset historical map labels.
# Switch to trocr-base-handwritten for manuscript annotations.
MODEL_ID = "microsoft/trocr-base-printed"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TextAnnotation:
    text: str
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in image pixels
    confidence: float


# ── Text region detection ─────────────────────────────────────────────────────

def detect_text_regions(
    image: Image.Image,
    min_area: int = 200,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect candidate text regions using connected component analysis
    on a binarised version of the map.

    This is intentionally lightweight — no deep learning, just OpenCV
    morphology — so it runs fast on CPU.

    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    import cv2

    gray  = np.array(image.convert("L"))
    # Adaptive threshold handles uneven lighting across scanned maps
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=8,
    )
    # Dilate horizontally to connect characters in the same word
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    h, w   = gray.shape
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bw / (bh + 1e-6)
        # Filter: must be wide (text-like aspect ratio) and large enough
        if area > min_area and 1.5 < aspect < 25:
            # Add small padding
            x1 = max(0, x - 4)
            y1 = max(0, y - 4)
            x2 = min(w, x + bw + 4)
            y2 = min(h, y + bh + 4)
            bboxes.append((x1, y1, x2, y2))

    return bboxes


# ── OCR model ─────────────────────────────────────────────────────────────────

class MapOCR:
    """
    Run TrOCR over detected text regions in a historical map image.
    Returns structured TextAnnotation objects with text + position.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MapOCR] Loading {model_id} on {self.device} …")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model     = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        print("[MapOCR] Ready.")

    def _ocr_crop(self, crop: Image.Image) -> Tuple[str, float]:
        """Run TrOCR on a single cropped region."""
        pixel_values = self.processor(
            images=crop.convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=32,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )[0].strip()

        # Rough confidence: mean of per-token max softmax scores
        if generated_ids.scores:
            scores = torch.stack(generated_ids.scores, dim=1)  # (1, T, V)
            probs  = scores.softmax(-1).max(-1).values          # (1, T)
            conf   = probs.mean().item()
        else:
            conf = 0.0

        return text, conf

    def extract(
        self,
        image: Image.Image,
        min_confidence: float = 0.4,
    ) -> List[TextAnnotation]:
        """
        Full pipeline: detect text regions → OCR each → return annotations.

        Args:
            image:          Full map PIL image.
            min_confidence: Drop predictions below this threshold.

        Returns:
            List of TextAnnotation (text, bbox, confidence).
        """
        bboxes      = detect_text_regions(image)
        annotations = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop            = image.crop(bbox)
            if crop.size[0] < 5 or crop.size[1] < 5:
                continue
            text, conf = self._ocr_crop(crop)
            # Basic sanity filter: must contain at least one letter
            if conf >= min_confidence and re.search(r"[A-Za-z]", text):
                annotations.append(TextAnnotation(text=text, bbox=bbox, confidence=conf))

        return annotations

    def to_dataframe(self, annotations: List[TextAnnotation]):
        """Convert annotations to a pandas DataFrame for CSV export."""
        import pandas as pd
        return pd.DataFrame([
            {
                "text":       a.text,
                "x1":         a.bbox[0],
                "y1":         a.bbox[1],
                "x2":         a.bbox[2],
                "y2":         a.bbox[3],
                "confidence": round(a.confidence, 4),
            }
            for a in annotations
        ])
