"""
pipeline.py
-----------
End-to-end HistoMap-Extract pipeline.

Usage:
    python pipeline.py --input data/sample_map.tiff --output outputs/

Produces:
    outputs/boundaries.geojson   — urban/building polygons
    outputs/roads.geojson        — road network linestrings
    outputs/place_names.csv      — OCR-extracted text annotations
    outputs/preview.png          — visual overlay for inspection
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from models.segmentation import MapSegmenter, CLASSES
from models.ocr import MapOCR
from utils.georeference import georef_mask_to_geojson, polygonise, skeletonise_roads
from utils.visualise import save_preview


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HistoMap-Extract pipeline")
    p.add_argument("--input",   required=True,  help="Path to input map image")
    p.add_argument("--output",  default="outputs/", help="Output directory")
    p.add_argument("--weights", default=None,   help="Path to segmentation model weights")
    p.add_argument("--no-ocr",  action="store_true", help="Skip OCR step (faster)")
    p.add_argument("--device",  default=None,   help="cuda or cpu (auto-detected if omitted)")
    return p.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str, weights: str = None,
        run_ocr: bool = True, device: str = None):

    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load image ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading image: {input_path}")
    image = Image.open(input_path).convert("RGB")
    W, H  = image.size
    print(f"      Size: {W}×{H} px")

    # ── 2. Segmentation ────────────────────────────────────────────────────
    print("\n[2/5] Running semantic segmentation …")
    segmenter = MapSegmenter(model_path=weights, device=device)
    mask      = segmenter.predict_full_image(image)
    # mask: np.ndarray (H, W), values in {0,1,2,3}

    building_mask = (mask == 1).astype(np.uint8)
    road_mask     = (mask == 2).astype(np.uint8)
    print(f"      Building pixels: {building_mask.sum():,}")
    print(f"      Road pixels:     {road_mask.sum():,}")

    # ── 3. Vectorise to GeoJSON ────────────────────────────────────────────
    print("\n[3/5] Vectorising masks → GeoJSON …")

    boundaries_path = out / "boundaries.geojson"
    roads_path      = out / "roads.geojson"

    boundary_polys  = polygonise(building_mask, label="building")
    road_lines      = skeletonise_roads(road_mask)

    georef_mask_to_geojson(boundary_polys, str(boundaries_path))
    georef_mask_to_geojson(road_lines,     str(roads_path), geom_type="LineString")

    print(f"      → {boundaries_path}  ({len(boundary_polys)} polygons)")
    print(f"      → {roads_path}  ({len(road_lines)} linestrings)")

    # ── 4. OCR ────────────────────────────────────────────────────────────
    if run_ocr:
        print("\n[4/5] Extracting place name annotations (TrOCR) …")
        ocr         = MapOCR(device=device)
        annotations = ocr.extract(image)
        df          = ocr.to_dataframe(annotations)
        csv_path    = out / "place_names.csv"
        df.to_csv(csv_path, index=False)
        print(f"      → {csv_path}  ({len(df)} annotations)")
    else:
        print("\n[4/5] OCR skipped (--no-ocr flag set).")

    # ── 5. Preview ─────────────────────────────────────────────────────────
    print("\n[5/5] Saving visual preview …")
    preview_path = out / "preview.png"
    save_preview(image, mask, str(preview_path))
    print(f"      → {preview_path}")

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs in: {out.resolve()}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_dir=args.output,
        weights=args.weights,
        run_ocr=not args.no_ocr,
        device=args.device,
    )
