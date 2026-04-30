"""
utils/georeference.py
---------------------
Convert segmentation masks → GeoJSON features.

Two key operations:
  polygonise()        — raster building mask → polygon features
  skeletonise_roads() — raster road mask → centreline linestrings
  georef_mask_to_geojson() — write features to a .geojson file

No external CRS transformation is applied by default (coordinates remain
in pixel space). For georeferenced TIFFs, rasterio will read the CRS
automatically and outputs will be in the map's native projection.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Union

# ── Optional rasterio import (falls back to pixel coords if unavailable) ──────
try:
    import rasterio
    from rasterio.features import shapes as rasterio_shapes
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from shapely.geometry import shape, mapping, MultiLineString, LineString
from shapely.ops import unary_union
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import cv2


# ── Polygonisation ────────────────────────────────────────────────────────────

def polygonise(
    mask: np.ndarray,
    label: str = "feature",
    min_area_px: int = 100,
) -> List[dict]:
    """
    Convert a binary mask to a list of GeoJSON-style feature dicts.

    Args:
        mask:        Binary (0/1) np.uint8 array.
        label:       Feature class label for the 'class' property.
        min_area_px: Discard polygons smaller than this (noise removal).

    Returns:
        List of dicts with keys: geometry (shapely), properties.
    """
    features = []

    if HAS_RASTERIO:
        # Rasterio gives clean, vectorised polygons directly from the mask
        mask_c = np.ascontiguousarray(mask.astype(np.uint8))
        for geom, val in rasterio_shapes(mask_c, mask=(mask_c == 1)):
            if val == 1:
                poly = shape(geom)
                if poly.area >= min_area_px:
                    features.append({
                        "geometry":   poly,
                        "properties": {"class": label, "area_px": poly.area},
                    })
    else:
        # Fallback: OpenCV contours → shapely polygons
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area_px:
                continue
            pts = cnt.squeeze()
            if pts.ndim < 2 or len(pts) < 3:
                continue
            from shapely.geometry import Polygon
            poly = Polygon(pts.tolist())
            if poly.is_valid and poly.area >= min_area_px:
                features.append({
                    "geometry":   poly,
                    "properties": {"class": label, "area_px": poly.area},
                })

    return features


# ── Road skeletonisation ──────────────────────────────────────────────────────

def skeletonise_roads(
    road_mask: np.ndarray,
    min_length_px: int = 30,
) -> List[dict]:
    """
    Thin a road pixel mask to centrelines and convert to LineString features.

    Uses skimage morphological skeletonisation — robust for the dense,
    interlocking road networks typical of historical OS maps.

    Args:
        road_mask:      Binary (0/1) np.uint8 array of road pixels.
        min_length_px:  Drop linestrings shorter than this.

    Returns:
        List of dicts with keys: geometry (shapely LineString), properties.
    """
    skeleton = skeletonize(road_mask > 0).astype(np.uint8)

    # Label connected components of the skeleton
    labelled = label(skeleton)
    features = []

    for region in regionprops(labelled):
        coords = region.coords  # (N, 2) array of (row, col)
        if len(coords) < 2:
            continue
        # Convert to (x, y) = (col, row) for GeoJSON convention
        pts = [(int(c[1]), int(c[0])) for c in coords]
        line = LineString(pts)
        if line.length >= min_length_px:
            features.append({
                "geometry":   line,
                "properties": {"class": "road", "length_px": line.length},
            })

    return features


# ── GeoJSON writer ────────────────────────────────────────────────────────────

def georef_mask_to_geojson(
    features: List[dict],
    output_path: str,
    geom_type: str = "Polygon",
) -> None:
    """
    Write a list of feature dicts to a GeoJSON file.

    Args:
        features:    Output of polygonise() or skeletonise_roads().
        output_path: Destination .geojson file path.
        geom_type:   "Polygon" or "LineString" — used only for validation.
    """
    geojson = {
        "type":     "FeatureCollection",
        "features": [],
    }

    for feat in features:
        geom = feat["geometry"]
        geojson["features"].append({
            "type":       "Feature",
            "geometry":   mapping(geom),
            "properties": feat.get("properties", {}),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"[GeoJSON] Written {len(features)} {geom_type} features → {output_path}")
