"""
utils/visualise.py
------------------
Generate visual overlays of segmentation masks on top of the original
map image — useful for rapid inspection and the project README/demo.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Class colour palette (RGBA)
PALETTE = {
    0: (0,   0,   0,   0),    # background — transparent
    1: (231, 76,  60,  160),  # building   — red
    2: (52,  152, 219, 160),  # road       — blue
    3: (46,  204, 113, 80),   # map area   — green (light)
}

CLASS_NAMES = {
    0: "Background",
    1: "Building",
    2: "Road",
    3: "Map area",
}


def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask to RGBA colour image."""
    H, W   = mask.shape
    rgba   = np.zeros((H, W, 4), dtype=np.uint8)
    for cls, colour in PALETTE.items():
        rgba[mask == cls] = colour
    return rgba


def save_preview(
    original: Image.Image,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
) -> None:
    """
    Blend segmentation mask over the original map image and save.

    Args:
        original:    Original map PIL image (RGB).
        mask:        Segmentation output (H, W) with class indices.
        output_path: Where to save the preview PNG.
        alpha:       Overlay opacity (0 = invisible, 1 = opaque).
    """
    # Resize mask to match original if tiling changed resolution
    if (mask.shape[1], mask.shape[0]) != original.size:
        mask_img = Image.fromarray(mask.astype(np.uint8)).resize(
            original.size, Image.NEAREST
        )
        mask = np.array(mask_img)

    overlay_rgba = Image.fromarray(mask_to_rgba(mask), mode="RGBA")
    base         = original.convert("RGBA")
    blended      = Image.blend(base, overlay_rgba, alpha=alpha)

    # Add legend
    blended      = _add_legend(blended)
    blended.convert("RGB").save(output_path)
    print(f"[Preview] Saved → {output_path}")


def _add_legend(image: Image.Image) -> Image.Image:
    """Paste a small colour legend in the bottom-left corner."""
    draw        = ImageDraw.Draw(image)
    x0, y0     = 10, image.height - (len(PALETTE) * 22 + 15)
    box_size    = 15
    line_height = 22

    # Semi-transparent background for readability
    bg_w = 140
    bg_h = len(PALETTE) * line_height + 10
    draw.rectangle([x0 - 5, y0 - 5, x0 + bg_w, y0 + bg_h], fill=(255, 255, 255, 180))

    for i, (cls, colour) in enumerate(PALETTE.items()):
        y = y0 + i * line_height
        draw.rectangle([x0, y, x0 + box_size, y + box_size], fill=colour[:3])
        draw.text((x0 + box_size + 6, y), CLASS_NAMES[cls], fill=(30, 30, 30))

    return image


def side_by_side(
    original: Image.Image,
    mask: np.ndarray,
    output_path: str,
) -> None:
    """
    Save original and colourised mask side by side — good for README figures.
    """
    W, H   = original.size
    canvas = Image.new("RGB", (W * 2 + 10, H), color=(240, 240, 240))
    canvas.paste(original, (0, 0))

    mask_rgb = Image.fromarray(mask_to_rgba(mask)[:, :, :3])
    if mask_rgb.size != original.size:
        mask_rgb = mask_rgb.resize(original.size, Image.NEAREST)
    canvas.paste(mask_rgb, (W + 10, 0))
    canvas.save(output_path)
    print(f"[Preview] Side-by-side saved → {output_path}")
