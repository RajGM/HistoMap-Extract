"""
segmentation.py
---------------
Semantic segmentation of historical map tiles using a U-Net backbone
fine-tuned on the ICDAR 2021 MapSeg dataset.

Classes predicted:
  0 - background
  1 - building block / urban footprint
  2 - road / transport network
  3 - map content area (vs. legend/margin)
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T


# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = {
    0: "background",
    1: "building",
    2: "road",
    3: "map_area",
}

MEAN = [0.485, 0.456, 0.406]   # ImageNet stats — good starting point for map images
STD  = [0.229, 0.224, 0.225]

TILE_SIZE = 512   # px — ICDAR tiles are 512×512


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = len(CLASSES), encoder: str = "resnet34") -> nn.Module:
    """
    Build a U-Net with a ResNet-34 encoder pretrained on ImageNet.
    ResNet-34 is lightweight enough to train on CPU over a weekend
    while still achieving competitive mIoU on map segmentation tasks.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model


# ── Preprocessing ────────────────────────────────────────────────────────────

def get_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((TILE_SIZE, TILE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


# ── Inference ────────────────────────────────────────────────────────────────

class MapSegmenter:
    """
    End-to-end segmentation on a single map image or tile.
    Handles tiling of large images automatically.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = build_model()
        self.model.to(self.device)
        self.transforms = get_transforms()

        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[MapSegmenter] Loaded weights from {model_path}")
        else:
            print("[MapSegmenter] No weights provided — using ImageNet init only.")
            print("               Run train.py first, or download pretrained weights.")

        self.model.eval()

    def predict_tile(self, image: Image.Image) -> np.ndarray:
        """
        Predict class mask for a single PIL image tile.
        Returns: np.ndarray of shape (H, W) with class indices.
        """
        tensor = self.transforms(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        with torch.no_grad():
            logits = self.model(tensor)                                # (1, C, H, W)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()          # (H, W)
        return mask

    def predict_full_image(self, image: Image.Image, overlap: int = 64) -> np.ndarray:
        """
        Slide a window over a large map image and stitch predictions.
        overlap: pixel overlap between adjacent tiles to reduce boundary artefacts.
        Returns: np.ndarray of shape (H, W) with class indices.
        """
        W, H   = image.size
        step   = TILE_SIZE - overlap
        output = np.zeros((H, W), dtype=np.uint8)
        count  = np.zeros((H, W), dtype=np.uint8)

        for y in range(0, H, step):
            for x in range(0, W, step):
                x1, y1 = x, y
                x2, y2 = min(x + TILE_SIZE, W), min(y + TILE_SIZE, H)
                tile   = image.crop((x1, y1, x2, y2)).resize((TILE_SIZE, TILE_SIZE))
                mask   = self.predict_tile(tile)
                # Resize back to original tile dimensions before pasting
                mask_resized = np.array(
                    Image.fromarray(mask).resize((x2 - x1, y2 - y1), Image.NEAREST)
                )
                output[y1:y2, x1:x2] = np.where(
                    count[y1:y2, x1:x2] == 0, mask_resized, output[y1:y2, x1:x2]
                )
                count[y1:y2, x1:x2] += 1

        return output


# ── Training loop (minimal, weekend-ready) ───────────────────────────────────

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: str = "models/segmentation_weights.pth",
):
    """
    Minimal training loop using combined Dice + Cross-Entropy loss.
    Dice loss handles class imbalance well — important for map data
    where background pixels dominate.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    ce_loss   = nn.CrossEntropyLoss()

    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).long()
            optimizer.zero_grad()
            logits = model(images)
            loss   = dice_loss(logits, masks) + ce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).long()
                logits    = model(images)
                val_loss  = dice_loss(logits, masks) + ce_loss(logits, masks)
                val_losses.append(val_loss.item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")

    return model
