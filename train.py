"""
train.py
--------
Fine-tune the U-Net segmentation model on ICDAR 2021 MapSeg data.

Usage:
    python train.py --data data/icdar21/ --epochs 15 --device cpu

Saves best model weights to models/segmentation_weights.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from models.segmentation import build_model, train, TILE_SIZE, MEAN, STD


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/icdar21/", help="ICDAR data root")
    p.add_argument("--epochs",  type=int, default=15)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--batch",   type=int, default=4)
    p.add_argument("--device",  default=None)
    p.add_argument("--save",    default="models/segmentation_weights.pth")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class ICDARMapDataset(Dataset):
    """
    ICDAR 2021 MapSeg dataset loader (Task 2: map area segmentation).

    Expects the original ICDAR folder structure:
        icdar21-mapseg-v1.0.0-full-20210527a/
            2-segmaparea/
                train/       101-INPUT.jpg, 101-OUTPUT-GT.png ...
                validation/  201-INPUT.jpg, 201-OUTPUT-GT.png ...
                test/        301-INPUT.jpg, 301-OUTPUT-GT.png ...
    """

    def __init__(self, data_dir: str, split: str = "train", augment: bool = True):
        # split = "train" | "validation" | "test"
        root        = Path(data_dir) / "2-segmaparea" / split
        self.images = sorted(root.glob("*-INPUT.jpg"))
        self.masks  = sorted(root.glob("*-OUTPUT-GT.png"))
        assert len(self.images) == len(self.masks), \
            f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"

        self.augment = augment
        self.img_transform = T.Compose([
            T.Resize((TILE_SIZE, TILE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask  = Image.open(self.masks[idx]).convert("L")

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask  = mask.transpose(Image.FLIP_TOP_BOTTOM)

        image_tensor = self.img_transform(image)
        mask_tensor  = torch.from_numpy(
            np.array(mask.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST))
        ).long()

        return image_tensor, mask_tensor


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset = ICDARMapDataset(args.data, split="train",      augment=True)
    val_dataset   = ICDARMapDataset(args.data, split="validation", augment=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False, num_workers=2)

    model = build_model()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save,
    )

    print(f"\n✓ Training complete. Best weights saved to: {args.save}")
