"""
data/download_icdar.py
----------------------
One-command downloader for the ICDAR 2021 MapSeg dataset.

Usage:
    python data/download_icdar.py

Downloads training + validation sets for all three tasks into data/icdar21/.
No account or registration required — dataset is openly licensed.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Official ICDAR 2021 MapSeg download links
DOWNLOADS = {
    "task1_train": "https://icdar21-mapseg.github.io/data/ICDAR21-MapSeg-Task1-train.zip",
    "task1_val":   "https://icdar21-mapseg.github.io/data/ICDAR21-MapSeg-Task1-val.zip",
    "task2_train": "https://icdar21-mapseg.github.io/data/ICDAR21-MapSeg-Task2-train.zip",
    "task2_val":   "https://icdar21-mapseg.github.io/data/ICDAR21-MapSeg-Task2-val.zip",
    "task3_train": "https://icdar21-mapseg.github.io/data/ICDAR21-MapSeg-Task3-train.zip",
    "task3_val":   "https://icdar21-mapseg.github.io/data/ICDAR21-MapSeg-Task3-val.zip",
}

DATA_DIR = Path("data/icdar21")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, dest: Path) -> None:
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in DOWNLOADS.items():
        zip_path = DATA_DIR / f"{name}.zip"
        out_dir  = DATA_DIR / name

        if out_dir.exists():
            print(f"[✓] {name} already downloaded — skipping.")
            continue

        print(f"\n[↓] Downloading {name} …")
        try:
            download(url, zip_path)
        except Exception as e:
            print(f"  [!] Download failed for {name}: {e}")
            print(f"      Please download manually from: {url}")
            print(f"      And extract to: {out_dir}")
            continue

        print(f"    Extracting …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR / name)
        zip_path.unlink()   # Remove zip after extraction
        print(f"    ✓ Extracted to {out_dir}")

    print(f"\n✓ All datasets ready in {DATA_DIR.resolve()}")
    print("\nDataset structure:")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name}/   ({n_files} files)")


if __name__ == "__main__":
    main()
