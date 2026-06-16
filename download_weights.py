#!/usr/bin/env python3
# ==============================================================================
# download_weights.py
#
# Downloads all required model checkpoints into the weights/ directory.
# Run directly:  python3 download_weights.py
# Or via setup:  ./setup.sh  (calls this automatically)
#
# Weights downloaded:
#   sam_vit_h_4b8939.pth       ~2.4 GB  SAM ViT-H
#   groundingdino_swint_ogc.pth ~700 MB  GroundingDINO SwinT
#   raft-things.pth             ~20 MB   RAFT optical flow
#   ProPainter.pth              ~800 MB  ProPainter inpainting
#
# HuggingFace models (SVD, ControlNet) are downloaded automatically by
# diffusers on first use and cached in ~/.cache/huggingface/.
# ==============================================================================

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# (filename, url, size_hint)
CHECKPOINTS: list[tuple[str, str, str]] = [
    (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "2.4 GB",
    ),
    (
        "groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "700 MB",
    ),
    (
        "raft-things.pth",
        "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth",
        "20 MB",
    ),
    (
        "ProPainter.pth",
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        "800 MB",
    ),
]


def _progress_bar(block: int, block_size: int, total: int) -> None:
    downloaded = block * block_size
    if total <= 0:
        sys.stdout.write(f"\r  {downloaded // 1_048_576} MB downloaded")
    else:
        pct  = min(100, downloaded * 100 // total)
        done = pct // 2
        bar  = "█" * done + "░" * (50 - done)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({downloaded//1_048_576}/{total//1_048_576} MB)")
    sys.stdout.flush()


def download_all(force: bool = False) -> None:
    print(f"\n{'━'*52}")
    print("  FlowEdit — Model Weight Downloader")
    print(f"{'━'*52}\n")

    for filename, url, size_hint in CHECKPOINTS:
        dest = WEIGHTS_DIR / filename

        if dest.exists() and not force:
            mb = dest.stat().st_size / 1_048_576
            print(f"  ✓  {filename}  ({mb:.0f} MB — already downloaded)")
            continue

        print(f"\n  ↓  {filename}  ({size_hint})")
        print(f"     {url}")
        try:
            urllib.request.urlretrieve(url, str(dest), reporthook=_progress_bar)
            sys.stdout.write("\n")
            mb = dest.stat().st_size / 1_048_576
            print(f"     Saved to {dest}  ({mb:.0f} MB)")
        except Exception as exc:
            sys.stdout.write("\n")
            print(f"  ✗  Failed: {exc}")
            print(f"     Download manually from:\n     {url}")
            print(f"     and place at: {dest}")
            if dest.exists():
                dest.unlink()   # remove partial download

    print(f"\n{'━'*52}")
    print("  Done. HuggingFace models (SVD, ControlNet, SD 1.5)")
    print("  will download automatically on first use via diffusers.")
    print(f"{'━'*52}\n")


if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        print("  --force: re-downloading all weights even if present")
    download_all(force=force)
