#!/usr/bin/env python3
"""
Download SAM 2 model checkpoints and configuration files.

Usage:
    python download_sam2_checkpoints.py [--model MODEL] [--output-dir DIR]

Models available:
    - sam2_hiera_tiny
    - sam2_hiera_small
    - sam2_hiera_base_plus
    - sam2_hiera_large (default)
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


# SAM 2 checkpoint URLs (official Facebook Research releases)
SAM2_CHECKPOINTS = {
    "sam2_hiera_tiny": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "config": "sam2_hiera_t.yaml",
        "size": "~38 MB"
    },
    "sam2_hiera_small": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "config": "sam2_hiera_s.yaml",
        "size": "~46 MB"
    },
    "sam2_hiera_base_plus": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "config": "sam2_hiera_b+.yaml",
        "size": "~80 MB"
    },
    "sam2_hiera_large": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "config": "sam2_hiera_l.yaml",
        "size": "~224 MB"
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download a file with progress bar."""
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    print(f"✓ Downloaded: {output_path}")


def download_sam2_checkpoint(model_name: str, output_dir: str = "checkpoints"):
    """
    Download SAM 2 checkpoint and config.

    Args:
        model_name: Model variant (e.g., 'sam2_hiera_large')
        output_dir: Directory to save checkpoints
    """
    if model_name not in SAM2_CHECKPOINTS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(SAM2_CHECKPOINTS.keys())}")
        return False

    model_info = SAM2_CHECKPOINTS[model_name]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Downloading SAM 2 Model: {model_name}")
    print(f"Size: {model_info['size']}")
    print(f"{'='*70}\n")

    # Download checkpoint
    checkpoint_url = model_info["checkpoint"]
    checkpoint_filename = checkpoint_url.split('/')[-1]
    checkpoint_path = output_path / checkpoint_filename

    if checkpoint_path.exists():
        print(f"✓ Checkpoint already exists: {checkpoint_path}")
    else:
        try:
            download_file(checkpoint_url, str(checkpoint_path))
        except Exception as e:
            print(f"✗ Failed to download checkpoint: {e}")
            return False

    # Create config info
    config_name = model_info["config"]
    config_info_path = output_path / f"{model_name}_info.txt"

    with open(config_info_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Config: {config_name}\n")
        f.write(f"Checkpoint: {checkpoint_filename}\n")
        f.write(f"Downloaded from: {checkpoint_url}\n")

    print(f"\n{'='*70}")
    print(f"✓ Download Complete!")
    print(f"{'='*70}\n")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_name}")
    print(f"\nTo use this checkpoint:")
    print(f"  from sam2.build_sam import build_sam2")
    print(f"  model = build_sam2('{model_name}', checkpoint='{checkpoint_path}')")
    print()

    return True


def download_all_checkpoints(output_dir: str = "checkpoints"):
    """Download all SAM 2 checkpoints."""
    print(f"\n{'='*70}")
    print(f"Downloading ALL SAM 2 Checkpoints")
    print(f"Total size: ~388 MB")
    print(f"{'='*70}\n")

    success_count = 0
    for model_name in SAM2_CHECKPOINTS.keys():
        if download_sam2_checkpoint(model_name, output_dir):
            success_count += 1
        print()

    print(f"\n{'='*70}")
    print(f"Downloaded {success_count}/{len(SAM2_CHECKPOINTS)} checkpoints")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download SAM 2 model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (large)
  python download_sam2_checkpoints.py

  # Download specific model
  python download_sam2_checkpoints.py --model sam2_hiera_tiny

  # Download to custom directory
  python download_sam2_checkpoints.py --output-dir /path/to/checkpoints

  # Download all models
  python download_sam2_checkpoints.py --all

Available models:
  - sam2_hiera_tiny       (~38 MB)  - Fastest, lowest quality
  - sam2_hiera_small      (~46 MB)  - Fast, good quality
  - sam2_hiera_base_plus  (~80 MB)  - Balanced
  - sam2_hiera_large      (~224 MB) - Best quality (default)
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sam2_hiera_large",
        choices=list(SAM2_CHECKPOINTS.keys()),
        help="SAM 2 model variant to download (default: sam2_hiera_large)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all SAM 2 checkpoints"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    # List models
    if args.list:
        print("\nAvailable SAM 2 Models:")
        print("-" * 70)
        for name, info in SAM2_CHECKPOINTS.items():
            print(f"  {name:25} - {info['size']:10} - {info['config']}")
        print()
        return 0

    # Download
    if args.all:
        download_all_checkpoints(args.output_dir)
    else:
        download_sam2_checkpoint(args.model, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
