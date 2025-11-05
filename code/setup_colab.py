"""
Setup script for Google Colab environment.

Run this first in Colab to install dependencies and download checkpoints.

Usage in Colab:
    !python setup_colab.py
"""

import os
import subprocess
import sys
from pathlib import Path
import urllib.request
import shutil


def run_command(cmd, description):
    """Run a shell command and print progress."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def check_file_integrity(filepath, min_size_mb=100):
    """Check if a file exists and has reasonable size."""
    if not os.path.exists(filepath):
        return False

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    if size_mb < min_size_mb:
        print(f"  ⚠️  Warning: File seems too small (< {min_size_mb} MB)")
        return False

    return True


def download_file(url, destination, description):
    """Download a file with progress."""
    print(f"\n[Download] {description}")
    print(f"  URL: {url}")
    print(f"  Destination: {destination}")

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download with progress
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f"\r  Progress: |{bar}| {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(url, destination, reporthook=progress)
        print()  # New line after progress

        # Verify file
        if check_file_integrity(destination):
            print(f"  ✅ Download complete and verified")
            return True
        else:
            print(f"  ❌ Downloaded file appears corrupted")
            return False

    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


def setup_colab_environment():
    """Complete setup for Colab environment."""
    print("="*70)
    print("COLAB ENVIRONMENT SETUP FOR LOFTUP + CLIP-GUIDED SEGMENTATION")
    print("="*70)

    # Step 1: Install Python packages
    packages = [
        "einops",              # Required by LoftUp
        "open_clip_torch",     # CLIP models
        "scipy",               # For connected components
        "opencv-python",       # Image processing
        "matplotlib",          # Visualization
        "huggingface_hub",     # For downloading models
    ]

    print("\n[Step 1/4] Installing Python packages...")
    for package in packages:
        print(f"\n  Installing {package}...")
        if not run_command(f"pip install -q {package}", f"Installing {package}"):
            print(f"  ⚠️  Warning: Failed to install {package}")

    print("\n✅ All packages installed")

    # Step 2: Clone/check SAM2 repository
    print("\n[Step 2/4] Setting up SAM2...")

    if not os.path.exists("sam2"):
        run_command(
            "git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo",
            "Cloning SAM2 repository"
        )
        run_command(
            "cd sam2_repo && pip install -q -e .",
            "Installing SAM2"
        )
    else:
        print("  SAM2 already installed")

    # Step 3: Download SAM2 checkpoint
    print("\n[Step 3/4] Downloading SAM2 checkpoint...")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "sam2_hiera_large.pt"

    # Check if checkpoint already exists and is valid
    if checkpoint_path.exists():
        print(f"  Checkpoint exists: {checkpoint_path}")
        if check_file_integrity(str(checkpoint_path), min_size_mb=800):
            print("  ✅ Checkpoint verified")
        else:
            print("  ⚠️  Checkpoint appears corrupted, re-downloading...")
            checkpoint_path.unlink()  # Delete corrupted file

    if not checkpoint_path.exists():
        sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

        success = download_file(
            sam2_url,
            str(checkpoint_path),
            "SAM2 Hiera Large checkpoint (~900 MB)"
        )

        if not success:
            print("\n  ❌ Failed to download SAM2 checkpoint")
            print("  Manual download instructions:")
            print(f"    1. Download from: {sam2_url}")
            print(f"    2. Upload to: {checkpoint_path}")
            return False

    # Step 4: Verify installation
    print("\n[Step 4/4] Verifying installation...")

    try:
        import einops
        print("  ✅ einops imported successfully")
    except ImportError:
        print("  ❌ einops not available")
        return False

    try:
        import sam2
        print("  ✅ sam2 imported successfully")
    except ImportError:
        print("  ❌ sam2 not available")
        return False

    try:
        from sam2.build_sam import build_sam2
        print("  ✅ SAM2 build functions available")
    except ImportError:
        print("  ❌ SAM2 build functions not available")
        return False

    # Verify checkpoint can be loaded
    try:
        import torch
        print(f"\n  Testing checkpoint load...")
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False  # Important for older checkpoints
        )
        if "model" in checkpoint:
            print(f"  ✅ Checkpoint loaded successfully")
            print(f"     Model keys: {len(checkpoint['model'])} parameters")
        else:
            print(f"  ❌ Checkpoint structure unexpected")
            return False
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        print(f"\n  Trying to re-download...")
        checkpoint_path.unlink()
        return False

    # Print summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\n✅ All components installed and verified:")
    print("   - Python packages (einops, sam2, etc.)")
    print("   - SAM2 checkpoint (verified)")
    print("   - LoftUp integration ready")
    print("\nYou can now run:")
    print("   !python clip_guided_segmentation_loftup.py \\")
    print("       --image examples/nba_frame.png \\")
    print("       --vocabulary person floor crowd background \\")
    print("       --use-loftup \\")
    print("       --output results/nba_loftup.png")
    print("="*70)

    return True


if __name__ == "__main__":
    success = setup_colab_environment()
    sys.exit(0 if success else 1)
