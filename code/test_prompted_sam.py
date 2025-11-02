#!/usr/bin/env python3
"""
Quick test script for prompted SAM2 improvement.

Tests both automatic and prompted SAM modes and compares:
- Speed (time per segmentation)
- Number of masks generated
- Segmentation quality (visual inspection)
"""

import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sclip_segmentor import SCLIPSegmentor

def test_prompted_sam():
    """Test prompted SAM2 vs automatic SAM2."""

    print("=" * 80)
    print("Testing Prompted SAM2 Improvement")
    print("=" * 80)
    print()

    # Load test image
    image_path = Path("photo.jpg")
    if not image_path.exists():
        print(f"ERROR: Test image not found: {image_path}")
        print("Please ensure photo.jpg exists in the code directory")
        return

    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Loaded image: {image.shape}")
    print()

    # Test vocabulary
    class_names = ["background", "car", "person", "road", "building", "sky"]
    print(f"Test vocabulary: {class_names}")
    print()

    # Initialize segmentor with SAM
    print("Initializing SCLIP segmentor with SAM refinement...")
    segmentor = SCLIPSegmentor(
        model_name="ViT-B/16",
        use_sam=True,  # Enable SAM refinement
        use_pamr=False,
        slide_inference=True,
        verbose=False  # Reduce noise for cleaner output
    )
    print("✓ Segmentor initialized")
    print()

    # Test 1: Prompted SAM (new approach)
    print("-" * 80)
    print("Test 1: Prompted SAM2 (NEW)")
    print("-" * 80)

    start_time = time.time()
    pred_prompted = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=True,
        min_coverage=0.6
    )
    time_prompted = time.time() - start_time

    print(f"✓ Segmentation complete")
    print(f"  Time: {time_prompted:.2f}s")
    print(f"  Unique classes: {np.unique(pred_prompted)}")
    print(f"  Prediction shape: {pred_prompted.shape}")
    print()

    # Test 2: Automatic SAM (legacy approach)
    print("-" * 80)
    print("Test 2: Automatic SAM2 (LEGACY)")
    print("-" * 80)

    start_time = time.time()
    pred_automatic = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=False,  # Use old automatic approach
        min_coverage=0.6
    )
    time_automatic = time.time() - start_time

    print(f"✓ Segmentation complete")
    print(f"  Time: {time_automatic:.2f}s")
    print(f"  Unique classes: {np.unique(pred_automatic)}")
    print(f"  Prediction shape: {pred_automatic.shape}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    print(f"Prompted SAM:   {time_prompted:.2f}s")
    print(f"Automatic SAM:  {time_automatic:.2f}s")
    print()

    if time_prompted < time_automatic:
        speedup = time_automatic / time_prompted
        print(f"✓ Speedup: {speedup:.2f}× faster with prompted SAM")
    else:
        slowdown = time_prompted / time_automatic
        print(f"⚠ Slowdown: {slowdown:.2f}× slower with prompted SAM")
    print()

    # Check if predictions are similar
    agreement = (pred_prompted == pred_automatic).sum() / pred_prompted.size
    print(f"Pixel agreement: {agreement * 100:.2f}%")
    print()

    # Save visualizations
    output_dir = Path("outputs/prompted_sam_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import save_image

    # Save predictions as colored masks
    save_image(image, output_dir / "original.png")

    # Create simple colored visualizations
    def mask_to_color(mask, num_classes=len(class_names)):
        """Convert class indices to RGB colors."""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Black background

        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx in range(num_classes):
            colored[mask == class_idx] = colors[class_idx]
        return colored

    save_image(mask_to_color(pred_prompted), output_dir / "prompted_sam.png")
    save_image(mask_to_color(pred_automatic), output_dir / "automatic_sam.png")

    print(f"✓ Visualizations saved to: {output_dir}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if time_prompted < time_automatic and agreement > 0.80:
        print("✅ SUCCESS: Prompted SAM is faster with similar quality")
        print("   The improvement is working as expected!")
    elif time_prompted < time_automatic:
        print("⚠️  PARTIAL SUCCESS: Prompted SAM is faster but predictions differ")
        print("   Check visualizations to verify quality")
    else:
        print("❌ ISSUE: Prompted SAM is not faster than automatic")
        print("   This may indicate a problem with the implementation")
    print()

    return {
        'time_prompted': time_prompted,
        'time_automatic': time_automatic,
        'speedup': time_automatic / time_prompted,
        'agreement': agreement
    }


if __name__ == "__main__":
    try:
        results = test_prompted_sam()

        # Exit code based on success
        if results and results['speedup'] > 1.0:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
