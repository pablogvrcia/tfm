#!/usr/bin/env python3
"""
Test script for improved SAM2 mask selection.

Compares:
- Old approach: All masks, no quality filtering
- New approach: Best mask per point + IoU filtering + NMS
"""

import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sclip_segmentor import SCLIPSegmentor

def test_improved_selection():
    """Test improved SAM2 mask selection."""

    print("=" * 80)
    print("Testing Improved SAM2 Mask Selection")
    print("=" * 80)
    print()

    # Load test image
    image_path = Path("photo.jpg")
    if not image_path.exists():
        print(f"ERROR: Test image not found: {image_path}")
        return

    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Loaded image: {image.shape}")
    print()

    # Test vocabulary
    class_names = ["background", "car", "person", "road", "building", "sky"]
    print(f"Test vocabulary: {class_names}")
    print()

    # Initialize segmentor
    print("Initializing SCLIP segmentor...")
    segmentor = SCLIPSegmentor(
        model_name="ViT-B/16",
        use_sam=True,
        use_pamr=False,
        slide_inference=True,
        verbose=True
    )
    print()

    # Test 1: OLD approach (no filtering, all 3 masks per point)
    print("-" * 80)
    print("Test 1: OLD Approach (All masks, no filtering)")
    print("-" * 80)
    print()

    start_time = time.time()
    pred_old = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=True,
        min_coverage=0.6,
        min_iou_score=0.0,  # Accept all masks
        nms_iou_threshold=1.0,  # No NMS
        use_best_mask_only=False  # Use all 3 masks per point
    )
    time_old = time.time() - start_time

    print(f"\n✓ Segmentation complete")
    print(f"  Time: {time_old:.2f}s")
    print(f"  Unique classes: {np.unique(pred_old)}")
    print()

    # Test 2: NEW approach (best mask + IoU filtering + NMS)
    print("-" * 80)
    print("Test 2: NEW Approach (Best mask + IoU filter + NMS)")
    print("-" * 80)
    print()

    start_time = time.time()
    pred_new = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=True,
        min_coverage=0.6,
        min_iou_score=0.75,  # Filter low-quality masks
        nms_iou_threshold=0.7,  # Remove 70%+ overlaps
        use_best_mask_only=True  # Use only best mask per point
    )
    time_new = time.time() - start_time

    print(f"\n✓ Segmentation complete")
    print(f"  Time: {time_new:.2f}s")
    print(f"  Unique classes: {np.unique(pred_new)}")
    print()

    # Test 3: More aggressive filtering
    print("-" * 80)
    print("Test 3: AGGRESSIVE Approach (Stricter filtering)")
    print("-" * 80)
    print()

    start_time = time.time()
    pred_aggressive = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=True,
        min_coverage=0.7,  # Higher coverage requirement
        min_iou_score=0.85,  # Only very high quality masks
        nms_iou_threshold=0.6,  # More aggressive NMS
        use_best_mask_only=True
    )
    time_aggressive = time.time() - start_time

    print(f"\n✓ Segmentation complete")
    print(f"  Time: {time_aggressive:.2f}s")
    print(f"  Unique classes: {np.unique(pred_aggressive)}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    print("Timing:")
    print(f"  Old (all masks):      {time_old:.2f}s")
    print(f"  New (filtered):       {time_new:.2f}s")
    print(f"  Aggressive:           {time_aggressive:.2f}s")
    print()

    if time_new < time_old:
        speedup = time_old / time_new
        print(f"✓ Speedup: {speedup:.2f}× faster with new approach")
    print()

    # Pixel agreement
    agreement_old_new = (pred_old == pred_new).sum() / pred_old.size
    agreement_old_agg = (pred_old == pred_aggressive).sum() / pred_old.size
    agreement_new_agg = (pred_new == pred_aggressive).sum() / pred_new.size

    print("Pixel Agreement:")
    print(f"  Old vs New:           {agreement_old_new * 100:.2f}%")
    print(f"  Old vs Aggressive:    {agreement_old_agg * 100:.2f}%")
    print(f"  New vs Aggressive:    {agreement_new_agg * 100:.2f}%")
    print()

    # Save visualizations
    output_dir = Path("outputs/improved_sam_selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils import save_image

    def mask_to_color(mask, num_classes=len(class_names)):
        """Convert class indices to RGB colors."""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Black background

        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx in range(num_classes):
            colored[mask == class_idx] = colors[class_idx]
        return colored

    save_image(image, output_dir / "original.png")
    save_image(mask_to_color(pred_old), output_dir / "old_all_masks.png")
    save_image(mask_to_color(pred_new), output_dir / "new_filtered.png")
    save_image(mask_to_color(pred_aggressive), output_dir / "aggressive.png")

    print(f"✓ Visualizations saved to: {output_dir}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    improvements = []

    if time_new < time_old:
        improvements.append(f"✓ Faster: {time_old/time_new:.2f}× speedup")

    if agreement_old_new >= 0.85:
        improvements.append(f"✓ Similar quality: {agreement_old_new*100:.1f}% agreement")

    if time_aggressive < time_old:
        improvements.append(f"✓ Aggressive mode even faster: {time_old/time_aggressive:.2f}×")

    if improvements:
        print("Improvements detected:")
        for imp in improvements:
            print(f"  {imp}")
    else:
        print("⚠ No clear improvements detected")

    print()

    return {
        'time_old': time_old,
        'time_new': time_new,
        'time_aggressive': time_aggressive,
        'agreement_old_new': agreement_old_new,
        'agreement_old_agg': agreement_old_agg,
        'agreement_new_agg': agreement_new_agg
    }


if __name__ == "__main__":
    try:
        results = test_improved_selection()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
