#!/usr/bin/env python3
"""
Test script for hierarchical SAM2 prompting with SCLIP confidence scores.

Compares:
- Standard prompting: All positive points
- Hierarchical prompting: Positive (high-conf) + Negative (competing classes)
"""

import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sclip_segmentor import SCLIPSegmentor

def test_hierarchical_prompting():
    """Test hierarchical prompting vs standard prompting."""

    print("=" * 80)
    print("Testing Hierarchical SAM2 Prompting")
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

    # Test 1: Standard prompting (all positive points)
    print("-" * 80)
    print("Test 1: STANDARD Prompting (All positive points)")
    print("-" * 80)
    print()

    start_time = time.time()
    pred_standard = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=True,
        use_hierarchical_prompts=False,  # Standard prompting
        min_coverage=0.6
    )
    time_standard = time.time() - start_time

    print(f"\n✓ Segmentation complete")
    print(f"  Time: {time_standard:.2f}s")
    print(f"  Unique classes: {np.unique(pred_standard)}")
    print()

    # Test 2: Hierarchical prompting (positive + negative)
    print("-" * 80)
    print("Test 2: HIERARCHICAL Prompting (Positive + Negative)")
    print("-" * 80)
    print()

    start_time = time.time()
    pred_hierarchical = segmentor.predict_with_sam(
        image,
        class_names,
        use_prompted_sam=True,
        use_hierarchical_prompts=True,  # Hierarchical prompting
        min_coverage=0.6
    )
    time_hierarchical = time.time() - start_time

    print(f"\n✓ Segmentation complete")
    print(f"  Time: {time_hierarchical:.2f}s")
    print(f"  Unique classes: {np.unique(pred_hierarchical)}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    print("Timing:")
    print(f"  Standard:      {time_standard:.2f}s")
    print(f"  Hierarchical:  {time_hierarchical:.2f}s")

    if time_hierarchical < time_standard:
        speedup = time_standard / time_hierarchical
        print(f"  Speedup: {speedup:.2f}×")
    else:
        slowdown = time_hierarchical / time_standard
        print(f"  Slowdown: {slowdown:.2f}×")
    print()

    # Pixel agreement
    agreement = (pred_standard == pred_hierarchical).sum() / pred_standard.size
    print(f"Pixel Agreement: {agreement * 100:.2f}%")
    print()

    # Per-class analysis
    print("Per-Class Pixel Counts:")
    print(f"  {'Class':<12} {'Standard':>10} {'Hierarchical':>12} {'Difference':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

    for i, class_name in enumerate(class_names):
        count_std = (pred_standard == i).sum()
        count_hier = (pred_hierarchical == i).sum()
        diff = count_hier - count_std
        diff_pct = (diff / count_std * 100) if count_std > 0 else 0

        print(f"  {class_name:<12} {count_std:>10} {count_hier:>12} {diff:>+12} ({diff_pct:+.1f}%)")
    print()

    # Save visualizations
    output_dir = Path("outputs/hierarchical_prompts")
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

    # Save visualizations
    save_image(image, output_dir / "original.png")
    save_image(mask_to_color(pred_standard), output_dir / "standard_prompts.png")
    save_image(mask_to_color(pred_hierarchical), output_dir / "hierarchical_prompts.png")

    # Create difference map
    diff_mask = (pred_standard != pred_hierarchical).astype(np.uint8) * 255
    diff_colored = np.zeros((*diff_mask.shape, 3), dtype=np.uint8)
    diff_colored[diff_mask > 0] = [255, 0, 0]  # Red where different

    # Overlay on original
    diff_overlay = image.copy()
    diff_overlay[diff_mask > 0] = (image[diff_mask > 0] * 0.5 + diff_colored[diff_mask > 0] * 0.5).astype(np.uint8)
    save_image(diff_overlay, output_dir / "differences.png")

    print(f"✓ Visualizations saved to: {output_dir}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    improvements = []

    if agreement >= 0.95:
        improvements.append(f"✓ Very similar results: {agreement*100:.1f}% agreement")
    elif agreement >= 0.85:
        improvements.append(f"✓ Similar results: {agreement*100:.1f}% agreement")
    else:
        improvements.append(f"⚠ Significant differences: {agreement*100:.1f}% agreement")

    if time_hierarchical < time_standard:
        improvements.append(f"✓ Faster: {time_standard/time_hierarchical:.2f}× speedup")

    # Check if hierarchical produces cleaner segmentation
    # (fewer class switches = more coherent)
    std_switches = ((pred_standard[:-1, :] != pred_standard[1:, :]).sum() +
                    (pred_standard[:, :-1] != pred_standard[:, 1:]).sum())
    hier_switches = ((pred_hierarchical[:-1, :] != pred_hierarchical[1:, :]).sum() +
                     (pred_hierarchical[:, :-1] != pred_hierarchical[:, 1:]).sum())

    if hier_switches < std_switches:
        reduction = (std_switches - hier_switches) / std_switches * 100
        improvements.append(f"✓ Cleaner segmentation: {reduction:.1f}% fewer boundary transitions")

    if improvements:
        print("Key Findings:")
        for imp in improvements:
            print(f"  {imp}")
    else:
        print("⚠ No clear improvements detected")

    print()
    print("Hierarchical Prompting Benefits:")
    print("  • Positive prompts guide SAM2 to high-confidence regions")
    print("  • Negative prompts suppress competing classes")
    print("  • Better semantic alignment with SCLIP confidence")
    print("  • Helps resolve boundary ambiguities")
    print()

    return {
        'time_standard': time_standard,
        'time_hierarchical': time_hierarchical,
        'agreement': agreement,
        'std_switches': std_switches,
        'hier_switches': hier_switches
    }


if __name__ == "__main__":
    try:
        results = test_hierarchical_prompting()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
