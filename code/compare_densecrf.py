#!/usr/bin/env python3
"""
Compare dense predictions with and without DenseCRF refinement.
Generates visualizations for thesis documentation.
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models.sclip_segmentor import SCLIPSegmentor

def create_colored_segmentation(segmentation_map, num_classes):
    """Create colored visualization of segmentation map."""
    # Generate distinct colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background black

    colored = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    for i in range(num_classes):
        colored[segmentation_map == i] = colors[i]

    return Image.fromarray(colored)

def main():
    # Configuration
    image_path = "examples/football_frame.png"
    vocabulary = ["Lionel Messi", "Luis Suarez", "Neymar Jr", "grass", "crowd", "background"]
    output_dir = "examples_results"

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Vocabulary: {vocabulary}")

    # Predict WITHOUT DenseCRF
    print("\n1. Running dense prediction WITHOUT DenseCRF...")
    segmentor_no_crf = SCLIPSegmentor(
        model_name="ViT-B/16",
        device=device,
        use_fp16=True,
        use_densecrf=False
    )
    seg_no_crf, _ = segmentor_no_crf.predict_dense(
        image=image_np,
        class_names=vocabulary
    )

    # Predict WITH DenseCRF
    print("\n2. Running dense prediction WITH DenseCRF...")
    segmentor_with_crf = SCLIPSegmentor(
        model_name="ViT-B/16",
        device=device,
        use_fp16=True,
        use_densecrf=True
    )
    seg_with_crf, _ = segmentor_with_crf.predict_dense(
        image=image_np,
        class_names=vocabulary
    )

    # Create visualizations
    print("\nCreating visualizations...")

    # Calculate differences first
    diff_mask = (seg_no_crf != seg_with_crf)
    num_changed = diff_mask.sum()
    total_pixels = seg_no_crf.size
    percent_changed = 100 * num_changed / total_pixels

    # Generate colored segmentations
    num_classes = len(vocabulary)
    colored_no_crf = create_colored_segmentation(seg_no_crf, num_classes)
    colored_with_crf = create_colored_segmentation(seg_with_crf, num_classes)

    # Save individual results
    colored_no_crf.save(f"{output_dir}/football_dense_no_crf.png", dpi=(150, 150))
    colored_with_crf.save(f"{output_dir}/football_dense_with_crf.png", dpi=(150, 150))
    print(f"Saved: {output_dir}/football_dense_no_crf.png")
    print(f"Saved: {output_dir}/football_dense_with_crf.png")

    # Create difference map
    diff_image = np.zeros((*diff_mask.shape, 3), dtype=np.uint8)
    diff_image[diff_mask] = [255, 0, 0]  # Red for differences

    # Create overlay on original image
    overlay_diff = np.array(image).copy()
    overlay_diff[diff_mask] = overlay_diff[diff_mask] // 2 + diff_image[diff_mask] // 2

    # Create side-by-side comparison with difference highlight
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Differences highlighted
    axes[0, 1].imshow(overlay_diff)
    axes[0, 1].set_title(f"Differences Highlighted (Red)\n{num_changed:,} pixels changed ({percent_changed:.2f}%)",
                        fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Without DenseCRF
    axes[1, 0].imshow(colored_no_crf)
    axes[1, 0].set_title("Without DenseCRF", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # With DenseCRF
    axes[1, 1].imshow(colored_with_crf)
    axes[1, 1].set_title("With DenseCRF", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Create legend
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    legend_patches = [
        mpatches.Patch(color=colors[i]/255.0, label=vocabulary[i])
        for i in range(num_classes)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/motogp_densecrf_comparison.png",
                dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/motogp_densecrf_comparison.png")

    print(f"\nStatistics:")
    print(f"  Pixels changed: {num_changed:,} / {total_pixels:,} ({percent_changed:.2f}%)")
    print(f"  Image size: {image.size}")

    print("\nDone!")

if __name__ == "__main__":
    main()
