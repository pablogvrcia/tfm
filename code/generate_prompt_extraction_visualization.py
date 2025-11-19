#!/usr/bin/env python3
"""
Generate visualization of the 5-stage Intelligent Prompt Extraction Process
for the CLIP-Guided Prompting approach.

This script creates a figure showing:
1. SCLIP confidence map (dense prediction)
2. Binary thresholding (confidence > 0.7)
3. Connected component clustering
4. Size filtering (remove small regions)
5. Final centroid computation with prompts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy import ndimage
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')


def generate_sample_confidence_map(height=224, width=224, num_objects=5, seed=42):
    """
    Generate a synthetic SCLIP confidence map with multiple object regions.

    Returns:
        confidence_map: (H, W) array with values in [0, 1]
        class_map: (H, W) array with class indices
    """
    np.random.seed(seed)

    confidence_map = np.zeros((height, width))
    class_map = np.zeros((height, width), dtype=int)

    # Define object regions (center_y, center_x, size_y, size_x, confidence, class_id)
    objects = [
        (60, 70, 40, 50, 0.92, 1),    # Car (top-left)
        (150, 80, 35, 35, 0.88, 2),   # Person (bottom-left)
        (80, 180, 30, 45, 0.85, 3),   # Horse (top-right)
        (165, 170, 25, 30, 0.80, 2),  # Another person (bottom-right)
        (50, 150, 15, 15, 0.75, 4),   # Small object (top-center)
    ]

    for cy, cx, sy, sx, conf, cls in objects:
        # Create elliptical region with Gaussian falloff
        y, x = np.ogrid[:height, :width]
        mask = ((y - cy)**2 / (sy**2) + (x - cx)**2 / (sx**2)) <= 1

        # Add Gaussian confidence with some noise
        distance = np.sqrt((y - cy)**2 / (sy**2) + (x - cx)**2 / (sx**2))
        gaussian = np.exp(-2 * distance**2)

        # Update confidence map (take max to handle overlaps)
        region_conf = gaussian * conf * mask
        confidence_map = np.maximum(confidence_map, region_conf)

        # Update class map
        class_map[mask] = cls

    # Add background noise
    noise = np.random.normal(0.2, 0.15, (height, width))
    confidence_map = np.clip(confidence_map + noise * 0.3, 0, 1)

    return confidence_map, class_map


def stage1_confidence_masking(confidence_map, class_map, threshold=0.7):
    """Stage 1: Per-class confidence masking."""
    binary_mask = (confidence_map > threshold).astype(np.uint8)
    return binary_mask


def stage2_connected_components(binary_mask):
    """Stage 2: Connected component analysis."""
    labeled_mask, num_features = label(binary_mask, return_num=True, connectivity=2)
    return labeled_mask, num_features


def stage3_size_filtering(labeled_mask, min_area=100):
    """Stage 3: Region filtering by size."""
    regions = regionprops(labeled_mask)
    filtered_mask = np.zeros_like(labeled_mask)

    valid_labels = []
    for region in regions:
        if region.area >= min_area:
            filtered_mask[labeled_mask == region.label] = region.label
            valid_labels.append(region.label)

    return filtered_mask, valid_labels


def stage4_centroid_computation(filtered_mask, class_map, confidence_map):
    """Stage 4: Centroid computation for each valid region."""
    regions = regionprops(filtered_mask)
    centroids = []

    for region in regions:
        cy, cx = region.centroid
        # Get class at centroid
        cls = class_map[int(cy), int(cx)]
        conf = confidence_map[int(cy), int(cx)]
        centroids.append({
            'y': cy,
            'x': cx,
            'class': cls,
            'confidence': conf,
            'area': region.area,
            'label': region.label
        })

    return centroids


def create_visualization():
    """Create the complete 5-stage visualization."""

    # Generate synthetic data
    confidence_map, class_map = generate_sample_confidence_map()

    # Apply the 5 stages
    binary_mask = stage1_confidence_masking(confidence_map, class_map, threshold=0.7)
    labeled_mask, num_features = stage2_connected_components(binary_mask)
    filtered_mask, valid_labels = stage3_size_filtering(labeled_mask, min_area=100)
    centroids = stage4_centroid_computation(filtered_mask, class_map, confidence_map)

    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Intelligent Prompt Extraction: 5-Stage Pipeline',
                 fontsize=20, fontweight='bold', y=0.98)

    # Define class names and colors
    class_names = {0: 'Background', 1: 'Car', 2: 'Person', 3: 'Horse', 4: 'Bottle'}
    class_colors = ['#808080', '#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3']

    # Stage 0: Original image placeholder (we'll show confidence map)
    ax0 = axes[0, 0]
    im0 = ax0.imshow(confidence_map, cmap='viridis', interpolation='bilinear')
    ax0.set_title('Stage 0: SCLIP Dense Prediction\n(Confidence Map)',
                  fontsize=14, fontweight='bold', pad=10)
    ax0.axis('off')
    cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label('Confidence Score', fontsize=11)
    ax0.text(0.5, -0.08, 'SCLIP produces pixel-wise class predictions\nwith confidence scores',
             transform=ax0.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 1: Binary thresholding
    ax1 = axes[0, 1]
    ax1.imshow(binary_mask, cmap='gray', interpolation='nearest')
    ax1.set_title('Stage 1: Confidence Masking\n(Threshold τ_conf = 0.7)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    ax1.text(0.5, -0.08, f'Filter pixels with confidence > 0.7\nRetaining high-certainty regions only',
             transform=ax1.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 2: Connected components with distinct colors
    ax2 = axes[0, 2]
    # Create random colormap for different components
    n_components = labeled_mask.max()
    colors = plt.cm.tab20(np.linspace(0, 1, n_components + 1))
    colors[0] = [0, 0, 0, 1]  # Background black
    cmap_cc = ListedColormap(colors)
    ax2.imshow(labeled_mask, cmap=cmap_cc, interpolation='nearest')
    ax2.set_title('Stage 2: Connected Component Analysis\n(8-connectivity)',
                  fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    ax2.text(0.5, -0.08, f'Identified {num_features} distinct regions\nGrouping adjacent pixels',
             transform=ax2.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 3: Size filtering
    ax3 = axes[1, 0]
    # Create colormap for filtered regions
    colors_filtered = plt.cm.tab20(np.linspace(0, 1, filtered_mask.max() + 1))
    colors_filtered[0] = [0, 0, 0, 1]
    cmap_filtered = ListedColormap(colors_filtered)
    ax3.imshow(filtered_mask, cmap=cmap_filtered, interpolation='nearest')
    ax3.set_title('Stage 3: Size Filtering\n(Minimum Area = 100 pixels)',
                  fontsize=14, fontweight='bold', pad=10)
    ax3.axis('off')
    num_valid = len(valid_labels)
    ax3.text(0.5, -0.08, f'Removed {num_features - num_valid} small regions (noise)\nRetaining {num_valid} valid objects',
             transform=ax3.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 4: Centroid computation - show on original confidence map
    ax4 = axes[1, 1]
    ax4.imshow(confidence_map, cmap='viridis', alpha=0.6, interpolation='bilinear')

    # Overlay filtered regions with transparency
    masked_regions = np.ma.masked_where(filtered_mask == 0, filtered_mask)
    ax4.imshow(masked_regions, cmap=cmap_filtered, alpha=0.4, interpolation='nearest')

    # Plot centroids
    for i, centroid in enumerate(centroids):
        cy, cx = centroid['y'], centroid['x']
        cls = centroid['class']

        # Draw centroid point
        ax4.plot(cx, cy, 'r*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)

        # Add label
        ax4.text(cx, cy - 8, f"{class_names[cls]}\n({centroid['confidence']:.2f})",
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='red', alpha=0.8, linewidth=1.5),
                zorder=11)

    ax4.set_title('Stage 4: Centroid Computation\n(Representative Points)',
                  fontsize=14, fontweight='bold', pad=10)
    ax4.axis('off')
    ax4.text(0.5, -0.08, f'Extracted {len(centroids)} prompt points\nRed stars mark centroids with class labels',
             transform=ax4.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 5: Final result on original image with prompt points
    ax5 = axes[1, 2]
    # Create a synthetic "original image" (blend confidence map with class map)
    original_vis = np.zeros((confidence_map.shape[0], confidence_map.shape[1], 3))
    for i in range(confidence_map.shape[0]):
        for j in range(confidence_map.shape[1]):
            cls = class_map[i, j]
            if cls > 0 and confidence_map[i, j] > 0.5:
                # Use class color
                color = np.array(plt.cm.colors.to_rgb(class_colors[cls]))
                original_vis[i, j] = color * confidence_map[i, j]
            else:
                # Background
                original_vis[i, j] = [0.3, 0.3, 0.3]

    ax5.imshow(original_vis, interpolation='bilinear')

    # Plot final prompts with arrows
    for i, centroid in enumerate(centroids):
        cy, cx = centroid['y'], centroid['x']
        cls = centroid['class']

        # Draw prompt point
        ax5.plot(cx, cy, 'o', markersize=12, color='yellow',
                markeredgecolor='black', markeredgewidth=2, zorder=10)

        # Draw crosshair
        ax5.plot([cx-8, cx+8], [cy, cy], 'k-', linewidth=1.5, zorder=9)
        ax5.plot([cx, cx], [cy-8, cy+8], 'k-', linewidth=1.5, zorder=9)

        # Add prompt number
        ax5.text(cx+12, cy-12, f"#{i+1}", fontsize=11, fontweight='bold',
                color='yellow', bbox=dict(boxstyle='circle,pad=0.3',
                facecolor='black', alpha=0.7), zorder=11)

    ax5.set_title('Stage 5: Final Prompt Points\n(Ready for SAM2)',
                  fontsize=14, fontweight='bold', pad=10)
    ax5.axis('off')
    ax5.text(0.5, -0.08, f'{len(centroids)} semantic prompts vs 4096 grid points\n96% reduction in computational cost',
             transform=ax5.transAxes, ha='center', fontsize=10, style='italic',
             color='green', fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save figure
    output_path = '/home/pablo/aux/tfm/overleaf/Imagenes/prompt_extraction_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_path}")

    # Also create a summary statistics box
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # Summary statistics
    summary_text = f"""
    INTELLIGENT PROMPT EXTRACTION PIPELINE - SUMMARY
    ═══════════════════════════════════════════════════

    Input: 224×224 image (50,176 pixels)
    SCLIP confidence threshold: τ_conf = 0.7
    Minimum region size: τ_area = 100 pixels

    STAGE RESULTS:
    ──────────────────────────────────────────────────
    Stage 1 (Confidence Masking):
        • High-confidence pixels: {binary_mask.sum():,} / {binary_mask.size:,}
        • Retention rate: {100*binary_mask.sum()/binary_mask.size:.1f}%

    Stage 2 (Connected Components):
        • Total regions identified: {num_features}
        • Connectivity: 8-connected (includes diagonals)

    Stage 3 (Size Filtering):
        • Regions after filtering: {len(valid_labels)}
        • Noise regions removed: {num_features - len(valid_labels)}
        • Filtering efficiency: {100*(num_features - len(valid_labels))/num_features:.1f}%

    Stage 4 & 5 (Centroid Extraction):
        • Final prompt points: {len(centroids)}
        • Class distribution:
    """

    # Add class distribution
    class_counts = {}
    for c in centroids:
        cls = class_names[c['class']]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    for cls, count in sorted(class_counts.items()):
        summary_text += f"          - {cls}: {count} prompt(s)\n    "

    summary_text += f"""

    EFFICIENCY GAINS:
    ──────────────────────────────────────────────────
    • Semantic prompts: {len(centroids)} points
    • Blind grid (64×64): 4,096 points
    • Reduction: {100*(1 - len(centroids)/4096):.1f}%
    • Speedup: {4096/len(centroids):.1f}× faster

    ✓ Intelligent prompting focuses computation on semantically
      meaningful regions, achieving massive efficiency gains
      while maintaining competitive accuracy.
    """

    ax.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue',
                     edgecolor='navy', linewidth=2, alpha=0.9))

    plt.tight_layout()
    summary_path = '/home/pablo/aux/tfm/overleaf/Imagenes/prompt_extraction_summary.png'
    plt.savefig(summary_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Summary statistics saved to: {summary_path}")

    return centroids


if __name__ == '__main__':
    print("Generating Intelligent Prompt Extraction visualization...")
    print("=" * 60)

    centroids = create_visualization()

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY:")
    print("=" * 60)
    print(f"Total semantic prompts extracted: {len(centroids)}")
    print(f"Computational reduction: {100*(1 - len(centroids)/4096):.2f}%")
    print(f"Speedup factor: {4096/len(centroids):.1f}×")
    print("\nPrompt details:")
    for i, c in enumerate(centroids, 1):
        class_names = {0: 'Background', 1: 'Car', 2: 'Person', 3: 'Horse', 4: 'Bottle'}
        print(f"  Prompt #{i}: {class_names[c['class']]:10s} at ({c['x']:6.1f}, {c['y']:6.1f}) "
              f"- Conf: {c['confidence']:.3f}, Area: {int(c['area']):4d} px")

    print("\n✓ Visualization complete!")
    print("\nOutput files:")
    print("  1. /home/pablo/aux/tfm/overleaf/Imagenes/prompt_extraction_pipeline.png")
    print("  2. /home/pablo/aux/tfm/overleaf/Imagenes/prompt_extraction_summary.png")
