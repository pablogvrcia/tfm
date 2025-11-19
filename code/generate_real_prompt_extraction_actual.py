#!/usr/bin/env python3
"""
Generate visualization of Intelligent Prompt Extraction using ACTUAL SCLIP output
from a PASCAL VOC 2012 image.

This demonstrates the real 5-stage pipeline on real data with real SCLIP predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import label, regionprops
from PIL import Image
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Import SCLIP segmentor
from models.sclip_segmentor import SCLIPSegmentor


def load_real_sclip_output(image_path, class_names):
    """
    Load a real PASCAL VOC image and run SCLIP to get actual predictions.

    Args:
        image_path: Path to VOC image
        class_names: List of class names

    Returns:
        img_array, confidence_map, class_map, image_name
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    print(f"✓ Loaded image: {os.path.basename(image_path)} ({img_array.shape[1]}x{img_array.shape[0]})")

    # Initialize SCLIP segmentor
    print("✓ Initializing SCLIP segmentor...")
    segmentor = SCLIPSegmentor(
        model_name="ViT-B/16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_sam=False,
        use_pamr=False,
        slide_inference=True,  # Use sliding window for better quality
        verbose=True,
        use_fp16=True,
        use_compile=False,
    )

    # Run SCLIP to get dense prediction with logits
    print("✓ Running SCLIP dense prediction...")
    class_map, logits = segmentor.predict_dense(img_array, class_names, return_logits=True)

    # Convert logits to confidence map
    # logits shape: (num_classes, H, W)
    # confidence_map: max confidence across all classes at each pixel
    probs = torch.softmax(logits, dim=0)  # (num_classes, H, W)
    confidence_map = probs.max(dim=0)[0].cpu().numpy()  # (H, W)

    # Resize image to match prediction size
    pred_H, pred_W = class_map.shape
    img_resized = Image.fromarray(img_array).resize((pred_W, pred_H), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0

    print(f"✓ SCLIP prediction complete: {pred_H}x{pred_W}")
    print(f"  Detected classes: {np.unique(class_map)}")
    print(f"  Confidence range: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]")

    return img_array, confidence_map, class_map, os.path.basename(image_path)


def apply_5stage_pipeline(confidence_map, class_map, threshold=0.7, min_area=100):
    """Apply the complete 5-stage intelligent prompt extraction pipeline."""

    # Stage 1: Confidence masking
    binary_mask = (confidence_map > threshold).astype(np.uint8)

    # Stage 2: Connected components
    labeled_mask, num_features = label(binary_mask, return_num=True, connectivity=2)

    # Stage 3: Size filtering
    regions = regionprops(labeled_mask)
    filtered_mask = np.zeros_like(labeled_mask)
    valid_regions = []

    for region in regions:
        if region.area >= min_area:
            filtered_mask[labeled_mask == region.label] = region.label
            valid_regions.append(region)

    # Stage 4 & 5: Centroid extraction
    centroids = []
    for region in valid_regions:
        cy, cx = region.centroid
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

    return binary_mask, labeled_mask, filtered_mask, centroids, num_features


def create_real_visualization():
    """Create visualization with real PASCAL VOC data and actual SCLIP output."""

    # PASCAL VOC class names
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Find a PASCAL VOC image
    voc_paths = [
        '/home/pablo/aux/tfm/code/data/benchmarks/pascal_voc/VOCdevkit/VOC2012/JPEGImages/',
        '/home/pablo/aux/tfm/data/VOCdevkit/VOC2012/JPEGImages/',
        '/home/pablo/aux/tfm/VOCdevkit/VOC2012/JPEGImages/',
    ]

    sample_image_path = None
    for path in voc_paths:
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.endswith('.jpg')]
            if images:
                # Use a specific image known to have good results
                target_images = ['2007_000129.jpg', '2007_000032.jpg', '2007_000039.jpg']
                for target in target_images:
                    if target in images:
                        sample_image_path = os.path.join(path, target)
                        break
                if not sample_image_path:
                    sample_image_path = os.path.join(path, images[0])
                break

    if not sample_image_path or not os.path.exists(sample_image_path):
        print("ERROR: No PASCAL VOC images found!")
        return None, None

    # Load data with REAL SCLIP output
    img_array, confidence_map, class_map, image_name = load_real_sclip_output(
        sample_image_path, class_names
    )

    # Apply pipeline
    binary_mask, labeled_mask, filtered_mask, centroids, num_features = apply_5stage_pipeline(
        confidence_map, class_map, threshold=0.7, min_area=100
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Real Example: Intelligent Prompt Extraction on PASCAL VOC\n({image_name})',
                 fontsize=20, fontweight='bold', y=0.98)

    # Stage 0: Original image
    ax0 = axes[0, 0]
    ax0.imshow(img_array)
    ax0.set_title('Input: PASCAL VOC 2012 Image', fontsize=14, fontweight='bold', pad=10)
    ax0.axis('off')
    ax0.text(0.5, -0.08, 'Real image from validation set',
             transform=ax0.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 1: SCLIP confidence map
    ax1 = axes[0, 1]
    im1 = ax1.imshow(confidence_map, cmap='viridis', interpolation='bilinear')
    ax1.set_title('Stage 0: SCLIP Confidence Map (REAL)', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Confidence', fontsize=10)
    ax1.text(0.5, -0.08, f'Dense pixel-wise predictions from ViT-B/16',
             transform=ax1.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 2: Binary thresholding
    ax2 = axes[0, 2]
    ax2.imshow(binary_mask, cmap='gray', interpolation='nearest')
    ax2.set_title('Stage 1: Binary Thresholding', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    retained_pct = 100 * binary_mask.sum() / binary_mask.size
    ax2.text(0.5, -0.08, f'Threshold = 0.7\nRetained: {retained_pct:.1f}% of pixels',
             transform=ax2.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 3: Connected components
    ax3 = axes[1, 0]
    n_components = labeled_mask.max()
    colors = plt.cm.tab20(np.linspace(0, 1, n_components + 1))
    colors[0] = [0, 0, 0, 1]
    cmap_cc = ListedColormap(colors)
    ax3.imshow(labeled_mask, cmap=cmap_cc, interpolation='nearest')
    ax3.set_title('Stage 2: Connected Components', fontsize=14, fontweight='bold', pad=10)
    ax3.axis('off')
    ax3.text(0.5, -0.08, f'Found {num_features} distinct regions',
             transform=ax3.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 4: Size filtering
    ax4 = axes[1, 1]
    colors_filtered = plt.cm.tab20(np.linspace(0, 1, filtered_mask.max() + 1))
    colors_filtered[0] = [0, 0, 0, 1]
    cmap_filtered = ListedColormap(colors_filtered)
    ax4.imshow(filtered_mask, cmap=cmap_filtered, interpolation='nearest')
    ax4.set_title('Stage 3: Size Filtering', fontsize=14, fontweight='bold', pad=10)
    ax4.axis('off')
    num_valid = len(centroids)
    ax4.text(0.5, -0.08, f'Min area = 100 px\nRetained: {num_valid} objects',
             transform=ax4.transAxes, ha='center', fontsize=10, style='italic')

    # Stage 5: Final prompts on original image
    ax5 = axes[1, 2]
    ax5.imshow(img_array)

    # Overlay prompts
    for i, centroid in enumerate(centroids):
        cy, cx = centroid['y'], centroid['x']
        cls = centroid['class']

        # Draw prompt point with crosshair
        ax5.plot(cx, cy, 'o', markersize=14, color='yellow',
                markeredgecolor='red', markeredgewidth=2.5, zorder=10)
        ax5.plot([cx-10, cx+10], [cy, cy], 'r-', linewidth=2, zorder=9)
        ax5.plot([cx, cx], [cy-10, cy+10], 'r-', linewidth=2, zorder=9)

        # Label
        ax5.text(cx, cy-15, f"#{i+1}\n{class_names[cls]}", fontsize=9,
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                         edgecolor='red', alpha=0.9, linewidth=2), zorder=11)

    ax5.set_title('Stage 4-5: Extracted Prompt Points', fontsize=14, fontweight='bold', pad=10)
    ax5.axis('off')
    reduction_pct = 100 * (1 - len(centroids) / 4096)
    ax5.text(0.5, -0.08, f'{len(centroids)} prompts extracted\n{reduction_pct:.1f}% reduction vs grid (4096)',
             transform=ax5.transAxes, ha='center', fontsize=10, style='italic',
             color='green', fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save
    output_path = '/home/pablo/aux/tfm/overleaf/Imagenes/real_prompt_extraction_example.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Real example visualization saved to: {output_path}")

    return centroids, image_name


if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING REAL PROMPT EXTRACTION EXAMPLE WITH ACTUAL SCLIP")
    print("=" * 70)

    centroids, image_name = create_real_visualization()

    if centroids is None:
        print("\nERROR: Failed to generate visualization")
        exit(1)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Image: {image_name}")
    print(f"Extracted prompts: {len(centroids)}")
    print(f"Reduction: {100 * (1 - len(centroids)/4096):.2f}%")
    print(f"Speedup: {4096/len(centroids):.1f}×")

    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    print("\nPrompt Details:")
    for i, c in enumerate(centroids, 1):
        print(f"  #{i}: {class_names[c['class']]:15s} at ({c['x']:6.1f}, {c['y']:6.1f}) "
              f"- Conf: {c['confidence']:.3f}, Area: {int(c['area']):4d} px")

    print("\n✓ Complete! Image saved to:")
    print("  /home/pablo/aux/tfm/overleaf/Imagenes/real_prompt_extraction_example.png")
