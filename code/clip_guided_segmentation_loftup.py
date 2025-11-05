"""
CLIP-guided SAM segmentation with LoftUp feature upsampling.

This is an enhanced version of clip_guided_segmentation.py that integrates
LoftUp feature upsampling for sharper semantic predictions.

Key improvements:
1. SCLIP features upsampled from 14× to full resolution with LoftUp
2. Sharper semantic boundaries → better prompt localization
3. Improved small object detection
4. Maintains 96% prompt reduction strategy
5. Keeps SAM2 for high-quality mask refinement

Usage:
    python clip_guided_segmentation_loftup.py \
        --image path/to/image.jpg \
        --vocabulary person car dog \
        --output loftup_results.png \
        --use-loftup
"""

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.loftup_sclip_segmentor import LoftUpSCLIPSegmentor, extract_prompt_points_from_upsampled
from scipy.ndimage import label, center_of_mass
import cv2


def generate_distinct_colors(n):
    """Generate visually distinct colors using a curated palette."""
    base_palette = [
        (0.12, 0.47, 0.71),  # Blue
        (1.00, 0.50, 0.05),  # Orange
        (0.17, 0.63, 0.17),  # Green
        (0.84, 0.15, 0.16),  # Red
        (0.58, 0.40, 0.74),  # Purple
        (0.55, 0.34, 0.29),  # Brown
        (0.89, 0.47, 0.76),  # Pink
        (0.50, 0.50, 0.50),  # Gray
        (0.74, 0.74, 0.13),  # Yellow
        (0.09, 0.75, 0.81),  # Cyan
    ]

    if n <= len(base_palette):
        return base_palette[:n]

    # Generate additional colors using HSV
    colors = list(base_palette)
    for i in range(n - len(base_palette)):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.6 + (i % 3) * 0.15
        value = 0.7 + (i % 2) * 0.2

        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors


def load_image(image_path):
    """Load image from path."""
    image = Image.open(image_path)
    return np.array(image.convert("RGB"))


def segment_with_guided_prompts(image, prompts, checkpoint_path=None, model_cfg=None, device=None):
    """
    Segment image using LoftUp-guided prompts.

    Args:
        image: (H, W, 3) RGB image
        prompts: List of prompt dictionaries with 'point', 'class_idx', etc.
        checkpoint_path: Path to SAM2 checkpoint
        model_cfg: SAM2 model config
        device: Device to use

    Returns:
        List of segmentation results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        checkpoint_path = "checkpoints/sam2_hiera_large.pt"

    if model_cfg is None:
        model_cfg = "sam2_hiera_l.yaml"

    print(f"\n[SAM2] Loading model from {checkpoint_path}...")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print("[SAM2] Setting image...")
    predictor.set_image(image)

    print(f"[SAM2] Generating masks for {len(prompts)} prompts...")
    results = []

    for i, prompt_info in enumerate(prompts):
        point = prompt_info['point']

        # Convert to numpy array format SAM expects
        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([1])  # 1 = foreground point

        # Get mask from SAM
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True  # Get 3 masks, pick best
        )

        # Pick mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]

        results.append({
            'mask': mask,
            'class_idx': prompt_info['class_idx'],
            'class_name': prompt_info['class_name'],
            'confidence': prompt_info['confidence'],
            'sam_score': float(score),
            'region_size': prompt_info['region_size'],
            'prompt_point': point
        })

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} masks...")

    print(f"[SAM2] Successfully generated {len(results)} masks")
    return results


def merge_overlapping_masks(results, iou_threshold=0.8):
    """
    Merge masks that overlap significantly and have the same class.

    Args:
        results: List of result dictionaries with 'mask', 'class_idx', etc.
        iou_threshold: IoU threshold for merging

    Returns:
        Filtered list of results with overlaps removed
    """
    print(f"\n[Merge] Applying NMS with IoU threshold: {iou_threshold}...")

    # Sort by confidence (keep highest confidence masks)
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    kept_results = []

    for result in sorted_results:
        mask = result['mask']
        class_idx = result['class_idx']

        # Check if this mask significantly overlaps with any kept mask of same class
        should_keep = True

        for kept in kept_results:
            if kept['class_idx'] != class_idx:
                continue

            kept_mask = kept['mask']

            # Calculate IoU
            mask_bool = mask.astype(bool)
            kept_mask_bool = kept_mask.astype(bool)
            intersection = (mask_bool & kept_mask_bool).sum()
            union = (mask_bool | kept_mask_bool).sum()
            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept_results.append(result)

    print(f"[Merge] Kept {len(kept_results)}/{len(results)} masks after NMS")
    return kept_results


def visualize_results(image, results, vocabulary, output_path=None):
    """Visualize segmentation results."""
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    # Create color map for classes
    distinct_colors = generate_distinct_colors(len(vocabulary))
    class_colors = {i: distinct_colors[i] for i in range(len(vocabulary))}

    # Sort by region size (draw larger regions first)
    sorted_results = sorted(results, key=lambda x: x['region_size'], reverse=True)

    # Draw masks
    for result in sorted_results:
        mask = result['mask']
        class_idx = result['class_idx']
        class_name = result['class_name']
        confidence = result['confidence']

        color = class_colors[class_idx]
        img_overlay = np.ones((mask.shape[0], mask.shape[1], 3))
        for i in range(3):
            img_overlay[:, :, i] = color[i]
        ax.imshow(np.dstack((img_overlay, mask * 0.7)))

        # Add label at prompt point
        point = result['prompt_point']
        ax.plot(point[0], point[1], 'r*', markersize=10)

        if result['region_size'] > 1000:  # Only label larger objects
            ax.text(
                point[0], point[1],
                f"{class_name}\n{confidence:.2f}",
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                ha='center', va='center'
            )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=class_colors[i], edgecolor='black', linewidth=1.5, label=vocabulary[i])
        for i in range(len(vocabulary))
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=12,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        shadow=True
    )

    ax.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"\n[Save] Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def print_statistics(results, vocabulary):
    """Print statistics."""
    print("\n" + "="*50)
    print("SEGMENTATION STATISTICS")
    print("="*50)

    # Count per class
    class_counts = {i: 0 for i in range(len(vocabulary))}
    for result in results:
        class_counts[result['class_idx']] += 1

    print(f"\nTotal segments: {len(results)}")
    print("\nClass distribution:")
    for i, class_name in enumerate(vocabulary):
        count = class_counts[i]
        percentage = (count / len(results) * 100) if len(results) > 0 else 0
        print(f"  {class_name:20s}: {count:4d} ({percentage:5.1f}%)")

    # Average confidence per class
    print("\nAverage CLIP confidence per class:")
    for i, class_name in enumerate(vocabulary):
        class_results = [r for r in results if r['class_idx'] == i]
        if class_results:
            avg_conf = np.mean([r['confidence'] for r in class_results])
            print(f"  {class_name:20s}: {avg_conf:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-guided SAM segmentation with LoftUp upsampling"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--vocabulary", nargs='+', required=True,
                       help="List of class names for CLIP")
    parser.add_argument("--use-loftup", action="store_true",
                       help="Enable LoftUp feature upsampling (recommended)")
    parser.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt",
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--model-cfg", default="sam2_hiera_l.yaml",
                       help="SAM2 model configuration")
    parser.add_argument("--output", default="loftup_segmentation.png",
                       help="Output visualization path")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum CLIP confidence for prompts")
    parser.add_argument("--min-region-size", type=int, default=100,
                       help="Minimum region size (pixels) for prompts")
    parser.add_argument("--iou-threshold", type=float, default=0.8,
                       help="IoU threshold for merging overlapping masks")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("="*70)
    print("LOFTUP-ENHANCED CLIP-GUIDED SEGMENTATION")
    print("="*70)
    print(f"Image: {args.image}")
    print(f"Vocabulary: {', '.join(args.vocabulary)}")
    print(f"LoftUp upsampling: {'ENABLED' if args.use_loftup else 'DISABLED'}")
    print("="*70)

    # Load image
    print(f"\n[Load] Loading image from {args.image}...")
    image = load_image(args.image)
    print(f"[Load] Image shape: {image.shape}")

    # Step 1: Get SCLIP predictions with optional LoftUp upsampling
    print("\n" + "="*70)
    if args.use_loftup:
        print("STEP 1: SCLIP + LoftUp Upsampling")
    else:
        print("STEP 1: SCLIP Dense Prediction (no upsampling)")
    print("="*70)

    segmentor = LoftUpSCLIPSegmentor(
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        use_loftup=args.use_loftup,
        loftup_model_name="loftup_clip",  # Use CLIP-trained LoftUp
        use_sam=False,
        use_pamr=False,
        verbose=True
    )

    seg_map, logits, upsampled_features = segmentor.predict_dense(
        image,
        args.vocabulary,
        return_logits=True,
        return_features=True
    )

    # Convert logits to probabilities for prompt extraction
    probs = torch.softmax(logits, dim=0).cpu().numpy()  # (num_classes, H, W)
    probs = probs.transpose(1, 2, 0)  # (H, W, num_classes)

    # Step 2: Extract prompt points from (upsampled) predictions
    print("\n" + "="*70)
    print("STEP 2: Extract Prompt Points")
    print("="*70)

    prompts = extract_prompt_points_from_upsampled(
        seg_map, probs, args.vocabulary,
        min_confidence=args.min_confidence,
        min_region_size=args.min_region_size
    )

    print(f"\n[Efficiency] Using {len(prompts)} prompts vs {64*64}=4096 in blind grid!")
    if len(prompts) > 0:
        print(f"[Efficiency] Speedup: {4096/len(prompts):.1f}x fewer SAM queries")
    else:
        print("[Warning] No prompts found! Try lowering --min-confidence")
        return

    # Step 3: Segment with guided prompts
    print("\n" + "="*70)
    print("STEP 3: SAM2 Segmentation")
    print("="*70)

    results = segment_with_guided_prompts(
        image, prompts,
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device
    )

    # Step 4: Merge overlapping masks
    results = merge_overlapping_masks(results, iou_threshold=args.iou_threshold)

    # Visualize results
    visualize_results(image, results, args.vocabulary, output_path=args.output)

    # Print statistics
    print_statistics(results, args.vocabulary)

    print("\n" + "="*70)
    print("SEGMENTATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
