"""
CLIP-guided SAM segmentation.
Uses CLIP dense predictions to intelligently place SAM prompts instead of blind grid.

Strategy:
1. Run CLIP dense prediction first (fast)
2. Extract high-confidence regions for each class
3. Use those as point prompts for SAM (much fewer prompts than 64x64 grid)
4. Get high-quality SAM masks only where CLIP says objects are
"""

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.sclip_segmentor import SCLIPSegmentor
from scipy.ndimage import label, center_of_mass
import cv2


def load_image(image_path):
    """Load image from path."""
    image = Image.open(image_path)
    return np.array(image.convert("RGB"))


def extract_prompt_points_from_clip(seg_map, probs, vocabulary, min_confidence=0.7, min_region_size=100):
    """
    Extract point prompts from CLIP predictions.

    Args:
        seg_map: (H, W) predicted class indices
        probs: (H, W, num_classes) probabilities
        vocabulary: List of class names
        min_confidence: Minimum confidence to consider a region
        min_region_size: Minimum pixel area for a region

    Returns:
        List of (point, label) tuples where point is (x, y) and label is class_idx
    """
    H, W = seg_map.shape
    prompts = []

    print("\nExtracting prompt points from CLIP predictions...")

    for class_idx, class_name in enumerate(vocabulary):
        # Get high-confidence regions for this class
        class_mask = (seg_map == class_idx)
        class_confidence = probs[:, :, class_idx]
        high_conf_mask = (class_mask & (class_confidence > min_confidence))

        # Find connected components
        labeled_regions, num_regions = label(high_conf_mask)

        if num_regions == 0:
            continue

        print(f"  {class_name}: found {num_regions} high-confidence regions")

        # For each region, extract centroid as prompt point
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Get centroid
            y_coords, x_coords = np.where(region_mask)
            centroid_x = int(x_coords.mean())
            centroid_y = int(y_coords.mean())

            # Get confidence at centroid
            confidence = class_confidence[centroid_y, centroid_x]

            prompts.append({
                'point': (centroid_x, centroid_y),
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': float(confidence),
                'region_size': int(region_size)
            })

    print(f"\nTotal prompt points extracted: {len(prompts)}")
    return prompts


def segment_with_guided_prompts(image, prompts, checkpoint_path=None, model_cfg=None, device=None,
                                seg_map=None, vocabulary=None):
    """
    Segment image using CLIP-guided prompts.

    Args:
        image: (H, W, 3) RGB image
        prompts: List of prompt dictionaries with 'point', 'class_idx', etc.
        seg_map: (H, W) CLIP dense prediction for re-classification (optional)
        vocabulary: List of class names for re-classification (optional)

    Returns:
        List of segmentation results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        checkpoint_path = "checkpoints/sam2_hiera_large.pt"

    if model_cfg is None:
        model_cfg = "sam2_hiera_l.yaml"

    print(f"\nLoading SAM 2 model from {checkpoint_path}...")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print("Setting image...")
    predictor.set_image(image)

    print(f"Generating masks for {len(prompts)} prompts...")
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

    print(f"Successfully generated {len(results)} masks")
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
    print(f"\nMerging overlapping masks (IoU threshold: {iou_threshold})...")

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

            # Calculate IoU (ensure boolean masks)
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

    print(f"Kept {len(kept_results)}/{len(results)} masks after merging")
    return kept_results


def visualize_results(image, results, vocabulary, output_path=None):
    """Visualize segmentation results."""
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    # Create color map for classes
    np.random.seed(42)
    class_colors = {i: np.random.random(3) for i in range(len(vocabulary))}

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
        ax.imshow(np.dstack((img_overlay, mask * 0.5)))

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

    ax.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"\nSaved visualization to {output_path}")
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
        description="CLIP-guided SAM segmentation (optimized)"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--vocabulary", nargs='+', required=True,
                       help="List of class names for CLIP")
    parser.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt",
                       help="Path to SAM 2 checkpoint")
    parser.add_argument("--model-cfg", default="sam2_hiera_l.yaml",
                       help="Model configuration")
    parser.add_argument("--output", default="clip_guided_segments.png",
                       help="Output visualization path")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum CLIP confidence for prompts")
    parser.add_argument("--min-region-size", type=int, default=100,
                       help="Minimum region size (pixels) for prompts")
    parser.add_argument("--iou-threshold", type=float, default=0.8,
                       help="IoU threshold for merging overlapping masks")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)

    # Step 1: Get CLIP dense predictions (fast)
    print("\n" + "="*50)
    print("STEP 1: CLIP Dense Prediction")
    print("="*50)

    segmentor = SCLIPSegmentor(
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        use_sam=False,
        use_pamr=False,
        verbose=True
    )

    seg_map, logits = segmentor.predict_dense(image, args.vocabulary, return_logits=True)
    probs = torch.softmax(logits, dim=0).cpu().numpy()  # (num_classes, H, W)
    probs = probs.transpose(1, 2, 0)  # (H, W, num_classes)

    # Step 2: Extract prompt points from CLIP predictions
    print("\n" + "="*50)
    print("STEP 2: Extract Prompt Points")
    print("="*50)

    prompts = extract_prompt_points_from_clip(
        seg_map, probs, args.vocabulary,
        min_confidence=args.min_confidence,
        min_region_size=args.min_region_size
    )

    print(f"\nUsing {len(prompts)} prompts vs {64*64}=4096 in blind grid!")
    if len(prompts) > 0:
        print(f"Speedup: {4096/len(prompts):.1f}x fewer SAM queries")
    else:
        print("WARNING: No prompts found! Try lowering --min-confidence")

    # Step 3: Segment with guided prompts
    if len(prompts) == 0:
        print("\nNo prompts to process. Exiting.")
        return

    print("\n" + "="*50)
    print("STEP 3: SAM Segmentation")
    print("="*50)

    results = segment_with_guided_prompts(
        image, prompts,
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device
    )

    # Step 4: Merge overlapping masks
    results = merge_overlapping_masks(results, iou_threshold=args.iou_threshold)

    # Visualize
    visualize_results(image, results, args.vocabulary, output_path=args.output)

    # Statistics
    print_statistics(results, args.vocabulary)


if __name__ == "__main__":
    main()
