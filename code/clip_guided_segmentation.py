"""
CLIP-guided SAM segmentation for images and videos.
Uses CLIP dense predictions to intelligently place SAM prompts instead of blind grid.

Strategy (Images):
1. Run CLIP dense prediction first (fast)
2. Extract high-confidence regions for each class
3. Use those as point prompts for SAM (much fewer prompts than 64x64 grid)
4. Get high-quality SAM masks only where CLIP says objects are

Strategy (Videos):
1. Extract first frame and run CLIP dense prediction
2. Get prompt points for detected objects
3. Use SAM2 video predictor to track across all frames
4. Generate segmented video output
"""

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.sclip_segmentor import SCLIPSegmentor
from models.video_segmentation import CLIPGuidedVideoSegmentor
from scipy.ndimage import label, center_of_mass
import cv2
import os
import sys


def generate_distinct_colors(n):
    """
    Generate visually distinct colors using a curated palette.

    Args:
        n: Number of colors needed

    Returns:
        List of RGB tuples in range [0, 1]
    """
    # Curated palette of visually distinct colors (RGB in 0-1 range)
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
        (1.00, 0.65, 0.00),  # Gold
        (0.20, 0.80, 0.20),  # Lime
        (0.80, 0.00, 0.80),  # Magenta
        (0.00, 0.50, 0.50),  # Teal
        (0.50, 0.00, 0.00),  # Maroon
        (0.00, 0.00, 0.50),  # Navy
    ]

    if n <= len(base_palette):
        return base_palette[:n]

    # If we need more colors, generate additional ones using HSV
    colors = list(base_palette)
    for i in range(n - len(base_palette)):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio for spacing
        saturation = 0.6 + (i % 3) * 0.15
        value = 0.7 + (i % 2) * 0.2

        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors


def is_video_file(file_path):
    """Check if file is a video."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return Path(file_path).suffix.lower() in video_extensions


def extract_first_frame(video_path):
    """Extract first frame from video as numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read first frame from: {video_path}")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def load_image(image_path):
    """Load image from path."""
    image = Image.open(image_path)
    return np.array(image.convert("RGB"))


def extract_prompt_points_from_clip(seg_map, probs, vocabulary, min_confidence=0.7, min_region_size=100,
                                    points_per_cluster=1, negative_points_per_cluster=0,
                                    negative_confidence_threshold=0.8):
    """
    Extract point prompts from CLIP predictions, including optional negative (background) points.

    Args:
        seg_map: (H, W) predicted class indices
        probs: (H, W, num_classes) probabilities
        vocabulary: List of class names
        min_confidence: Minimum confidence to consider a region
        min_region_size: Minimum pixel area for a region
        points_per_cluster: Number of positive points to extract per cluster (1=centroid only, >1=multiple points)
        negative_points_per_cluster: Number of negative (background) points per cluster (0=disabled)
        negative_confidence_threshold: Minimum confidence for regions to extract negative points from

    Returns:
        List of prompt dictionaries with 'point', 'class_idx', 'class_name', 'confidence', 'region_size', and 'negative_points'
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

        # For each region, extract point(s) as prompt point(s)
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Get all points in the region
            y_coords, x_coords = np.where(region_mask)

            # Extract negative (background) points if enabled
            negative_points = []
            if negative_points_per_cluster > 0:
                # Find high-confidence regions of OTHER classes (not this class)
                # These serve as negative examples to help SAM distinguish foreground from background
                negative_mask = np.zeros((H, W), dtype=bool)

                for other_idx in range(len(vocabulary)):
                    if other_idx == class_idx:
                        continue  # Skip current class

                    other_confidence = probs[:, :, other_idx]
                    other_high_conf = (seg_map == other_idx) & (other_confidence > negative_confidence_threshold)
                    negative_mask |= other_high_conf

                # Only consider negative points that are outside the current region
                # and have a minimum distance from the positive region
                from scipy.ndimage import distance_transform_edt

                # Calculate distance from positive region
                distance_from_region = distance_transform_edt(~region_mask)

                # Negative points should be at least some distance away (e.g., 10 pixels)
                min_distance = 10
                valid_negative_mask = negative_mask & (distance_from_region > min_distance)

                if valid_negative_mask.sum() > 0:
                    neg_y, neg_x = np.where(valid_negative_mask)

                    if len(neg_x) > 0:
                        # Sample negative points with spatial diversity
                        num_neg_points = min(negative_points_per_cluster, len(neg_x))

                        if num_neg_points > 0:
                            # Select negative points spread out spatially
                            # Start with the point farthest from the positive region
                            neg_distances = distance_from_region[neg_y, neg_x]

                            # Sort by distance (farthest first)
                            sorted_neg_indices = np.argsort(neg_distances)[::-1]

                            # Select points with spatial diversity
                            neg_coords = np.stack([neg_x, neg_y], axis=1)

                            # Start with farthest point
                            first_idx = sorted_neg_indices[0]
                            negative_points.append((int(neg_x[first_idx]), int(neg_y[first_idx])))

                            # Select additional points with maximum distance from already selected
                            for _ in range(num_neg_points - 1):
                                max_min_dist = -1
                                best_idx = -1

                                for idx in sorted_neg_indices:
                                    point = neg_coords[idx]
                                    # Calculate minimum distance to already selected negative points
                                    min_dist = min(
                                        np.sqrt((point[0] - np[0])**2 + (point[1] - np[1])**2)
                                        for np in negative_points
                                    )

                                    if min_dist > max_min_dist:
                                        max_min_dist = min_dist
                                        best_idx = idx

                                if best_idx >= 0:
                                    negative_points.append((int(neg_coords[best_idx][0]), int(neg_coords[best_idx][1])))
                                    # Remove selected point from consideration
                                    sorted_neg_indices = sorted_neg_indices[sorted_neg_indices != best_idx]

            if points_per_cluster == 1:
                # Extract only centroid (default behavior)
                centroid_x = int(x_coords.mean())
                centroid_y = int(y_coords.mean())
                confidence = class_confidence[centroid_y, centroid_x]

                prompts.append({
                    'point': (centroid_x, centroid_y),
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'region_size': int(region_size),
                    'negative_points': negative_points
                })
            else:
                # Extract multiple points using k-means or spatial sampling
                num_points = min(points_per_cluster, len(x_coords))

                if num_points <= 1:
                    # Fallback to centroid if region too small
                    centroid_x = int(x_coords.mean())
                    centroid_y = int(y_coords.mean())
                    confidence = class_confidence[centroid_y, centroid_x]

                    prompts.append({
                        'point': (centroid_x, centroid_y),
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'region_size': int(region_size),
                        'negative_points': negative_points
                    })
                else:
                    # Sample points evenly across the region
                    # Use stratified sampling based on confidence
                    region_confidences = class_confidence[y_coords, x_coords]

                    # Sort by confidence and select top points spread across region
                    sorted_indices = np.argsort(region_confidences)[::-1]

                    # Select points with spatial diversity
                    selected_points = []
                    coords = np.stack([x_coords, y_coords], axis=1)

                    # Always include centroid as first point
                    centroid_x = int(x_coords.mean())
                    centroid_y = int(y_coords.mean())
                    selected_points.append((centroid_x, centroid_y))

                    # Select additional points with maximum distance from already selected
                    remaining_points = num_points - 1
                    for _ in range(remaining_points):
                        max_min_dist = -1
                        best_idx = -1

                        for idx in sorted_indices:
                            point = coords[idx]
                            # Calculate minimum distance to already selected points
                            min_dist = min(
                                np.sqrt((point[0] - sp[0])**2 + (point[1] - sp[1])**2)
                                for sp in selected_points
                            )

                            if min_dist > max_min_dist:
                                max_min_dist = min_dist
                                best_idx = idx

                        if best_idx >= 0:
                            selected_points.append((int(coords[best_idx][0]), int(coords[best_idx][1])))
                            # Remove selected point from consideration
                            sorted_indices = sorted_indices[sorted_indices != best_idx]

                    # Add all selected points as separate prompts
                    for point in selected_points:
                        confidence = class_confidence[point[1], point[0]]
                        prompts.append({
                            'point': point,
                            'class_idx': class_idx,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'region_size': int(region_size),
                            'negative_points': negative_points
                        })

    print(f"\nTotal prompt points extracted (before NMS): {len(prompts)}")
    return prompts


def apply_nms_to_prompts(prompts, nms_threshold=30, max_prompts_per_class=None):
    """
    Apply Non-Maximum Suppression to prompts to remove redundant nearby points.

    Args:
        prompts: List of prompt dicts with 'point', 'class_idx', 'confidence'
        nms_threshold: Minimum distance (pixels) between prompts of same class
        max_prompts_per_class: Maximum prompts to keep per class (keeps highest confidence)

    Returns:
        Filtered list of prompts
    """
    if not prompts:
        return prompts

    # Group prompts by class
    prompts_by_class = {}
    for p in prompts:
        class_idx = p['class_idx']
        if class_idx not in prompts_by_class:
            prompts_by_class[class_idx] = []
        prompts_by_class[class_idx].append(p)

    # Apply NMS per class
    filtered_prompts = []

    for class_idx, class_prompts in prompts_by_class.items():
        # Sort by confidence (highest first)
        class_prompts = sorted(class_prompts, key=lambda p: p['confidence'], reverse=True)

        # Apply max_prompts_per_class limit
        if max_prompts_per_class is not None:
            class_prompts = class_prompts[:max_prompts_per_class]

        # NMS: keep prompts that are far enough from already selected ones
        selected = []
        for prompt in class_prompts:
            px, py = prompt['point']

            # Check distance to all already selected prompts of this class
            too_close = False
            for sel in selected:
                sx, sy = sel['point']
                dist = np.sqrt((px - sx)**2 + (py - sy)**2)

                if dist < nms_threshold:
                    too_close = True
                    break

            if not too_close:
                selected.append(prompt)

        filtered_prompts.extend(selected)

    print(f"After NMS: {len(filtered_prompts)} prompts (removed {len(prompts) - len(filtered_prompts)} redundant)")
    return filtered_prompts


def segment_with_guided_prompts(image, prompts, checkpoint_path=None, model_cfg=None, device=None,
                                seg_map=None, probs=None, vocabulary=None, use_hybrid_voting=True):
    """
    Segment image using CLIP-guided prompts.

    Args:
        image: (H, W, 3) RGB image
        prompts: List of prompt dictionaries with 'point', 'class_idx', etc.
        seg_map: (H, W) CLIP dense prediction for hybrid voting (optional)
        probs: (H, W, num_classes) CLIP probabilities for hybrid voting (optional)
        vocabulary: List of class names for hybrid voting (optional)
        use_hybrid_voting: If True and seg_map/probs provided, use hybrid voting policy

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
        negative_points = prompt_info.get('negative_points', [])

        # Convert to numpy array format SAM expects
        # Combine positive and negative points
        all_coords = [point]
        all_labels = [1]  # 1 = foreground point

        # Add negative points if present
        if negative_points:
            for neg_point in negative_points:
                all_coords.append(neg_point)
                all_labels.append(0)  # 0 = background point

        point_coords = np.array(all_coords)
        point_labels = np.array(all_labels)

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

        # Assign class using hybrid voting if enabled and data available
        prompt_class = prompt_info['class_idx']
        assigned_class = prompt_class
        correction_info = None

        if use_hybrid_voting and seg_map is not None and probs is not None and vocabulary is not None:
            assigned_class, correction_info = assign_class_hybrid_voting(
                mask, prompt_class, seg_map, probs, vocabulary
            )

            # Log corrections for debugging
            if correction_info is not None:
                print(f"  [VOTING CORRECTION] Mask {i+1}: "
                      f"{correction_info['from_name']} → {correction_info['to_name']} "
                      f"(conf: {correction_info['conf_ratio']:.2f}x, "
                      f"coverage: {correction_info['coverage']:.1%})")

        results.append({
            'mask': mask,
            'class_idx': assigned_class,
            'class_name': vocabulary[assigned_class] if vocabulary else prompt_info['class_name'],
            'confidence': prompt_info['confidence'],
            'sam_score': float(score),
            'region_size': prompt_info['region_size'],
            'prompt_point': point,
            'original_class': prompt_class if correction_info else None,
            'voting_correction': correction_info
        })

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} masks...")

    print(f"Successfully generated {len(results)} masks")
    return results


def assign_class_hybrid_voting(mask, prompt_class, seg_map, probs, vocabulary,
                               conf_threshold=1.2, coverage_threshold=0.25, agreement_threshold=0.6):
    """
    Hybrid voting policy: Conservative but corrects obvious CLIP errors.

    Strategy:
    1. Start with prompt_class (baseline guarantee)
    2. Calculate confidence-weighted winner from all pixels in mask
    3. Only switch if there's STRONG evidence (3 conditions must hold)

    This prevents degrading quality while fixing obvious mistakes like "bear → hair drier".

    Args:
        mask: SAM mask (H, W) boolean array
        prompt_class: Class index from the prompt
        seg_map: CLIP argmax predictions (H, W)
        probs: CLIP probabilities (H, W, num_classes)
        vocabulary: List of class names
        conf_threshold: Ratio of confidence required to switch (default 1.2 = 20% better)
        coverage_threshold: Min pixel coverage for new class (default 0.25 = 25%)
        agreement_threshold: Max agreement with prompt class to allow switch (default 0.6)

    Returns:
        assigned_class: Class index to assign to this mask
        correction_info: Dict with debugging info (or None if no correction)
    """
    # Baseline: use prompt class
    assigned_class = prompt_class
    correction_info = None

    # Ensure mask is boolean for indexing
    mask_bool = mask.astype(bool)

    # Extract predictions inside mask
    masked_seg = seg_map[mask_bool]
    unique_classes = np.unique(masked_seg)

    if len(unique_classes) == 1:
        # Homogeneous mask, no correction needed
        return assigned_class, None

    # Calculate average confidence for each class in mask
    class_confidences = {}
    class_pixel_counts = {}

    for class_idx in unique_classes:
        # Pixels where this class is argmax
        class_pixels_mask = (masked_seg == class_idx)
        pixel_count = class_pixels_mask.sum()

        # Get coordinates of these pixels in original image
        mask_coords = np.argwhere(mask_bool)
        class_mask_indices = np.where(class_pixels_mask)[0]
        class_coords = mask_coords[class_mask_indices]

        # Calculate average confidence for this class at those pixels
        confidences = [probs[y, x, class_idx] for y, x in class_coords]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        class_confidences[class_idx] = avg_confidence
        class_pixel_counts[class_idx] = pixel_count

    # Handle empty mask case (no valid pixels)
    if not class_confidences:
        return assigned_class, None

    # Find confidence-weighted winner
    conf_winner = max(class_confidences.keys(), key=lambda k: class_confidences[k])

    # If winner is same as prompt, no correction needed
    if conf_winner == prompt_class:
        return assigned_class, None

    # Calculate metrics for decision
    prompt_conf = class_confidences.get(prompt_class, 0.0)
    winner_conf = class_confidences[conf_winner]

    # Avoid division by zero
    if prompt_conf < 1e-6:
        conf_ratio = float('inf')
    else:
        conf_ratio = winner_conf / prompt_conf

    pixel_coverage = class_pixel_counts[conf_winner] / mask_bool.sum()
    agreement_with_prompt = class_pixel_counts.get(prompt_class, 0) / mask_bool.sum()

    # Decision: Switch only if ALL three conditions hold
    should_switch = (
        conf_ratio > conf_threshold and
        pixel_coverage > coverage_threshold and
        agreement_with_prompt < agreement_threshold
    )

    if should_switch:
        assigned_class = conf_winner
        correction_info = {
            'from_class': prompt_class,
            'from_name': vocabulary[prompt_class],
            'to_class': conf_winner,
            'to_name': vocabulary[conf_winner],
            'conf_ratio': conf_ratio,
            'coverage': pixel_coverage,
            'agreement': agreement_with_prompt,
            'from_confidence': prompt_conf,
            'to_confidence': winner_conf
        }

    return assigned_class, correction_info


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

    # Create color map for classes using distinct colors
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

    # Add legend with color-to-class mapping
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
        print(f"\nSaved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_filtered_results(image, results, target_class, vocabulary, output_path=None, edit_style="segment"):
    """
    Visualize only masks corresponding to a specific target class.

    Args:
        image: (H, W, 3) RGB image
        results: List of result dictionaries with 'mask', 'class_idx', etc.
        target_class: str, the target class name to filter for
        vocabulary: List of class names
        output_path: Optional path to save the visualization
        edit_style: str, one of "segment", "replace", "remove"
            - "segment": show segmentation overlays (default)
            - "replace": show white masks on black background
            - "remove": show image with target class removed (black)
    """
    # Filter results for target class
    filtered_results = [r for r in results if r['class_name'] == target_class]

    if len(filtered_results) == 0:
        print(f"\nWARNING: No masks found for target class '{target_class}'")
        return

    print(f"\nFiltered {len(filtered_results)} masks for target class '{target_class}'")

    # Combine all filtered masks
    H, W = image.shape[:2]
    combined_mask = np.zeros((H, W), dtype=bool)
    for result in filtered_results:
        combined_mask |= result['mask'].astype(bool)

    if edit_style == "segment":
        # Show segmentation overlay (original behavior)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)

        # Use a single color for the target class
        target_class_idx = vocabulary.index(target_class)
        distinct_colors = generate_distinct_colors(len(vocabulary))
        class_colors = {i: distinct_colors[i] for i in range(len(vocabulary))}
        target_color = class_colors[target_class_idx]

        # Sort by region size (draw larger regions first)
        sorted_results = sorted(filtered_results, key=lambda x: x['region_size'], reverse=True)

        # Draw masks
        for result in sorted_results:
            mask = result['mask']
            confidence = result['confidence']

            img_overlay = np.ones((mask.shape[0], mask.shape[1], 3))
            for i in range(3):
                img_overlay[:, :, i] = target_color[i]
            ax.imshow(np.dstack((img_overlay, mask * 0.7)))

            # Add label at prompt point
            point = result['prompt_point']
            ax.plot(point[0], point[1], 'r*', markersize=10)

            if result['region_size'] > 1000:  # Only label larger objects
                ax.text(
                    point[0], point[1],
                    f"{target_class}\n{confidence:.2f}",
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor=target_color, alpha=0.8),
                    ha='center', va='center'
                )

        ax.set_title(f"Filtered Segmentation: {target_class} only ({len(filtered_results)} instances)",
                     fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()

    elif edit_style == "replace":
        # Show white masks on black background
        fig, ax = plt.subplots(figsize=(20, 20))
        mask_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask_image[combined_mask] = [255, 255, 255]
        ax.imshow(mask_image)
        ax.set_title(f"Mask for replacement: {target_class} ({len(filtered_results)} instances)",
                     fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()

    elif edit_style == "remove":
        # Show image with target class removed (blackened)
        fig, ax = plt.subplots(figsize=(20, 20))
        removed_image = image.copy()
        removed_image[combined_mask] = [0, 0, 0]
        ax.imshow(removed_image)
        ax.set_title(f"Image with {target_class} removed ({len(filtered_results)} instances)",
                     fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"Saved filtered visualization ({edit_style} style) to {output_path}")
    else:
        plt.show()

    plt.close()

    return combined_mask


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


def print_statistics_video(video_segments, prompts):
    """Print statistics for video segmentation."""
    print("\n" + "="*50)
    print("VIDEO SEGMENTATION STATISTICS")
    print("="*50)

    # Count frames where each object appears
    num_frames = len(video_segments)
    class_names = {p['class_idx']: p['class_name'] for p in prompts}

    print(f"\nTotal frames: {num_frames}")
    print(f"Tracked objects: {len(prompts)}")

    print("\nObject tracking across frames:")
    for prompt in prompts:
        obj_id = prompt['class_idx']
        class_name = prompt['class_name']

        # Count frames where this object is present
        frames_present = sum(
            1 for frame_idx, masks in video_segments.items()
            if obj_id in masks and masks[obj_id].sum() > 0
        )

        percentage = (frames_present / num_frames * 100) if num_frames > 0 else 0
        print(f"  {class_name:20s}: {frames_present:4d}/{num_frames} frames ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-guided SAM segmentation (optimized)"
    )
    parser.add_argument("--image", required=True, help="Path to input image or video")
    parser.add_argument("--vocabulary", nargs='+', required=True,
                       help="List of class names for CLIP")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Target class name to filter and visualize (must be in vocabulary)")
    parser.add_argument("--edit", type=str, choices=["segment", "replace", "remove"],
                       default="segment",
                       help="Edit style for filtered visualization: segment (overlay), replace (mask), remove (blacken)")
    parser.add_argument("--use-inpainting", action="store_true",
                       help="Use Stable Diffusion inpainting for remove/replace modes")
    parser.add_argument("--edit-prompt", type=str, default=None,
                       help="Prompt for replace mode (e.g., 'a red sports car')")
    parser.add_argument("--square-mask", action="store_true",
                       help="Generate square/bounding box masks instead of fine-grained masks for inpainting")
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
    parser.add_argument("--points-per-cluster", type=int, default=1,
                       help="Number of points per cluster (1=centroid only, >1=multiple points)")
    parser.add_argument("--negative-points-per-cluster", type=int, default=0,
                       help="Number of negative (background) points per cluster (0=disabled)")
    parser.add_argument("--negative-confidence-threshold", type=float, default=0.8,
                       help="Minimum confidence for negative point regions")
    parser.add_argument("--iou-threshold", type=float, default=0.8,
                       help="IoU threshold for merging overlapping masks")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Validate prompt if provided
    if args.prompt and args.prompt not in args.vocabulary:
        raise ValueError(
            f"Prompt '{args.prompt}' is not in vocabulary. "
            f"Available classes: {', '.join(args.vocabulary)}"
        )

    # Check if input is video or image
    is_video = is_video_file(args.image)

    if is_video:
        print(f"Detected video input: {args.image}")

        # Check for mask generation modes
        if args.edit in ["replace", "remove"] or args.use_inpainting:
            # For these modes, we'll generate a B/W mask based on --edit-prompt
            if not args.prompt:
                print("\nERROR: --prompt is required for replace/remove/inpainting modes")
                print("Specify which object to mask using --prompt <class_name>")
                return

            print(f"\nGenerating B/W mask for '{args.prompt}' in video mode...")
            print("Note: Full video inpainting/editing is not yet supported.")
            print("This will generate a mask for the first frame.\n")

        # Extract first frame for CLIP analysis
        print("Extracting first frame for CLIP analysis...")
        image = extract_first_frame(args.image)
    else:
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
        min_region_size=args.min_region_size,
        points_per_cluster=args.points_per_cluster,
        negative_points_per_cluster=args.negative_points_per_cluster,
        negative_confidence_threshold=args.negative_confidence_threshold
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

    # Handle video vs image differently
    if is_video:
        # Check if we're in mask generation mode
        if args.edit in ["replace", "remove"] or args.use_inpainting:
            print("\n" + "="*50)
            print("STEP 3: Generate B/W Mask Video")
            print("="*50)

            # Initialize video segmentor to track object across all frames
            video_segmentor = CLIPGuidedVideoSegmentor(
                checkpoint_path=args.checkpoint,
                model_cfg=args.model_cfg,
                device=args.device
            )

            # Filter prompts to only track the target class
            filtered_prompts = [p for p in prompts if p['class_name'] == args.prompt]

            if len(filtered_prompts) == 0:
                print(f"\nWARNING: No prompts found for target class '{args.prompt}'")
                return

            print(f"\nTracking {len(filtered_prompts)} instances of '{args.prompt}' across video")

            # Generate output filename for mask video
            base_name = args.output.rsplit('.', 1)[0]
            mask_output = f"{base_name}_mask_{args.prompt}.mp4"

            # Segment video and generate mask video
            video_segments = video_segmentor.segment_video(
                video_path=args.image,
                prompts=filtered_prompts,
                output_path=mask_output,
                visualize=False  # We'll create custom mask visualization
            )

            # Create B/W mask video from segmentation results
            print("\nGenerating B/W mask video...")
            if args.square_mask:
                print("Using bounding box masks (--square-mask enabled)")
            cap = cv2.VideoCapture(args.image)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # First, save frames as temporary images, then use ffmpeg to create video
            import tempfile
            import shutil
            import subprocess

            temp_dir = tempfile.mkdtemp()
            print(f"Saving mask frames to temporary directory: {temp_dir}")

            try:
                # Generate mask frames and save as images
                # Use sequential numbering starting from 0 for ffmpeg
                for seq_idx, frame_idx in enumerate(sorted(video_segments.keys())):
                    masks = video_segments[frame_idx]

                    # Combine all masks for this frame
                    combined_mask = np.zeros((height, width), dtype=np.uint8)

                    # Debug: print first mask shape
                    if seq_idx == 0 and masks:
                        first_mask = next(iter(masks.values()))
                        print(f"Debug: First mask shape: {first_mask.shape}, dtype: {first_mask.dtype}")
                        print(f"Debug: Expected shape: (height={height}, width={width})")

                    for obj_id, mask in masks.items():
                        if mask.sum() > 0:  # Only if mask has content
                            # Handle dimension issues first
                            if len(mask.shape) == 3:
                                # If mask has shape (1, H, W) or (H, W, 1), squeeze it
                                if mask.shape[0] == 1:
                                    mask = mask[0]  # Remove leading dimension
                                elif mask.shape[2] == 1:
                                    mask = mask[:, :, 0]  # Remove trailing dimension
                                else:
                                    print(f"Warning: 3D mask with shape {mask.shape}, cannot handle")
                                    continue

                            # After squeezing, check if dimensions need transposing
                            if mask.shape == (width, height):
                                # If dimensions are swapped, transpose
                                mask = mask.T
                            elif mask.shape != (height, width):
                                print(f"Warning: mask shape {mask.shape} doesn't match expected ({height}, {width}), skipping")
                                continue

                            mask_binary = mask.astype(bool)

                            # Convert to square/bounding box mask if requested
                            if args.square_mask:
                                # Find bounding box of the mask
                                rows = np.any(mask_binary, axis=1)
                                cols = np.any(mask_binary, axis=0)
                                if rows.any() and cols.any():
                                    y_min, y_max = np.where(rows)[0][[0, -1]]
                                    x_min, x_max = np.where(cols)[0][[0, -1]]

                                    # Create square mask by filling bounding box
                                    mask_binary = np.zeros((height, width), dtype=bool)
                                    mask_binary[y_min:y_max+1, x_min:x_max+1] = True

                            mask_binary = (mask_binary.astype(np.uint8) * 255)
                            combined_mask = np.maximum(combined_mask, mask_binary)

                    # Save as PNG with sequential numbering starting from 0
                    frame_path = f"{temp_dir}/frame_{seq_idx:06d}.png"
                    success = cv2.imwrite(frame_path, combined_mask)
                    if not success:
                        print(f"Warning: Failed to write frame {seq_idx} to {frame_path}")
                    elif seq_idx == 0:
                        print(f"Debug: Successfully wrote first frame, shape: {combined_mask.shape}")

                print(f"Saved {len(video_segments)} mask frames")

                # Also create a preview video with original frames + gray overlay on tracked area
                print("\nGenerating preview video with gray overlay on tracked area...")
                preview_output = f"{base_name}_preview_{args.prompt}.mp4"
                preview_dir = tempfile.mkdtemp()

                # Read original video and apply gray overlay
                cap = cv2.VideoCapture(args.image)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count in video_segments:
                        masks = video_segments[frame_count]

                        # Combine all masks for this frame
                        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                        for obj_id, mask in masks.items():
                            # Handle 3D masks
                            if len(mask.shape) == 3 and mask.shape[0] == 1:
                                mask = mask[0]

                            if mask.sum() > 0:
                                combined_mask |= mask.astype(bool)

                        # Convert to square/bounding box mask if requested
                        if args.square_mask and combined_mask.any():
                            rows = np.any(combined_mask, axis=1)
                            cols = np.any(combined_mask, axis=0)
                            if rows.any() and cols.any():
                                y_min, y_max = np.where(rows)[0][[0, -1]]
                                x_min, x_max = np.where(cols)[0][[0, -1]]

                                # Create square mask by filling bounding box
                                combined_mask_bbox = np.zeros_like(combined_mask, dtype=bool)
                                combined_mask_bbox[y_min:y_max+1, x_min:x_max+1] = True
                                combined_mask = combined_mask_bbox

                        # Apply gray overlay (128, 128, 128) to tracked areas
                        frame_preview = frame.copy()
                        frame_preview[combined_mask] = [128, 128, 128]

                        # Save preview frame
                        preview_frame_path = f"{preview_dir}/frame_{frame_count:06d}.png"
                        cv2.imwrite(preview_frame_path, frame_preview)
                    else:
                        # No mask for this frame, save original
                        preview_frame_path = f"{preview_dir}/frame_{frame_count:06d}.png"
                        cv2.imwrite(preview_frame_path, frame)

                    frame_count += 1

                cap.release()
                print(f"Generated {frame_count} preview frames")

                # Create preview video using ffmpeg
                preview_cmd = [
                    'ffmpeg',
                    '-y',
                    '-framerate', str(fps),
                    '-i', f'{preview_dir}/frame_%06d.png',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    preview_output
                ]

                result_preview = subprocess.run(preview_cmd, capture_output=True, text=True)
                if result_preview.returncode == 0:
                    print(f"Saved preview video to {preview_output}")
                else:
                    print(f"Warning: Failed to create preview video: {result_preview.stderr}")

                # Clean up preview frames
                shutil.rmtree(preview_dir)

                # Use ffmpeg to create video from frames
                print("\nCreating B/W mask video from frames using ffmpeg...")
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file
                    '-framerate', str(fps),
                    '-i', f'{temp_dir}/frame_%06d.png',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    mask_output
                ]

                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    print(f"\nSaved B/W mask video to {mask_output}")
                else:
                    print(f"Error creating video with ffmpeg: {result.stderr}")
                    print("Trying alternative method with OpenCV...")

                    # Fallback: try OpenCV with mp4v codec
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    mask_writer = cv2.VideoWriter(mask_output, fourcc, fps, (width, height), isColor=False)

                    for seq_idx in range(len(video_segments)):
                        frame_path = f"{temp_dir}/frame_{seq_idx:06d}.png"
                        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                        if frame is not None:
                            mask_writer.write(frame)

                    mask_writer.release()
                    print(f"\nSaved B/W mask video to {mask_output} (using OpenCV fallback)")

            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary files")

            # Statistics
            print_statistics_video(video_segments, filtered_prompts)

            # Run VACE inference if edit_prompt is provided
            if args.edit_prompt:
                print("\n" + "="*50)
                print("STEP 4: VACE Video Inpainting")
                print("="*50)
                print(f"Running VACE inference with prompt: '{args.edit_prompt}'")

                # Clean up memory before running VACE
                print("\nCleaning up GPU memory before VACE inference...")
                try:
                    # Delete video segmentor to free memory
                    if 'video_segmentor' in locals():
                        del video_segmentor
                        print("  Deleted video_segmentor")

                    # Delete CLIP segmentor to free memory
                    if 'segmentor' in locals():
                        del segmentor
                        print("  Deleted CLIP segmentor")

                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print("  Cleared CUDA cache")

                    # Run garbage collection
                    import gc
                    collected = gc.collect()
                    print(f"  Garbage collected {collected} objects")

                    # Print memory stats
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"  GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                except Exception as e:
                    print(f"Warning: Error during memory cleanup: {e}")

                try:
                    import subprocess

                    # Prepare paths
                    vace_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VACE')
                    vace_script = os.path.join(vace_root, 'vace', 'vace_wan_inference.py')

                    # Prepare output paths (absolute)
                    base_name = args.output.rsplit('.', 1)[0]
                    abs_preview_output = os.path.abspath(preview_output)
                    abs_mask_output = os.path.abspath(mask_output)
                    abs_save_file = os.path.abspath(f"{base_name}_inpainted_{args.prompt}.mp4")

                    # Build VACE command
                    vace_cmd = [
                        sys.executable,  # Use same Python interpreter
                        vace_script,
                        '--model_name', 'vace-1.3B',
                        '--size', '480p',
                        '--frame_num', str(min(81, len(video_segments))),
                        '--ckpt_dir', 'models/Wan2.1-VACE-1.3B/',
                        '--src_video', abs_preview_output,
                        '--src_mask', abs_mask_output,
                        '--prompt', args.edit_prompt,
                        '--use_prompt_extend', 'plain',
                        '--base_seed', '2025',
                        '--sample_solver', 'unipc',
                        '--sample_guide_scale', '5.0',
                        '--save_file', abs_save_file,
                    ]

                    print(f"VACE output will be saved to: {abs_save_file}")
                    print("This may take several minutes depending on your hardware...")

                    # Print the exact command for debugging
                    print(f"\nRunning VACE command:")
                    print(f"  Working directory: {vace_root}")
                    print(f"  Command: {' '.join(vace_cmd)}")
                    print()

                    # Run VACE as subprocess from VACE root directory
                    result = subprocess.run(
                        vace_cmd,
                        cwd=vace_root,
                        capture_output=False,  # Show output in real-time
                        text=True
                    )

                    if result.returncode == 0:
                        print(f"\nVACE inpainting complete!")
                        print(f"Output video: {abs_save_file}")
                    else:
                        print(f"\nVACE inference failed with return code {result.returncode}")

                except Exception as e:
                    print(f"\nError during VACE inference: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\nSkipping VACE inference (no --edit-prompt provided)")
                print("To run inpainting, add --edit-prompt 'your description here'")

        else:
            # Normal video segmentation mode
            print("\n" + "="*50)
            print("STEP 3: SAM2 Video Segmentation")
            print("="*50)

            # Initialize video segmentor
            video_segmentor = CLIPGuidedVideoSegmentor(
                checkpoint_path=args.checkpoint,
                model_cfg=args.model_cfg,
                device=args.device
            )

            # Determine output path (change extension to .mp4)
            output_path = args.output
            if not output_path.endswith('.mp4'):
                output_path = output_path.rsplit('.', 1)[0] + '.mp4'

            # Segment video
            video_segments = video_segmentor.segment_video(
                video_path=args.image,
                prompts=prompts,
                output_path=output_path,
                visualize=True
            )

            print(f"\n{'='*50}")
            print("VIDEO SEGMENTATION COMPLETE!")
            print(f"{'='*50}")
            print(f"Output saved to: {output_path}")
            print(f"Tracked {len(prompts)} objects across video")

            # Statistics
            print_statistics_video(video_segments, prompts)

    else:
        # Image segmentation (original workflow)
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

        # Visualize all results
        visualize_results(image, results, args.vocabulary, output_path=args.output)

    # If prompt is specified, create filtered visualization (images only)
    if args.prompt and not is_video:
        print("\n" + "="*50)
        print(f"FILTERED VISUALIZATION: {args.prompt} (style: {args.edit})")
        print("="*50)

        # Generate filtered output filename
        base_name = args.output.rsplit('.', 1)[0]
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'png'
        filtered_output = f"{base_name}_filtered_{args.prompt}_{args.edit}.{ext}"

        combined_mask = visualize_filtered_results(image, results, args.prompt, args.vocabulary,
                                                   output_path=filtered_output, edit_style=args.edit)

        # Save binary mask for potential inpainting use
        if combined_mask is not None and args.edit in ["replace", "remove"]:
            mask_output = f"{base_name}_mask_{args.prompt}.png"

            # Convert to square/bounding box mask if requested
            mask_to_save = combined_mask.astype(bool)
            if args.square_mask:
                rows = np.any(mask_to_save, axis=1)
                cols = np.any(mask_to_save, axis=0)
                if rows.any() and cols.any():
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]

                    # Create square mask by filling bounding box
                    H, W = mask_to_save.shape
                    mask_to_save = np.zeros((H, W), dtype=bool)
                    mask_to_save[y_min:y_max+1, x_min:x_max+1] = True
                    print(f"Converted to bounding box mask: ({x_min}, {y_min}) to ({x_max}, {y_max})")

            mask_image = (mask_to_save.astype(np.uint8) * 255)
            cv2.imwrite(mask_output, mask_image)
            print(f"Saved binary mask to {mask_output}")

            # Perform Stable Diffusion inpainting if requested
            if args.use_inpainting:
                print("\n" + "="*50)
                print("STABLE DIFFUSION INPAINTING")
                print("="*50)

                try:
                    from models.inpainting import StableDiffusionInpainter

                    # Initialize Stable Diffusion
                    inpainter = StableDiffusionInpainter(
                        model_id="stabilityai/stable-diffusion-2-inpainting",
                        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
                    )

                    # Perform inpainting
                    if args.edit == "remove":
                        inpainted = inpainter.remove_object(image, mask_image)
                        inpainted_output = f"{base_name}_inpainted_removed_{args.prompt}.{ext}"
                    else:  # replace
                        if args.edit_prompt is None:
                            print("Warning: --edit-prompt not specified for replace mode")
                            replacement_prompt = f"a {args.prompt}"
                        else:
                            replacement_prompt = args.edit_prompt
                        inpainted = inpainter.replace_object(image, mask_image, replacement_prompt)
                        inpainted_output = f"{base_name}_inpainted_replaced_{args.prompt}.{ext}"

                    # Convert PIL to numpy if needed
                    if not isinstance(inpainted, np.ndarray):
                        inpainted = np.array(inpainted)

                    # Save result
                    cv2.imwrite(inpainted_output, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
                    print(f"Saved inpainted result to {inpainted_output}")

                    # Create comparison visualization
                    comparison = inpainter.compare_results(image, inpainted, mask_image)
                    comparison_output = f"{base_name}_inpainting_comparison_{args.prompt}.{ext}"
                    cv2.imwrite(comparison_output, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                    print(f"Saved comparison to {comparison_output}")

                except ImportError as e:
                    print(f"Error: Could not import Stable Diffusion inpainter: {e}")
                    print("Install with: pip install diffusers transformers accelerate")
                except Exception as e:
                    print(f"Error during inpainting: {e}")
                    import traceback
                    traceback.print_exc()

    # Statistics (for images only, video stats already printed)
    if not is_video:
        print_statistics(results, args.vocabulary)


if __name__ == "__main__":
    main()
