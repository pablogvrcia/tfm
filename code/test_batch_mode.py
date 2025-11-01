#!/usr/bin/env python3
"""
Test script for batch prompt mode on first COCO-Stuff image.

Tests the new batch processing feature using pipeline.segment_batch()
that scores all 171 classes simultaneously.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import OpenVocabSegmentationPipeline
from datasets import COCOStuffDataset, load_dataset
from benchmarks.metrics import compute_all_metrics


def test_batch_mode():
    """Test batch mode on first COCO-Stuff image with all 171 classes."""

    print("="*80)
    print("Testing Batch Prompt Mode on COCO-Stuff")
    print("="*80)
    print()

    # Load COCO-Stuff dataset
    data_dir = Path(__file__).parent / "data" / "benchmarks"
    print(f"Loading COCO-Stuff dataset from: {data_dir}")

    try:
        # dataset = COCOStuffDataset(data_dir=data_dir, split='val2017', max_samples=1)
        dataset = load_dataset('pascal-voc', data_dir)
    except Exception as e:
        print(f"Error loading COCO-Stuff dataset: {e}")
        print("Make sure COCO-Stuff is downloaded and in the correct location.")
        return

    # Load first image
    print(f"Loading first image from COCO-Stuff...")
    sample = dataset[0]
    image = sample['image']
    gt_mask = sample['mask']
    class_names = sample['class_names']
    image_id = sample['image_id']

    print(f"Image ID: {image_id}")
    print(f"Image shape: {image.shape}")
    print(f"Total classes in dataset: {len(class_names)}")
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=True)
    print()

    # Get unique classes present in ground truth (just for info)
    unique_classes_in_gt = np.unique(gt_mask)
    unique_classes_in_gt = unique_classes_in_gt[unique_classes_in_gt != 255]  # Remove ignore index
    unique_classes_in_gt = unique_classes_in_gt[unique_classes_in_gt != 0]  # Remove background

    print(f"Classes present in ground truth: {len(unique_classes_in_gt)}")
    print(f"Classes: {[class_names[c] for c in unique_classes_in_gt if c < len(class_names)]}")
    print()

    # Use ALL 171 classes (skip class 0 which is 'unlabeled')
    text_prompts = [class_names[i] for i in range(1, len(class_names))]

    print(f"Testing with ALL {len(text_prompts)} classes in batch mode")
    print("This tests the full COCO-Stuff vocabulary!")
    print()

    # Test batch mode with background suppression
    print("="*80)
    print("Running BATCH MODE (with background suppression):")
    print("="*80)
    t0 = time.time()
    class_to_masks = pipeline.segment_batch(
        image,
        text_prompts,
        use_background_suppression=False,
        score_threshold=0.15,
        top_k_per_class=5  # Keep top 5 masks per class,
    )
    batch_total_time = time.time() - t0
    print(f"\nTotal time: {batch_total_time:.1f}s")
    print()

    # Build prediction mask from results
    print("Building prediction mask...")
    pred_mask = np.zeros_like(gt_mask)
    confidence_map = np.zeros(gt_mask.shape, dtype=np.float32)

    # Assign masks to prediction (highest confidence wins per pixel)
    for class_name, masks_list in class_to_masks.items():
        # Find class ID
        try:
            cls_id = class_names.index(class_name)
        except ValueError:
            continue

        for scored_mask in masks_list:
            mask = scored_mask.mask_candidate.mask
            score = scored_mask.final_score

            # Assign pixels where this mask has higher confidence
            update_mask = (mask > 0) & (score > confidence_map)
            pred_mask[update_mask] = cls_id
            confidence_map[update_mask] = score

    # Assign background to unassigned pixels
    pred_mask[confidence_map == 0] = 0

    # Analyze results
    print("="*80)
    print("Results Analysis:")
    print("="*80)
    print("Top classes by pixel count:")

    # Count pixels per class
    class_pixels = []
    for cls_id in range(1, len(class_names)):
        mask_pixels = (pred_mask == cls_id).sum()
        if mask_pixels > 0:
            percentage = 100.0 * mask_pixels / pred_mask.size
            class_pixels.append((cls_id, mask_pixels, percentage))

    # Sort by pixel count
    class_pixels.sort(key=lambda x: x[1], reverse=True)

    # Show top 15 classes
    for cls_id, mask_pixels, percentage in class_pixels[:15]:
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        num_masks = len(class_to_masks.get(class_name, []))
        print(f"  {class_name:20s}: {mask_pixels:8d} pixels ({percentage:5.2f}%) [{num_masks} masks]")

    background_pixels = (pred_mask == 0).sum()
    background_pct = 100.0 * background_pixels / pred_mask.size
    print(f"  {'background':20s}: {background_pixels:8d} pixels ({background_pct:5.2f}%)")
    print()

    # Compute metrics vs ground truth
    print("="*80)
    print("Metrics vs Ground Truth:")
    print("="*80)

    # Compute per-class IoU
    metrics = compute_all_metrics(pred_mask, gt_mask, num_classes=len(class_names))

    print(f"  Mean IoU:         {metrics['miou']:.2f}%")
    print(f"  Pixel Accuracy:   {metrics['pixel_accuracy']:.2f}%")
    print(f"  Mean F1:          {metrics['f1']:.2f}%")
    print()

    # Save visualization
    output_path = Path(__file__).parent / "output" / "batch_mode_cocostuff_test.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create color visualization with random colors
    print("Creating visualization with labels...")
    import cv2

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(class_names), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Create colored masks
    pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

    for cls_id in range(len(class_names)):
        pred_colored[pred_mask == cls_id] = colors[cls_id]
        if cls_id != 255:  # Skip ignore index
            gt_colored[gt_mask == cls_id] = colors[cls_id]

    # Create overlays
    pred_overlay = cv2.addWeighted(image, 0.5, pred_colored, 0.5, 0)
    gt_overlay = cv2.addWeighted(image, 0.5, gt_colored, 0.5, 0)

    # Add labels to prediction overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    padding = 5

    # Find representative positions for top detected classes
    for cls_id, mask_pixels, percentage in class_pixels[:20]:  # Top 20
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        # Find centroid of the class mask
        class_mask = (pred_mask == cls_id)
        if class_mask.sum() > 0:
            y_coords, x_coords = np.where(class_mask)
            centroid_y = int(y_coords.mean())
            centroid_x = int(x_coords.mean())

            # Create label text
            label_text = f"{class_name}"
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

            # Ensure label is within image bounds
            label_x = max(padding, min(centroid_x - text_size[0]//2, pred_overlay.shape[1] - text_size[0] - padding))
            label_y = max(text_size[1] + padding, min(centroid_y, pred_overlay.shape[0] - padding))

            # Draw background rectangle
            cv2.rectangle(pred_overlay,
                         (label_x - padding, label_y - text_size[1] - padding),
                         (label_x + text_size[0] + padding, label_y + padding),
                         (0, 0, 0), -1)

            # Draw text
            color_bgr = tuple(int(c) for c in colors[cls_id][::-1])  # RGB to BGR
            cv2.putText(pred_overlay, label_text, (label_x, label_y),
                       font, font_scale, color_bgr, font_thickness, cv2.LINE_AA)

    # Create legend
    legend_width = 300
    legend_height = min(800, 30 + len(class_pixels[:30]) * 25)
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    # Draw legend title
    cv2.putText(legend, "Detected Classes:", (10, 20),
               font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw legend entries
    for i, (cls_id, mask_pixels, percentage) in enumerate(class_pixels[:30]):
        y_pos = 50 + i * 25
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        # Draw color box
        color_bgr = tuple(int(c) for c in colors[cls_id][::-1])
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 5), color_bgr, -1)
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 5), (0, 0, 0), 1)

        # Draw text
        label = f"{class_name} ({percentage:.1f}%)"
        cv2.putText(legend, label, (35, y_pos),
                   font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Stack: Original | Prediction with labels | Ground Truth | Legend
    comparison = np.hstack([image, pred_overlay, gt_overlay])

    # Resize legend to match height
    if legend.shape[0] != comparison.shape[0]:
        legend_resized = cv2.resize(legend, (legend_width, comparison.shape[0]))
    else:
        legend_resized = legend

    final_image = np.hstack([comparison, legend_resized])

    Image.fromarray(final_image).save(output_path)
    print(f"Visualization saved to: {output_path}")
    print("  (Left to Right: Original | Prediction with labels | Ground Truth | Legend)")
    print()

    # Performance summary
    print("="*80)
    print("Performance Summary:")
    print("="*80)
    print(f"  Total time:       {batch_total_time:.1f}s")
    print(f"  Classes tested:   {len(text_prompts)}")
    print(f"  Classes detected: {len([m for m in class_to_masks.values() if len(m) > 0])}")
    print(f"  Total masks:      {sum(len(m) for m in class_to_masks.values())}")
    print()
    print(f"✓ Batch mode runs SAM once for all {len(text_prompts)} classes!")
    print(f"  Sequential mode would run SAM {len(text_prompts)} times = ~{len(text_prompts)}x slower")
    print()


if __name__ == "__main__":
    test_batch_mode()
