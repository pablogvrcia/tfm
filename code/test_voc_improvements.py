#!/usr/bin/env python
"""Test improvements for Pascal VOC performance"""

import numpy as np
import sys
from datasets import PASCALVOCDataset
from sclip_segmentor import SCLIPSegmentor
from benchmarks.metrics import compute_all_metrics
from models.sam2_segmentation import SAM2MaskGenerator

# Load 3 test samples
dataset = PASCALVOCDataset('data/benchmarks', 'val', max_samples=3)

print("="*80)
print("Testing Pascal VOC Improvements")
print("="*80)

# Baseline: Current SAM settings
print("\n1. Baseline (points_per_side=32, IoU=0.7)")
print("-" * 40)

segmentor_baseline = SCLIPSegmentor(
    use_sam=True,
    verbose=False
)

results_baseline = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    pred = segmentor_baseline.segment(sample['image'], dataset.class_names)
    metrics = compute_all_metrics(pred, sample['mask'], dataset.num_classes)
    results_baseline.append(metrics)
    print(f"  Sample {idx}: {metrics['miou']*100:.2f}% mIoU")

baseline_miou = np.mean([r['miou'] for r in results_baseline])
print(f"Baseline mIoU: {baseline_miou*100:.2f}%")

# Strategy 1: More granular SAM masks
print("\n2. Fine-grained SAM (points_per_side=48)")
print("-" * 40)

sam_generator_fine = SAM2MaskGenerator(
    points_per_side=48,  # More fine-grained
    pred_iou_thresh=0.65,  # Lower threshold
    stability_score_thresh=0.80
)

segmentor_fine = SCLIPSegmentor(
    use_sam=True,
    verbose=False
)
segmentor_fine.sam_generator = sam_generator_fine

results_fine = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    pred = segmentor_fine.segment(sample['image'], dataset.class_names)
    metrics = compute_all_metrics(pred, sample['mask'], dataset.num_classes)
    results_fine.append(metrics)
    print(f"  Sample {idx}: {metrics['miou']*100:.2f}% mIoU")

fine_miou = np.mean([r['miou'] for r in results_fine])
print(f"Fine-grained mIoU: {fine_miou*100:.2f}%")
print(f"Improvement: {(fine_miou - baseline_miou)*100:+.2f}%")

# Strategy 2: Ultra-fine SAM masks
print("\n3. Ultra-fine SAM (points_per_side=64)")
print("-" * 40)

sam_generator_ultra = SAM2MaskGenerator(
    points_per_side=64,  # Very fine-grained
    pred_iou_thresh=0.6,
    stability_score_thresh=0.75
)

segmentor_ultra = SCLIPSegmentor(
    use_sam=True,
    verbose=False
)
segmentor_ultra.sam_generator = sam_generator_ultra

results_ultra = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    pred = segmentor_ultra.segment(sample['image'], dataset.class_names)
    metrics = compute_all_metrics(pred, sample['mask'], dataset.num_classes)
    results_ultra.append(metrics)
    print(f"  Sample {idx}: {metrics['miou']*100:.2f}% mIoU")

ultra_miou = np.mean([r['miou'] for r in results_ultra])
print(f"Ultra-fine mIoU: {ultra_miou*100:.2f}%")
print(f"Improvement: {(ultra_miou - baseline_miou)*100:+.2f}%")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Baseline (32 points):       {baseline_miou*100:.2f}% mIoU")
print(f"Fine-grained (48 points):   {fine_miou*100:.2f}% mIoU ({(fine_miou-baseline_miou)*100:+.2f}%)")
print(f"Ultra-fine (64 points):     {ultra_miou*100:.2f}% mIoU ({(ultra_miou-baseline_miou)*100:+.2f}%)")

if fine_miou > baseline_miou:
    print(f"\n✓ Best improvement: {max(fine_miou, ultra_miou)*100:.2f}% mIoU")
else:
    print(f"\n✗ No improvement found - baseline is best")
