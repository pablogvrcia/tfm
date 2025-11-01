#!/usr/bin/env python
"""Debug metrics computation to understand why mIoU is so low"""

import numpy as np
from datasets import COCOStuffDataset
from sclip_segmentor import SCLIPSegmentor
from benchmarks.metrics import compute_all_metrics

# Load 3 samples
dataset = COCOStuffDataset('data/benchmarks', 'val2017', max_samples=3)
segmentor = SCLIPSegmentor(verbose=False)

all_results = []

for idx in range(len(dataset)):
    print(f"\n=== Sample {idx} ===")
    sample = dataset[idx]

    # Predict
    pred_mask = segmentor.segment(sample['image'], dataset.class_names)

    # Compute metrics
    results = compute_all_metrics(pred_mask, sample['mask'], num_classes=171)

    print(f"Sample mIoU: {results['miou']:.4f} ({results['miou']*100:.2f}%)")
    print(f"Valid classes: {results['num_valid_classes']}")

    # Show non-nan per-class IoUs
    non_nan_ious = [(k, v) for k, v in results['per_class_iou'].items() if not np.isnan(v)]
    print(f"Non-nan class IoUs: {len(non_nan_ious)}")
    if len(non_nan_ious) > 0:
        print(f"  Sample values: {[f'{v:.3f}' for k, v in non_nan_ious[:5]]}")

    all_results.append(results)

# Aggregate
print("\n=== Aggregation ===")
print(f"Sample mIoUs: {[r['miou'] for r in all_results]}")
print(f"Mean of sample mIoUs: {np.mean([r['miou'] for r in all_results]):.4f}")

# Check per-class aggregation
print("\nPer-class IoU aggregation (class 0 'person'):")
class_0_ious = [r['per_class_iou'].get(0, np.nan) for r in all_results]
print(f"  Sample IoUs: {class_0_ious}")
print(f"  np.mean: {np.mean(class_0_ious)}")
print(f"  np.nanmean: {np.nanmean(class_0_ious)}")
