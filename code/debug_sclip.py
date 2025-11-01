#!/usr/bin/env python
"""Debug SCLIP to understand why mIoU is so low"""

import numpy as np
from datasets import COCOStuffDataset
from sclip_segmentor import SCLIPSegmentor

# Load one sample
dataset = COCOStuffDataset('data/benchmarks', 'val2017', max_samples=1)
sample = dataset[0]

print('GT mask stats:')
gt_mask = sample['mask']
unique_gt = np.unique(gt_mask)
valid_gt = unique_gt[unique_gt != 255]
print(f'  Shape: {gt_mask.shape}')
print(f'  Classes present: {len(valid_gt)}')
print(f'  Sample classes: {valid_gt[:10].tolist()}')

# Predict
print('\nRunning SCLIP prediction...')
segmentor = SCLIPSegmentor(verbose=False)
pred_mask = segmentor.segment(sample['image'], dataset.class_names)

print('\nPrediction mask stats:')
unique_pred = np.unique(pred_mask)
print(f'  Shape: {pred_mask.shape}')
print(f'  Classes predicted: {len(unique_pred)}')
print(f'  Sample classes: {unique_pred[:10].tolist()}')

# Check specific class overlap
print('\nClass overlap check:')
overlap = set(valid_gt.tolist()) & set(unique_pred.tolist())
print(f'  Classes in both GT and pred: {len(overlap)}')
print(f'  Overlapping classes: {sorted(list(overlap))[:10]}')

# Compute IoU for one overlapping class
if len(overlap) > 0:
    test_class = list(overlap)[0]
    gt_pixels = (gt_mask == test_class).sum()
    pred_pixels = (pred_mask == test_class).sum()
    intersection = ((gt_mask == test_class) & (pred_mask == test_class)).sum()
    union = ((gt_mask == test_class) | (pred_mask == test_class)).sum()
    iou = intersection / union if union > 0 else 0

    print(f'\nIoU for class {test_class} ({dataset.class_names[test_class]}):')
    print(f'  GT pixels: {gt_pixels}')
    print(f'  Pred pixels: {pred_pixels}')
    print(f'  Intersection: {intersection}')
    print(f'  Union: {union}')
    print(f'  IoU: {iou:.4f} ({iou*100:.2f}%)')

# Check what the issue might be
print('\n=== Debugging potential issues ===')
print(f'Pred mask dtype: {pred_mask.dtype}')
print(f'GT mask dtype: {gt_mask.dtype}')
print(f'Pred mask min/max: {pred_mask.min()}/{pred_mask.max()}')
print(f'GT mask min/max: {gt_mask.min()}/{gt_mask.max()}')

# Check if predictions are at original resolution
print(f'\nResolution match: {pred_mask.shape == gt_mask.shape}')
