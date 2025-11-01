#!/usr/bin/env python
"""Test text feature caching speedup"""

import time
import numpy as np
from datasets import PASCALVOCDataset
from sclip_segmentor import SCLIPSegmentor

# Load 5 test samples
dataset = PASCALVOCDataset('data/benchmarks', 'val', max_samples=5)

print("="*80)
print("Testing Text Feature Caching Speedup")
print("="*80)

# Test with SAM (includes both dense SCLIP + SAM refinement)
segmentor = SCLIPSegmentor(use_sam=True, verbose=False)

times = []
for idx in range(len(dataset)):
    sample = dataset[idx]

    start = time.time()
    pred = segmentor.segment(sample['image'], dataset.class_names)
    elapsed = time.time() - start

    times.append(elapsed)
    print(f"Sample {idx}: {elapsed:.2f}s")

print("\n" + "="*80)
print("Results:")
print("="*80)
print(f"First image (no cache):  {times[0]:.2f}s")
print(f"Avg images 2-5 (cached): {np.mean(times[1:]):.2f}s")
print(f"Speedup: {times[0] / np.mean(times[1:]):.2f}x faster with cache")
print(f"\nTotal time: {sum(times):.2f}s")
print(f"Avg time per image: {np.mean(times):.2f}s")

# Check cache stats
print(f"\nCache info:")
print(f"  Cached text features: {len(segmentor._text_feature_cache)} sets")
print(f"  Classes: {len(dataset.class_names)} classes")
