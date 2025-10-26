#!/usr/bin/env python3
"""
Debug script to see actual similarity scores
"""

import numpy as np
from PIL import Image

from models.sam2_segmentation import SAM2MaskGenerator
from models.clip_features import CLIPFeatureExtractor
from models.mask_alignment import MaskTextAligner

# Load models
print("Loading models...")
sam = SAM2MaskGenerator(model_type='sam2_hiera_tiny', device='cuda')
clip = CLIPFeatureExtractor(device='cuda')
aligner = MaskTextAligner(clip)

# Load image
print("Loading image...")
image = np.array(Image.open('photo.jpg').convert('RGB'))

# Generate masks
print("Generating masks...")
all_masks = sam.generate_masks(image)
filtered = sam.filter_by_size(all_masks, min_area=1024)
print(f"Found {len(filtered)} masks")

# Score without threshold
print("\nScoring masks for 'car'...")
prompt = "car"

# Temporarily set threshold to -999 to see all scores
aligner.similarity_threshold = -999.0

scored, _ = aligner.align_masks_with_text(
    filtered,
    prompt,
    image,
    top_k=20
)

print(f"\n{'='*70}")
print(f"ALL SCORES for '{prompt}' (sorted by final_score):")
print(f"{'='*70}")

for i, sm in enumerate(scored[:20], 1):
    print(f"{i:2d}. Final: {sm.final_score:7.4f} | "
          f"Sim: {sm.similarity_score:6.4f} | "
          f"BG: {sm.background_score:6.4f} | "
          f"Area: {sm.mask_candidate.area:6d} | "
          f"IoU: {sm.mask_candidate.predicted_iou:.2f}")

print(f"\n{'='*70}")
print("Analysis:")
print(f"{'='*70}")
print(f"Highest final score: {scored[0].final_score:.4f}")
print(f"Current threshold:   0.15")
print()

if scored[0].final_score < 0.15:
    print(f"⚠ PROBLEM: Best score ({scored[0].final_score:.4f}) < threshold (0.15)")
    print(f"  Recommendation: Lower threshold to {scored[0].final_score - 0.05:.2f}")
else:
    print(f"✓ Best score ({scored[0].final_score:.4f}) > threshold (0.15)")
    print("  Should have found matches!")

print()
print("Try different prompts:")
for test_prompt in ["vehicle", "road", "blue", "car", "automobile", "ocean", "sky"]:
    aligner.similarity_threshold = -999.0
    test_scored, _ = aligner.align_masks_with_text(
        filtered[:10],  # Just check first 10 masks
        test_prompt,
        image,
        top_k=1
    )
    if test_scored:
        print(f"  '{test_prompt:12s}' -> score: {test_scored[0].final_score:.4f}")
