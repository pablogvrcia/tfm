#!/usr/bin/env python3
"""Debug script to check dimension mismatch."""

import torch
import numpy as np
from PIL import Image
from models.clip_features import CLIPFeatureExtractor
from models.sam2_segmentation import SAM2MaskGenerator
from models.mask_alignment import MaskTextAligner

print("Testing dimension compatibility...\n")

# Initialize components
clip = CLIPFeatureExtractor(device='cpu')
sam = SAM2MaskGenerator(device='cpu', points_per_side=8)
aligner = MaskTextAligner(clip_extractor=clip)

# Create test image
test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
pil_image = Image.fromarray(test_image)

# Generate masks
print("1. Generating masks...")
masks = sam.generate_masks(test_image)
print(f"   Generated {len(masks)} masks\n")

# Extract text features
print("2. Extracting text features...")
text_emb = clip.extract_text_features('blue area')
print(f"   Text embedding shape: {text_emb.shape}\n")

# Extract image features
print("3. Extracting image features...")
global_emb, dense_features = clip.extract_image_features(pil_image)
print(f"   Global embedding shape: {global_emb.shape}")
print(f"   Dense features count: {len(dense_features)}")
for i, feat in enumerate(dense_features):
    print(f"   Layer {i}: {feat.shape}")

# Try mask alignment
print("\n4. Testing mask alignment...")
try:
    scored_masks, vis_data = aligner.align_masks_with_text(
        masks[:1],  # Just test with one mask
        'blue area',
        test_image,
        top_k=1,
        return_similarity_maps=True
    )
    print(f"   ✓ Alignment successful! Scored {len(scored_masks)} masks")
except Exception as e:
    print(f"   ✗ Alignment failed: {e}")
    import traceback
    traceback.print_exc()
