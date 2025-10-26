#!/usr/bin/env python3
"""
Debug hook registration and feature extraction.
"""

import numpy as np
from PIL import Image
import torch

from models.clip_features import CLIPFeatureExtractor

# Load CLIP
print("Loading CLIP...")
clip = CLIPFeatureExtractor(device='cuda')

print(f"\nExpected layers to extract: {clip.extract_layers}")
print(f"Embed dim: {clip.embed_dim}")

# Check model structure
print("\nChecking model structure...")
if hasattr(clip.model.visual.transformer, 'resblocks'):
    num_blocks = len(clip.model.visual.transformer.resblocks)
    print(f"  Found {num_blocks} transformer blocks")
    for i in range(num_blocks):
        print(f"    Block {i}: {type(clip.model.visual.transformer.resblocks[i])}")
else:
    print("  No resblocks found!")

# Extract features
print("\nExtracting features...")
image = np.array(Image.open('photo.jpg').convert('RGB'))
global_feat, dense_features = clip.extract_image_features(image)

print(f"\nFeatures captured: {list(clip.features.keys())}")
print(f"Number of dense features returned: {len(dense_features)}")

for key, feat in clip.features.items():
    print(f"\n  {key}:")
    print(f"    Shape: {feat.shape}")
    print(f"    Embed dim: {feat.shape[-1]}")
    print(f"    Matches target? {feat.shape[-1] == clip.embed_dim}")
