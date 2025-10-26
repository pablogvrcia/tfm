#!/usr/bin/env python3
"""
Debug CLIP feature extraction to understand why all scores are identical.
"""

import numpy as np
from PIL import Image
import torch

from models.clip_features import CLIPFeatureExtractor

# Load models
print("Loading CLIP...")
clip = CLIPFeatureExtractor(device='cuda')

# Load image
print("Loading image...")
image = np.array(Image.open('photo.jpg').convert('RGB'))
print(f"Image shape: {image.shape}")

# Extract features
print("\nExtracting image features...")
global_feat, dense_features = clip.extract_image_features(image)

print(f"\nGlobal feature shape: {global_feat.shape}")
print(f"Number of dense feature maps: {len(dense_features)}")

for i, feat in enumerate(dense_features):
    print(f"  Layer {i}: {feat.shape}")

# Extract text features
print("\nExtracting text features for 'car'...")
text_embedding = clip.extract_text_features("car", use_prompt_ensemble=True)
print(f"Text embedding shape: {text_embedding.shape}")

# Compute similarity map
print("\nComputing similarity map...")
h, w = image.shape[:2]
similarity_map = clip.compute_similarity_map(
    dense_features,
    text_embedding,
    target_size=(h, w),
    aggregation="mean"
)

print(f"Similarity map shape: {similarity_map.shape}")
print(f"Similarity map stats:")
print(f"  Min:    {similarity_map.min():.6f}")
print(f"  Max:    {similarity_map.max():.6f}")
print(f"  Mean:   {similarity_map.mean():.6f}")
print(f"  Std:    {similarity_map.std():.6f}")
print(f"  Unique values: {len(np.unique(similarity_map))}")

# Check if map is uniform
if similarity_map.std() < 0.001:
    print("\n⚠ WARNING: Similarity map is nearly uniform!")
    print("  This explains why all masks get identical scores.")
    print("\nPossible causes:")
    print("  1. Dense features not extracted properly")
    print("  2. Feature dimension mismatch")
    print("  3. Normalization issue")

    # Debug dense features
    print("\nChecking dense features...")
    for i, feat in enumerate(dense_features):
        if feat.shape[0] == text_embedding.shape[0]:
            print(f"\n  Layer {i} matches text embedding dimension!")
            print(f"    Feature shape: {feat.shape}")
            print(f"    Text shape: {text_embedding.shape}")

            # Compute similarity manually
            D, H, W = feat.shape
            feat_flat = feat.reshape(D, -1).T  # (H*W, D)
            feat_norm = torch.nn.functional.normalize(feat_flat, dim=1)
            text_norm = torch.nn.functional.normalize(text_embedding.unsqueeze(0), dim=1)

            sim = torch.matmul(text_norm, feat_norm.T).squeeze().cpu().numpy()
            print(f"    Manual similarity stats:")
            print(f"      Min:  {sim.min():.6f}")
            print(f"      Max:  {sim.max():.6f}")
            print(f"      Mean: {sim.mean():.6f}")
            print(f"      Std:  {sim.std():.6f}")
else:
    print("\n✓ Similarity map has variation")
    print(f"  This is good - scores should differ across masks")

    # Sample some locations
    print("\n  Sample values at different locations:")
    print(f"    Top-left (0,0):       {similarity_map[0, 0]:.6f}")
    print(f"    Center ({h//2},{w//2}):       {similarity_map[h//2, w//2]:.6f}")
    print(f"    Bottom-right ({h-1},{w-1}): {similarity_map[h-1, w-1]:.6f}")
