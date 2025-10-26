#!/usr/bin/env python3
"""
Debug background suppression to understand scoring.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.clip_features import CLIPFeatureExtractor

# Load CLIP
print("Loading CLIP...")
clip = CLIPFeatureExtractor(device='cuda')

# Load image
print("Loading image...")
image = np.array(Image.open('photo.jpg').convert('RGB'))
h, w = image.shape[:2]
print(f"Image shape: {h}x{w}")

# Extract features
print("\nExtracting features...")
_, dense_features = clip.extract_image_features(image)
text_car = clip.extract_text_features("car", use_prompt_ensemble=True)

# Compute similarity maps
print("\nComputing similarity maps...")
sim_map_car = clip.compute_similarity_map(dense_features, text_car, (h, w), "mean")
bg_map = clip.compute_background_suppression(dense_features, (h, w))

print(f"\nSimilarity to 'car':")
print(f"  Min:  {sim_map_car.min():.6f}")
print(f"  Max:  {sim_map_car.max():.6f}")
print(f"  Mean: {sim_map_car.mean():.6f}")
print(f"  Std:  {sim_map_car.std():.6f}")

print(f"\nBackground score:")
print(f"  Min:  {bg_map.min():.6f}")
print(f"  Max:  {bg_map.max():.6f}")
print(f"  Mean: {bg_map.mean():.6f}")
print(f"  Std:  {bg_map.std():.6f}")

# Check specific regions
# Sky region (top): y=50, x=643
# Car region (center): y=300, x=190
# Road region (bottom): y=500, x=643

print("\n" + "="*60)
print("Region Analysis:")
print("="*60)

regions = [
    ("Sky (top-center)", 50, w//2),
    ("Car body (center)", 300, 190),
    ("Road (bottom-center)", 500, w//2),
]

for name, y, x in regions:
    sim = sim_map_car[y, x]
    bg = bg_map[y, x]
    final = sim - 0.3 * bg  # background_weight = 0.3
    print(f"\n{name} [{y}, {x}]:")
    print(f"  Similarity to 'car':  {sim:.6f}")
    print(f"  Background score:     {bg:.6f}")
    print(f"  Final score:          {final:.6f}")

# Visualize
print("\nCreating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

im1 = axes[0, 1].imshow(sim_map_car, cmap='RdYlGn', vmin=-0.1, vmax=0.15)
axes[0, 1].set_title(f"Similarity: 'car'\nRange: [{sim_map_car.min():.3f}, {sim_map_car.max():.3f}]")
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(bg_map, cmap='RdYlGn_r', vmin=-0.05, vmax=0.15)
axes[1, 0].set_title(f"Background Score\nRange: [{bg_map.min():.3f}, {bg_map.max():.3f}]")
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0])

final_map = sim_map_car - 0.3 * bg_map
im3 = axes[1, 1].imshow(final_map, cmap='RdYlGn', vmin=-0.05, vmax=0.1)
axes[1, 1].set_title(f"Final Score (sim - 0.3*bg)\nRange: [{final_map.min():.3f}, {final_map.max():.3f}]")
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('debug_background.png', dpi=150, bbox_inches='tight')
print("✓ Saved: debug_background.png")
