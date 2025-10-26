#!/usr/bin/env python3
"""Check CLIP feature resolution."""

import numpy as np
from PIL import Image
from models.clip_features import CLIPFeatureExtractor

clip = CLIPFeatureExtractor(device='cuda', extract_layers=[24])

image = np.array(Image.open('photo.jpg').convert('RGB'))
print(f"Image: {image.shape}")

_, dense_features = clip.extract_image_features(image)

for i, feat in enumerate(dense_features):
    print(f"Layer {i}: {feat.shape}")
    _, h, w = feat.shape
    print(f"  Spatial resolution: {h}×{w}")
    print(f"  Patch size in original image: {image.shape[0]/h:.1f} × {image.shape[1]/w:.1f} pixels")
