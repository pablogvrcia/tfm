#!/usr/bin/env python3
"""Check CLIP model for projection layers."""

import open_clip
import torch

# Load model
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai",
    device='cuda'
)

print("Visual model structure:")
print(model.visual)

print("\n\nLooking for projection layers...")
if hasattr(model.visual, 'proj'):
    print(f"Found proj: {model.visual.proj}")
    if model.visual.proj is not None:
        print(f"  Shape: {model.visual.proj.shape}")

if hasattr(model.visual.transformer, 'width'):
    print(f"Transformer width: {model.visual.transformer.width}")

if hasattr(model.visual, 'width'):
    print(f"Visual width: {model.visual.width}")

if hasattr(model.visual, 'output_dim'):
    print(f"Output dim: {model.visual.output_dim}")
