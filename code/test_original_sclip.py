#!/usr/bin/env python
"""Test SCLIP's original implementation directly"""

import sys
sys.path.insert(0, '/home/pablo/aux/tfm/code/SCLIP')

import torch
import numpy as np
from PIL import Image
import clip

# Load a sample image
from datasets import COCOStuffDataset

print("Loading dataset...")
dataset = COCOStuffDataset('data/benchmarks', 'val2017', max_samples=1)
sample = dataset[0]
image = sample['image']
gt_mask = sample['mask']

print(f"Image shape: {image.shape}")
print(f"GT mask shape: {gt_mask.shape}")

# Resize to 2048px (SCLIP preprocessing)
from PIL import Image as PILImage
h, w = image.shape[:2]
scale = 2048 / max(h, w)
new_h, new_w = int(h * scale), int(w * scale)
pil_img = PILImage.fromarray(image)
pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
image_resized = np.array(pil_img)

print(f"Resized image shape: {image_resized.shape}")

# Load SCLIP's CLIP model
print("\nLoading CLIP with CSA...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _ = clip.load('ViT-B/16', device=device, jit=False)

# Preprocess image to tensor (matching SCLIP)
image_float = image_resized.astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)  # (3, H, W)
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
image_tensor = (image_tensor - mean) / std
image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

print(f"Image tensor shape: {image_tensor.shape}")

# Get text features for a few classes
from SCLIP.prompts.imagenet_template import openai_imagenet_template

test_classes = ['person', 'chair', 'table', 'wall', 'floor']
print(f"\nEncoding text for {len(test_classes)} classes...")

text_features = []
for cls_name in test_classes:
    # Use ImageNet templates
    texts = [template(cls_name) for template in openai_imagenet_template]
    tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.mean(dim=0)  # Average over templates
        feats = feats / feats.norm()
        text_features.append(feats)

text_features = torch.stack(text_features)  # (num_classes, D)
print(f"Text features shape: {text_features.shape}")

# Test on a 224x224 crop (sliding window approach)
print("\nTesting on 224x224 crop...")
crop = image_tensor[:, :, :224, :224]

with torch.no_grad():
    # Encode image with CSA
    img_features = model.encode_image(crop, return_all=True, csa=True)
    img_features = img_features[:, 1:]  # Remove CLS token
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    print(f"Image features shape: {img_features.shape}")

    # Compute similarities
    logits = img_features @ text_features.T  # (1, num_patches, num_classes)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")

    # Apply logit scaling (SCLIP uses scale=40)
    logits_scaled = logits * 40.0

    # Softmax
    probs = torch.softmax(logits_scaled, dim=-1)

    # Predicted classes
    pred_classes = probs.argmax(dim=-1)
    unique, counts = torch.unique(pred_classes, return_counts=True)

    print(f"\nPredicted classes in crop:")
    for cls_idx, count in zip(unique.tolist(), counts.tolist()):
        pct = 100 * count / pred_classes.numel()
        print(f"  {test_classes[cls_idx]}: {count} patches ({pct:.1f}%)")

print("\n✓ Original SCLIP code works!")
