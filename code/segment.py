#!/usr/bin/env python3
"""
Simple segmentation script - works on GTX 1060 6GB
No Stable Diffusion loaded, so no memory issues!

Usage:
    python segment.py --image photo.jpg --prompt "car"
    python segment.py --image photo.jpg --prompt "person" --top-k 5
    python segment.py --image photo.jpg --prompt "dog" --visualize
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import time
import torch

from models.sam2_segmentation import SAM2MaskGenerator
from models.clip_features import CLIPFeatureExtractor
from models.mask_alignment import MaskTextAligner


def main():
    parser = argparse.ArgumentParser(description="GTX 1060-friendly segmentation")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt (e.g., 'car', 'person')")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top results (default: 5)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Save visualization")
    parser.add_argument("--output", "-o", default="output", help="Output directory")

    args = parser.parse_args()

    print("="*70)
    print("Open-Vocabulary Segmentation (GTX 1060 Optimized)")
    print("="*70)
    print(f"Image:   {args.image}")
    print(f"Prompt:  '{args.prompt}'")
    print(f"Top-K:   {args.top_k}")
    print("="*70)
    print()

    # Load models (NO Stable Diffusion!)
    print("Loading models...")
    t0 = time.time()
    sam = SAM2MaskGenerator(model_type='sam2_hiera_base_plus', device='cuda')
    clip = CLIPFeatureExtractor(device='cuda')
    aligner = MaskTextAligner(clip)
    print(f"✓ Models loaded ({time.time()-t0:.1f}s)")
    print()

    # Load image
    print("Loading image...")
    image = np.array(Image.open(args.image).convert('RGB'))
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    print()

    # Generate masks
    print(f"Segmenting '{args.prompt}'...")
    print("  Stage 1: SAM 2 mask generation...")
    t0 = time.time()
    all_masks = sam.generate_masks(image)
    t_sam = time.time() - t0
    print(f"    ✓ Generated {len(all_masks)} masks ({t_sam:.2f}s)")

    filtered = sam.filter_by_size(all_masks, min_area=1024)
    print(f"    ✓ Filtered to {len(filtered)} masks (min: 32x32 px)")
    print()

    # Align with text
    print("  Stage 2: CLIP alignment...")
    t0 = time.time()
    scored, vis_data = aligner.align_masks_with_text(
        filtered,
        args.prompt,
        image,
        top_k=args.top_k,
        return_similarity_maps=args.visualize
    )
    t_clip = time.time() - t0
    print(f"    ✓ Found {len(scored)} matches ({t_clip:.2f}s)")
    print()

    # Results
    if scored:
        print("="*70)
        print(f"Top {len(scored)} Results:")
        print("="*70)
        for i, sm in enumerate(scored, 1):
            print(f"Rank #{i}:")
            print(f"  Score:       {sm.final_score:.4f}")
            print(f"  Similarity:  {sm.similarity_score:.4f}")
            print(f"  Background:  {sm.background_score:.4f}")
            print(f"  Area:        {sm.mask_candidate.area:,} pixels")
            print(f"  BBox:        {sm.mask_candidate.bbox}")
            print(f"  IoU:         {sm.mask_candidate.predicted_iou:.3f}")
            print()
    else:
        print("="*70)
        print("No matches found!")
        print("="*70)
        print("Try:")
        print("  - Different prompt matching image content")
        print("  - Lower threshold in models/mask_alignment.py")
        print("  - Different image with clear objects")
        print()

    # Save visualization
    if args.visualize and scored:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        # Visualize top masks
        import cv2
        vis = aligner.visualize_scored_masks(image, scored, max_display=args.top_k)

        output_path = output_dir / f"segmentation_{args.prompt.replace(' ', '_')}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved visualization: {output_path}")
        print()

    # Performance summary
    print("="*70)
    print("Performance Summary")
    print("="*70)
    print(f"  SAM 2:      {t_sam:.2f}s")
    print(f"  CLIP:       {t_clip:.2f}s")
    print(f"  Total:      {t_sam + t_clip:.2f}s")
    print()

    # Memory info
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory: {mem_allocated:.2f} GB / {mem_total:.2f} GB")
        print(f"  Usage:      {100*mem_allocated/mem_total:.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()
