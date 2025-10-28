#!/usr/bin/env python3
"""
Segmentation-only test script for GTX 1060 6GB
This works within memory constraints by not loading Stable Diffusion.
"""

from models.sam2_segmentation import SAM2MaskGenerator
from models.clip_features import CLIPFeatureExtractor
from models.mask_alignment import MaskTextAligner
from PIL import Image
import numpy as np
import time

def main():
    print("="*70)
    print("Open-Vocabulary Segmentation Test (GTX 1060 optimized)")
    print("="*70)
    print()

    # Load models
    print("Loading SAM 2 (tiny model for 6GB VRAM)...")
    t0 = time.time()
    sam = SAM2MaskGenerator(model_type='sam2_hiera_base_plus', device='cuda')
    print(f"  ✓ SAM 2 loaded ({time.time()-t0:.1f}s)")

    print("Loading CLIP...")
    t0 = time.time()
    clip = CLIPFeatureExtractor(device='cuda')
    aligner = MaskTextAligner(clip)
    print(f"  ✓ CLIP loaded ({time.time()-t0:.1f}s)")
    print()

    # Load image
    print("Loading image: photo.jpg")
    image = np.array(Image.open('photo.jpg'))
    print(f"  Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print()

    # Test prompt
    prompt = "person"
    print(f"Searching for: '{prompt}'")
    print()

    # Generate masks with SAM 2
    print("Stage 1: Generating masks with SAM 2...")
    t0 = time.time()
    all_masks = sam.generate_masks(image)
    t_sam = time.time()-t0
    print(f"  ✓ Generated {len(all_masks)} mask candidates ({t_sam:.2f}s)")

    # Filter by size
    filtered_masks = sam.filter_by_size(all_masks, min_area=1024)
    print(f"  ✓ Filtered to {len(filtered_masks)} masks (min area: 32x32 pixels)")
    print()

    # Align masks with text
    print("Stage 2: Aligning masks with CLIP...")
    t0 = time.time()
    scored_masks, vis_data = aligner.align_masks_with_text(
        filtered_masks,
        prompt,
        image,
        top_k=5,
        return_similarity_maps=False
    )
    t_clip = time.time()-t0
    print(f"  ✓ Aligned {len(scored_masks)} matches ({t_clip:.2f}s)")
    print()

    # Show results
    if scored_masks:
        print(f"Top {len(scored_masks)} matches for '{prompt}':")
        print()
        for i, scored_mask in enumerate(scored_masks, 1):
            m = scored_mask.mask_candidate
            print(f"  Rank #{i}:")
            print(f"    Final score:     {scored_mask.final_score:.4f}")
            print(f"    Similarity:      {scored_mask.similarity_score:.4f}")
            print(f"    Background:      {scored_mask.background_score:.4f}")
            print(f"    Mask area:       {m.area:,} pixels")
            print(f"    Bounding box:    {m.bbox}")
            print(f"    IoU confidence:  {m.predicted_iou:.3f}")
            print()
    else:
        print(f"No matches found for '{prompt}'")
        print("Try a different prompt or lower the similarity threshold")
        print()

    # Timing summary
    print("="*70)
    print("Performance Summary")
    print("="*70)
    print(f"  SAM 2 mask generation:  {t_sam:.2f}s")
    print(f"  CLIP alignment:         {t_clip:.2f}s")
    print(f"  Total:                  {t_sam+t_clip:.2f}s")
    print()
    print("✓ Segmentation complete!")
    print()

    # Memory info
    import torch
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {mem_allocated:.2f} GB / {mem_total:.2f} GB")
        print(f"  Reserved:  {mem_reserved:.2f} GB")
        print()

if __name__ == "__main__":
    main()
