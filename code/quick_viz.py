#!/usr/bin/env python3
"""
Quick visualization script - generates a single comprehensive figure.
Faster than create_visualizations.py, good for testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path

from models.sam2_segmentation import SAM2MaskGenerator
from models.clip_features import CLIPFeatureExtractor
from models.mask_alignment import MaskTextAligner
from PIL import Image


def main():
    # Configuration
    image_path = "photo.jpg"
    text_prompt = "tree"  # Change to match your image
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Quick Visualization Generator")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Prompt: '{text_prompt}'")
    print()

    # Load models
    print("Loading models...")
    sam = SAM2MaskGenerator(model_type='sam2_hiera_base_plus', device='cuda')
    clip = CLIPFeatureExtractor(device='cuda')
    aligner = MaskTextAligner(clip)
    print()

    # Load image
    print("Loading image...")
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print()

    # Generate masks
    print("Generating masks with SAM 2...")
    all_masks = sam.generate_masks(image)
    print(f"Generated {len(all_masks)} masks")

    filtered_masks = sam.filter_by_size(all_masks, min_area=1024)
    print(f"Filtered to {len(filtered_masks)} masks")
    print()

    # Extract CLIP features
    print("Computing CLIP similarity...")
    _, dense_features = clip.extract_image_features(image)
    text_embedding = clip.extract_text_features(text_prompt)
    similarity_map = clip.compute_similarity_map(
        dense_features,
        text_embedding,
        target_size=(image.shape[0], image.shape[1])
    )
    print()

    # Align masks
    print("Aligning masks...")
    scored_masks, _ = aligner.align_masks_with_text(
        filtered_masks,
        text_prompt,
        image,
        top_k=3
    )
    print(f"Found {len(scored_masks)} matches")
    print()

    # Create comprehensive figure
    print("Creating visualization...")
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Original, SAM masks, CLIP similarity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('(a) Original Image', fontweight='bold', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    mask_vis = sam.visualize_masks(image, all_masks[:50], alpha=0.5)
    ax2.imshow(mask_vis)
    ax2.set_title(f'(b) SAM 2 Masks (50/{len(all_masks)})', fontweight='bold', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    overlay = clip.visualize_similarity_map(image, similarity_map, alpha=0.6)
    ax3.imshow(overlay)
    ax3.set_title(f'(c) CLIP Similarity: "{text_prompt}"', fontweight='bold', fontsize=12)
    ax3.axis('off')

    # Row 2: Top 3 matches
    for idx in range(3):
        ax = fig.add_subplot(gs[1, idx])

        if idx < len(scored_masks):
            scored_mask = scored_masks[idx]
            mask = scored_mask.mask_candidate.mask

            # Create overlay
            result = image.copy()
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = [255, 0, 0]
            result = cv2.addWeighted(result, 0.6, colored_mask, 0.4, 0)

            # Draw bbox
            x, y, w, h = scored_mask.mask_candidate.bbox
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)

            ax.imshow(result)
            ax.set_title(
                f'({"def"[idx]}) Rank #{idx+1}: Score={scored_mask.final_score:.3f}\n'
                f'Area={scored_mask.mask_candidate.area:,} px',
                fontweight='bold',
                fontsize=11
            )
        else:
            ax.imshow(image)
            ax.set_title(f'({"def"[idx]}) No match #{idx+1}', fontweight='bold', fontsize=11)

        ax.axis('off')

    # Overall title
    fig.suptitle(
        f'Open-Vocabulary Segmentation: "{text_prompt}"',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # Save
    output_path = output_dir / f"segmentation_{text_prompt.replace(' ', '_')}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    print(f"✓ Saved: {output_path}")
    print()

    # Print results
    if scored_masks:
        print("Top matches:")
        for i, sm in enumerate(scored_masks[:3], 1):
            print(f"  #{i}: score={sm.final_score:.4f}, "
                  f"similarity={sm.similarity_score:.4f}, "
                  f"bg={sm.background_score:.4f}, "
                  f"area={sm.mask_candidate.area:,} px")
    else:
        print("No matches found. Try:")
        print("  - Different prompt")
        print("  - Lower similarity threshold")
        print("  - Different image")

    print()
    print("="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
