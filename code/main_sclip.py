"""
Main entry point for SCLIP-based Open-Vocabulary Segmentation and Editing.

This uses the extended SCLIP+SAM2 approach (Approach 2 from Chapter 2):
1. SCLIP dense prediction with CSA attention
2. SAM2 mask refinement via majority voting
3. Stable Diffusion inpainting for image editing

Usage example:
    python main_sclip.py --image photo.jpg --prompt "car" --mode replace --edit "Rayo McQueen"
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))

from sclip_segmentor import SCLIPSegmentor
from models.sam2_segmentation import SAM2MaskGenerator
from models.inpainting import StableDiffusionInpainter
from utils import save_image
import cv2


def create_combined_mask_from_sclip(
    sclip_prediction: np.ndarray,
    target_class_idx: int,
    sam_generator: SAM2MaskGenerator,
    image: np.ndarray,
    min_coverage: float = 0.6
) -> np.ndarray:
    """
    Combine SCLIP dense prediction with SAM2 masks via majority voting.

    This implements our novel SAM2 refinement layer (Chapter 2, Section 2.2.5).

    Args:
        sclip_prediction: Dense prediction from SCLIP (H, W) with class indices
        target_class_idx: Index of the target class
        sam_generator: SAM2 mask generator
        image: Original image
        min_coverage: Minimum coverage of target class to keep a mask

    Returns:
        Combined binary mask (H, W) with SAM2-refined boundaries
    """
    # Get dense SCLIP mask for target class
    sclip_mask = (sclip_prediction == target_class_idx).astype(np.uint8)

    # Generate SAM2 masks
    sam_masks = sam_generator.generate_masks(image)

    # Majority voting: keep SAM masks where >60% pixels match SCLIP prediction
    refined_masks = []
    for sam_mask in sam_masks:
        mask_array = sam_mask.mask.astype(bool)
        overlap = np.logical_and(mask_array, sclip_mask).sum()
        total = mask_array.sum()

        if total > 0:
            coverage = overlap / total
            if coverage >= min_coverage:
                refined_masks.append(sam_mask.mask)

    # Combine all refined masks
    if not refined_masks:
        # Fallback to SCLIP prediction if no SAM masks match
        return sclip_mask

    combined_mask = np.zeros_like(sclip_mask)
    for mask in refined_masks:
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask.astype(np.uint8)


def visualize_sclip_segmentation(
    image: np.ndarray,
    sclip_prediction: np.ndarray,
    class_names: list,
    target_class: str
) -> np.ndarray:
    """Create visualization of SCLIP segmentation."""
    # Create colored segmentation map
    h, w = sclip_prediction.shape
    vis = image.copy()

    # Find target class index
    target_idx = class_names.index(target_class)

    # Create mask for target class
    mask = (sclip_prediction == target_idx)

    # Overlay red color on target class
    overlay = vis.copy()
    overlay[mask] = [255, 0, 0]  # Red

    # Blend
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="SCLIP-based Open-Vocabulary Segmentation and Editing"
    )

    # Input/output
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_sclip",
        help="Output directory"
    )

    # Operation mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["segment", "remove", "replace", "style"],
        default="segment",
        help="Operation mode"
    )

    # Prompts
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt describing target object"
    )
    parser.add_argument(
        "--edit", "-e",
        type=str,
        help="Edit prompt (for replace/style modes)"
    )

    # Additional vocabulary (optional)
    parser.add_argument(
        "--vocabulary",
        type=str,
        nargs="+",
        help="Additional class names (default: use prompt only)"
    )

    # Configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Computation device"
    )
    parser.add_argument(
        "--use-sam-refinement",
        action="store_true",
        default=True,
        help="Use SAM2 mask refinement (default: True)"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show visualizations"
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if args.mode in ["replace", "style"] and not args.edit:
        print(f"Error: --edit required for {args.mode} mode")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("\n" + "="*70)
    print("SCLIP-based Open-Vocabulary Segmentation (Extended with SAM2)")
    print("Approach 2: Dense Prediction + Novel SAM2 Refinement")
    print("="*70 + "\n")

    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    image_np = np.array(image)

    # Build vocabulary
    if args.vocabulary:
        class_names = list(args.vocabulary)
        if args.prompt not in class_names:
            class_names.append(args.prompt)
    else:
        # Use prompt + background
        class_names = ["background", args.prompt]

    print(f"Vocabulary: {class_names}")
    print(f"Target class: '{args.prompt}'\n")

    # Initialize SCLIP segmentor
    print("Initializing SCLIP segmentor...")
    sclip = SCLIPSegmentor(
        model_name="ViT-B/16",
        device=args.device,
        use_sam=False,  # We'll use SAM separately for refinement
        use_pamr=False,  # Disabled by default in SCLIP paper
        slide_inference=True,
        verbose=True
    )

    # Stage 1: SCLIP dense prediction
    print(f"\n{'='*70}")
    print("Stage 1: SCLIP Dense Prediction (CSA features)")
    print(f"{'='*70}\n")

    prediction, logits = sclip.predict_dense(
        image_np,
        class_names,
        return_logits=True
    )

    # Save SCLIP visualization
    if args.visualize:
        sclip_vis = visualize_sclip_segmentation(
            image_np,
            prediction,
            class_names,
            args.prompt
        )
        save_image(sclip_vis, output_dir / "sclip_prediction.png")
        print(f"Saved SCLIP visualization: {output_dir / 'sclip_prediction.png'}")

    # Stage 2: SAM2 mask refinement (our novel contribution)
    if args.use_sam_refinement:
        print(f"\n{'='*70}")
        print("Stage 2: SAM2 Mask Refinement (Novel Contribution)")
        print(f"{'='*70}\n")

        sam_generator = SAM2MaskGenerator(device=args.device)

        target_idx = class_names.index(args.prompt)
        refined_mask = create_combined_mask_from_sclip(
            prediction,
            target_idx,
            sam_generator,
            image_np,
            min_coverage=0.6
        )

        print(f"Refined mask coverage: {refined_mask.sum() / refined_mask.size * 100:.2f}%")

        # Save refined mask visualization
        if args.visualize:
            mask_vis = image_np.copy()
            overlay = mask_vis.copy()
            overlay[refined_mask > 0] = [0, 255, 0]  # Green
            mask_vis = cv2.addWeighted(mask_vis, 0.6, overlay, 0.4, 0)
            save_image(mask_vis, output_dir / "sam2_refined_mask.png")
            print(f"Saved SAM2 refined mask: {output_dir / 'sam2_refined_mask.png'}")
    else:
        # Use SCLIP prediction directly
        target_idx = class_names.index(args.prompt)
        refined_mask = (prediction == target_idx).astype(np.uint8)

    # Save original and mask
    save_image(image_np, output_dir / "original.png")

    # Check if we found anything
    if refined_mask.sum() == 0:
        print(f"\nWarning: No pixels found for class '{args.prompt}'")
        print("Try:")
        print("  1. Adding more related classes with --vocabulary")
        print("  2. Using a different prompt")
        sys.exit(0)

    # Stage 3: Inpainting (if editing mode)
    if args.mode != "segment":
        print(f"\n{'='*70}")
        print(f"Stage 3: Stable Diffusion Inpainting ({args.mode})")
        print(f"{'='*70}\n")

        inpainter = StableDiffusionInpainter(
            model_id="stabilityai/stable-diffusion-2-inpainting",
            device=args.device
        )

        # Convert mask to uint8 (0 or 255)
        mask_uint8 = (refined_mask * 255).astype(np.uint8)

        # Perform inpainting
        if args.mode == "remove":
            edited = inpainter.remove_object(image_np, mask_uint8)
            print("Object removed via inpainting")

        elif args.mode == "replace":
            edited = inpainter.replace_object(
                image_np,
                mask_uint8,
                args.edit
            )
            print(f"Object replaced with: '{args.edit}'")

        elif args.mode == "style":
            edited = inpainter.style_transfer(
                image_np,
                mask_uint8,
                args.edit
            )
            print(f"Style transferred: '{args.edit}'")

        # Save edited image
        save_image(np.array(edited), output_dir / "edited.png")
        print(f"Saved edited image: {output_dir / 'edited.png'}")

        # Create comparison
        if args.visualize:
            comparison = inpainter.compare_results(
                image_np,
                edited,
                mask_uint8
            )
            save_image(comparison, output_dir / "comparison.png")
            print(f"Saved comparison: {output_dir / 'comparison.png'}")

    print(f"\n{'='*70}")
    print("Pipeline Complete!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}/")

    if args.mode == "segment":
        print("\nGenerated files:")
        print("  - original.png: Input image")
        print("  - sclip_prediction.png: Dense SCLIP segmentation")
        if args.use_sam_refinement:
            print("  - sam2_refined_mask.png: SAM2-refined mask")
    else:
        print("\nGenerated files:")
        print("  - original.png: Input image")
        print("  - sclip_prediction.png: Dense SCLIP segmentation")
        if args.use_sam_refinement:
            print("  - sam2_refined_mask.png: SAM2-refined mask")
        print("  - edited.png: Final edited image")
        print("  - comparison.png: Side-by-side comparison")

    print("\nDone!\n")


if __name__ == "__main__":
    main()
