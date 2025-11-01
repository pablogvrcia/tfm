"""
Main entry point for the Open-Vocabulary Semantic Segmentation Pipeline.

DEFAULT APPROACH: Dense SCLIP + SAM2 Refinement (Chapter 2, Section 2.2)
OPTIONAL: Proposal-based SAM2+CLIP (use --use-proposals flag)

Usage examples:
    # Default: Dense SCLIP + SAM2 refinement
    python main.py --image image.jpg --prompt "car" --mode replace --edit "sports car"

    # Use proposal-based approach instead (faster, better for discrete objects)
    python main.py --image image.jpg --prompt "car" --mode replace --edit "sports car" --use-proposals

    # Segmentation only
    python main.py --image image.jpg --prompt "red car" --mode segment

    # Object removal with dense approach
    python main.py --image image.jpg --prompt "person" --mode remove

    # Style transfer with extended vocabulary
    python main.py --image landscape.jpg --prompt "sky" --mode style --edit "sunset" --vocabulary sky clouds ocean
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import cv2

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import both pipelines
from pipeline import OpenVocabSegmentationPipeline, PipelineResult
from sclip_segmentor import SCLIPSegmentor
from models.sam2_segmentation import SAM2MaskGenerator
from models.inpainting import StableDiffusionInpainter
from config import PipelineConfig, get_fast_config, get_quality_config
from utils import save_image, plot_results, print_timing_summary


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
    vis = image.copy()
    target_idx = class_names.index(target_class)
    mask = (sclip_prediction == target_idx)

    overlay = vis.copy()
    overlay[mask] = [255, 0, 0]  # Red
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Open-Vocabulary Semantic Segmentation and Editing\n"
                    "Default: Dense SCLIP + SAM2 Refinement (Approach 2)\n"
                    "Optional: Proposal-based SAM2+CLIP (use --use-proposals)",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default="output",
        help="Output directory (default: output/)"
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

    # Method selection
    parser.add_argument(
        "--use-proposals",
        action="store_true",
        help="Use proposal-based approach (SAM2+CLIP) instead of default dense SCLIP+SAM2"
    )

    # Vocabulary (for dense approach)
    parser.add_argument(
        "--vocabulary",
        type=str,
        nargs="+",
        help="Additional class names for dense approach (e.g., --vocabulary sky ocean road)"
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
        "--top-k",
        type=int,
        default=5,
        help="Number of top masks to return (proposal-based only)"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive mask selection (proposal-based only)"
    )

    # Visualization
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Save visualizations"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save outputs"
    )

    args = parser.parse_args()

    # Validate inputs
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
    if args.use_proposals:
        print("Proposal-Based Segmentation (SAM2+CLIP)")
        print("Approach 1: Fast, discrete objects (Chapter 2, Section 2.1)")
    else:
        print("Dense SCLIP + SAM2 Refinement")
        print("Approach 2: Semantic scenes, stuff classes (Chapter 2, Section 2.2)")
    print("="*70 + "\n")

    # Route to appropriate pipeline
    if args.use_proposals:
        # Use proposal-based approach
        run_proposal_based(args, output_dir)
    else:
        # Use dense SCLIP + SAM2 approach (default)
        run_dense_sclip(args, output_dir)

    print(f"\n{'='*70}")
    print("Pipeline Complete!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}/")
    print("Done!\n")


def run_dense_sclip(args, output_dir: Path):
    """Run dense SCLIP + SAM2 refinement approach (default)."""
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
        use_sam=False,
        use_pamr=False,
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
    if args.visualize and not args.no_save:
        sclip_vis = visualize_sclip_segmentation(
            image_np,
            prediction,
            class_names,
            args.prompt
        )
        save_image(sclip_vis, output_dir / "sclip_prediction.png")
        print(f"Saved SCLIP visualization")

    # Stage 2: SAM2 mask refinement
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
    if args.visualize and not args.no_save:
        mask_vis = image_np.copy()
        overlay = mask_vis.copy()
        overlay[refined_mask > 0] = [0, 255, 0]  # Green
        mask_vis = cv2.addWeighted(mask_vis, 0.6, overlay, 0.4, 0)
        save_image(mask_vis, output_dir / "sam2_refined_mask.png")
        print(f"Saved SAM2 refined mask")

    # Save original
    if not args.no_save:
        save_image(image_np, output_dir / "original.png")

    # Check if we found anything
    if refined_mask.sum() == 0:
        print(f"\nWarning: No pixels found for class '{args.prompt}'")
        print("Suggestions:")
        print("  1. Add more context classes: --vocabulary sky ocean road building")
        print("  2. Try proposal-based approach: --use-proposals")
        print("  3. Try a different prompt")
        return

    # Stage 3: Inpainting (if editing mode)
    if args.mode != "segment":
        print(f"\n{'='*70}")
        print(f"Stage 3: Stable Diffusion Inpainting ({args.mode})")
        print(f"{'='*70}\n")

        inpainter = StableDiffusionInpainter(
            model_id="stabilityai/stable-diffusion-2-inpainting",
            device=args.device
        )

        mask_uint8 = (refined_mask * 255).astype(np.uint8)

        if args.mode == "remove":
            edited = inpainter.remove_object(image_np, mask_uint8)
            print("Object removed")
        elif args.mode == "replace":
            edited = inpainter.replace_object(image_np, mask_uint8, args.edit)
            print(f"Object replaced with: '{args.edit}'")
        elif args.mode == "style":
            edited = inpainter.style_transfer(image_np, mask_uint8, args.edit)
            print(f"Style transferred: '{args.edit}'")

        # Save outputs
        if not args.no_save:
            save_image(np.array(edited), output_dir / "edited.png")
            print(f"Saved edited image")

            if args.visualize:
                comparison = inpainter.compare_results(image_np, edited, mask_uint8)
                save_image(comparison, output_dir / "comparison.png")
                print(f"Saved comparison")


def run_proposal_based(args, output_dir: Path):
    """Run proposal-based SAM2+CLIP approach."""
    from pipeline import OpenVocabSegmentationPipeline

    pipeline = OpenVocabSegmentationPipeline(
        device=args.device,
        verbose=True
    )

    if args.mode == "segment":
        result = pipeline.segment(
            args.image,
            args.prompt,
            top_k=args.top_k,
            return_visualization=True,
            use_adaptive_selection=args.adaptive
        )

        # Save outputs
        if not args.no_save:
            save_image(result.original_image, output_dir / "original.png")

            visualizations = pipeline.visualize_results(result)
            if 'scored_masks' in visualizations:
                save_image(visualizations['scored_masks'], output_dir / "segmentation.png")
            if 'similarity_map' in visualizations:
                save_image(visualizations['similarity_map'], output_dir / "similarity_map.png")

        print_timing_summary(result.timing)

    else:  # Editing modes
        result = pipeline.segment_and_edit(
            args.image,
            args.prompt,
            args.mode,
            edit_prompt=args.edit,
            top_k=1,
            return_visualization=True
        )

        # Save outputs
        if not args.no_save:
            save_image(result.original_image, output_dir / "original.png")
            if result.edited_image:
                save_image(result.edited_image, output_dir / "edited.png")

            visualizations = pipeline.visualize_results(result)
            if 'edit_comparison' in visualizations:
                save_image(visualizations['edit_comparison'], output_dir / "comparison.png")

        print_timing_summary(result.timing)


if __name__ == "__main__":
    main()
