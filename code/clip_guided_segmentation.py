"""
CLIP-guided SAM segmentation for images and videos.
Uses CLIP dense predictions to intelligently place SAM prompts instead of blind grid.

Strategy (Images):
1. Run CLIP dense prediction first (fast)
2. Extract high-confidence regions for each class
3. Use those as point prompts for SAM (much fewer prompts than 64x64 grid)
4. Get high-quality SAM masks only where CLIP says objects are

Strategy (Videos):
1. Extract first frame and run CLIP dense prediction
2. Get prompt points for detected objects
3. Use SAM2 video predictor to track across all frames
4. Generate segmented video output
"""

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.sclip_segmentor import SCLIPSegmentor
from models.video_segmentation import CLIPGuidedVideoSegmentor
from scipy.ndimage import label, center_of_mass
import cv2


def is_video_file(file_path):
    """Check if file is a video."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return Path(file_path).suffix.lower() in video_extensions


def extract_first_frame(video_path):
    """Extract first frame from video as numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read first frame from: {video_path}")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def load_image(image_path):
    """Load image from path."""
    image = Image.open(image_path)
    return np.array(image.convert("RGB"))


def extract_prompt_points_from_clip(seg_map, probs, vocabulary, min_confidence=0.7, min_region_size=100):
    """
    Extract point prompts from CLIP predictions.

    Args:
        seg_map: (H, W) predicted class indices
        probs: (H, W, num_classes) probabilities
        vocabulary: List of class names
        min_confidence: Minimum confidence to consider a region
        min_region_size: Minimum pixel area for a region

    Returns:
        List of (point, label) tuples where point is (x, y) and label is class_idx
    """
    H, W = seg_map.shape
    prompts = []

    print("\nExtracting prompt points from CLIP predictions...")

    for class_idx, class_name in enumerate(vocabulary):
        # Get high-confidence regions for this class
        class_mask = (seg_map == class_idx)
        class_confidence = probs[:, :, class_idx]
        high_conf_mask = (class_mask & (class_confidence > min_confidence))

        # Find connected components
        labeled_regions, num_regions = label(high_conf_mask)

        if num_regions == 0:
            continue

        print(f"  {class_name}: found {num_regions} high-confidence regions")

        # For each region, extract centroid as prompt point
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Get centroid
            y_coords, x_coords = np.where(region_mask)
            centroid_x = int(x_coords.mean())
            centroid_y = int(y_coords.mean())

            # Get confidence at centroid
            confidence = class_confidence[centroid_y, centroid_x]

            prompts.append({
                'point': (centroid_x, centroid_y),
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': float(confidence),
                'region_size': int(region_size)
            })

    print(f"\nTotal prompt points extracted: {len(prompts)}")
    return prompts


def segment_with_guided_prompts(image, prompts, checkpoint_path=None, model_cfg=None, device=None,
                                seg_map=None, vocabulary=None):
    """
    Segment image using CLIP-guided prompts.

    Args:
        image: (H, W, 3) RGB image
        prompts: List of prompt dictionaries with 'point', 'class_idx', etc.
        seg_map: (H, W) CLIP dense prediction for re-classification (optional)
        vocabulary: List of class names for re-classification (optional)

    Returns:
        List of segmentation results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        checkpoint_path = "checkpoints/sam2_hiera_large.pt"

    if model_cfg is None:
        model_cfg = "sam2_hiera_l.yaml"

    print(f"\nLoading SAM 2 model from {checkpoint_path}...")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print("Setting image...")
    predictor.set_image(image)

    print(f"Generating masks for {len(prompts)} prompts...")
    results = []

    for i, prompt_info in enumerate(prompts):
        point = prompt_info['point']

        # Convert to numpy array format SAM expects
        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([1])  # 1 = foreground point

        # Get mask from SAM
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True  # Get 3 masks, pick best
        )

        # Pick mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]

        results.append({
            'mask': mask,
            'class_idx': prompt_info['class_idx'],
            'class_name': prompt_info['class_name'],
            'confidence': prompt_info['confidence'],
            'sam_score': float(score),
            'region_size': prompt_info['region_size'],
            'prompt_point': point
        })

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} masks...")

    print(f"Successfully generated {len(results)} masks")
    return results


def merge_overlapping_masks(results, iou_threshold=0.8):
    """
    Merge masks that overlap significantly and have the same class.

    Args:
        results: List of result dictionaries with 'mask', 'class_idx', etc.
        iou_threshold: IoU threshold for merging

    Returns:
        Filtered list of results with overlaps removed
    """
    print(f"\nMerging overlapping masks (IoU threshold: {iou_threshold})...")

    # Sort by confidence (keep highest confidence masks)
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    kept_results = []

    for result in sorted_results:
        mask = result['mask']
        class_idx = result['class_idx']

        # Check if this mask significantly overlaps with any kept mask of same class
        should_keep = True

        for kept in kept_results:
            if kept['class_idx'] != class_idx:
                continue

            kept_mask = kept['mask']

            # Calculate IoU (ensure boolean masks)
            mask_bool = mask.astype(bool)
            kept_mask_bool = kept_mask.astype(bool)
            intersection = (mask_bool & kept_mask_bool).sum()
            union = (mask_bool | kept_mask_bool).sum()
            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept_results.append(result)

    print(f"Kept {len(kept_results)}/{len(results)} masks after merging")
    return kept_results


def visualize_results(image, results, vocabulary, output_path=None):
    """Visualize segmentation results."""
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    # Create color map for classes
    np.random.seed(42)
    class_colors = {i: np.random.random(3) for i in range(len(vocabulary))}

    # Sort by region size (draw larger regions first)
    sorted_results = sorted(results, key=lambda x: x['region_size'], reverse=True)

    # Draw masks
    for result in sorted_results:
        mask = result['mask']
        class_idx = result['class_idx']
        class_name = result['class_name']
        confidence = result['confidence']

        color = class_colors[class_idx]
        img_overlay = np.ones((mask.shape[0], mask.shape[1], 3))
        for i in range(3):
            img_overlay[:, :, i] = color[i]
        ax.imshow(np.dstack((img_overlay, mask * 0.5)))

        # Add label at prompt point
        point = result['prompt_point']
        ax.plot(point[0], point[1], 'r*', markersize=10)

        if result['region_size'] > 1000:  # Only label larger objects
            ax.text(
                point[0], point[1],
                f"{class_name}\n{confidence:.2f}",
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                ha='center', va='center'
            )

    ax.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"\nSaved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_filtered_results(image, results, target_class, vocabulary, output_path=None, edit_style="segment"):
    """
    Visualize only masks corresponding to a specific target class.

    Args:
        image: (H, W, 3) RGB image
        results: List of result dictionaries with 'mask', 'class_idx', etc.
        target_class: str, the target class name to filter for
        vocabulary: List of class names
        output_path: Optional path to save the visualization
        edit_style: str, one of "segment", "replace", "remove"
            - "segment": show segmentation overlays (default)
            - "replace": show white masks on black background
            - "remove": show image with target class removed (black)
    """
    # Filter results for target class
    filtered_results = [r for r in results if r['class_name'] == target_class]

    if len(filtered_results) == 0:
        print(f"\nWARNING: No masks found for target class '{target_class}'")
        return

    print(f"\nFiltered {len(filtered_results)} masks for target class '{target_class}'")

    # Combine all filtered masks
    H, W = image.shape[:2]
    combined_mask = np.zeros((H, W), dtype=bool)
    for result in filtered_results:
        combined_mask |= result['mask'].astype(bool)

    if edit_style == "segment":
        # Show segmentation overlay (original behavior)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)

        # Use a single color for the target class
        np.random.seed(42)
        target_class_idx = vocabulary.index(target_class)
        class_colors = {i: np.random.random(3) for i in range(len(vocabulary))}
        target_color = class_colors[target_class_idx]

        # Sort by region size (draw larger regions first)
        sorted_results = sorted(filtered_results, key=lambda x: x['region_size'], reverse=True)

        # Draw masks
        for result in sorted_results:
            mask = result['mask']
            confidence = result['confidence']

            img_overlay = np.ones((mask.shape[0], mask.shape[1], 3))
            for i in range(3):
                img_overlay[:, :, i] = target_color[i]
            ax.imshow(np.dstack((img_overlay, mask * 0.5)))

            # Add label at prompt point
            point = result['prompt_point']
            ax.plot(point[0], point[1], 'r*', markersize=10)

            if result['region_size'] > 1000:  # Only label larger objects
                ax.text(
                    point[0], point[1],
                    f"{target_class}\n{confidence:.2f}",
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor=target_color, alpha=0.8),
                    ha='center', va='center'
                )

        ax.set_title(f"Filtered Segmentation: {target_class} only ({len(filtered_results)} instances)",
                     fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()

    elif edit_style == "replace":
        # Show white masks on black background
        fig, ax = plt.subplots(figsize=(20, 20))
        mask_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask_image[combined_mask] = [255, 255, 255]
        ax.imshow(mask_image)
        ax.set_title(f"Mask for replacement: {target_class} ({len(filtered_results)} instances)",
                     fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()

    elif edit_style == "remove":
        # Show image with target class removed (blackened)
        fig, ax = plt.subplots(figsize=(20, 20))
        removed_image = image.copy()
        removed_image[combined_mask] = [0, 0, 0]
        ax.imshow(removed_image)
        ax.set_title(f"Image with {target_class} removed ({len(filtered_results)} instances)",
                     fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"Saved filtered visualization ({edit_style} style) to {output_path}")
    else:
        plt.show()

    plt.close()

    return combined_mask


def print_statistics(results, vocabulary):
    """Print statistics."""
    print("\n" + "="*50)
    print("SEGMENTATION STATISTICS")
    print("="*50)

    # Count per class
    class_counts = {i: 0 for i in range(len(vocabulary))}
    for result in results:
        class_counts[result['class_idx']] += 1

    print(f"\nTotal segments: {len(results)}")
    print("\nClass distribution:")
    for i, class_name in enumerate(vocabulary):
        count = class_counts[i]
        percentage = (count / len(results) * 100) if len(results) > 0 else 0
        print(f"  {class_name:20s}: {count:4d} ({percentage:5.1f}%)")

    # Average confidence per class
    print("\nAverage CLIP confidence per class:")
    for i, class_name in enumerate(vocabulary):
        class_results = [r for r in results if r['class_idx'] == i]
        if class_results:
            avg_conf = np.mean([r['confidence'] for r in class_results])
            print(f"  {class_name:20s}: {avg_conf:.3f}")


def print_statistics_video(video_segments, prompts):
    """Print statistics for video segmentation."""
    print("\n" + "="*50)
    print("VIDEO SEGMENTATION STATISTICS")
    print("="*50)

    # Count frames where each object appears
    num_frames = len(video_segments)
    class_names = {p['class_idx']: p['class_name'] for p in prompts}

    print(f"\nTotal frames: {num_frames}")
    print(f"Tracked objects: {len(prompts)}")

    print("\nObject tracking across frames:")
    for prompt in prompts:
        obj_id = prompt['class_idx']
        class_name = prompt['class_name']

        # Count frames where this object is present
        frames_present = sum(
            1 for frame_idx, masks in video_segments.items()
            if obj_id in masks and masks[obj_id].sum() > 0
        )

        percentage = (frames_present / num_frames * 100) if num_frames > 0 else 0
        print(f"  {class_name:20s}: {frames_present:4d}/{num_frames} frames ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-guided SAM segmentation (optimized)"
    )
    parser.add_argument("--image", required=True, help="Path to input image or video")
    parser.add_argument("--vocabulary", nargs='+', required=True,
                       help="List of class names for CLIP")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Target class name to filter and visualize (must be in vocabulary)")
    parser.add_argument("--edit", type=str, choices=["segment", "replace", "remove"],
                       default="segment",
                       help="Edit style for filtered visualization: segment (overlay), replace (mask), remove (blacken)")
    parser.add_argument("--use-inpainting", action="store_true",
                       help="Use Stable Diffusion inpainting for remove/replace modes")
    parser.add_argument("--edit-prompt", type=str, default=None,
                       help="Prompt for replace mode (e.g., 'a red sports car')")
    parser.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt",
                       help="Path to SAM 2 checkpoint")
    parser.add_argument("--model-cfg", default="sam2_hiera_l.yaml",
                       help="Model configuration")
    parser.add_argument("--output", default="clip_guided_segments.png",
                       help="Output visualization path")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum CLIP confidence for prompts")
    parser.add_argument("--min-region-size", type=int, default=100,
                       help="Minimum region size (pixels) for prompts")
    parser.add_argument("--iou-threshold", type=float, default=0.8,
                       help="IoU threshold for merging overlapping masks")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Validate prompt if provided
    if args.prompt and args.prompt not in args.vocabulary:
        raise ValueError(
            f"Prompt '{args.prompt}' is not in vocabulary. "
            f"Available classes: {', '.join(args.vocabulary)}"
        )

    # Check if input is video or image
    is_video = is_video_file(args.image)

    if is_video:
        print(f"Detected video input: {args.image}")

        # Check for unsupported features
        if args.edit in ["replace", "remove"] or args.use_inpainting:
            print("\nWARNING: Video inpainting/editing is not yet supported!")
            print("Only segmentation visualization is available for videos.")
            print("Use --edit segment (default) for video segmentation.\n")
            if args.edit != "segment":
                return

        # Extract first frame for CLIP analysis
        print("Extracting first frame for CLIP analysis...")
        image = extract_first_frame(args.image)
    else:
        # Load image
        print(f"Loading image from {args.image}...")
        image = load_image(args.image)

    # Step 1: Get CLIP dense predictions (fast)
    print("\n" + "="*50)
    print("STEP 1: CLIP Dense Prediction")
    print("="*50)

    segmentor = SCLIPSegmentor(
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        use_sam=False,
        use_pamr=False,
        verbose=True
    )

    seg_map, logits = segmentor.predict_dense(image, args.vocabulary, return_logits=True)
    probs = torch.softmax(logits, dim=0).cpu().numpy()  # (num_classes, H, W)
    probs = probs.transpose(1, 2, 0)  # (H, W, num_classes)

    # Step 2: Extract prompt points from CLIP predictions
    print("\n" + "="*50)
    print("STEP 2: Extract Prompt Points")
    print("="*50)

    prompts = extract_prompt_points_from_clip(
        seg_map, probs, args.vocabulary,
        min_confidence=args.min_confidence,
        min_region_size=args.min_region_size
    )

    print(f"\nUsing {len(prompts)} prompts vs {64*64}=4096 in blind grid!")
    if len(prompts) > 0:
        print(f"Speedup: {4096/len(prompts):.1f}x fewer SAM queries")
    else:
        print("WARNING: No prompts found! Try lowering --min-confidence")

    # Step 3: Segment with guided prompts
    if len(prompts) == 0:
        print("\nNo prompts to process. Exiting.")
        return

    # Handle video vs image differently
    if is_video:
        print("\n" + "="*50)
        print("STEP 3: SAM2 Video Segmentation")
        print("="*50)

        # Initialize video segmentor
        video_segmentor = CLIPGuidedVideoSegmentor(
            checkpoint_path=args.checkpoint,
            model_cfg=args.model_cfg,
            device=args.device
        )

        # Determine output path (change extension to .mp4)
        output_path = args.output
        if not output_path.endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'

        # Segment video
        video_segments = video_segmentor.segment_video(
            video_path=args.image,
            prompts=prompts,
            output_path=output_path,
            visualize=True
        )

        print(f"\n{'='*50}")
        print("VIDEO SEGMENTATION COMPLETE!")
        print(f"{'='*50}")
        print(f"Output saved to: {output_path}")
        print(f"Tracked {len(prompts)} objects across video")

        # Statistics
        print_statistics_video(video_segments, prompts)

    else:
        # Image segmentation (original workflow)
        print("\n" + "="*50)
        print("STEP 3: SAM Segmentation")
        print("="*50)

        results = segment_with_guided_prompts(
            image, prompts,
            checkpoint_path=args.checkpoint,
            model_cfg=args.model_cfg,
            device=args.device
        )

        # Step 4: Merge overlapping masks
        results = merge_overlapping_masks(results, iou_threshold=args.iou_threshold)

        # Visualize all results
        visualize_results(image, results, args.vocabulary, output_path=args.output)

    # If prompt is specified, create filtered visualization (images only)
    if args.prompt and not is_video:
        print("\n" + "="*50)
        print(f"FILTERED VISUALIZATION: {args.prompt} (style: {args.edit})")
        print("="*50)

        # Generate filtered output filename
        base_name = args.output.rsplit('.', 1)[0]
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'png'
        filtered_output = f"{base_name}_filtered_{args.prompt}_{args.edit}.{ext}"

        combined_mask = visualize_filtered_results(image, results, args.prompt, args.vocabulary,
                                                   output_path=filtered_output, edit_style=args.edit)

        # Save binary mask for potential inpainting use
        if combined_mask is not None and args.edit in ["replace", "remove"]:
            mask_output = f"{base_name}_mask_{args.prompt}.png"
            mask_image = (combined_mask.astype(np.uint8) * 255)
            cv2.imwrite(mask_output, mask_image)
            print(f"Saved binary mask to {mask_output}")

            # Perform Stable Diffusion inpainting if requested
            if args.use_inpainting:
                print("\n" + "="*50)
                print("STABLE DIFFUSION INPAINTING")
                print("="*50)

                try:
                    from models.inpainting import StableDiffusionInpainter

                    # Initialize Stable Diffusion
                    inpainter = StableDiffusionInpainter(
                        model_id="stabilityai/stable-diffusion-2-inpainting",
                        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
                    )

                    # Perform inpainting
                    if args.edit == "remove":
                        inpainted = inpainter.remove_object(image, mask_image)
                        inpainted_output = f"{base_name}_inpainted_removed_{args.prompt}.{ext}"
                    else:  # replace
                        if args.edit_prompt is None:
                            print("Warning: --edit-prompt not specified for replace mode")
                            replacement_prompt = f"a {args.prompt}"
                        else:
                            replacement_prompt = args.edit_prompt
                        inpainted = inpainter.replace_object(image, mask_image, replacement_prompt)
                        inpainted_output = f"{base_name}_inpainted_replaced_{args.prompt}.{ext}"

                    # Convert PIL to numpy if needed
                    if not isinstance(inpainted, np.ndarray):
                        inpainted = np.array(inpainted)

                    # Save result
                    cv2.imwrite(inpainted_output, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
                    print(f"Saved inpainted result to {inpainted_output}")

                    # Create comparison visualization
                    comparison = inpainter.compare_results(image, inpainted, mask_image)
                    comparison_output = f"{base_name}_inpainting_comparison_{args.prompt}.{ext}"
                    cv2.imwrite(comparison_output, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                    print(f"Saved comparison to {comparison_output}")

                except ImportError as e:
                    print(f"Error: Could not import Stable Diffusion inpainter: {e}")
                    print("Install with: pip install diffusers transformers accelerate")
                except Exception as e:
                    print(f"Error during inpainting: {e}")
                    import traceback
                    traceback.print_exc()

    # Statistics (for images only, video stats already printed)
    if not is_video:
        print_statistics(results, args.vocabulary)


if __name__ == "__main__":
    main()
