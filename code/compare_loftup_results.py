"""
Compare LoftUp-enhanced vs. standard CLIP-guided segmentation.

This script runs both approaches side-by-side and generates a comparison visualization.

Usage:
    python compare_loftup_results.py --image path/to/image.jpg --vocabulary person car dog
"""

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time

from models.loftup_sclip_segmentor import LoftUpSCLIPSegmentor
from models.sclip_segmentor import SCLIPSegmentor


def load_image(image_path):
    """Load image from path."""
    image = Image.open(image_path)
    return np.array(image.convert("RGB"))


def compare_segmentations(image, vocabulary, device="cuda"):
    """
    Compare standard SCLIP vs. LoftUp-enhanced SCLIP.

    Returns:
        dict with comparison metrics and visualizations
    """
    results = {}

    # Test 1: Standard SCLIP (no upsampling)
    print("\n" + "="*70)
    print("TEST 1: Standard SCLIP (Bilinear Upsampling)")
    print("="*70)

    start_time = time.time()
    segmentor_standard = LoftUpSCLIPSegmentor(
        device=device,
        use_loftup=False,  # Disable LoftUp
        use_sam=False,
        use_pamr=False,
        verbose=True
    )

    seg_map_standard, logits_standard, _ = segmentor_standard.predict_dense(
        image, vocabulary,
        return_logits=True,
        return_features=False
    )
    time_standard = time.time() - start_time

    # Extract prompts from standard predictions
    probs_standard = torch.softmax(logits_standard, dim=0).cpu().numpy().transpose(1, 2, 0)
    from models.loftup_sclip_segmentor import extract_prompt_points_from_upsampled
    prompts_standard = extract_prompt_points_from_upsampled(
        seg_map_standard, probs_standard, vocabulary
    )

    results['standard'] = {
        'seg_map': seg_map_standard,
        'logits': logits_standard,
        'prompts': prompts_standard,
        'time': time_standard,
        'num_prompts': len(prompts_standard)
    }

    print(f"[Standard] Time: {time_standard:.2f}s, Prompts: {len(prompts_standard)}")

    # Test 2: LoftUp-enhanced SCLIP
    print("\n" + "="*70)
    print("TEST 2: LoftUp-Enhanced SCLIP")
    print("="*70)

    start_time = time.time()
    segmentor_loftup = LoftUpSCLIPSegmentor(
        device=device,
        use_loftup=True,  # Enable LoftUp
        loftup_model_name="loftup_clip",
        use_sam=False,
        use_pamr=False,
        verbose=True
    )

    seg_map_loftup, logits_loftup, features_loftup = segmentor_loftup.predict_dense(
        image, vocabulary,
        return_logits=True,
        return_features=True
    )
    time_loftup = time.time() - start_time

    # Extract prompts from LoftUp predictions
    probs_loftup = torch.softmax(logits_loftup, dim=0).cpu().numpy().transpose(1, 2, 0)
    prompts_loftup = extract_prompt_points_from_upsampled(
        seg_map_loftup, probs_loftup, vocabulary
    )

    results['loftup'] = {
        'seg_map': seg_map_loftup,
        'logits': logits_loftup,
        'features': features_loftup,
        'prompts': prompts_loftup,
        'time': time_loftup,
        'num_prompts': len(prompts_loftup)
    }

    print(f"[LoftUp] Time: {time_loftup:.2f}s, Prompts: {len(prompts_loftup)}")

    return results


def visualize_comparison(image, results, vocabulary, output_path):
    """Create side-by-side comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, weight='bold')
    axes[0, 0].axis('off')

    # Standard SCLIP segmentation
    seg_standard = results['standard']['seg_map']
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(seg_standard, alpha=0.5, cmap='tab20')
    axes[0, 1].set_title(f"Standard SCLIP\n{results['standard']['num_prompts']} prompts, "
                         f"{results['standard']['time']:.2f}s",
                         fontsize=14, weight='bold')
    axes[0, 1].axis('off')

    # LoftUp SCLIP segmentation
    seg_loftup = results['loftup']['seg_map']
    axes[0, 2].imshow(image)
    axes[0, 2].imshow(seg_loftup, alpha=0.5, cmap='tab20')
    axes[0, 2].set_title(f"LoftUp-Enhanced SCLIP\n{results['loftup']['num_prompts']} prompts, "
                         f"{results['loftup']['time']:.2f}s",
                         fontsize=14, weight='bold')
    axes[0, 2].axis('off')

    # Prompt points comparison
    axes[1, 0].imshow(image)
    for prompt in results['standard']['prompts']:
        point = prompt['point']
        axes[1, 0].plot(point[0], point[1], 'ro', markersize=5, alpha=0.7)
    axes[1, 0].set_title(f"Standard Prompts ({results['standard']['num_prompts']})",
                         fontsize=14, weight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image)
    for prompt in results['loftup']['prompts']:
        point = prompt['point']
        axes[1, 1].plot(point[0], point[1], 'go', markersize=5, alpha=0.7)
    axes[1, 1].set_title(f"LoftUp Prompts ({results['loftup']['num_prompts']})",
                         fontsize=14, weight='bold')
    axes[1, 1].axis('off')

    # Difference map
    diff = (seg_loftup != seg_standard).astype(float)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title("Difference Map\n(Red = Different Predictions)",
                         fontsize=14, weight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n[Save] Comparison saved to {output_path}")
    plt.close()


def print_comparison_metrics(results):
    """Print detailed comparison metrics."""
    print("\n" + "="*70)
    print("COMPARISON METRICS")
    print("="*70)

    print("\n1. Efficiency:")
    print(f"   Standard:  {results['standard']['time']:.2f}s")
    print(f"   LoftUp:    {results['loftup']['time']:.2f}s")
    overhead = results['loftup']['time'] - results['standard']['time']
    print(f"   Overhead:  +{overhead:.2f}s ({overhead/results['standard']['time']*100:.1f}%)")

    print("\n2. Prompt Extraction:")
    print(f"   Standard:  {results['standard']['num_prompts']} prompts")
    print(f"   LoftUp:    {results['loftup']['num_prompts']} prompts")
    diff = results['loftup']['num_prompts'] - results['standard']['num_prompts']
    print(f"   Difference: {diff:+d} prompts")

    print("\n3. Segmentation Differences:")
    seg_standard = results['standard']['seg_map']
    seg_loftup = results['loftup']['seg_map']
    diff_pixels = (seg_standard != seg_loftup).sum()
    total_pixels = seg_standard.size
    diff_percent = diff_pixels / total_pixels * 100
    print(f"   Different pixels: {diff_pixels}/{total_pixels} ({diff_percent:.2f}%)")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard vs. LoftUp-enhanced CLIP-guided segmentation"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--vocabulary", nargs='+', required=True,
                       help="List of class names for CLIP")
    parser.add_argument("--output", default="comparison_loftup.png",
                       help="Output comparison visualization path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("="*70)
    print("LOFTUP COMPARISON TEST")
    print("="*70)
    print(f"Image: {args.image}")
    print(f"Vocabulary: {', '.join(args.vocabulary)}")
    print("="*70)

    # Load image
    image = load_image(args.image)
    print(f"\nImage shape: {image.shape}")

    # Run comparison
    results = compare_segmentations(image, args.vocabulary, device=args.device)

    # Visualize comparison
    visualize_comparison(image, results, args.vocabulary, args.output)

    # Print metrics
    print_comparison_metrics(results)

    print("\n[Complete] Comparison complete!")


if __name__ == "__main__":
    main()
