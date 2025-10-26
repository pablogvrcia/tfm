#!/usr/bin/env python3
"""
Demo: Adaptive Mask Selection

This script demonstrates the adaptive mask selection feature that automatically
determines how many masks to select based on the semantic granularity of the query.

Examples:
- "car" → Selects 1 complete vehicle
- "tire" → Selects all 4 tires (parts)
- "mountain" → Selects all mountains (instances)
- "sky" → Selects the large sky region (stuff)

Usage:
    python demo_adaptive_selection.py --image street.jpg
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import OpenVocabSegmentationPipeline
from utils import load_image, save_image, create_mask_overlay


def demo_adaptive_vs_fixed(image_path: str):
    """
    Compare adaptive selection vs. fixed top-K selection.
    """
    print("\n" + "="*80)
    print("DEMO: Adaptive Mask Selection")
    print("="*80 + "\n")

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}\n")

    # Initialize pipeline
    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=False)

    # Test queries with different semantic granularities
    test_cases = [
        {
            "prompt": "car",
            "description": "Singular object (should select 1 complete car)",
            "expected": "1 mask"
        },
        {
            "prompt": "tires",
            "description": "Object parts (should select all tires, typically 2-4)",
            "expected": "2-4 masks"
        },
        {
            "prompt": "windows",
            "description": "Multiple parts (should select all windows)",
            "expected": "4-8 masks"
        },
        {
            "prompt": "people",
            "description": "Multiple instances (should select all people)",
            "expected": "N masks"
        },
        {
            "prompt": "mountains",
            "description": "Multiple instances (should select all mountains)",
            "expected": "N masks"
        },
        {
            "prompt": "sky",
            "description": "Stuff category (should select largest region)",
            "expected": "1 large mask"
        },
    ]

    output_dir = Path("output/adaptive_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    save_image(image, output_dir / "00_original.png")

    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case["prompt"]
        description = test_case["description"]
        expected = test_case["expected"]

        print(f"\n{'='*80}")
        print(f"Test Case {i}: '{prompt}'")
        print(f"Description: {description}")
        print(f"Expected: {expected}")
        print('='*80)

        # Run with FIXED top-K=5
        print("\n[1] Fixed selection (top-K=5):")
        result_fixed = pipeline.segment(
            image,
            prompt,
            top_k=5,
            use_adaptive_selection=False,
            return_visualization=True
        )
        print(f"    Selected: {len(result_fixed.segmentation_masks)} masks")
        if result_fixed.segmentation_masks:
            print(f"    Top score: {result_fixed.segmentation_masks[0].final_score:.3f}")

        # Run with ADAPTIVE selection
        print("\n[2] Adaptive selection:")
        result_adaptive = pipeline.segment(
            image,
            prompt,
            use_adaptive_selection=True,
            return_visualization=True
        )
        print(f"    Selected: {len(result_adaptive.segmentation_masks)} masks")
        if result_adaptive.segmentation_masks:
            print(f"    Top score: {result_adaptive.segmentation_masks[0].final_score:.3f}")
            if result_adaptive.visualization_data and 'adaptive_info' in result_adaptive.visualization_data:
                info = result_adaptive.visualization_data['adaptive_info']
                print(f"    Method: {info['method']}")
                print(f"    Score range: {info['score_range'][0]:.3f} - {info['score_range'][1]:.3f}")

        # Create visualizations
        vis_fixed = create_comparison_image(image, result_fixed, "Fixed (top-5)")
        vis_adaptive = create_comparison_image(image, result_adaptive, "Adaptive")

        # Save comparison
        comparison = np.hstack([vis_fixed, vis_adaptive])
        filename = f"{i:02d}_{prompt.replace(' ', '_')}_comparison.png"
        save_image(comparison, output_dir / filename)
        print(f"\n    Saved: {filename}")

    print("\n" + "="*80)
    print(f"All results saved to: {output_dir}/")
    print("="*80 + "\n")


def create_comparison_image(image, result, title):
    """Create a visualization with title and masks."""
    # Create visualization
    vis = image.copy()

    # Colors for masks
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]

    for i, scored_mask in enumerate(result.segmentation_masks[:8]):
        mask = scored_mask.mask_candidate.mask
        color = colors[i % len(colors)]

        # Create overlay
        overlay = vis.copy()
        overlay[mask > 0] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Draw bounding box
        x, y, w, h = scored_mask.mask_candidate.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Add label
        label = f"#{i+1}: {scored_mask.final_score:.2f}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Add title
    title_height = 40
    title_img = np.ones((title_height, vis.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add mask count
    count_text = f"Masks: {len(result.segmentation_masks)}"
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(title_img, count_text, (vis.shape[1] - text_size[0] - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return np.vstack([title_img, vis])


def demo_query_analysis():
    """
    Demonstrate how the system analyzes different types of queries.
    """
    print("\n" + "="*80)
    print("Query Analysis Examples")
    print("="*80 + "\n")

    from models.adaptive_selection import AdaptiveMaskSelector

    selector = AdaptiveMaskSelector()

    test_queries = [
        "car",
        "the red car",
        "tires",
        "all the windows",
        "people walking",
        "mountain",
        "mountains in the background",
        "sky",
        "trees",
        "the door handle",
    ]

    print(f"{'Query':<30} {'Type':<15} {'Plural':<8} {'Part':<8} {'Stuff':<8}")
    print("-" * 80)

    for query in test_queries:
        semantic_type, info = selector._analyze_prompt(query)
        print(f"{query:<30} {semantic_type:<15} {str(info['is_plural']):<8} "
              f"{str(info['is_part']):<8} {str(info['is_stuff']):<8}")

    print("\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo: Adaptive Mask Selection")
    parser.add_argument("--image", "-i", type=str, help="Path to test image")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only run query analysis (no image needed)")

    args = parser.parse_args()

    if args.analysis_only:
        demo_query_analysis()
    elif args.image:
        # Check if image exists
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        demo_query_analysis()
        demo_adaptive_vs_fixed(args.image)
    else:
        print("Usage:")
        print("  python demo_adaptive_selection.py --image path/to/image.jpg")
        print("  python demo_adaptive_selection.py --analysis-only")
        sys.exit(1)


if __name__ == "__main__":
    main()
