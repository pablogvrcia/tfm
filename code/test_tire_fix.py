#!/usr/bin/env python3
"""
Test script to verify tire detection improvements.

This tests the fixes for CLIP confusing tires with grilles, license plates, etc.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_image, save_image
from pipeline import OpenVocabSegmentationPipeline


def test_tire_detection(image_path: str):
    """
    Test tire detection before and after fixes.
    """
    print("\n" + "="*80)
    print("TEST: Tire Detection Fix")
    print("="*80 + "\n")

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        print("\nUsage: python test_tire_fix.py <path_to_car_image>")
        return

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")

    # Initialize pipeline
    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=True)

    # Test queries that commonly fail
    test_queries = [
        "tire",
        "car tire",
        "black tire",
        "wheel",
    ]

    output_dir = Path("output/tire_fix_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    save_image(image, output_dir / "00_original.png")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}: '{query}'")
        print('='*80)

        # Segment
        result = pipeline.segment(
            image,
            query,
            top_k=5,
            return_visualization=True
        )

        # Analyze results
        print(f"\nResults for '{query}':")
        print(f"  Total masks found: {len(result.segmentation_masks)}")

        if result.segmentation_masks:
            print("\n  Top 5 masks:")
            for j, mask in enumerate(result.segmentation_masks[:5], 1):
                area = mask.mask_candidate.area
                score = mask.final_score
                sim = mask.similarity_score
                bg = mask.background_score
                print(f"    #{j}: score={score:.3f} (sim={sim:.3f}, bg={bg:.3f}), area={area}")

            # Visualize
            vis = pipeline.visualize_results(result)

            if 'scored_masks' in vis:
                filename = f"{i:02d}_{query.replace(' ', '_')}_masks.png"
                save_image(vis['scored_masks'], output_dir / filename)
                print(f"\n  Saved: {filename}")

            if 'similarity_map' in vis:
                filename = f"{i:02d}_{query.replace(' ', '_')}_similarity.png"
                save_image(vis['similarity_map'], output_dir / filename)

        else:
            print("  No masks found!")

    print("\n" + "="*80)
    print(f"Test complete! Results saved to: {output_dir}/")
    print("="*80 + "\n")

    # Analysis
    print("\nExpected improvements:")
    print("  ✓ Tire masks should have HIGHER scores than before")
    print("  ✓ Grille/license plate should have LOWER scores (confuser penalty)")
    print("  ✓ More accurate mask selection for 'tire' query")
    print("\nCheck the output images to verify improvements.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test tire detection fix")
    parser.add_argument("image", type=str, nargs='?',
                       help="Path to car image (should show tires)")

    args = parser.parse_args()

    if args.image:
        test_tire_detection(args.image)
    else:
        # Try to find a test image in output
        test_img = Path("output/original.png")
        if test_img.exists():
            print(f"Using existing test image: {test_img}")
            test_tire_detection(str(test_img))
        else:
            print("Usage: python test_tire_fix.py <path_to_car_image>")
            print("\nThis script tests the improved tire detection that fixes:")
            print("  - CLIP confusing tires with grilles")
            print("  - License plates getting high scores")
            print("  - Headlights being selected instead of tires")
            print("\nThe fixes include:")
            print("  1. Black background (focus on object)")
            print("  2. Minimum size upscaling (better CLIP features)")
            print("  3. Confuser penalty (distinguish from similar objects)")
            sys.exit(1)


if __name__ == "__main__":
    main()
