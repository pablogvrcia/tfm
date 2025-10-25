"""
Basic usage examples for the Open-Vocabulary Segmentation Pipeline.

This script demonstrates the core functionality with simple examples.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import OpenVocabSegmentationPipeline
from utils import save_image, plot_results
import numpy as np
from PIL import Image


def example_1_simple_segmentation():
    """
    Example 1: Basic object segmentation
    Segment a specific object using a text prompt.
    """
    print("\n" + "="*70)
    print("Example 1: Simple Segmentation")
    print("="*70)

    # Initialize pipeline
    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=True)

    # Create a simple test image (or load your own)
    # For demo purposes, creating a synthetic image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Segment objects
    result = pipeline.segment(
        test_image,
        text_prompt="object in center",
        top_k=3,
        return_visualization=True
    )

    print(f"\nFound {len(result.segmentation_masks)} matching masks")
    for i, mask in enumerate(result.segmentation_masks[:3], 1):
        print(f"  Mask {i}: score={mask.final_score:.3f}, area={mask.mask_candidate.area}")

    # Visualize results
    visualizations = pipeline.visualize_results(result)

    # Save outputs
    output_dir = Path("output/example1")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, img in visualizations.items():
        save_image(img, output_dir / f"{name}.png")

    print(f"\nOutputs saved to: {output_dir}/")


def example_2_object_removal():
    """
    Example 2: Remove an object from an image
    """
    print("\n" + "="*70)
    print("Example 2: Object Removal")
    print("="*70)

    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=True)

    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Remove object
    result = pipeline.segment_and_edit(
        test_image,
        text_prompt="unwanted object",
        edit_operation="remove",
        top_k=1
    )

    if result.edited_image:
        output_dir = Path("output/example2")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_image(result.original_image, output_dir / "original.png")
        save_image(result.edited_image, output_dir / "removed.png")

        print(f"\nOutputs saved to: {output_dir}/")


def example_3_object_replacement():
    """
    Example 3: Replace an object with something else
    """
    print("\n" + "="*70)
    print("Example 3: Object Replacement")
    print("="*70)

    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=True)

    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Replace object
    result = pipeline.segment_and_edit(
        test_image,
        text_prompt="old object",
        edit_operation="replace",
        edit_prompt="modern sleek object",
        top_k=1
    )

    if result.edited_image:
        output_dir = Path("output/example3")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_image(result.original_image, output_dir / "original.png")
        save_image(result.edited_image, output_dir / "replaced.png")

        # Create comparison
        visualizations = pipeline.visualize_results(result)
        if 'edit_comparison' in visualizations:
            save_image(visualizations['edit_comparison'], output_dir / "comparison.png")

        print(f"\nOutputs saved to: {output_dir}/")


def example_4_batch_processing():
    """
    Example 4: Process multiple images
    """
    print("\n" + "="*70)
    print("Example 4: Batch Processing")
    print("="*70)

    pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=False)

    # Create multiple test images
    test_images = [
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        for _ in range(3)
    ]

    prompts = ["object A", "object B", "object C"]

    results = []
    for i, (image, prompt) in enumerate(zip(test_images, prompts), 1):
        print(f"\nProcessing image {i}/3: '{prompt}'")
        result = pipeline.segment(image, prompt, top_k=1, return_visualization=False)
        results.append(result)

        if result.segmentation_masks:
            print(f"  Top score: {result.segmentation_masks[0].final_score:.3f}")

    print(f"\nProcessed {len(results)} images successfully")


def example_5_custom_config():
    """
    Example 5: Use custom configuration
    """
    print("\n" + "="*70)
    print("Example 5: Custom Configuration")
    print("="*70)

    from config import PipelineConfig, get_fast_config, get_quality_config

    # Try different configs
    configs = {
        "Fast": get_fast_config(),
        "Quality": get_quality_config(),
        "Balanced": PipelineConfig()
    }

    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    for name, config in configs.items():
        print(f"\nTesting {name} config:")
        print(f"  SAM points per side: {config.sam2.points_per_side}")
        print(f"  CLIP layers: {config.clip.extract_layers}")
        print(f"  Inpainting steps: {config.inpainting.num_inference_steps}")

        pipeline = OpenVocabSegmentationPipeline(device="cuda", verbose=False)
        result = pipeline.segment(test_image, "test object", return_visualization=False)

        print(f"  Time: {sum(result.timing.values()):.2f}s")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Open-Vocabulary Segmentation Pipeline - Basic Examples")
    print("="*70)

    # Note: These examples use synthetic images for demonstration
    # Replace with your own images by loading them:
    # from utils import load_image
    # image = load_image("path/to/your/image.jpg")

    try:
        # Run examples
        example_1_simple_segmentation()
        example_2_object_removal()
        example_3_object_replacement()
        example_4_batch_processing()
        example_5_custom_config()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
