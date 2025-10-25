"""
Main entry point for the Open-Vocabulary Semantic Segmentation Pipeline.

Usage examples:
    # Segmentation only
    python main.py --image image.jpg --prompt "red car" --mode segment

    # Object removal
    python main.py --image image.jpg --prompt "person" --mode remove

    # Object replacement
    python main.py --image image.jpg --prompt "old TV" --mode replace --edit "modern flat screen TV"

    # Benchmark
    python main.py --image image.jpg --mode benchmark
"""

import argparse
from pathlib import Path
import sys

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import OpenVocabSegmentationPipeline, PipelineResult
from config import PipelineConfig, get_fast_config, get_quality_config
from utils import save_image, plot_results, print_timing_summary


def main():
    parser = argparse.ArgumentParser(
        description="Open-Vocabulary Semantic Segmentation and Editing"
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
        choices=["segment", "remove", "replace", "style", "benchmark"],
        default="segment",
        help="Operation mode"
    )

    # Prompts
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt describing target object"
    )
    parser.add_argument(
        "--edit", "-e",
        type=str,
        help="Edit prompt (for replace/style modes)"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Configuration preset"
    )
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
        help="Number of top masks to return"
    )

    # Visualization
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show visualizations"
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

    if args.mode != "benchmark" and not args.prompt:
        print("Error: --prompt required for this mode")
        sys.exit(1)

    if args.mode in ["replace", "style"] and not args.edit:
        print(f"Error: --edit required for {args.mode} mode")
        sys.exit(1)

    # Load configuration
    if args.config == "fast":
        config = get_fast_config()
    elif args.config == "quality":
        config = get_quality_config()
    else:  # balanced
        config = PipelineConfig()

    config.device = args.device

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    print("\n" + "="*70)
    print("Open-Vocabulary Semantic Segmentation Pipeline")
    print("Master Thesis Implementation")
    print("="*70 + "\n")

    pipeline = OpenVocabSegmentationPipeline(
        device=config.device,
        verbose=True
    )

    # Run based on mode
    if args.mode == "segment":
        result = run_segmentation(pipeline, args, output_dir)

    elif args.mode == "remove":
        result = run_editing(pipeline, args, output_dir, "remove")

    elif args.mode == "replace":
        result = run_editing(pipeline, args, output_dir, "replace")

    elif args.mode == "style":
        result = run_editing(pipeline, args, output_dir, "style")

    elif args.mode == "benchmark":
        run_benchmark(pipeline, args)
        return

    # Print timing summary
    print_timing_summary(result.timing)

    # Visualize if requested
    if args.visualize and result is not None:
        visualizations = pipeline.visualize_results(result)
        plot_results(visualizations)

    print(f"\nOutputs saved to: {output_dir}/")
    print("Done!\n")


def run_segmentation(
    pipeline: OpenVocabSegmentationPipeline,
    args,
    output_dir: Path
) -> PipelineResult:
    """Run segmentation-only mode."""
    print(f"\nMode: Segmentation")
    print(f"Image: {args.image}")
    print(f"Prompt: '{args.prompt}'\n")

    # Run pipeline
    result = pipeline.segment(
        args.image,
        args.prompt,
        top_k=args.top_k,
        return_visualization=True
    )

    # Save outputs
    if not args.no_save:
        # Save original
        save_image(result.original_image, output_dir / "original.png")

        # Save visualizations
        visualizations = pipeline.visualize_results(result)

        if 'scored_masks' in visualizations:
            save_image(visualizations['scored_masks'], output_dir / "segmentation.png")

        if 'similarity_map' in visualizations:
            save_image(visualizations['similarity_map'], output_dir / "similarity_map.png")

        if 'comparison_grid' in visualizations:
            save_image(visualizations['comparison_grid'], output_dir / "comparison_grid.png")

    return result


def run_editing(
    pipeline: OpenVocabSegmentationPipeline,
    args,
    output_dir: Path,
    operation: str
) -> PipelineResult:
    """Run editing mode (remove/replace/style)."""
    print(f"\nMode: {operation.capitalize()}")
    print(f"Image: {args.image}")
    print(f"Target: '{args.prompt}'")
    if args.edit:
        print(f"Edit: '{args.edit}'")
    print()

    # Run pipeline
    result = pipeline.segment_and_edit(
        args.image,
        args.prompt,
        operation,
        edit_prompt=args.edit,
        top_k=1,  # Usually process top mask for editing
        return_visualization=True
    )

    # Save outputs
    if not args.no_save and result.edited_image:
        # Save original and edited
        save_image(result.original_image, output_dir / "original.png")
        save_image(result.edited_image, output_dir / "edited.png")

        # Save comparison
        visualizations = pipeline.visualize_results(result)
        if 'edit_comparison' in visualizations:
            save_image(visualizations['edit_comparison'], output_dir / "comparison.png")

        # Save segmentation visualization
        if 'scored_masks' in visualizations:
            save_image(visualizations['scored_masks'], output_dir / "segmentation.png")

    return result


def run_benchmark(pipeline: OpenVocabSegmentationPipeline, args):
    """Run benchmark mode."""
    print("\nMode: Benchmark")
    print(f"Image: {args.image}\n")

    # Predefined test prompts covering different scenarios
    test_prompts = [
        "person",
        "car",
        "chair",
        "dog",
        "laptop",
    ]

    # Run benchmark
    pipeline.benchmark(
        args.image,
        test_prompts,
        num_runs=3
    )

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
