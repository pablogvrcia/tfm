"""
Simple Results Comparison Script for COCO-Stuff (and other datasets)

Compares different hyperparameter configurations without external dependencies.

Usage:
    python3 compare_results.py                    # Compare all results
    python3 compare_results.py --dataset coco-stuff  # Specific dataset
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_result(result_file: Path) -> Dict:
    """Load a single result JSON file."""
    try:
        with open(result_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return None


def extract_config_summary(result: Dict) -> Dict:
    """Extract key configuration parameters from result."""
    args = result.get('args', {})

    config = {
        # Core metrics
        'miou': result.get('miou', 0) * 100,
        'pixel_acc': result.get('pixel_accuracy', 0) * 100,
        'f1': result.get('f1', 0) * 100,
        'boundary_f1': result.get('boundary_f1', 0) * 100,
        'elapsed_time': result.get('elapsed_time', 0),
        'num_samples': args.get('num_samples', 'unknown'),

        # Key hyperparameters
        'use_sam': args.get('use_sam', False),
        'use_clip_guided_sam': args.get('use_clip_guided_sam', False),
        'use_pamr': args.get('use_pamr', False),
        'slide_inference': args.get('slide_inference', True),
        'logit_scale': args.get('logit_scale', 40.0),
        'min_confidence': args.get('min_confidence', 0.3),
        'min_region_size': args.get('min_region_size', 100),
        'iou_threshold': args.get('iou_threshold', 0.8),

        # Phase improvements
        'use_all_phase1': args.get('use_all_phase1', False),
        'use_loftup': args.get('use_loftup', False),
        'use_resclip': args.get('use_resclip', False),
        'use_densecrf': args.get('use_densecrf', False),

        'use_all_phase2a': args.get('use_all_phase2a', False),
        'use_cliptrase': args.get('use_cliptrase', False),
        'use_clip_rc': args.get('use_clip_rc', False),

        'template_strategy': args.get('template_strategy', 'imagenet80'),
        'use_confidence_sharpening': args.get('use_confidence_sharpening', False),
        'use_hierarchical_prediction': args.get('use_hierarchical_prediction', False),

        # Performance opts
        'use_fp16': args.get('use_fp16', True),
        'batch_prompts': args.get('batch_prompts', True),
    }

    # Add per-class IoU for important classes
    per_class_iou = result.get('per_class_iou', {})
    config['person_iou'] = per_class_iou.get('person', None)

    return config


def analyze_results(results_dir: Path, dataset_filter: str = None) -> List[Tuple[str, Dict, Dict]]:
    """Analyze all results in the directory."""

    # Find all result files
    if dataset_filter:
        pattern = f'**/{dataset_filter}_results.json'
    else:
        pattern = '**/*_results.json'

    result_files = list(results_dir.glob(pattern))

    if not result_files:
        print(f"No results found in {results_dir}")
        if dataset_filter:
            print(f"  Filter: {dataset_filter}")
        return []

    print(f"Found {len(result_files)} result file(s)")
    print()

    # Load and process results
    results = []
    for result_file in result_files:
        result = load_result(result_file)
        if result is None:
            continue

        config_name = result_file.parent.name
        dataset_name = result_file.stem.replace('_results', '')
        config_summary = extract_config_summary(result)

        results.append((config_name, dataset_name, config_summary))

    return results


def print_comparison_table(results: List[Tuple[str, Dict, Dict]]):
    """Print formatted comparison table."""

    if not results:
        print("No results to compare!")
        return

    # Sort by mIoU descending
    results = sorted(results, key=lambda x: x[2]['miou'], reverse=True)

    print("=" * 140)
    print("RESULTS COMPARISON")
    print("=" * 140)
    print()

    # Header
    header = f"{'Config Name':<40} {'Dataset':<15} {'Samples':>7} {'mIoU':>8} {'F1':>8} {'Bound-F1':>8} {'Time':>8} {'T/Samp':>8}"
    print(header)
    print("-" * 140)

    # Rows
    for config_name, dataset_name, config in results:
        time_per_sample = config['elapsed_time'] / config['num_samples'] if config['num_samples'] != 'unknown' else 0

        row = (
            f"{config_name:<40} "
            f"{dataset_name:<15} "
            f"{str(config['num_samples']):>7} "
            f"{config['miou']:>7.2f}% "
            f"{config['f1']:>7.2f}% "
            f"{config['boundary_f1']:>7.2f}% "
            f"{config['elapsed_time']:>7.1f}s "
            f"{time_per_sample:>7.2f}s"
        )
        print(row)

    print()


def print_hyperparameter_details(results: List[Tuple[str, str, Dict]]):
    """Print detailed hyperparameter settings for each config."""

    print("\n" + "=" * 140)
    print("HYPERPARAMETER DETAILS")
    print("=" * 140)

    for config_name, dataset_name, config in results:
        print(f"\n[{config_name}] ({dataset_name})")
        print("-" * 80)

        # Core settings
        print("Core Settings:")
        print(f"  Samples: {config['num_samples']}")
        print(f"  SAM: {config['use_sam']}")
        print(f"  CLIP-guided SAM: {config['use_clip_guided_sam']}")
        print(f"  PAMR: {config['use_pamr']}")
        print(f"  Slide inference: {config['slide_inference']}")

        # Hyperparameters
        print("\nKey Hyperparameters:")
        print(f"  Logit scale: {config['logit_scale']}")
        print(f"  Min confidence: {config['min_confidence']}")
        print(f"  Min region size: {config['min_region_size']}")
        print(f"  IoU threshold: {config['iou_threshold']}")
        print(f"  Template strategy: {config['template_strategy']}")

        # Phase improvements
        print("\nPhase Improvements:")
        print(f"  Phase 1 (all): {config['use_all_phase1']}")
        print(f"    - LoftUp: {config['use_loftup']}")
        print(f"    - ResCLIP: {config['use_resclip']}")
        print(f"    - DenseCRF: {config['use_densecrf']}")
        print(f"  Phase 2A (all): {config['use_all_phase2a']}")
        print(f"    - CLIPtrase: {config['use_cliptrase']}")
        print(f"    - CLIP-RC: {config['use_clip_rc']}")
        print(f"  Phase 2C:")
        print(f"    - Confidence sharpening: {config['use_confidence_sharpening']}")
        print(f"    - Hierarchical prediction: {config['use_hierarchical_prediction']}")

        # Performance
        print("\nPerformance:")
        print(f"  FP16: {config['use_fp16']}")
        print(f"  Batch prompts: {config['batch_prompts']}")

        # Metrics
        print("\nMetrics:")
        print(f"  mIoU: {config['miou']:.2f}%")
        print(f"  F1: {config['f1']:.2f}%")
        print(f"  Boundary F1: {config['boundary_f1']:.2f}%")
        if config['person_iou'] is not None:
            print(f"  Person IoU: {config['person_iou']*100:.2f}%")


def print_best_config(results: List[Tuple[str, str, Dict]]):
    """Print the best configuration."""

    if not results:
        return

    # Sort by mIoU
    results = sorted(results, key=lambda x: x[2]['miou'], reverse=True)

    print("\n" + "=" * 140)
    print("BEST CONFIGURATION")
    print("=" * 140)

    config_name, dataset_name, config = results[0]

    print(f"\nConfiguration: {config_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {config['num_samples']}")
    print()

    print("Performance:")
    print(f"  mIoU: {config['miou']:.2f}%")
    print(f"  F1: {config['f1']:.2f}%")
    print(f"  Boundary F1: {config['boundary_f1']:.2f}%")
    print(f"  Time per sample: {config['elapsed_time'] / config['num_samples']:.2f}s")
    print()

    print("Key Settings:")
    print(f"  CLIP-guided SAM: {config['use_clip_guided_sam']}")
    print(f"  Min confidence: {config['min_confidence']}")
    print(f"  Min region size: {config['min_region_size']}")
    print(f"  IoU threshold: {config['iou_threshold']}")
    print(f"  Logit scale: {config['logit_scale']}")
    print(f"  Template strategy: {config['template_strategy']}")
    print(f"  Phase 1: {config['use_all_phase1']}")
    print(f"  Phase 2A: {config['use_all_phase2a']}")
    print(f"  CLIP-RC: {config['use_clip_rc']}")


def generate_recommendations(results: List[Tuple[str, str, Dict]]):
    """Generate hyperparameter recommendations based on results."""

    if len(results) < 2:
        print("\nNeed at least 2 configurations to generate recommendations.")
        return

    print("\n" + "=" * 140)
    print("RECOMMENDATIONS FOR HYPERPARAMETER TUNING")
    print("=" * 140)

    print("\nBased on the current results, here are configurations to try for 50 samples:")
    print()

    # Get best config
    best = sorted(results, key=lambda x: x[2]['miou'], reverse=True)[0][2]

    print("1. BASELINE (from best current config):")
    print(f"   min_confidence={best['min_confidence']}")
    print(f"   min_region_size={best['min_region_size']}")
    print(f"   iou_threshold={best['iou_threshold']}")
    print(f"   logit_scale={best['logit_scale']}")
    print(f"   template_strategy='{best['template_strategy']}'")
    print()

    print("2. SWEEP MIN_CONFIDENCE (if using CLIP-guided SAM):")
    print("   Try: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]")
    print("   Lower values = more regions but potentially noisier")
    print()

    print("3. SWEEP MIN_REGION_SIZE:")
    print("   Try: [50, 100, 200, 300, 500]")
    print("   Lower values = detect smaller objects, but slower")
    print()

    print("4. SWEEP IOU_THRESHOLD:")
    print("   Try: [0.5, 0.6, 0.7, 0.8, 0.9]")
    print("   Lower values = merge more overlapping regions")
    print()

    print("5. SWEEP LOGIT_SCALE:")
    print("   Try: [20.0, 30.0, 40.0, 50.0, 60.0]")
    print("   Higher values = sharper predictions")
    print()

    print("6. TEMPLATE STRATEGIES:")
    print("   Try: ['imagenet80', 'top7', 'top3', 'spatial', 'adaptive']")
    print("   'top7' is usually 3-4x faster with +2-3% mIoU")
    print()

    print("7. PHASE COMBINATIONS:")
    print("   - Phase 1 only (LoftUp + ResCLIP + DenseCRF)")
    print("   - Phase 2A only (CLIPtrase + CLIP-RC)")
    print("   - All phases combined")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description='Compare hyperparameter results')

    parser.add_argument('--results-dir', type=str, default='benchmarks/results',
                        help='Results directory')

    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter by dataset (e.g., coco-stuff, pascal-voc)')

    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed hyperparameter settings')

    parser.add_argument('--recommendations', action='store_true',
                        help='Generate hyperparameter recommendations')

    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Analyze results
    results = analyze_results(results_dir, args.dataset)

    if not results:
        sys.exit(1)

    # Print comparison table
    print_comparison_table(results)

    # Print details if requested
    if args.detailed:
        print_hyperparameter_details(results)

    # Print best config
    print_best_config(results)

    # Generate recommendations
    if args.recommendations:
        generate_recommendations(results)


if __name__ == '__main__':
    main()
