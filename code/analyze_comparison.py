#!/usr/bin/env python3
"""
Analyze and compare benchmark results from different methods.

Usage:
    python analyze_comparison.py benchmarks/results/comparison_coco-stuff
"""

import argparse
import json
from pathlib import Path
import sys


def load_results(results_dir):
    """Load results from all methods."""
    results_dir = Path(results_dir)

    methods = {
        'Dense SCLIP': results_dir / 'dense_sclip',
        'Blind Grid 32×32': results_dir / 'blind_grid_32x32',
        'Blind Grid 64×64': results_dir / 'blind_grid_64x64',
        'CLIP-Guided SAM': results_dir / 'clip_guided',
    }

    loaded = {}
    for name, path in methods.items():
        result_file = path / 'coco-stuff_results.json'
        if not result_file.exists():
            result_file = path / 'pascal-voc_results.json'
        if not result_file.exists():
            print(f"Warning: No results found for {name} at {path}")
            continue

        with open(result_file) as f:
            loaded[name] = json.load(f)

    return loaded


def print_comparison_table(results):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*100)
    print()

    # Header
    print(f"{'Method':<25} {'Prompts':<12} {'Time (s)':<12} {'GFLOPs':<12} {'mIoU':<10} {'Pixel Acc':<12} {'F1':<10} {'Speedup':<10}")
    print("-"*115)

    # Extract baseline time
    baseline_time = None
    if 'Dense SCLIP' in results:
        baseline_time = results['Dense SCLIP']['elapsed_time'] / len(results['Dense SCLIP'].get('args', {}).get('num_samples', 1))

    # Print each method
    for method, data in results.items():
        # Extract metrics
        miou = data.get('miou', 0) * 100
        pixel_acc = data.get('pixel_accuracy', 0) * 100
        f1 = data.get('f1', 0) * 100

        # Calculate time per image
        num_samples = data.get('args', {}).get('num_samples', 1)
        total_time = data.get('elapsed_time', 0)
        time_per_image = total_time / num_samples if num_samples > 0 else 0

        # Get GFLOPs if available
        profiling = data.get('profiling', {})
        total_gflops = profiling.get('total_gflops', 0)
        gflops_per_image = total_gflops / num_samples if num_samples > 0 and total_gflops > 0 else 0

        # Estimate number of prompts
        if 'blind_grid_32x32' in str(data.get('args', {}).get('output_dir', '')):
            num_prompts = "~800-1024"
        elif 'blind_grid_64x64' in str(data.get('args', {}).get('output_dir', '')):
            num_prompts = "~3000-4096"
        elif 'clip_guided' in str(data.get('args', {}).get('output_dir', '')):
            num_prompts = "50-300"
        else:
            num_prompts = "0"

        # Calculate speedup
        if baseline_time and baseline_time > 0:
            speedup = baseline_time / time_per_image
        else:
            speedup = 1.0

        # Print row with GFLOPs if available
        if gflops_per_image > 0:
            print(f"{method:<25} {num_prompts:<12} {time_per_image:>10.2f}s {gflops_per_image:>10.1f} {miou:>8.2f}% {pixel_acc:>10.2f}% {f1:>8.2f}% {speedup:>8.2f}×")
        else:
            print(f"{method:<25} {num_prompts:<12} {time_per_image:>10.2f}s {'N/A':>10} {miou:>8.2f}% {pixel_acc:>10.2f}% {f1:>8.2f}% {speedup:>8.2f}×")

    print("-"*115)
    print()


def analyze_efficiency(results):
    """Analyze efficiency metrics."""
    print("="*100)
    print("EFFICIENCY ANALYSIS")
    print("="*100)
    print()

    if 'CLIP-Guided SAM' not in results or 'Blind Grid 64×64' not in results:
        print("Warning: Missing methods for efficiency comparison")
        return

    guided = results['CLIP-Guided SAM']
    blind = results['Blind Grid 64×64']

    # Extract metrics
    guided_samples = guided.get('args', {}).get('num_samples', 1)
    blind_samples = blind.get('args', {}).get('num_samples', 1)

    guided_time = guided.get('elapsed_time', 0) / guided_samples
    blind_time = blind.get('elapsed_time', 0) / blind_samples

    guided_miou = guided.get('miou', 0) * 100
    blind_miou = blind.get('miou', 0) * 100

    # Get GFLOPs
    guided_profiling = guided.get('profiling', {})
    blind_profiling = blind.get('profiling', {})

    guided_gflops = guided_profiling.get('total_gflops', 0) / guided_samples if guided_samples > 0 else 0
    blind_gflops = blind_profiling.get('total_gflops', 0) / blind_samples if blind_samples > 0 else 0

    # Calculate improvements
    time_speedup = blind_time / guided_time if guided_time > 0 else 0
    gflops_reduction = (blind_gflops - guided_gflops) / blind_gflops * 100 if blind_gflops > 0 else 0
    prompt_reduction = (4096 - 200) / 4096 * 100  # Assuming ~200 prompts for guided
    miou_diff = guided_miou - blind_miou

    print(f"CLIP-Guided SAM vs Blind Grid 64×64:")
    print(f"  Time speedup:      {time_speedup:.1f}× faster")
    print(f"  Prompt reduction:  ~{prompt_reduction:.0f}% fewer prompts (200 vs 4096)")
    if guided_gflops > 0 and blind_gflops > 0:
        print(f"  GFLOPs reduction:  {gflops_reduction:.1f}% ({guided_gflops:.1f} vs {blind_gflops:.1f})")
    print(f"  mIoU difference:   {miou_diff:+.2f}% ({guided_miou:.2f}% vs {blind_miou:.2f}%)")
    print()

    if miou_diff >= 0 and time_speedup > 1:
        print(f"✓ CLIP-Guided achieves BETTER or EQUAL mIoU with {time_speedup:.1f}× speedup")
    elif miou_diff < 0 and abs(miou_diff) < 2 and time_speedup > 5:
        print(f"✓ CLIP-Guided achieves comparable mIoU (-{abs(miou_diff):.2f}%) with {time_speedup:.1f}× speedup")
    else:
        print(f"⚠ Results need analysis - check configurations")

    print()


def print_per_class_comparison(results):
    """Print per-class IoU comparison for top classes."""
    print("="*100)
    print("PER-CLASS IoU COMPARISON (Top 10 Classes)")
    print("="*100)
    print()

    if not results:
        return

    # Get first method to extract class names
    first_method = list(results.values())[0]
    per_class = first_method.get('per_class_iou', {})

    if not per_class:
        print("No per-class results available")
        return

    # Sort classes by average IoU across all methods
    class_avgs = {}
    for class_name in per_class.keys():
        ious = []
        for method_results in results.values():
            iou = method_results.get('per_class_iou', {}).get(class_name, 0)
            if not (iou is None or (isinstance(iou, float) and iou != iou)):  # Check for nan
                ious.append(iou)
        if ious:
            class_avgs[class_name] = sum(ious) / len(ious)

    # Get top 10 classes
    top_classes = sorted(class_avgs.items(), key=lambda x: x[1], reverse=True)[:10]

    # Print header
    print(f"{'Class':<20}", end='')
    for method in results.keys():
        print(f"{method[:18]:>20}", end='')
    print()
    print("-"*100)

    # Print each class
    for class_name, _ in top_classes:
        print(f"{class_name:<20}", end='')
        for method_results in results.values():
            iou = method_results.get('per_class_iou', {}).get(class_name, 0)
            if iou is None or (isinstance(iou, float) and iou != iou):  # Check for nan
                print(f"{'N/A':>20}", end='')
            else:
                print(f"{iou*100:>18.2f}%", end='')
        print()

    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark comparison results')
    parser.add_argument('results_dir', type=str, help='Directory containing comparison results')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print("Error: No results found!")
        sys.exit(1)

    print(f"Found {len(results)} method(s): {', '.join(results.keys())}")

    # Print comparisons
    print_comparison_table(results)
    analyze_efficiency(results)
    print_per_class_comparison(results)

    print("="*100)
    print("Analysis complete!")
    print("="*100)


if __name__ == '__main__':
    main()
