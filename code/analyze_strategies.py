#!/usr/bin/env python3
"""
Analyze and compare improved prompt extraction strategies.

Usage:
    python analyze_strategies.py benchmarks/results/improved_strategies_coco-stuff
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np


def load_strategy_results(results_dir):
    """Load results from all strategy experiments."""
    results_dir = Path(results_dir)

    strategies = {
        'Baseline (Current)': results_dir / 'baseline_current',
        'Strategy 1: Adaptive Threshold': results_dir / 'strategy1_adaptive',
        'Strategy 2: Conf-Weighted': results_dir / 'strategy2_conf_weighted',
        'Strategy 3: Density K-means': results_dir / 'strategy3_density',
        'Strategy 4: Prob Map (BEST)': results_dir / 'strategy4_prob_map',
    }

    loaded = {}
    for name, path in strategies.items():
        # Try different dataset names
        for dataset in ['coco-stuff', 'pascal-voc', 'cityscapes']:
            result_file = path / f'{dataset}_results.json'
            if result_file.exists():
                with open(result_file) as f:
                    loaded[name] = json.load(f)
                break
        else:
            print(f"Warning: No results found for {name} at {path}")

    return loaded


def print_comparison_table(results):
    """Print formatted comparison table."""
    print("\n" + "="*120)
    print("IMPROVED STRATEGIES COMPARISON")
    print("="*120)
    print()

    # Header
    header = f"{'Strategy':<35} {'Prompts':<12} {'Time/img':<12} {'GFLOPs':<12} {'mIoU':<10} {'Pixel Acc':<12} {'vs Baseline':<12}"
    print(header)
    print("-"*120)

    # Get baseline mIoU
    baseline_miou = None
    if 'Baseline (Current)' in results:
        baseline_miou = results['Baseline (Current)'].get('miou', 0) * 100

    # Print each strategy
    for strategy_name, data in results.items():
        # Extract metrics
        miou = data.get('miou', 0) * 100
        pixel_acc = data.get('pixel_accuracy', 0) * 100

        # Calculate time per image
        num_samples = data.get('args', {}).get('num_samples', 1)
        total_time = data.get('elapsed_time', 0)
        time_per_image = total_time / num_samples if num_samples > 0 else 0

        # Get GFLOPs
        profiling = data.get('profiling', {})
        total_gflops = profiling.get('total_gflops', 0)
        gflops_per_image = total_gflops / num_samples if num_samples > 0 and total_gflops > 0 else 0

        # Get num prompts
        num_prompts = profiling.get('num_sam_prompts', 0)
        prompts_per_image = num_prompts / num_samples if num_samples > 0 else 0

        # Calculate improvement vs baseline
        if baseline_miou and baseline_miou > 0:
            improvement = miou - baseline_miou
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"

        # Print row
        if gflops_per_image > 0:
            print(f"{strategy_name:<35} {prompts_per_image:>10.0f}  {time_per_image:>10.2f}s {gflops_per_image:>10.1f} {miou:>8.2f}% {pixel_acc:>10.2f}% {improvement_str:>10}")
        else:
            print(f"{strategy_name:<35} {prompts_per_image:>10.0f}  {time_per_image:>10.2f}s {'N/A':>10} {miou:>8.2f}% {pixel_acc:>10.2f}% {improvement_str:>10}")

    print("-"*120)
    print()


def analyze_best_strategy(results):
    """Identify and analyze the best performing strategy."""
    print("="*120)
    print("BEST STRATEGY ANALYSIS")
    print("="*120)
    print()

    # Find strategy with highest mIoU
    best_strategy = None
    best_miou = 0

    for strategy_name, data in results.items():
        miou = data.get('miou', 0) * 100
        if miou > best_miou:
            best_miou = miou
            best_strategy = strategy_name

    if best_strategy:
        print(f"🏆 Best Strategy: {best_strategy}")
        print(f"   mIoU: {best_miou:.2f}%")

        data = results[best_strategy]
        profiling = data.get('profiling', {})
        num_samples = data.get('args', {}).get('num_samples', 1)

        print(f"\nKey Metrics:")
        print(f"  Prompts/image: {profiling.get('num_sam_prompts', 0) / num_samples:.0f}")
        print(f"  Time/image: {data.get('elapsed_time', 0) / num_samples:.2f}s")

        if profiling.get('total_gflops', 0) > 0:
            print(f"  GFLOPs/image: {profiling.get('total_gflops', 0) / num_samples:.1f}")

        # Compare to baseline
        if 'Baseline (Current)' in results:
            baseline = results['Baseline (Current)']
            baseline_miou = baseline.get('miou', 0) * 100
            improvement = best_miou - baseline_miou

            print(f"\nImprovement over Baseline:")
            print(f"  mIoU: {improvement:+.2f}% ({baseline_miou:.2f}% → {best_miou:.2f}%)")

            if improvement > 0:
                print(f"  ✓ Successfully improved over baseline!")
            else:
                print(f"  ⚠ Still below baseline - needs more tuning")

    print()


def print_per_class_insights(results):
    """Analyze per-class performance improvements."""
    print("="*120)
    print("PER-CLASS IMPROVEMENTS (Top 10 Classes)")
    print("="*120)
    print()

    if 'Baseline (Current)' not in results:
        print("Baseline results not found - skipping per-class analysis")
        return

    baseline = results['Baseline (Current)']
    baseline_per_class = baseline.get('per_class_iou', {})

    # Find strategy with best overall mIoU
    best_strategy = max(results.items(), key=lambda x: x[1].get('miou', 0))[0]
    best_data = results[best_strategy]
    best_per_class = best_data.get('per_class_iou', {})

    # Calculate improvements per class
    improvements = {}
    for class_name in baseline_per_class.keys():
        baseline_iou = baseline_per_class.get(class_name, 0)
        best_iou = best_per_class.get(class_name, 0)

        if baseline_iou > 0 and not (isinstance(baseline_iou, float) and baseline_iou != baseline_iou):
            improvement = (best_iou - baseline_iou) * 100
            improvements[class_name] = {
                'baseline': baseline_iou,
                'best': best_iou,
                'improvement': improvement
            }

    # Sort by improvement
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1]['improvement'], reverse=True)

    # Print top improvers
    print("Top 10 Most Improved Classes:")
    print(f"{'Class':<25} {'Baseline':<12} {'Best Strategy':<15} {'Improvement':<12}")
    print("-"*70)

    for class_name, stats in sorted_improvements[:10]:
        baseline_iou = stats['baseline'] * 100
        best_iou = stats['best'] * 100
        improvement = stats['improvement']

        print(f"{class_name:<25} {baseline_iou:>10.2f}% {best_iou:>13.2f}% {improvement:>10.2f}%")

    print()

    # Print worst performers
    print("Top 10 Classes Still Needing Improvement:")
    print(f"{'Class':<25} {'Baseline':<12} {'Best Strategy':<15} {'Change':<12}")
    print("-"*70)

    for class_name, stats in sorted_improvements[-10:]:
        baseline_iou = stats['baseline'] * 100
        best_iou = stats['best'] * 100
        improvement = stats['improvement']

        print(f"{class_name:<25} {baseline_iou:>10.2f}% {best_iou:>13.2f}% {improvement:>10.2f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze improved strategy results')
    parser.add_argument('results_dir', type=str, help='Directory containing strategy results')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_strategy_results(args.results_dir)

    if not results:
        print("Error: No results found!")
        sys.exit(1)

    print(f"Found {len(results)} strategy result(s)")

    # Print comparisons
    print_comparison_table(results)
    analyze_best_strategy(results)
    print_per_class_insights(results)

    print("="*120)
    print("Analysis complete!")
    print("="*120)


if __name__ == '__main__':
    main()
