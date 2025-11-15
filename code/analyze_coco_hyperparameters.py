"""
Hyperparameter Analysis for COCO-Stuff Dataset

This script helps analyze and compare different hyperparameter configurations
for COCO-Stuff semantic segmentation benchmarks.

Usage:
    python analyze_coco_hyperparameters.py --mode analyze  # Analyze existing results
    python analyze_coco_hyperparameters.py --mode sweep    # Run hyperparameter sweep
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import subprocess
import sys


def load_result(result_file: Path) -> Dict:
    """Load a single result JSON file."""
    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_existing_results(results_dir: Path = Path('benchmarks/results')):
    """Analyze all existing COCO-Stuff results."""

    # Find all COCO-Stuff result files
    coco_results = list(results_dir.glob('**/coco-stuff_results.json'))

    if not coco_results:
        print("No COCO-Stuff results found!")
        return None

    print(f"Found {len(coco_results)} COCO-Stuff result file(s)\n")

    # Extract key metrics and hyperparameters
    data = []
    for result_file in coco_results:
        result = load_result(result_file)
        args = result.get('args', {})

        row = {
            'config_name': result_file.parent.name,
            'num_samples': args.get('num_samples', 'unknown'),
            'miou': result.get('miou', 0) * 100,
            'pixel_acc': result.get('pixel_accuracy', 0) * 100,
            'f1': result.get('f1', 0) * 100,
            'boundary_f1': result.get('boundary_f1', 0) * 100,
            'elapsed_time': result.get('elapsed_time', 0),

            # Hyperparameters
            'use_sam': args.get('use_sam', False),
            'use_clip_guided_sam': args.get('use_clip_guided_sam', False),
            'use_pamr': args.get('use_pamr', False),
            'slide_inference': args.get('slide_inference', True),
            'logit_scale': args.get('logit_scale', 40.0),
            'prob_threshold': args.get('prob_threshold', 0.0),
            'min_confidence': args.get('min_confidence', 0.3),
            'min_region_size': args.get('min_region_size', 100),
            'iou_threshold': args.get('iou_threshold', 0.8),

            # Phase 1 improvements
            'use_loftup': args.get('use_loftup', False) or args.get('use_all_phase1', False),
            'use_resclip': args.get('use_resclip', False) or args.get('use_all_phase1', False),
            'use_densecrf': args.get('use_densecrf', False) or args.get('use_all_phase1', False),

            # Phase 2A improvements
            'use_cliptrase': args.get('use_cliptrase', False) or args.get('use_all_phase2a', False),
            'use_clip_rc': args.get('use_clip_rc', False) or args.get('use_all_phase2a', False),

            # Phase 2B improvements
            'template_strategy': args.get('template_strategy', 'imagenet80'),

            # Phase 2C improvements
            'use_confidence_sharpening': args.get('use_confidence_sharpening', False),
            'use_hierarchical_prediction': args.get('use_hierarchical_prediction', False),

            # Performance optimizations
            'use_fp16': args.get('use_fp16', True),
            'batch_prompts': args.get('batch_prompts', True),
        }

        # Add per-class IoU for important classes
        per_class_iou = result.get('per_class_iou', {})
        row['person_iou'] = per_class_iou.get('person', np.nan)

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values('miou', ascending=False)

    return df


def print_analysis(df: pd.DataFrame):
    """Print formatted analysis of results."""

    print("=" * 100)
    print("COCO-STUFF HYPERPARAMETER ANALYSIS")
    print("=" * 100)
    print()

    # Summary table
    print("Performance Summary (sorted by mIoU):")
    print("-" * 100)

    summary_cols = ['config_name', 'num_samples', 'miou', 'pixel_acc', 'f1', 'boundary_f1', 'elapsed_time']
    summary_df = df[summary_cols].copy()
    summary_df['time_per_sample'] = summary_df['elapsed_time'] / summary_df['num_samples']

    print(summary_df.to_string(index=False))
    print()

    # Hyperparameter impact analysis
    print("\n" + "=" * 100)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("=" * 100)

    # Binary hyperparameters
    binary_params = [
        'use_sam', 'use_clip_guided_sam', 'use_pamr', 'slide_inference',
        'use_loftup', 'use_resclip', 'use_densecrf',
        'use_cliptrase', 'use_clip_rc',
        'use_confidence_sharpening', 'use_hierarchical_prediction'
    ]

    print("\nBinary Hyperparameters:")
    print("-" * 100)
    print(f"{'Parameter':<30} {'Enabled mIoU':<15} {'Disabled mIoU':<15} {'Delta':<10}")
    print("-" * 100)

    for param in binary_params:
        if param in df.columns:
            enabled = df[df[param] == True]['miou'].mean() if (df[param] == True).any() else np.nan
            disabled = df[df[param] == False]['miou'].mean() if (df[param] == False).any() else np.nan
            delta = enabled - disabled if not (np.isnan(enabled) or np.isnan(disabled)) else np.nan

            if not np.isnan(delta):
                symbol = "+" if delta > 0 else ""
                print(f"{param:<30} {enabled:>14.2f}% {disabled:>14.2f}% {symbol}{delta:>9.2f}%")

    print()

    # Continuous hyperparameters
    print("\nContinuous Hyperparameters:")
    print("-" * 100)
    continuous_params = ['logit_scale', 'prob_threshold', 'min_confidence', 'min_region_size', 'iou_threshold']

    for param in continuous_params:
        if param in df.columns and df[param].nunique() > 1:
            print(f"\n{param}:")
            param_analysis = df.groupby(param)['miou'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            print(param_analysis)

    print()

    # Template strategy analysis
    if 'template_strategy' in df.columns and df['template_strategy'].nunique() > 1:
        print("\nTemplate Strategy Analysis:")
        print("-" * 100)
        template_analysis = df.groupby('template_strategy').agg({
            'miou': 'mean',
            'elapsed_time': 'mean',
            'f1': 'mean'
        }).sort_values('miou', ascending=False)
        print(template_analysis)
        print()

    # Best configuration
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION")
    print("=" * 100)

    best_idx = df['miou'].idxmax()
    best_config = df.loc[best_idx]

    print(f"\nConfiguration: {best_config['config_name']}")
    print(f"Samples: {best_config['num_samples']}")
    print(f"mIoU: {best_config['miou']:.2f}%")
    print(f"F1: {best_config['f1']:.2f}%")
    print(f"Boundary F1: {best_config['boundary_f1']:.2f}%")
    print(f"Time per sample: {best_config['elapsed_time'] / best_config['num_samples']:.2f}s")
    print()

    print("Hyperparameters:")
    for param in binary_params + continuous_params + ['template_strategy']:
        if param in best_config.index:
            print(f"  {param}: {best_config[param]}")


def generate_sweep_configs(base_config: Dict) -> List[Dict]:
    """Generate hyperparameter sweep configurations."""

    configs = []

    # Baseline (your provided config)
    baseline = base_config.copy()
    baseline['name'] = 'baseline-clip-rc'
    configs.append(baseline)

    # Test different min_confidence values (for CLIP-guided SAM)
    for min_conf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        cfg = base_config.copy()
        cfg['name'] = f'min-conf-{min_conf}'
        cfg['min_confidence'] = min_conf
        configs.append(cfg)

    # Test different min_region_size values
    for min_size in [50, 100, 200, 300, 500]:
        cfg = base_config.copy()
        cfg['name'] = f'min-region-{min_size}'
        cfg['min_region_size'] = min_size
        configs.append(cfg)

    # Test different iou_threshold values
    for iou_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        cfg = base_config.copy()
        cfg['name'] = f'iou-thresh-{iou_thresh}'
        cfg['iou_threshold'] = iou_thresh
        configs.append(cfg)

    # Test different logit_scale values
    for logit_scale in [20.0, 30.0, 40.0, 50.0, 60.0]:
        cfg = base_config.copy()
        cfg['name'] = f'logit-scale-{int(logit_scale)}'
        cfg['logit_scale'] = logit_scale
        configs.append(cfg)

    # Test different template strategies
    for template in ['imagenet80', 'top7', 'top3', 'spatial', 'adaptive']:
        cfg = base_config.copy()
        cfg['name'] = f'template-{template}'
        cfg['template_strategy'] = template
        configs.append(cfg)

    # Test phase combinations
    phase_combos = [
        {'name': 'no-phases', 'use_all_phase1': False, 'use_all_phase2a': False, 'use_clip_rc': False},
        {'name': 'phase1-only', 'use_all_phase1': True, 'use_all_phase2a': False, 'use_clip_rc': False},
        {'name': 'phase2a-only', 'use_all_phase1': False, 'use_all_phase2a': True, 'use_clip_rc': False},
        {'name': 'clip-rc-only', 'use_all_phase1': False, 'use_all_phase2a': False, 'use_clip_rc': True},
        {'name': 'all-phases', 'use_all_phase1': True, 'use_all_phase2a': True, 'use_clip_rc': True},
    ]

    for combo in phase_combos:
        cfg = base_config.copy()
        cfg.update(combo)
        configs.append(cfg)

    # Test with/without PAMR
    cfg_pamr = base_config.copy()
    cfg_pamr['name'] = 'with-pamr'
    cfg_pamr['use_pamr'] = True
    configs.append(cfg_pamr)

    # Test different slide window settings
    slide_configs = [
        {'crop': 224, 'stride': 56},   # More overlap
        {'crop': 224, 'stride': 112},  # Default
        {'crop': 224, 'stride': 168},  # Less overlap
        {'crop': 336, 'stride': 168},  # Larger window
    ]

    for slide_cfg in slide_configs:
        cfg = base_config.copy()
        cfg['name'] = f"slide-{slide_cfg['crop']}-{slide_cfg['stride']}"
        cfg['slide_crop'] = slide_cfg['crop']
        cfg['slide_stride'] = slide_cfg['stride']
        configs.append(cfg)

    return configs


def run_sweep(configs: List[Dict], num_samples: int = 50, dataset: str = 'coco-stuff'):
    """Run hyperparameter sweep."""

    print(f"Running hyperparameter sweep with {len(configs)} configurations...")
    print(f"Dataset: {dataset}, Samples: {num_samples}")
    print()

    results = []

    for i, config in enumerate(configs, 1):
        config_name = config.get('name', f'config_{i}')
        print(f"\n[{i}/{len(configs)}] Running configuration: {config_name}")
        print("-" * 80)

        # Build command
        cmd = [
            'python', 'run_benchmarks.py',
            '--dataset', dataset,
            '--num-samples', str(num_samples),
            '--output-dir', f'benchmarks/results/sweep/{config_name}',
            '--save-vis'
        ]

        # Add config parameters
        for key, value in config.items():
            if key == 'name':
                continue

            param_name = '--' + key.replace('_', '-')

            if isinstance(value, bool):
                if value:
                    cmd.append(param_name)
            else:
                cmd.extend([param_name, str(value)])

        print(f"Command: {' '.join(cmd)}")

        # Run benchmark
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            results.append({'config': config_name, 'status': 'success'})
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Configuration {config_name} failed!")
            print(e.stderr)
            results.append({'config': config_name, 'status': 'failed', 'error': str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"Total configurations: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed configurations:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['config']}: {r.get('error', 'Unknown error')}")


def parse_args():
    parser = argparse.ArgumentParser(description='COCO-Stuff Hyperparameter Analysis')

    parser.add_argument('--mode', type=str, default='analyze',
                        choices=['analyze', 'sweep'],
                        help='Mode: analyze existing results or run sweep')

    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples for sweep (default: 50)')

    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for analysis results')

    parser.add_argument('--results-dir', type=str, default='benchmarks/results',
                        help='Results directory for analysis')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'analyze':
        # Analyze existing results
        results_dir = Path(args.results_dir)
        df = analyze_existing_results(results_dir)

        if df is not None:
            print_analysis(df)

            if args.output:
                output_path = Path(args.output)
                df.to_csv(output_path, index=False)
                print(f"\nResults saved to: {output_path}")

    elif args.mode == 'sweep':
        # Generate and run sweep
        base_config = {
            'dataset': 'coco-stuff',
            'model': 'ViT-B/16',
            'use_sam': False,
            'use_clip_guided_sam': False,
            'min_confidence': 0.3,
            'min_region_size': 100,
            'iou_threshold': 0.8,
            'use_pamr': False,
            'logit_scale': 40.0,
            'prob_threshold': 0.0,
            'slide_inference': True,
            'slide_crop': 224,
            'slide_stride': 112,
            'use_fp16': True,
            'use_compile': False,
            'batch_prompts': True,
            'use_loftup': False,
            'use_resclip': False,
            'use_densecrf': False,
            'use_all_phase1': False,
            'use_cliptrase': False,
            'use_clip_rc': True,
            'use_all_phase2a': False,
            'template_strategy': 'imagenet80',
            'use_confidence_sharpening': False,
            'use_hierarchical_prediction': False,
        }

        configs = generate_sweep_configs(base_config)
        run_sweep(configs, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
