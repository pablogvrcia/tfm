"""
SCLIP Benchmark Runner with LoFTup Support

Evaluates SCLIP-based semantic segmentation on standard benchmarks.

Usage:
    # Dense mode (pure SCLIP, fastest, best performance)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10

    # With LoFTup enhancement (improved features)
    python run_benchmarks.py --dataset pascal-voc --num-samples 10 --use-loftup

    # Compare with and without LoFTup (A/B testing)
    python run_benchmarks.py --dataset pascal-voc --num-samples 10 --compare-loftup

    # Hybrid mode (SCLIP + SAM)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10 --use-sam

    # With PAMR refinement (better boundaries)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10 --use-pamr

    # Full comparison: baseline vs LoFTup on Pascal VOC
    python run_benchmarks.py --dataset pascal-voc --compare-loftup --save-vis
"""

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import time

from models.sclip_segmentor import SCLIPSegmentor
from datasets import COCOStuffDataset, PASCALVOCDataset
from benchmarks.metrics import compute_all_metrics

# Import CLIP-guided segmentation functions
try:
    from clip_guided_segmentation import (
        extract_prompt_points_from_clip,
        segment_with_guided_prompts,
        merge_overlapping_masks
    )
    CLIP_GUIDED_AVAILABLE = True
except ImportError:
    CLIP_GUIDED_AVAILABLE = False
    print("Warning: clip_guided_segmentation module not available")


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Evaluation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='coco-stuff',
                        choices=['coco-stuff', 'pascal-voc'],
                        help='Dataset to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/benchmarks',
                        help='Path to dataset directory')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')

    # SCLIP settings
    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help='CLIP model to use (SCLIP paper uses ViT-B/16)')
    parser.add_argument('--use-sam', action='store_true',
                        help='Use SAM for mask proposals (hybrid mode)')
    parser.add_argument('--use-clip-guided-sam', action='store_true',
                        help='Use CLIP-guided SAM with intelligent prompting and overlap resolution')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                        help='Minimum CLIP confidence for guided prompts (--use-clip-guided-sam only)')
    parser.add_argument('--min-region-size', type=int, default=100,
                        help='Minimum region size for guided prompts (--use-clip-guided-sam only)')
    parser.add_argument('--iou-threshold', type=float, default=0.8,
                        help='IoU threshold for merging overlaps (--use-clip-guided-sam only)')
    parser.add_argument('--use-pamr', action='store_true', default=False,
                        help='Use PAMR refinement (default: False, SCLIP disables by default)')
    parser.add_argument('--pamr-steps', type=int, default=10,
                        help='Number of PAMR iterations')
    parser.add_argument('--logit-scale', type=float, default=40.0,
                        help='Temperature scaling for logits')
    parser.add_argument('--prob-threshold', type=float, default=0.0,
                        help='Probability threshold for predictions')

    # Inference settings
    parser.add_argument('--slide-inference', action='store_true', default=True,
                        help='Use sliding window inference (default: True)')
    parser.add_argument('--slide-crop', type=int, default=224,
                        help='Crop size for sliding window (SCLIP default: 224)')
    parser.add_argument('--slide-stride', type=int, default=112,
                        help='Stride for sliding window (SCLIP default: 112)')

    # LoFTup settings
    parser.add_argument('--use-loftup', action='store_true', default=False,
                        help='Enable LoFTup feature upsampling (improved spatial resolution)')
    parser.add_argument('--no-loftup', action='store_false', dest='use_loftup',
                        help='Disable LoFTup feature upsampling')
    parser.add_argument('--loftup-adaptive', action='store_true', default=True,
                        help='Use adaptive LoFTup upsampling (adjusts factor based on feature size)')
    parser.add_argument('--loftup-factor', type=float, default=2.0,
                        help='Fixed LoFTup upsampling factor (if not adaptive, default: 2.0)')
    parser.add_argument('--compare-loftup', action='store_true',
                        help='Run comparison: baseline vs LoFTup (A/B testing)')

    # Output
    parser.add_argument('--output-dir', type=str, default='benchmarks/results',
                        help='Output directory for results')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualizations')

    return parser.parse_args()


def load_dataset(dataset_name, data_dir, num_samples):
    """Load dataset."""
    data_dir = Path(data_dir)

    if dataset_name == 'coco-stuff':
        dataset = COCOStuffDataset(
            data_dir=data_dir,
            split='val2017',
            max_samples=num_samples
        )
    elif dataset_name == 'pascal-voc':
        dataset = PASCALVOCDataset(
            data_dir=data_dir,
            split='val',
            max_samples=num_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def segment_with_clip_guided_sam(image, class_names, segmentor, args):
    """
    Perform CLIP-guided SAM segmentation.

    Uses the improved method from clip_guided_segmentation.py:
    1. CLIP dense prediction
    2. Extract intelligent prompts
    3. SAM segmentation at prompts
    4. Merge overlaps with cross-class resolution
    """
    import torch

    # Step 1: Get CLIP dense predictions
    seg_map, logits = segmentor.predict_dense(image, class_names, return_logits=True)
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    probs = probs.transpose(1, 2, 0)  # (H, W, num_classes)

    # Step 2: Extract prompt points
    prompts = extract_prompt_points_from_clip(
        seg_map, probs, class_names,
        min_confidence=args.min_confidence,
        min_region_size=args.min_region_size
    )

    if len(prompts) == 0:
        # Fallback to dense prediction if no prompts
        return seg_map

    # Step 3: Segment with guided prompts
    results = segment_with_guided_prompts(
        image, prompts,
        checkpoint_path="checkpoints/sam2_hiera_large.pt",
        model_cfg="sam2_hiera_l.yaml",
        device=segmentor.device
    )

    # Step 4: Merge overlapping masks
    results = merge_overlapping_masks(results, iou_threshold=args.iou_threshold)

    # Step 5: Convert results to dense segmentation map
    H, W = image.shape[:2]
    final_seg_map = np.zeros((H, W), dtype=np.int64)

    # Sort by confidence (higher confidence masks overwrite lower)
    sorted_results = sorted(results, key=lambda x: x['confidence'])

    for result in sorted_results:
        mask = result['mask']
        class_idx = result['class_idx']

        # Ensure mask is boolean and correct shape
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.shape != (H, W):
            # Resize mask if needed
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        final_seg_map[mask] = class_idx

    return final_seg_map


def run_single_evaluation(segmentor, dataset, args, mode_name="default"):
    """
    Run evaluation on dataset with given segmentor.

    Args:
        segmentor: SCLIPSegmentor instance
        dataset: Dataset instance
        args: Arguments
        mode_name: Name for this evaluation mode (e.g., "baseline", "loftup")

    Returns:
        dict: Results including metrics and timing
    """
    all_preds = []
    all_gts = []

    print(f"\n{'='*80}")
    print(f"Evaluating: {mode_name}")
    print(f"{'='*80}")

    start_time = time.time()

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {mode_name}"):
        # Load sample
        sample = dataset[idx]
        image = sample['image']
        gt_mask = sample['mask']

        # Predict
        if args.use_clip_guided_sam:
            pred_mask = segment_with_clip_guided_sam(image, dataset.class_names, segmentor, args)
        else:
            pred_mask = segmentor.segment(image, dataset.class_names)

        # Collect predictions
        all_preds.append(pred_mask)
        all_gts.append(gt_mask)

        # Save visualization if requested
        if args.save_vis:
            save_visualization(image, gt_mask, pred_mask, dataset, idx, args, mode_name)

    # Compute metrics per sample and average
    all_results = []
    for pred_mask, gt_mask in zip(all_preds, all_gts):
        sample_results = compute_all_metrics(
            pred_mask,
            gt_mask,
            num_classes=dataset.num_classes
        )
        all_results.append(sample_results)

    # Average metrics across all samples
    results = {
        'miou': np.mean([r['miou'] for r in all_results]),
        'pixel_accuracy': np.mean([r['pixel_accuracy'] for r in all_results]),
        'f1': np.mean([r['f1'] for r in all_results]),
        'precision': np.mean([r['precision'] for r in all_results]),
        'recall': np.mean([r['recall'] for r in all_results]),
        'boundary_f1': np.mean([r['boundary_f1'] for r in all_results]),
        'per_class_iou': {}
    }

    # Average per-class IoU
    for class_idx in range(dataset.num_classes):
        class_name = dataset.class_names[class_idx]
        class_ious = [r['per_class_iou'].get(class_idx, np.nan) for r in all_results]
        results['per_class_iou'][class_name] = np.nanmean(class_ious)

    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    results['time_per_image'] = elapsed_time / len(dataset)

    return results, all_preds, all_gts


def save_visualization(image, gt_mask, pred_mask, dataset, idx, args, mode_name="default"):
    """Save visualization of segmentation results."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    vis_dir = Path(args.output_dir) / 'visualizations' / args.dataset / mode_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(image)
    axes[0].set_title(f"Image {idx}", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Choose colormap based on dataset
    if dataset.num_classes <= 21:
        from matplotlib.colors import ListedColormap
        import matplotlib.cm as cm
        base_cmap = cm.get_cmap('tab20', 20)
        colors = [base_cmap(i) for i in range(20)]
        colors.insert(0, (0, 0, 0, 1))
        voc_cmap = ListedColormap(colors)
        cmap_viz = voc_cmap
        vmax_viz = dataset.num_classes - 1
        cmap_legend = cm.get_cmap('tab20', 20)
    else:
        cmap_viz = 'tab20'
        vmax_viz = 170
        cmap_legend = plt.cm.get_cmap('tab20', 171)

    # Ground truth
    axes[1].imshow(gt_mask, cmap=cmap_viz, vmin=0, vmax=vmax_viz)
    axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(pred_mask, cmap=cmap_viz, vmin=0, vmax=vmax_viz)
    axes[2].set_title(f"Prediction ({mode_name})", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Create legend
    gt_classes = np.unique(gt_mask[gt_mask != 255])
    pred_classes = np.unique(pred_mask)
    all_classes = sorted(set(gt_classes.tolist()) | set(pred_classes.tolist()))

    legend_elements = []
    for cls in all_classes:
        if dataset.num_classes <= 21:
            color = (0, 0, 0, 1) if cls == 0 else cmap_legend((cls - 1) / 19)
        else:
            color = cmap_legend(cls / 170)

        in_gt = "✓" if cls in gt_classes else ""
        in_pred = "✓" if cls in pred_classes else ""
        label = f"{dataset.class_names[cls]} {in_gt}{in_pred}"
        legend_elements.append(mpatches.Patch(color=color, label=label))

    num_cols = min(6, len(legend_elements))
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=num_cols, fontsize=8, frameon=True,
              title="Classes (✓✓ = in both, ✓ = in one)")

    num_rows = (len(legend_elements) + num_cols - 1) // num_cols
    legend_height = 0.08 + (num_rows - 1) * 0.03
    plt.tight_layout(rect=[0, legend_height, 1, 1])
    plt.savefig(vis_dir / f'sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_comparison_results(baseline_results, loftup_results, dataset):
    """Print comparison between baseline and LoFTup results."""
    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs LoFTup")
    print("=" * 80)

    # Overall metrics
    print("\nOverall Metrics:")
    print(f"{'Metric':<20} {'Baseline':<12} {'LoFTup':<12} {'Improvement':<12}")
    print("-" * 60)

    metrics = ['miou', 'pixel_accuracy', 'f1', 'precision', 'recall', 'boundary_f1']
    for metric in metrics:
        baseline_val = baseline_results[metric] * 100
        loftup_val = loftup_results[metric] * 100
        improvement = loftup_val - baseline_val
        improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"

        print(f"{metric:<20} {baseline_val:>6.2f}%     {loftup_val:>6.2f}%     {improvement_str:>12}")

    # Timing
    print("\nTiming:")
    baseline_time = baseline_results['time_per_image']
    loftup_time = loftup_results['time_per_image']
    overhead = ((loftup_time / baseline_time) - 1) * 100

    print(f"{'Time per image':<20} {baseline_time:>6.2f}s     {loftup_time:>6.2f}s     +{overhead:>5.1f}%")

    # Per-class improvements
    print("\nTop 10 Classes with Largest Improvements:")
    print(f"{'Class':<20} {'Baseline':<12} {'LoFTup':<12} {'Improvement':<12}")
    print("-" * 60)

    class_improvements = {}
    for class_name in baseline_results['per_class_iou'].keys():
        baseline_iou = baseline_results['per_class_iou'][class_name]
        loftup_iou = loftup_results['per_class_iou'][class_name]

        if not (np.isnan(baseline_iou) or np.isnan(loftup_iou)):
            improvement = (loftup_iou - baseline_iou) * 100
            class_improvements[class_name] = {
                'baseline': baseline_iou * 100,
                'loftup': loftup_iou * 100,
                'improvement': improvement
            }

    # Sort by improvement
    sorted_classes = sorted(class_improvements.items(),
                           key=lambda x: x[1]['improvement'],
                           reverse=True)

    for class_name, metrics in sorted_classes[:10]:
        baseline_val = metrics['baseline']
        loftup_val = metrics['loftup']
        improvement = metrics['improvement']
        improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"

        print(f"{class_name:<20} {baseline_val:>6.2f}%     {loftup_val:>6.2f}%     {improvement_str:>12}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    miou_improvement = (loftup_results['miou'] - baseline_results['miou']) * 100

    if miou_improvement > 0:
        print(f"✓ LoFTup improves mIoU by {miou_improvement:.2f} percentage points")
        print(f"  ({baseline_results['miou']*100:.2f}% → {loftup_results['miou']*100:.2f}%)")
    else:
        print(f"✗ LoFTup decreases mIoU by {abs(miou_improvement):.2f} percentage points")
        print(f"  ({baseline_results['miou']*100:.2f}% → {loftup_results['miou']*100:.2f}%)")

    print(f"✓ Computational overhead: +{overhead:.1f}%")

    # Count improved classes
    num_improved = sum(1 for v in class_improvements.values() if v['improvement'] > 0)
    num_total = len(class_improvements)
    print(f"✓ Improved classes: {num_improved}/{num_total} ({num_improved/num_total*100:.1f}%)")


def save_comparison_report(baseline_results, loftup_results, dataset, args):
    """Save detailed comparison report to JSON."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'dataset': args.dataset,
        'num_samples': len(dataset),
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'baseline': baseline_results,
        'loftup': loftup_results,
        'comparison': {
            'miou_improvement': (loftup_results['miou'] - baseline_results['miou']) * 100,
            'f1_improvement': (loftup_results['f1'] - baseline_results['f1']) * 100,
            'overhead_percent': ((loftup_results['time_per_image'] / baseline_results['time_per_image']) - 1) * 100
        }
    }

    report_file = output_dir / f"{args.dataset}_loftup_comparison.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison report saved to: {report_file}")

    return report_file


def main():
    args = parse_args()

    # Validate arguments
    if args.use_clip_guided_sam and not CLIP_GUIDED_AVAILABLE:
        print("Error: --use-clip-guided-sam requires clip_guided_segmentation module")
        return

    if args.use_clip_guided_sam and args.use_sam:
        print("Error: Cannot use both --use-sam and --use-clip-guided-sam")
        return

    print("=" * 80)
    print(f"Benchmark: {args.dataset.upper()}")
    print("=" * 80)

    if args.compare_loftup:
        print(f"Mode: COMPARISON (Baseline vs LoFTup)")
    elif args.use_clip_guided_sam:
        print(f"Mode: CLIP-Guided SAM (Intelligent prompting + overlap resolution)")
        print(f"  Min confidence: {args.min_confidence}")
        print(f"  Min region size: {args.min_region_size}")
        print(f"  IoU threshold: {args.iou_threshold}")
    elif args.use_sam:
        print(f"Mode: Hybrid (SAM + SCLIP)")
    else:
        print(f"Mode: Dense (SCLIP only)")

    print(f"PAMR: {args.use_pamr}")
    print(f"Slide inference: {args.slide_inference}")
    print(f"LoFTup: {args.use_loftup if not args.compare_loftup else 'Comparison mode'}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, args.data_dir, args.num_samples)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.num_classes}")
    print()

    # Run comparison mode if requested
    if args.compare_loftup:
        print("\n" + "=" * 80)
        print("RUNNING COMPARISON MODE")
        print("=" * 80)

        # 1. Run baseline (without LoFTup)
        print("\nInitializing baseline segmentor (without LoFTup)...")
        baseline_segmentor = SCLIPSegmentor(
            model_name=args.model,
            use_sam=args.use_sam if not args.use_clip_guided_sam else False,
            use_pamr=args.use_pamr,
            pamr_steps=args.pamr_steps,
            logit_scale=args.logit_scale,
            prob_threshold=args.prob_threshold,
            slide_inference=args.slide_inference,
            slide_crop=args.slide_crop,
            slide_stride=args.slide_stride,
            use_loftup=False,  # Disable LoFTup for baseline
            verbose=True
        )

        baseline_results, baseline_preds, baseline_gts = run_single_evaluation(
            baseline_segmentor, dataset, args, mode_name="baseline"
        )

        # 2. Run with LoFTup
        print("\nInitializing LoFTup-enhanced segmentor...")
        loftup_segmentor = SCLIPSegmentor(
            model_name=args.model,
            use_sam=args.use_sam if not args.use_clip_guided_sam else False,
            use_pamr=args.use_pamr,
            pamr_steps=args.pamr_steps,
            logit_scale=args.logit_scale,
            prob_threshold=args.prob_threshold,
            slide_inference=args.slide_inference,
            slide_crop=args.slide_crop,
            slide_stride=args.slide_stride,
            use_loftup=True,  # Enable LoFTup
            loftup_adaptive=args.loftup_adaptive,
            loftup_upsample_factor=args.loftup_factor,
            verbose=True
        )

        loftup_results, loftup_preds, loftup_gts = run_single_evaluation(
            loftup_segmentor, dataset, args, mode_name="loftup"
        )

        # 3. Print comparison
        print_comparison_results(baseline_results, loftup_results, dataset)

        # 4. Save comparison report
        save_comparison_report(baseline_results, loftup_results, dataset, args)

        return  # Exit after comparison

    # Single evaluation mode
    print("Initializing SCLIP segmentor...")
    segmentor = SCLIPSegmentor(
        model_name=args.model,
        use_sam=args.use_sam if not args.use_clip_guided_sam else False,
        use_pamr=args.use_pamr,
        pamr_steps=args.pamr_steps,
        logit_scale=args.logit_scale,
        prob_threshold=args.prob_threshold,
        slide_inference=args.slide_inference,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
        use_loftup=args.use_loftup,
        loftup_adaptive=args.loftup_adaptive,
        loftup_upsample_factor=args.loftup_factor,
        verbose=True
    )

    # Run single evaluation
    mode_name = "loftup" if args.use_loftup else "baseline"
    results, all_preds, all_gts = run_single_evaluation(
        segmentor, dataset, args, mode_name=mode_name
    )

    print(f"\nEvaluation completed in {results['elapsed_time']:.2f}s")
    print(f"Average time per image: {results['time_per_image']:.2f}s")

    # Print results
    print()
    print("=" * 80)
    print(f"Results for {args.dataset.upper()}")
    print("=" * 80)
    print(f"mIoU: {results['miou']*100:.2f}%")
    print(f"Pixel Accuracy: {results['pixel_accuracy']*100:.2f}%")
    print(f"F1 Score: {results['f1']*100:.2f}%")
    if 'boundary_f1' in results:
        print(f"Boundary F1: {results['boundary_f1']*100:.2f}%")
    print()

    # Print per-class IoU (top classes)
    print("Per-Class IoU:")
    per_class_iou = results['per_class_iou']
    sorted_classes = sorted(per_class_iou.items(), key=lambda x: (np.isnan(x[1]), -x[1]))
    for class_name, iou in sorted_classes[:30]:  # Top 30 classes
        if np.isnan(iou):
            print(f"  {class_name:20s}:    nan%")
        else:
            print(f"  {class_name:20s}: {iou*100:6.2f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{args.dataset}_results.json"
    results['timestamp'] = datetime.now().isoformat()
    results['args'] = vars(args)
    results['elapsed_time'] = elapsed_time

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    print()
    print("=" * 80)
    print("✓ Benchmark evaluation complete!")
    print("=" * 80)

    # Summary
    print()
    print("Results:")
    print()
    print(f"{args.dataset.upper()}:")
    print(f"  mIoU: {results['miou']*100:.2f}%")
    print(f"  F1: {results['f1']*100:.2f}%")


if __name__ == '__main__':
    main()
