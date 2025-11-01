"""
SCLIP Benchmark Runner

Evaluates SCLIP-based semantic segmentation on standard benchmarks.

Usage:
    # Dense mode (pure SCLIP, fastest, best performance)
    python run_sclip_benchmarks.py --dataset coco-stuff --num-samples 10

    # Hybrid mode (SCLIP + SAM)
    python run_sclip_benchmarks.py --dataset coco-stuff --num-samples 10 --use-sam

    # With PAMR refinement (better boundaries)
    python run_sclip_benchmarks.py --dataset coco-stuff --num-samples 10 --use-pamr

    # Sliding window inference (slower but better)
    python run_sclip_benchmarks.py --dataset coco-stuff --num-samples 10 --slide-inference
"""

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import time

from sclip_segmentor import SCLIPSegmentor
from datasets import COCOStuffDataset, PASCALVOCDataset
from benchmarks.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='SCLIP Benchmark Evaluation')

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


def main():
    args = parse_args()

    print("=" * 80)
    print(f"SCLIP Benchmark: {args.dataset.upper()}")
    print("=" * 80)
    print(f"Mode: {'Hybrid (SAM + SCLIP)' if args.use_sam else 'Dense (SCLIP only)'}")
    print(f"PAMR: {args.use_pamr}")
    print(f"Slide inference: {args.slide_inference}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, args.data_dir, args.num_samples)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.num_classes}")
    print()

    # Initialize SCLIP segmentor
    print("Initializing SCLIP segmentor...")
    segmentor = SCLIPSegmentor(
        model_name=args.model,
        use_sam=args.use_sam,
        use_pamr=args.use_pamr,
        pamr_steps=args.pamr_steps,
        logit_scale=args.logit_scale,
        prob_threshold=args.prob_threshold,
        slide_inference=args.slide_inference,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
        verbose=True
    )

    # Collect predictions and ground truth
    all_preds = []
    all_gts = []

    # Evaluation loop
    print("Starting evaluation...")
    start_time = time.time()

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        # Load sample
        sample = dataset[idx]
        image = sample['image']
        gt_mask = sample['mask']

        # Predict
        pred_mask = segmentor.segment(image, dataset.class_names)

        # Collect predictions
        all_preds.append(pred_mask)
        all_gts.append(gt_mask)

        # Save visualization if requested
        if args.save_vis:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            vis_dir = Path(args.output_dir) / 'visualizations' / args.dataset
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create visualization with adaptive height for legend
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))

            axes[0].imshow(image)
            axes[0].set_title(f"Image {idx}", fontsize=14, fontweight='bold')
            axes[0].axis('off')

            # Choose colormap based on dataset
            if dataset.num_classes <= 21:
                # Pascal VOC - use distinctive colors for few classes
                from matplotlib.colors import ListedColormap
                import matplotlib.cm as cm
                # Use a colormap that works well for small number of classes
                base_cmap = cm.get_cmap('tab20', 20)
                colors = [base_cmap(i) for i in range(20)]
                colors.insert(0, (0, 0, 0, 1))  # Black for background/class 0
                voc_cmap = ListedColormap(colors)
                cmap_viz = voc_cmap
                vmax_viz = dataset.num_classes - 1
                cmap_legend = cm.get_cmap('tab20', 20)
            else:
                # COCO-Stuff - use extended colormap
                cmap_viz = 'tab20'
                vmax_viz = 170
                cmap_legend = plt.cm.get_cmap('tab20', 171)

            # Ground truth with legend
            axes[1].imshow(gt_mask, cmap=cmap_viz, vmin=0, vmax=vmax_viz)
            axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # Prediction with legend
            axes[2].imshow(pred_mask, cmap=cmap_viz, vmin=0, vmax=vmax_viz)
            axes[2].set_title("SCLIP Prediction", fontsize=14, fontweight='bold')
            axes[2].axis('off')

            # Create legend showing present classes
            gt_classes = np.unique(gt_mask[gt_mask != 255])
            pred_classes = np.unique(pred_mask)

            # Show ALL classes that appear in either GT or predictions
            all_classes = sorted(set(gt_classes.tolist()) | set(pred_classes.tolist()))
            display_classes = all_classes  # Show all present classes, no limit

            # Create color patches for legend
            legend_elements = []
            for cls in display_classes:
                if dataset.num_classes <= 21:
                    # Pascal VOC coloring
                    if cls == 0:
                        color = (0, 0, 0, 1)  # Black for background
                    else:
                        color = cmap_legend((cls - 1) / 19)
                else:
                    # COCO-Stuff coloring
                    color = cmap_legend(cls / 170)

                in_gt = "✓" if cls in gt_classes else ""
                in_pred = "✓" if cls in pred_classes else ""
                label = f"{dataset.class_names[cls]} {in_gt}{in_pred}"
                legend_elements.append(mpatches.Patch(color=color, label=label))

            # Add legend below the images with adaptive columns
            num_cols = min(6, len(legend_elements))  # Up to 6 columns
            fig.legend(handles=legend_elements, loc='lower center',
                      ncol=num_cols, fontsize=8, frameon=True,
                      title="Classes (✓✓ = in both GT and pred, ✓ = in one only)")

            # Calculate space needed for legend based on number of rows
            num_rows = (len(legend_elements) + num_cols - 1) // num_cols
            legend_height = 0.08 + (num_rows - 1) * 0.03  # More space per row
            plt.tight_layout(rect=[0, legend_height, 1, 1])  # Leave space for legend
            plt.savefig(vis_dir / f'sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()

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

    # Average per-class IoU (use nanmean to ignore classes not in GT)
    for class_idx in range(dataset.num_classes):
        class_name = dataset.class_names[class_idx]
        class_ious = [r['per_class_iou'].get(class_idx, np.nan) for r in all_results]
        results['per_class_iou'][class_name] = np.nanmean(class_ious)

    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f}s")
    print(f"Average time per image: {elapsed_time / len(dataset):.2f}s")

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

    results_file = output_dir / f"{args.dataset}_sclip_results.json"
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
    print(f"  mIoU: {results['miou']:.2f}%")
    print(f"  F1: {results['f1']:.2f}%")


if __name__ == '__main__':
    main()
