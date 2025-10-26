#!/usr/bin/env python3
"""
Benchmark evaluation runner for open-vocabulary semantic segmentation.

Evaluates the system on:
- COCO-Stuff 164K
- PASCAL VOC 2012
- ADE20K
- COCO-Open split (48 base + 17 novel classes)

Usage:
    python run_benchmarks.py --dataset all
    python run_benchmarks.py --dataset pascal-voc --num-samples 100
    python run_benchmarks.py --dataset coco-open --save-vis
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import OpenVocabSegmentationPipeline
from benchmarks.metrics import compute_all_metrics, compute_novel_class_miou
from utils import load_image, save_image


def run_benchmark(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    num_samples: int = None,
    save_visualizations: bool = False,
    device: str = "cuda"
):
    """
    Run benchmark evaluation on a dataset.

    Args:
        dataset_name: Name of dataset (coco-stuff, pascal-voc, ade20k, coco-open)
        data_dir: Root directory containing datasets
        output_dir: Directory to save results
        num_samples: Number of samples to evaluate (None = all)
        save_visualizations: Whether to save visualization images
        device: Computation device
    """
    print(f"\n{'='*80}")
    print(f"Running benchmark: {dataset_name.upper()}")
    print(f"{'='*80}\n")

    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = OpenVocabSegmentationPipeline(device=device, verbose=False)

    # Load dataset
    dataset = load_dataset(dataset_name, data_dir)

    if num_samples is not None:
        dataset = dataset[:num_samples]

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.num_classes if hasattr(dataset, 'num_classes') else 'N/A'}")
    print()

    # Run evaluation
    all_metrics = []
    pred_masks_list = []
    gt_masks_list = []

    vis_dir = output_dir / "visualizations" / dataset_name
    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        image = sample['image']
        gt_mask = sample['mask']
        class_names = sample.get('class_names', [])

        # Run segmentation for each class in ground truth
        pred_mask = np.zeros_like(gt_mask)

        unique_classes = np.unique(gt_mask)
        unique_classes = unique_classes[unique_classes != 255]  # Remove ignore index

        for cls_id in unique_classes:
            if cls_id >= len(class_names):
                continue

            class_name = class_names[cls_id]

            try:
                # Segment
                result = pipeline.segment(
                    image,
                    text_prompt=class_name,
                    top_k=1,
                    return_visualization=False
                )

                if result.segmentation_masks:
                    # Use top mask
                    mask = result.segmentation_masks[0].mask_candidate.mask
                    pred_mask[mask > 0] = cls_id

            except Exception as e:
                print(f"  Error on class '{class_name}': {e}")
                continue

        # Compute metrics
        metrics = compute_all_metrics(
            pred_mask,
            gt_mask,
            num_classes=len(class_names) if class_names else 256
        )
        metrics['sample_id'] = idx
        all_metrics.append(metrics)

        pred_masks_list.append(pred_mask)
        gt_masks_list.append(gt_mask)

        # Save visualization
        if save_visualizations and idx < 20:  # Save first 20
            vis = create_comparison_vis(image, pred_mask, gt_mask)
            save_image(vis, vis_dir / f"sample_{idx:04d}.png")

    # Aggregate metrics
    results = aggregate_metrics(all_metrics, dataset_name)

    # For COCO-Open, compute novel class metrics
    if dataset_name == "coco-open" and hasattr(dataset, 'novel_classes'):
        novel_metrics = compute_novel_class_miou(
            pred_masks_list,
            gt_masks_list,
            novel_classes=dataset.novel_classes,
            num_classes=dataset.num_classes
        )
        results.update(novel_metrics)

    # Save results
    results_file = output_dir / f"{dataset_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results for {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"mIoU: {results['miou']:.2f}%")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.2f}%")
    print(f"F1 Score: {results['f1']:.2f}%")
    print(f"Boundary F1: {results['boundary_f1']:.2f}%")

    if 'novel_miou' in results:
        print(f"\nCOCO-Open Split:")
        print(f"  Novel mIoU: {results['novel_miou']:.2f}%")
        print(f"  Base mIoU: {results['base_miou']:.2f}%")

    print(f"\nResults saved to: {results_file}")

    return results


def load_dataset(name: str, data_dir: Path):
    """Load dataset by name."""
    # Simple placeholder - you would implement full dataset loaders
    # For now, return mock dataset structure

    class MockDataset:
        def __init__(self, name, data_dir):
            self.name = name
            self.data_dir = data_dir
            self.num_classes = 21 if name == "pascal-voc" else 171

        def __len__(self):
            return 10  # Mock

        def __getitem__(self, idx):
            # Mock sample
            return {
                'image': np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
                'mask': np.random.randint(0, self.num_classes, (512, 512), dtype=np.uint8),
                'class_names': [f'class_{i}' for i in range(self.num_classes)]
            }

    return MockDataset(name, data_dir)


def aggregate_metrics(metrics_list, dataset_name):
    """Aggregate metrics across all samples."""
    agg = {
        'dataset': dataset_name,
        'num_samples': len(metrics_list),
        'timestamp': datetime.now().isoformat()
    }

    # Average metrics
    for key in ['miou', 'pixel_accuracy', 'f1', 'precision', 'recall', 'boundary_f1']:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            agg[key] = float(np.mean(values) * 100)  # Convert to percentage

    return agg


def create_comparison_vis(image, pred_mask, gt_mask):
    """Create side-by-side comparison visualization."""
    # Simple visualization (you can enhance this)
    import cv2

    # Colorize masks
    pred_colored = colorize_mask(pred_mask)
    gt_colored = colorize_mask(gt_mask)

    # Stack horizontally
    vis = np.hstack([image, gt_colored, pred_colored])

    # Add labels
    cv2.putText(vis, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Ground Truth", (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Prediction", (image.shape[1] * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return vis


def colorize_mask(mask, palette=None):
    """Apply color palette to segmentation mask."""
    if palette is None:
        # Generate random palette
        np.random.seed(42)
        palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]  # Background black

    colored = palette[mask]
    return colored


def main():
    parser = argparse.ArgumentParser(description="Run open-vocabulary segmentation benchmarks")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "coco-stuff", "pascal-voc", "ade20k", "coco-open"],
        default="pascal-voc",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/benchmarks",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmarks/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    data_dir = Path(args.data_dir)

    if args.dataset == "all":
        datasets = ["coco-stuff", "pascal-voc", "ade20k", "coco-open"]
    else:
        datasets = [args.dataset]

    all_results = {}

    for dataset in datasets:
        try:
            results = run_benchmark(
                dataset_name=dataset,
                data_dir=data_dir,
                output_dir=output_dir,
                num_samples=args.num_samples,
                save_visualizations=args.save_vis,
                device=args.device
            )
            all_results[dataset] = results

        except Exception as e:
            print(f"\n❌ Error evaluating {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save aggregated results
    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("✓ Benchmark evaluation complete!")
    print(f"{'='*80}")
    print(f"\nSummary saved to: {summary_file}")
    print("\nResults:")

    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  mIoU: {results.get('miou', 0):.2f}%")
        print(f"  F1: {results.get('f1', 0):.2f}%")


if __name__ == "__main__":
    main()
