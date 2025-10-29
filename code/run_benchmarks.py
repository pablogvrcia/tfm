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
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import OpenVocabSegmentationPipeline
from benchmarks.metrics import compute_all_metrics, compute_novel_class_miou
from utils import save_image
from datasets import load_dataset


def run_benchmark(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    num_samples: int = None,
    save_visualizations: bool = False,
    device: str = "cuda",
    mode: str = "oracle"
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
        mode: Evaluation mode:
            - "oracle": Only prompt classes present in GT (faster, upper bound)
            - "open-vocab": Prompt ALL vocabulary classes (realistic, slower)
    """
    print(f"\n{'='*80}")
    print(f"Running benchmark: {dataset_name.upper()}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*80}\n")

    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = OpenVocabSegmentationPipeline(device=device, verbose=False)

    # Load dataset
    dataset = load_dataset(dataset_name, data_dir, max_samples=num_samples)

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.num_classes if hasattr(dataset, 'num_classes') else 'N/A'}")
    print(f"Evaluation mode: {mode}")
    if mode == "oracle":
        print("  → Only prompting classes present in each image's ground truth")
    else:
        print(f"  → Prompting ALL {dataset.num_classes} classes for every image (true open-vocabulary)")
    print()

    # Run evaluation
    all_metrics = []
    pred_masks_list = []
    gt_masks_list = []

    vis_dir = output_dir / "visualizations" / dataset_name
    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        image = sample['image']
        gt_mask = sample['mask']
        class_names = sample.get('class_names', [])

        # Initialize prediction mask
        pred_mask = np.zeros_like(gt_mask)
        confidence_map = np.zeros(gt_mask.shape, dtype=np.float32)

        # Determine which classes to evaluate
        if mode == "oracle":
            # Oracle mode: Only prompt classes present in GT
            unique_classes = np.unique(gt_mask)
            unique_classes = unique_classes[unique_classes != 255]  # Remove ignore index
            unique_classes = unique_classes[unique_classes != 0]  # Skip background
        else:
            # Open-vocabulary mode: Prompt ALL classes
            unique_classes = np.arange(1, len(class_names))  # All classes except background (0)

        # Choose evaluation method based on mode
        if mode == "open-vocab":
            # Use batch mode for open-vocabulary (much faster for many classes)
            text_prompts = [class_names[cls_id] for cls_id in unique_classes if cls_id < len(class_names)]

            class_to_masks = pipeline.segment_batch(
                image,
                text_prompts,
                use_background_suppression=True,
                score_threshold=0.12,
                top_k_per_class=5
            )

            # Build prediction mask from batch results
            for class_name, masks_list in class_to_masks.items():
                try:
                    cls_id = class_names.index(class_name)
                except ValueError:
                    continue

                for scored_mask in masks_list:
                    mask = scored_mask.mask_candidate.mask
                    score = scored_mask.final_score
                    mask_size = mask.sum()

                    # Skip tiny masks
                    min_size = (gt_mask.shape[0] * gt_mask.shape[1]) * 0.001
                    if mask_size < min_size:
                        continue

                    # Check overlap with already-assigned regions of THIS class
                    already_assigned_this_class = (pred_mask == cls_id)
                    overlap_with_class = (mask & already_assigned_this_class).sum() / max(mask_size, 1)

                    # Skip if mostly overlaps (>70%)
                    if overlap_with_class > 0.7:
                        continue

                    # Assign pixels where confidence is higher
                    update_mask = (mask > 0) & (score > confidence_map)
                    pred_mask[update_mask] = cls_id
                    confidence_map[update_mask] = score

        else:
            # Oracle mode: Sequential processing (original method)
            for cls_id in unique_classes:
                if cls_id >= len(class_names):
                    continue

                class_name = class_names[cls_id]

                try:
                    # Get all high-confidence masks without adaptive selection
                    # For benchmarks, we want ALL instances of the class
                    result = pipeline.segment(
                        image,
                        text_prompt=class_name,
                        top_k=20,  # Get many candidates
                        use_adaptive_selection=False,  # Don't use adaptive - get all masks
                        return_visualization=False
                    )

                    if result.segmentation_masks:
                        # Sort masks by size (largest first) to prioritize complete objects
                        sorted_masks = sorted(result.segmentation_masks,
                                            key=lambda x: x.mask_candidate.mask.sum(),
                                            reverse=True)

                        # Strategy: Take largest mask with high score, then add non-overlapping instances
                        for seg_result in sorted_masks:
                            mask = seg_result.mask_candidate.mask
                            score = seg_result.final_score
                            mask_size = mask.sum()

                            # Skip tiny masks (noise) - must be at least 0.1% of image
                            min_size = (gt_mask.shape[0] * gt_mask.shape[1]) * 0.001
                            if mask_size < min_size:
                                continue

                            # Score threshold: adaptive based on mask size
                            # Larger masks can have slightly lower scores
                            score_threshold = 0.15 if mask_size > min_size * 50 else 0.20

                            if score < score_threshold:
                                continue

                            # Calculate overlap with already-assigned regions of THIS class
                            already_assigned_this_class = (pred_mask == cls_id)
                            overlap_with_class = (mask & already_assigned_this_class).sum() / max(mask_size, 1)

                            # Skip if mostly overlaps with existing mask of same class (>70%)
                            if overlap_with_class > 0.7:
                                continue

                            # Assign pixels where confidence is higher
                            update_mask = (mask > 0) & (score > confidence_map)
                            pred_mask[update_mask] = cls_id
                            confidence_map[update_mask] = score

                except Exception as e:
                    print(f"  Error on class '{class_name}': {e}")
                    continue

        # Assign background to unassigned pixels (confidence = 0)
        pred_mask[confidence_map == 0] = 0

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
        if save_visualizations:  # Save first 20
            vis = create_comparison_vis(image, pred_mask, gt_mask)
            save_image(vis, vis_dir / f"sample_{idx:04d}.png")

    # Aggregate metrics
    results = aggregate_metrics(all_metrics, dataset_name)

    # Compute per-class metrics across all samples
    if hasattr(dataset, 'class_names'):
        per_class_metrics = compute_per_class_metrics(
            pred_masks_list,
            gt_masks_list,
            dataset.class_names
        )
        results['per_class_iou'] = per_class_metrics

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

    if 'per_class_iou' in results and results['per_class_iou']:
        print(f"\nPer-Class IoU:")
        per_class = results['per_class_iou']
        # Sort by IoU descending
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        for class_name, iou in sorted_classes:
            print(f"  {class_name:15s}: {iou:5.2f}%")

    print(f"\nResults saved to: {results_file}")

    return results


class PASCALVOCDataset:
    """PASCAL VOC 2012 Segmentation Dataset Loader."""

    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]

    def __init__(self, data_dir: Path, split='val', max_samples=None):
        self.data_dir = Path(data_dir) / "pascal_voc" / "VOCdevkit" / "VOC2012"
        self.split = split
        self.num_classes = 21
        self.class_names = self.CLASSES

        # Load image IDs from split file
        split_file = self.data_dir / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing by creating new dataset with subset
            start, stop, step = idx.indices(len(self))
            new_dataset = PASCALVOCDataset.__new__(PASCALVOCDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.num_classes = self.num_classes
            new_dataset.class_names = self.class_names
            new_dataset.image_ids = self.image_ids[idx]
            return new_dataset

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        img_id = self.image_ids[idx]

        # Load image
        img_path = self.data_dir / "JPEGImages" / f"{img_id}.jpg"
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load segmentation mask
        mask_path = self.data_dir / "SegmentationClass" / f"{img_id}.png"
        mask = np.array(Image.open(mask_path))

        return {
            'image': image,
            'mask': mask,
            'class_names': self.class_names,
            'image_id': img_id
        }


class MockDataset:
    """Mock dataset for testing."""
    def __init__(self, name, data_dir, max_samples=10):
        self.name = name
        self.data_dir = data_dir
        self.num_classes = 21 if name == "pascal-voc" else 171
        self.max_samples = max_samples
        self.class_names = [f'class_{i}' for i in range(self.num_classes)]

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            num_samples = len(range(start, stop, step))
            return MockDataset(self.name, self.data_dir, max_samples=num_samples)

        if idx >= self.max_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.max_samples}")

        return {
            'image': np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            'mask': np.random.randint(0, self.num_classes, (512, 512), dtype=np.uint8),
            'class_names': self.class_names
        }


def compute_per_class_metrics(pred_masks_list, gt_masks_list, class_names):
    """Compute per-class IoU across all samples."""
    num_classes = len(class_names)
    class_intersections = np.zeros(num_classes, dtype=np.float64)
    class_unions = np.zeros(num_classes, dtype=np.float64)
    class_present = np.zeros(num_classes, dtype=np.bool_)

    for pred_mask, gt_mask in zip(pred_masks_list, gt_masks_list):
        for cls_id in range(num_classes):
            pred_cls = (pred_mask == cls_id)
            gt_cls = (gt_mask == cls_id)

            if gt_cls.sum() > 0:  # Class present in ground truth
                class_present[cls_id] = True
                intersection = np.logical_and(pred_cls, gt_cls).sum()
                union = np.logical_or(pred_cls, gt_cls).sum()

                class_intersections[cls_id] += intersection
                class_unions[cls_id] += union

    # Compute per-class IoU
    per_class_iou = {}
    for cls_id in range(num_classes):
        if class_present[cls_id] and class_unions[cls_id] > 0:
            iou = (class_intersections[cls_id] / class_unions[cls_id]) * 100
            per_class_iou[class_names[cls_id]] = float(iou)

    return per_class_iou


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
        # PASCAL VOC standard color palette
        palette = np.array([
            [0, 0, 0],       # 0: background
            [128, 0, 0],     # 1: aeroplane
            [0, 128, 0],     # 2: bicycle
            [128, 128, 0],   # 3: bird
            [0, 0, 128],     # 4: boat
            [128, 0, 128],   # 5: bottle
            [0, 128, 128],   # 6: bus
            [128, 128, 128], # 7: car
            [64, 0, 0],      # 8: cat
            [192, 0, 0],     # 9: chair
            [64, 128, 0],    # 10: cow
            [192, 128, 0],   # 11: diningtable
            [64, 0, 128],    # 12: dog
            [192, 0, 128],   # 13: horse
            [64, 128, 128],  # 14: motorbike
            [192, 128, 128], # 15: person
            [0, 64, 0],      # 16: pottedplant
            [128, 64, 0],    # 17: sheep
            [0, 192, 0],     # 18: sofa
            [128, 192, 0],   # 19: train
            [0, 64, 128],    # 20: tvmonitor
        ], dtype=np.uint8)

        # Extend palette for higher indices
        if mask.max() >= len(palette):
            np.random.seed(42)
            extra = np.random.randint(0, 255, (256 - len(palette), 3), dtype=np.uint8)
            palette = np.vstack([palette, extra])

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
    parser.add_argument(
        "--mode",
        type=str,
        default="oracle",
        choices=["oracle", "open-vocab"],
        help="Evaluation mode: 'oracle' (only GT classes) or 'open-vocab' (all classes)"
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
                device=args.device,
                mode=args.mode
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
