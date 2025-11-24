"""
SCLIP Benchmark Runner

Evaluates SCLIP-based semantic segmentation on standard benchmarks.

Usage:
    # Dense mode (pure SCLIP, fastest, best performance)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10
    python run_benchmarks.py --dataset cityscapes --num-samples 10

    # Hybrid mode (SCLIP + SAM)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10 --use-sam

    # With PAMR refinement (better boundaries)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10 --use-pamr

    # Sliding window inference (slower but better)
    python run_benchmarks.py --dataset coco-stuff --num-samples 10 --slide-inference
"""

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import time
import psutil
import torch
from collections import defaultdict

from models.sclip_segmentor import SCLIPSegmentor
from datasets import COCOStuffDataset, PASCALVOCDataset, CityscapesDataset
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


class PerformanceProfiler:
    """
    Performance profiling for benchmarks - inspired by 2025 optimization papers.

    Tracks: timing, memory usage, GPU utilization, throughput
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.gpu_memory = defaultdict(list)
        self.start_times = {}

    def start(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.time()

    def end(self, name: str):
        """End timing a section and record metrics."""
        if name not in self.start_times:
            return

        elapsed = time.time() - self.start_times[name]
        self.timings[name].append(elapsed)

        # Record memory usage
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage[name].append(mem_mb)

        # Record GPU memory if available
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.gpu_memory[name].append(gpu_mem_mb)

        del self.start_times[name]

    def get_summary(self):
        """Get performance summary."""
        summary = {}

        for name in self.timings:
            times = self.timings[name]
            summary[name] = {
                'count': len(times),
                'total_time': sum(times),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'mean_memory_mb': np.mean(self.memory_usage[name]) if self.memory_usage[name] else 0,
                'mean_gpu_memory_mb': np.mean(self.gpu_memory[name]) if self.gpu_memory[name] else 0,
            }

        return summary

    def print_summary(self):
        """Print formatted performance summary."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("PERFORMANCE PROFILING SUMMARY (2025 Optimizations)")
        print("="*80)

        for name, stats in sorted(summary.items()):
            print(f"\n{name}:")
            print(f"  Count:        {stats['count']}")
            print(f"  Total time:   {stats['total_time']:.2f}s")
            print(f"  Mean time:    {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
            print(f"  Min/Max time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
            print(f"  Throughput:   {1.0/stats['mean_time']:.2f} samples/sec")
            if stats['mean_memory_mb'] > 0:
                print(f"  Mean RAM:     {stats['mean_memory_mb']:.1f} MB")
            if stats['mean_gpu_memory_mb'] > 0:
                print(f"  Mean GPU RAM: {stats['mean_gpu_memory_mb']:.1f} MB")

        print("\n" + "="*80)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Evaluation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='coco-stuff',
                        choices=['coco-stuff', 'pascal-voc', 'cityscapes'],
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
    parser.add_argument('--points-per-cluster', type=int, default=1,
                        help='Number of points per cluster (1=centroid only, >1=multiple points) (--use-clip-guided-sam only)')
    parser.add_argument('--negative-points-per-cluster', type=int, default=0,
                        help='Number of negative (background) points per cluster (0=disabled) (--use-clip-guided-sam only)')
    parser.add_argument('--negative-confidence-threshold', type=float, default=0.8,
                        help='Minimum confidence for negative point regions (--use-clip-guided-sam only)')
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

    # Output
    parser.add_argument('--output-dir', type=str, default='benchmarks/results',
                        help='Output directory for results')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualizations')

    # Performance optimizations (2025 papers)
    parser.add_argument('--use-fp16', action='store_true', default=True,
                        help='Enable FP16 mixed precision (inspired by TernaryCLIP 2025)')
    parser.add_argument('--use-compile', action='store_true', default=False,
                        help='Enable torch.compile() for JIT optimization (PyTorch 2.0+)')
    parser.add_argument('--batch-prompts', action='store_true', default=True,
                        help='Enable batch processing for SAM prompts (inspired by EfficientViT-SAM 2024)')
    parser.add_argument('--enable-profiling', action='store_true', default=False,
                        help='Enable detailed performance profiling')

    # Phase 1 improvements (ICCV/CVPR 2025 papers for mIoU improvement)
    parser.add_argument('--use-loftup', action='store_true', default=False,
                        help='Enable LoftUp feature upsampling (+2-4%% mIoU, ICCV 2025)')
    parser.add_argument('--loftup-mode', type=str, default='fast', choices=['fast', 'accurate'],
                        help='LoftUp mode: fast (2x upsample, faster) or accurate (full SCLIP approach, +2%% mIoU)')
    parser.add_argument('--use-resclip', action='store_true', default=False,
                        help='Enable ResCLIP residual attention (+8-13%% mIoU, CVPR 2025)')
    parser.add_argument('--use-densecrf', action='store_true', default=False,
                        help='Enable DenseCRF boundary refinement (+1-2%% mIoU, +3-5%% boundary F1)')
    parser.add_argument('--use-all-phase1', action='store_true', default=False,
                        help='Enable all Phase 1 improvements (LoftUp + ResCLIP + DenseCRF)')

    # Multi-descriptor support (SCLIP's cls_voc21.txt approach)
    parser.add_argument('--descriptor-file', type=str, default=None,
                        help='Path to SCLIP descriptor file (e.g., configs/cls_voc21.txt)')

    # Phase 2A improvements (2025 - training-free for human parsing)
    parser.add_argument('--use-cliptrase', action='store_true', default=False,
                        help='Enable CLIPtrase self-correlation recalibration (+5-10%% mIoU person, ECCV 2024)')
    parser.add_argument('--use-clip-rc', action='store_true', default=False,
                        help='Enable CLIP-RC regional clues extraction (+8-12%% mIoU person, CVPR 2024)')
    parser.add_argument('--use-all-phase2a', action='store_true', default=False,
                        help='Enable all Phase 2A improvements (CLIPtrase + CLIP-RC)')

    # Phase 2B improvements (2025 - prompt engineering)
    parser.add_argument('--template-strategy', type=str, default='imagenet80',
                        choices=['imagenet80', 'top7', 'spatial', 'top3', 'adaptive'],
                        help='Prompt template strategy (Phase 2B):\n'
                             '  imagenet80: Original 80 ImageNet templates (baseline)\n'
                             '  top7: Top-7 dense prediction templates (recommended, 3-4x faster, +2-3%% mIoU)\n'
                             '  spatial: Spatial context templates (+1-2%% mIoU)\n'
                             '  top3: Ultra-fast top-3 templates (5x faster)\n'
                             '  adaptive: Adaptive per-class (stuff vs thing, +3-5%% mIoU)')

    # Phase 2C improvements (2025 - confidence sharpening for flat predictions)
    parser.add_argument('--use-confidence-sharpening', action='store_true', default=False,
                        help='Enable confidence sharpening for flat predictions (+5-8%% mIoU, Phase 2C)')
    parser.add_argument('--use-hierarchical-prediction', action='store_true', default=False,
                        help='Enable hierarchical class grouping (+3-5%% mIoU, Phase 2C)')

    # Phase 3 improvements (MHQR - Multi-scale Hierarchical Query-based Refinement)
    parser.add_argument('--use-mhqr', action='store_true', default=False,
                        help='Enable full MHQR pipeline (+8-15%% mIoU expected, Phase 3)')
    parser.add_argument('--mhqr-dynamic-queries', action='store_true', default=True,
                        help='Use dynamic multi-scale query generation (Phase 3, default: True)')
    parser.add_argument('--mhqr-hierarchical-decoder', action='store_true', default=False,
                        help='Use hierarchical mask decoder (Phase 3, default: False - simplified MHQR does not use this)')
    parser.add_argument('--mhqr-semantic-merging', action='store_true', default=False,
                        help='Use semantic-guided mask merging (Phase 3, default: False - simplified MHQR does not use this)')
    parser.add_argument('--mhqr-scales', type=float, nargs='+', default=[0.25, 0.5, 1.0],
                        help='Multi-scale pyramid scales (Phase 3, default: [0.25, 0.5, 1.0])')
    parser.add_argument('--use-all-phase3', action='store_true', default=False,
                        help='Enable all Phase 3 MHQR components')

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
    elif dataset_name == 'cityscapes':
        dataset = CityscapesDataset(
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
        min_region_size=args.min_region_size,
        points_per_cluster=args.points_per_cluster,
        negative_points_per_cluster=args.negative_points_per_cluster,
        negative_confidence_threshold=args.negative_confidence_threshold
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
    if args.use_clip_guided_sam:
        print(f"Mode: CLIP-Guided SAM (Intelligent prompting + overlap resolution)")
        print(f"  Min confidence: {args.min_confidence}")
        print(f"  Min region size: {args.min_region_size}")
        print(f"  Points per cluster: {args.points_per_cluster}")
        if args.negative_points_per_cluster > 0:
            print(f"  Negative points per cluster: {args.negative_points_per_cluster}")
            print(f"  Negative confidence threshold: {args.negative_confidence_threshold}")
        print(f"  IoU threshold: {args.iou_threshold}")
    elif args.use_sam:
        print(f"Mode: Hybrid (SAM + SCLIP)")
    else:
        print(f"Mode: Dense (SCLIP only)")
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

    # Initialize performance profiler
    profiler = PerformanceProfiler() if args.enable_profiling else None

    # Initialize SCLIP segmentor with 2025 optimizations
    print("Initializing SCLIP segmentor with 2025 optimizations...")
    if args.use_fp16:
        print("  ✓ FP16 mixed precision enabled (TernaryCLIP 2025)")
    if args.use_compile:
        print("  ✓ torch.compile() enabled")
    if args.batch_prompts:
        print("  ✓ Batch prompt processing enabled (EfficientViT-SAM 2024)")

    # Handle --use-all-phase1 flag
    use_loftup = args.use_loftup or args.use_all_phase1
    use_resclip = args.use_resclip or args.use_all_phase1
    use_densecrf = args.use_densecrf or args.use_all_phase1

    # Handle --use-all-phase2a flag
    use_cliptrase = args.use_cliptrase or args.use_all_phase2a
    use_clip_rc = args.use_clip_rc or args.use_all_phase2a

    # Handle --use-all-phase3 flag
    use_mhqr = args.use_mhqr or args.use_all_phase3

    # Print Phase 1 status
    if use_loftup or use_resclip or use_densecrf:
        print("\n Phase 1 Improvements (ICCV/CVPR 2025 for mIoU):")
        if use_loftup:
            print("  ✓ LoftUp feature upsampling (+2-4% mIoU expected)")
        if use_resclip:
            print("  ✓ ResCLIP residual attention (+8-13% mIoU expected)")
        if use_densecrf:
            print("  ✓ DenseCRF boundary refinement (+1-2% mIoU, +3-5% boundary F1 expected)")
        total_expected = sum([
            3 if use_loftup else 0,
            10 if use_resclip else 0,
            1.5 if use_densecrf else 0
        ])
        print(f"  → Total expected improvement: +{total_expected:.1f}% mIoU")

    # Print Phase 2A status (training-free human parsing)
    if use_cliptrase or use_clip_rc:
        print("\n Phase 2A Improvements (Training-Free Human Parsing):")
        if use_cliptrase:
            print("  ✓ CLIPtrase self-correlation recalibration (+5-10% mIoU person expected)")
        if use_clip_rc:
            print("  ✓ CLIP-RC regional clues extraction (+8-12% mIoU person expected)")
        total_expected_person = sum([
            7.5 if use_cliptrase else 0,
            10 if use_clip_rc else 0
        ])
        print(f"  → Total expected improvement for person class: +{total_expected_person:.1f}% mIoU")

    # Print Phase 3 status (MHQR - Multi-scale Hierarchical Query-based Refinement)
    if use_mhqr:
        print("\n Phase 3 Improvements (MHQR - Multi-scale Hierarchical Query-based Refinement):")
        print("  ✓ Dynamic multi-scale query generation" if args.mhqr_dynamic_queries else "  ✗ Dynamic queries disabled")
        print("  ✓ Hierarchical mask decoder" if args.mhqr_hierarchical_decoder else "  ✗ Hierarchical decoder disabled")
        print("  ✓ Semantic-guided mask merging" if args.mhqr_semantic_merging else "  ✗ Semantic merging disabled")
        print(f"  → Scales: {args.mhqr_scales}")
        print(f"  → Total expected improvement: +8-15% mIoU")

    segmentor = SCLIPSegmentor(
        model_name=args.model,
        use_sam=args.use_sam if not args.use_clip_guided_sam else False,  # Disable built-in SAM for clip-guided
        use_pamr=args.use_pamr,
        pamr_steps=args.pamr_steps,
        logit_scale=args.logit_scale,
        prob_threshold=args.prob_threshold,
        slide_inference=args.slide_inference,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
        verbose=True,
        # Multi-descriptor support
        descriptor_file=args.descriptor_file,
        # 2025 optimization parameters
        use_fp16=args.use_fp16,
        use_compile=args.use_compile,
        batch_prompts=args.batch_prompts,
        # Phase 1 improvements (ICCV/CVPR 2025 for mIoU)
        use_loftup=use_loftup,
        loftup_mode=args.loftup_mode,
        use_resclip=use_resclip,
        use_densecrf=use_densecrf,
        # Phase 2A improvements (2025 - training-free human parsing)
        use_cliptrase=use_cliptrase,
        use_clip_rc=use_clip_rc,
        # Phase 2B improvements (2025 - prompt engineering)
        template_strategy=args.template_strategy,
        # Phase 2C improvements (2025 - confidence sharpening)
        use_confidence_sharpening=args.use_confidence_sharpening,
        use_hierarchical_prediction=args.use_hierarchical_prediction,
        # Phase 3 improvements (MHQR - Multi-scale Hierarchical Query-based Refinement)
        use_mhqr=use_mhqr,
        mhqr_dynamic_queries=args.mhqr_dynamic_queries,
        mhqr_hierarchical_decoder=args.mhqr_hierarchical_decoder,
        mhqr_semantic_merging=args.mhqr_semantic_merging,
        mhqr_scales=args.mhqr_scales,
    )
    print()

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

        # Profile prediction performance
        if profiler:
            profiler.start('total_inference')

        # Predict
        if args.use_clip_guided_sam:
            if profiler:
                profiler.start('clip_guided_sam')
            pred_mask = segment_with_clip_guided_sam(image, dataset.class_names, segmentor, args)
            if profiler:
                profiler.end('clip_guided_sam')
        else:
            if profiler:
                profiler.start('sclip_segment')
            pred_mask = segmentor.segment(image, dataset.class_names)
            if profiler:
                profiler.end('sclip_segment')

        if profiler:
            profiler.end('total_inference')

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
            axes[2].set_title("Prediction", fontsize=14, fontweight='bold')
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

    # Print performance profiling summary
    if profiler:
        profiler.print_summary()

    # Summary
    print()
    print("Results:")
    print()
    print(f"{args.dataset.upper()}:")
    print(f"  mIoU: {results['miou']*100:.2f}%")
    print(f"  F1: {results['f1']*100:.2f}%")


if __name__ == '__main__':
    main()
