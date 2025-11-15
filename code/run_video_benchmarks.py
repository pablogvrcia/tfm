"""
Video Segmentation Benchmark Runner

Evaluates CLIP-guided SAM2 video segmentation on standard video benchmarks.

Usage:
    # DAVIS 2017 validation set
    python run_video_benchmarks.py --dataset davis-2017 --num-samples 5

    # YouTube-VOS validation set
    python run_video_benchmarks.py --dataset youtube-vos --num-samples 10

    # With specific settings
    python run_video_benchmarks.py --dataset davis-2017 --min-confidence 0.5 --save-vis
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
from models.video_segmentation import CLIPGuidedVideoSegmentor
from video_datasets import load_video_dataset
from benchmarks.video_metrics import compute_all_video_metrics, aggregate_video_metrics

# Import CLIP-guided segmentation functions
try:
    from clip_guided_segmentation import (
        extract_prompt_points_from_clip,
        generate_distinct_colors
    )
    CLIP_GUIDED_AVAILABLE = True
except ImportError:
    CLIP_GUIDED_AVAILABLE = False
    print("Warning: clip_guided_segmentation module not available")


class PerformanceProfiler:
    """Performance profiling for video benchmarks."""

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
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*80)

        for name, stats in sorted(summary.items()):
            print(f"\n{name}:")
            print(f"  Count:        {stats['count']}")
            print(f"  Total time:   {stats['total_time']:.2f}s")
            print(f"  Mean time:    {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
            print(f"  Min/Max time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
            if stats['mean_memory_mb'] > 0:
                print(f"  Mean RAM:     {stats['mean_memory_mb']:.1f} MB")
            if stats['mean_gpu_memory_mb'] > 0:
                print(f"  Mean GPU RAM: {stats['mean_gpu_memory_mb']:.1f} MB")

        print("\n" + "="*80)


def parse_args():
    parser = argparse.ArgumentParser(description='Video Segmentation Benchmark Evaluation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='davis-2017',
                        choices=['davis-2016', 'davis-2017', 'youtube-vos'],
                        help='Video dataset to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/video_benchmarks',
                        help='Path to video dataset directory')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of videos to evaluate (None = all)')

    # SCLIP settings
    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help='CLIP model to use')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                        help='Minimum CLIP confidence for guided prompts')
    parser.add_argument('--min-region-size', type=int, default=100,
                        help='Minimum region size for guided prompts')
    parser.add_argument('--iou-threshold', type=float, default=0.8,
                        help='IoU threshold for merging overlaps')

    # SAM2 settings
    parser.add_argument('--checkpoint', default='checkpoints/sam2_hiera_large.pt',
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--model-cfg', default='sam2_hiera_l.yaml',
                        help='SAM2 model configuration')

    # Output
    parser.add_argument('--output-dir', type=str, default='benchmarks/video_results',
                        help='Output directory for results')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualizations')

    # Performance optimizations
    parser.add_argument('--enable-profiling', action='store_true', default=False,
                        help='Enable detailed performance profiling')

    # Video-specific settings
    parser.add_argument('--use-first-frame-only', action='store_true',
                        help='Extract prompts from first frame only (faster)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames per video to process (None = all)')

    # Class vocabulary (optional - if not provided, will be inferred from dataset)
    parser.add_argument('--vocabulary', nargs='+', default=None,
                        help='Custom class vocabulary for CLIP (optional)')

    return parser.parse_args()


def get_default_vocabulary(dataset_name: str) -> list:
    """
    Get default vocabulary for video datasets.

    For video object segmentation, we use generic object categories
    since we don't have predefined class labels.
    """
    # Generic vocabulary for open-vocabulary video segmentation
    # Based on common objects in DAVIS and YouTube-VOS
    return [
        'person', 'car', 'dog', 'cat', 'horse', 'bike', 'motorbike',
        'airplane', 'boat', 'train', 'bird', 'bear', 'rabbit', 'elephant',
        'giraffe', 'cow', 'camel', 'skateboard', 'surfboard', 'tennis racket',
        'backpack', 'umbrella', 'suitcase', 'bottle', 'cup', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'tv', 'laptop',
        'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'ball', 'kite', 'baseball bat', 'baseball glove', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'sheep', 'zebra', 'truck', 'bus'
    ]


def extract_prompts_from_first_frame(
    video_data: dict,
    segmentor: SCLIPSegmentor,
    vocabulary: list,
    args: argparse.Namespace
) -> list:
    """
    Extract CLIP-guided prompts from first frame of video.

    Args:
        video_data: Video data dictionary
        segmentor: SCLIP segmentor
        vocabulary: Class vocabulary
        args: Command line arguments

    Returns:
        List of prompt dictionaries
    """
    first_frame = video_data['frames'][0]

    # Run CLIP dense prediction on first frame
    seg_map, logits = segmentor.predict_dense(first_frame, vocabulary, return_logits=True)
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    probs = probs.transpose(1, 2, 0)  # (H, W, num_classes)

    # Extract prompt points
    prompts = extract_prompt_points_from_clip(
        seg_map, probs, vocabulary,
        min_confidence=args.min_confidence,
        min_region_size=args.min_region_size
    )

    return prompts


def segment_video(
    video_data: dict,
    prompts: list,
    video_segmentor: CLIPGuidedVideoSegmentor,
    video_path: str,
    args: argparse.Namespace
) -> dict:
    """
    Segment video using CLIP-guided SAM2.

    Args:
        video_data: Video data dictionary
        prompts: List of prompt dictionaries
        video_segmentor: Video segmentor
        video_path: Path to video (or None for frame sequences)
        args: Command line arguments

    Returns:
        Dictionary mapping frame_idx -> predictions
    """
    # For DAVIS/YouTube-VOS, we need to create a temporary video from frames
    # or use the frame sequence directly

    if video_path is None:
        # Create temporary video from frames
        import tempfile
        import cv2

        temp_dir = tempfile.mkdtemp()
        temp_video_path = f"{temp_dir}/temp_video.mp4"

        # Write frames to video
        height, width = video_data['frames'][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))

        for frame in video_data['frames']:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        video_path = temp_video_path

    # Segment video
    video_segments = video_segmentor.segment_video(
        video_path=video_path,
        prompts=prompts,
        output_path=None,
        visualize=False
    )

    # Convert to dense prediction format
    predictions = {}
    height, width = video_data['frames'][0].shape[:2]

    for frame_idx, masks in video_segments.items():
        # Create dense prediction map
        pred_map = np.zeros((height, width), dtype=np.int64)

        for obj_id, mask in masks.items():
            # Handle 3D masks
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask[0]

            mask_bool = mask.astype(bool)
            pred_map[mask_bool] = obj_id

        predictions[frame_idx] = pred_map

    # Clean up temporary video if created
    if 'temp_video' in video_path:
        import shutil
        shutil.rmtree(temp_dir)

    return predictions


def main():
    args = parse_args()

    print("=" * 80)
    print(f"Video Benchmark: {args.dataset.upper()}")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Min region size: {args.min_region_size}")
    print()

    # Load dataset
    print("Loading video dataset...")
    dataset = load_video_dataset(
        args.dataset,
        Path(args.data_dir),
        split='val',
        max_samples=args.num_samples
    )
    print(f"Dataset: {args.dataset}")
    print(f"Videos: {len(dataset)}")
    print()

    # Get vocabulary
    if args.vocabulary is None:
        vocabulary = get_default_vocabulary(args.dataset)
        print(f"Using default vocabulary with {len(vocabulary)} classes")
    else:
        vocabulary = args.vocabulary
        print(f"Using custom vocabulary with {len(vocabulary)} classes")
    print()

    # Initialize performance profiler
    profiler = PerformanceProfiler() if args.enable_profiling else None

    # Initialize SCLIP segmentor
    print("Initializing SCLIP segmentor...")
    segmentor = SCLIPSegmentor(
        model_name=args.model,
        use_sam=False,
        use_pamr=False,
        verbose=True
    )
    print()

    # Initialize video segmentor
    print("Initializing SAM2 video segmentor...")
    video_segmentor = CLIPGuidedVideoSegmentor(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print()

    # Collect metrics for all videos
    all_video_metrics = []

    # Evaluation loop
    print("Starting evaluation...")
    start_time = time.time()

    for idx in tqdm(range(len(dataset)), desc="Evaluating videos"):
        # Load video
        video_data = dataset[idx]
        video_name = video_data['video_name']

        print(f"\n[{idx+1}/{len(dataset)}] Processing: {video_name}")
        print(f"  Frames: {video_data['num_frames']}, Objects: {video_data['num_objects']}")

        # Limit frames if requested
        if args.max_frames is not None and video_data['num_frames'] > args.max_frames:
            video_data['frames'] = video_data['frames'][:args.max_frames]
            video_data['annotations'] = video_data['annotations'][:args.max_frames]
            video_data['num_frames'] = args.max_frames
            print(f"  Limited to {args.max_frames} frames")

        # Profile CLIP prompt extraction
        if profiler:
            profiler.start('clip_prompt_extraction')

        # Extract prompts from first frame
        prompts = extract_prompts_from_first_frame(
            video_data, segmentor, vocabulary, args
        )

        if profiler:
            profiler.end('clip_prompt_extraction')

        if len(prompts) == 0:
            print(f"  WARNING: No prompts extracted for {video_name}, skipping...")
            continue

        print(f"  Extracted {len(prompts)} prompts")

        # Profile video segmentation
        if profiler:
            profiler.start('video_segmentation')

        # Segment video
        predictions_dict = segment_video(
            video_data,
            prompts,
            video_segmentor,
            video_path=None,  # Will create temp video from frames
            args=args
        )

        if profiler:
            profiler.end('video_segmentation')

        # Convert predictions to list format
        predictions = []
        for frame_idx in sorted(predictions_dict.keys()):
            predictions.append(predictions_dict[frame_idx])

        # Ensure we have predictions for all frames
        if len(predictions) < len(video_data['annotations']):
            print(f"  WARNING: Only {len(predictions)}/{len(video_data['annotations'])} frames predicted")
            # Pad with zeros
            height, width = video_data['annotations'][0].shape
            while len(predictions) < len(video_data['annotations']):
                predictions.append(np.zeros((height, width), dtype=np.int64))

        # Profile metrics computation
        if profiler:
            profiler.start('metrics_computation')

        # Compute metrics
        video_metrics = compute_all_video_metrics(
            annotations=video_data['annotations'],
            predictions=predictions,
            object_ids=video_data['object_ids']
        )

        if profiler:
            profiler.end('metrics_computation')

        all_video_metrics.append(video_metrics)

        print(f"  J: {video_metrics['J']:.3f}, F: {video_metrics['F']:.3f}, "
              f"J&F: {video_metrics['J&F']:.3f}, T: {video_metrics['T']:.3f}")

        # Save visualization if requested
        if args.save_vis:
            vis_dir = Path(args.output_dir) / 'visualizations' / args.dataset / video_name
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Save a few sample frames
            num_vis_frames = min(5, len(predictions))
            vis_indices = np.linspace(0, len(predictions)-1, num_vis_frames, dtype=int)

            for i, frame_idx in enumerate(vis_indices):
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original frame
                axes[0].imshow(video_data['frames'][frame_idx])
                axes[0].set_title(f"Frame {frame_idx}")
                axes[0].axis('off')

                # Ground truth
                axes[1].imshow(video_data['annotations'][frame_idx], cmap='tab20')
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')

                # Prediction
                axes[2].imshow(predictions[frame_idx], cmap='tab20')
                axes[2].set_title("Prediction")
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(vis_dir / f'frame_{frame_idx:04d}.png', dpi=100, bbox_inches='tight')
                plt.close()

    # Aggregate metrics across all videos
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    aggregated_metrics = aggregate_video_metrics(all_video_metrics)

    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f}s")
    print(f"Average time per video: {elapsed_time / len(dataset):.2f}s")

    # Print results
    print()
    print("=" * 80)
    print(f"Results for {args.dataset.upper()}")
    print("=" * 80)
    print(f"J (Region Similarity):   {aggregated_metrics['J_mean']*100:.2f}% ± {aggregated_metrics['J_std']*100:.2f}%")
    print(f"F (Boundary Accuracy):   {aggregated_metrics['F_mean']*100:.2f}% ± {aggregated_metrics['F_std']*100:.2f}%")
    print(f"J&F (Overall):           {aggregated_metrics['J&F_mean']*100:.2f}% ± {aggregated_metrics['J&F_std']*100:.2f}%")
    print(f"T (Temporal Stability):  {aggregated_metrics['T_mean']*100:.2f}% ± {aggregated_metrics['T_std']*100:.2f}%")
    print(f"Videos evaluated:        {aggregated_metrics['num_videos']}")
    print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{args.dataset}_results.json"

    # Convert all_video_metrics to JSON-serializable format
    serializable_metrics = []
    for video_metrics in all_video_metrics:
        # Convert numpy types to Python types
        serializable_video_metrics = {}
        for key, value in video_metrics.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (per-object metrics)
                serializable_video_metrics[key] = {
                    str(k): {str(k2): float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
                             for k2, v2 in v.items()} if isinstance(v, dict) else
                    (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in value.items()
                }
            else:
                serializable_video_metrics[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        serializable_metrics.append(serializable_video_metrics)

    results = {
        'aggregated_metrics': {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in aggregated_metrics.items()
        },
        'per_video_metrics': serializable_metrics,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'elapsed_time': elapsed_time,
        'num_videos': len(dataset)
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")

    print()
    print("=" * 80)
    print("✓ Video benchmark evaluation complete!")
    print("=" * 80)

    # Print performance profiling summary
    if profiler:
        profiler.print_summary()


if __name__ == '__main__':
    main()
