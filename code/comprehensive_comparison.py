#!/usr/bin/env python3
"""
Comprehensive Comparison: SCLIP-Guided vs SAM-CLIP vs Enhanced Methods

This script provides a systematic comparison of:
1. Baseline: SAM-CLIP (CVPR 2024)
2. Current: SCLIP-Guided (your thesis)
3. Enhanced: Dual-Prompt Strategy
4. Enhanced: Adaptive Thresholding
5. Enhanced: Multi-Scale Prompting
6. Combined: All enhancements together

For Master's Thesis: Extending contribution from 15 ECTS to 30 ECTS

Usage:
    python comprehensive_comparison.py \
        --image examples/2007_000033.jpg \
        --vocabulary aeroplane person background \
        --output comparison_results/

Author: Pablo García García
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import time
import json

# Import all methods
from models.sclip_segmentor import SCLIPSegmentor
from models.samclip_segmentor import SAMCLIPSegmentor
from models.dual_prompt_strategy import DualPromptExtractor
from models.adaptive_thresholding import AdaptiveThresholdCalculator, extract_prompts_with_adaptive_thresholds
from models.multiscale_prompting import MultiScalePrompter
from models.sam2_segmentation import SAM2MaskGenerator


class ComprehensiveComparison:
    """
    Runs comprehensive comparison of all methods.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        device: str = None,
        verbose: bool = True
    ):
        """
        Initialize comparison framework.

        Args:
            model_name: CLIP model variant
            device: Computation device
            verbose: Print progress
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.verbose = verbose

        # Results storage
        self.results = {}
        self.timings = {}

    def run_samclip_baseline(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Run SAM-CLIP baseline (CVPR 2024 approach)."""
        if self.verbose:
            print("\n" + "="*70)
            print("METHOD 1: SAM-CLIP Baseline (CVPR 2024)")
            print("="*70)

        start_time = time.time()

        segmentor = SAMCLIPSegmentor(
            model_name=self.model_name,
            device=self.device,
            verbose=self.verbose
        )

        result = segmentor.predict_with_sam(image, class_names)

        elapsed = time.time() - start_time

        stats = {
            'method': 'SAM-CLIP Baseline',
            'time': elapsed,
            'description': 'Standard CLIP + SAM without CSA enhancement'
        }

        return result, stats

    def run_sclip_guided(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Run SCLIP-Guided (current thesis approach)."""
        if self.verbose:
            print("\n" + "="*70)
            print("METHOD 2: SCLIP-Guided (Current Thesis)")
            print("="*70)

        start_time = time.time()

        segmentor = SCLIPSegmentor(
            model_name=self.model_name,
            device=self.device,
            use_sam=True,
            use_pamr=False,
            use_densecrf=True,  # Enable DenseCRF
            verbose=self.verbose,
            has_background_class=True
        )

        result = segmentor.predict_with_sam(
            image,
            class_names,
            use_prompted_sam=True,
            use_hierarchical_prompts=False
        )

        elapsed = time.time() - start_time

        stats = {
            'method': 'SCLIP-Guided (Thesis)',
            'time': elapsed,
            'description': 'SCLIP with CSA + intelligent prompting + DenseCRF'
        }

        return result, stats

    def run_dual_prompt(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Run dual-prompt strategy enhancement."""
        if self.verbose:
            print("\n" + "="*70)
            print("METHOD 3: Dual-Prompt Strategy")
            print("="*70)

        start_time = time.time()

        # Initialize dual-prompt extractor
        dual_prompter = DualPromptExtractor(
            model_name=self.model_name,
            device=self.device,
            sclip_weight=0.6,
            clip_weight=0.4,
            verbose=self.verbose
        )

        # Get dual-stream prompts
        prompts = dual_prompter.extract_dual_prompts(
            image,
            class_names,
            min_confidence=0.7,
            use_confidence_fusion=True
        )

        # Use SAM with dual prompts
        sam_generator = SAM2MaskGenerator(device=self.device, use_fp16=True)

        if len(prompts) == 0:
            result, _ = dual_prompter.compute_dual_predictions(image, class_names, logit_scale=40.0)
        else:
            # Get base prediction
            result, _, _, _ = dual_prompter.compute_dual_predictions(image, class_names, logit_scale=40.0)

            # Refine with SAM
            points = [(int(p['point'][0]), int(p['point'][1])) for p in prompts]
            labels = [1] * len(points)

            sam_masks = sam_generator.segment_with_points(image, points, labels)

            # Merge with base prediction
            for i, mask_cand in enumerate(sam_masks[::3]):
                mask_region = mask_cand.mask > 0.5
                if mask_region.sum() > 0:
                    # Get majority class from prompts
                    prompt_idx = i
                    if prompt_idx < len(prompts):
                        class_idx = prompts[prompt_idx]['class_idx']
                        result[mask_region] = class_idx

        elapsed = time.time() - start_time

        stats = {
            'method': 'Dual-Prompt (SCLIP+CLIP)',
            'time': elapsed,
            'num_prompts': len(prompts),
            'description': 'Fused SCLIP and CLIP features for better prompting'
        }

        return result, stats

    def run_adaptive_threshold(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Run adaptive thresholding enhancement."""
        if self.verbose:
            print("\n" + "="*70)
            print("METHOD 4: Adaptive Confidence Thresholding")
            print("="*70)

        start_time = time.time()

        # Get base SCLIP prediction
        segmentor = SCLIPSegmentor(
            model_name=self.model_name,
            device=self.device,
            use_sam=False,
            verbose=self.verbose,
            has_background_class=True
        )

        seg_map, logits = segmentor.predict_dense(image, class_names, return_logits=True)
        probs = torch.softmax(logits, dim=0).cpu().numpy().transpose(1, 2, 0)

        # Extract prompts with adaptive thresholds
        threshold_calc = AdaptiveThresholdCalculator(
            base_threshold=0.7,
            adaptation_strength=0.3,
            verbose=self.verbose
        )

        prompts = extract_prompts_with_adaptive_thresholds(
            seg_map,
            probs,
            class_names,
            threshold_calculator=threshold_calc,
            min_region_size=100
        )

        # Refine with SAM
        sam_generator = SAM2MaskGenerator(device=self.device, use_fp16=True)

        result = seg_map.copy()

        if len(prompts) > 0:
            points = [(int(p['point'][0]), int(p['point'][1])) for p in prompts]
            labels = [1] * len(points)

            sam_masks = sam_generator.segment_with_points(image, points, labels)

            for i, mask_cand in enumerate(sam_masks[::3]):
                mask_region = mask_cand.mask > 0.5
                if mask_region.sum() > 0 and i < len(prompts):
                    class_idx = prompts[i]['class_idx']
                    result[mask_region] = class_idx

        elapsed = time.time() - start_time

        # Get threshold statistics
        thresholds = threshold_calc.class_thresholds
        threshold_range = f"{min(thresholds.values()):.2f}-{max(thresholds.values()):.2f}"

        stats = {
            'method': 'Adaptive Thresholding',
            'time': elapsed,
            'num_prompts': len(prompts),
            'threshold_range': threshold_range,
            'description': 'Per-class adaptive confidence thresholds'
        }

        return result, stats

    def run_multiscale(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Run multi-scale prompting enhancement."""
        if self.verbose:
            print("\n" + "="*70)
            print("METHOD 5: Multi-Scale Prompting")
            print("="*70)

        start_time = time.time()

        # Initialize segmentor
        segmentor = SCLIPSegmentor(
            model_name=self.model_name,
            device=self.device,
            use_sam=False,
            verbose=False,  # Reduce verbosity for multi-scale
            has_background_class=True
        )

        # Initialize multi-scale prompter
        ms_prompter = MultiScalePrompter(
            scales=[0.75, 1.0, 1.5],
            min_confidence=0.7,
            nms_distance=30,
            verbose=self.verbose
        )

        # Extract multi-scale prompts
        prompts = ms_prompter.extract_multiscale_prompts(
            image,
            class_names,
            segmentor,
            use_nms=True
        )

        # Get base prediction at original scale
        result, _ = segmentor.predict_dense(image, class_names, return_logits=False)

        # Refine with SAM using multi-scale prompts
        if len(prompts) > 0:
            sam_generator = SAM2MaskGenerator(device=self.device, use_fp16=True)

            points = [(int(p['point'][0]), int(p['point'][1])) for p in prompts]
            labels = [1] * len(points)

            sam_masks = sam_generator.segment_with_points(image, points, labels)

            for i, mask_cand in enumerate(sam_masks[::3]):
                mask_region = mask_cand.mask > 0.5
                if mask_region.sum() > 0 and i < len(prompts):
                    class_idx = prompts[i]['class_idx']
                    result[mask_region] = class_idx

        elapsed = time.time() - start_time

        # Get scale statistics
        scale_counts = {}
        for p in prompts:
            scale = p['scale']
            scale_counts[scale] = scale_counts.get(scale, 0) + 1

        stats = {
            'method': 'Multi-Scale Prompting',
            'time': elapsed,
            'num_prompts': len(prompts),
            'scales': list(scale_counts.keys()),
            'scale_distribution': scale_counts,
            'description': 'Prompts extracted at multiple scales (0.75x, 1.0x, 1.5x)'
        }

        return result, stats

    def run_all_combined(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Run all enhancements combined."""
        if self.verbose:
            print("\n" + "="*70)
            print("METHOD 6: ALL ENHANCEMENTS COMBINED")
            print("="*70)

        start_time = time.time()

        # Use dual-prompt + adaptive thresholds + multi-scale
        dual_prompter = DualPromptExtractor(
            model_name=self.model_name,
            device=self.device,
            verbose=False
        )

        # Get base prediction with dual streams
        result, fused_logits, _, _ = dual_prompter.compute_dual_predictions(
            image, class_names, logit_scale=40.0
        )

        # Prepare for adaptive thresholding
        probs = torch.softmax(fused_logits, dim=0).cpu().numpy().transpose(1, 2, 0)

        # Extract prompts with adaptive thresholds
        threshold_calc = AdaptiveThresholdCalculator(
            base_threshold=0.7,
            adaptation_strength=0.3,
            verbose=False
        )

        prompts = extract_prompts_with_adaptive_thresholds(
            result,
            probs,
            class_names,
            threshold_calculator=threshold_calc
        )

        # Add multi-scale prompts
        segmentor = SCLIPSegmentor(
            model_name=self.model_name,
            device=self.device,
            use_sam=False,
            verbose=False,
            has_background_class=True
        )

        ms_prompter = MultiScalePrompter(
            scales=[0.75, 1.0, 1.5],
            verbose=False
        )

        ms_prompts = ms_prompter.extract_multiscale_prompts(
            image, class_names, segmentor, use_nms=True
        )

        # Combine all prompts
        all_prompts = prompts + ms_prompts

        # Refine with SAM
        if len(all_prompts) > 0:
            sam_generator = SAM2MaskGenerator(device=self.device, use_fp16=True)

            points = [(int(p['point'][0]), int(p['point'][1])) for p in all_prompts]
            labels = [1] * len(points)

            sam_masks = sam_generator.segment_with_points(image, points, labels)

            for i, mask_cand in enumerate(sam_masks[::3]):
                mask_region = mask_cand.mask > 0.5
                if mask_region.sum() > 0 and i < len(all_prompts):
                    class_idx = all_prompts[i]['class_idx']
                    result[mask_region] = class_idx

        elapsed = time.time() - start_time

        stats = {
            'method': 'All Enhancements Combined',
            'time': elapsed,
            'num_prompts': len(all_prompts),
            'components': ['Dual-Prompt', 'Adaptive Thresholds', 'Multi-Scale'],
            'description': 'All improvements together'
        }

        return result, stats

    def visualize_comparison(
        self,
        image: np.ndarray,
        results_dict: Dict[str, np.ndarray],
        stats_dict: Dict[str, Dict],
        output_path: str
    ):
        """
        Create comprehensive visualization of all methods.

        Args:
            image: Original image
            results_dict: Dictionary of method name -> segmentation result
            stats_dict: Dictionary of method name -> statistics
            output_path: Path to save visualization
        """
        num_methods = len(results_dict)
        fig = plt.figure(figsize=(24, 4 * ((num_methods + 1) // 2)))

        gs = GridSpec((num_methods + 1) // 2, 2, hspace=0.3, wspace=0.15)

        for idx, (method_name, result) in enumerate(results_dict.items()):
            row = idx // 2
            col = idx % 2

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(result, cmap='tab20', interpolation='nearest')

            # Get statistics
            stats = stats_dict[method_name]
            time_str = f"{stats['time']:.2f}s"
            num_prompts = stats.get('num_prompts', 'N/A')

            title = f"{method_name}\n{time_str} | {num_prompts} prompts"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n[Visualization] Saved to {output_path}")
        plt.close()

    def save_results_json(
        self,
        stats_dict: Dict[str, Dict],
        output_path: str
    ):
        """Save results statistics as JSON."""
        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"[Results] Saved statistics to {output_path}")

    def print_summary_table(self, stats_dict: Dict[str, Dict]):
        """Print summary table of all methods."""
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print("="*70)

        print(f"\n{'Method':<30} {'Time (s)':>10} {'Prompts':>10}")
        print("-" * 52)

        for method_name, stats in stats_dict.items():
            time_val = f"{stats['time']:.2f}"
            prompts_val = str(stats.get('num_prompts', 'N/A'))
            print(f"{method_name:<30} {time_val:>10} {prompts_val:>10}")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive comparison of segmentation methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on a single image
  python comprehensive_comparison.py \\
      --image examples/2007_000033.jpg \\
      --vocabulary aeroplane person background \\
      --output comparison_results/

  # Run specific methods
  python comprehensive_comparison.py \\
      --image examples/football.jpg \\
      --vocabulary person grass "Lionel Messi" background \\
      --methods sclip dual adaptive \\
      --output results_football/
        """
    )

    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--vocabulary', type=str, nargs='+', required=True,
                       help='Class vocabulary (space-separated)')
    parser.add_argument('--output', type=str, default='comparison_results/',
                       help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+',
                       choices=['samclip', 'sclip', 'dual', 'adaptive', 'multiscale', 'combined', 'all'],
                       default=['all'],
                       help='Which methods to run (default: all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu, default: auto-detect)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image: {args.image}")
    image = np.array(Image.open(args.image).convert('RGB'))
    print(f"  Image size: {image.shape[1]}×{image.shape[0]}")

    # Initialize comparison
    comparison = ComprehensiveComparison(
        model_name="ViT-B/16",
        device=args.device,
        verbose=True
    )

    # Determine which methods to run
    methods_to_run = args.methods
    if 'all' in methods_to_run:
        methods_to_run = ['samclip', 'sclip', 'dual', 'adaptive', 'multiscale', 'combined']

    # Run all methods
    results_dict = {}
    stats_dict = {}

    method_functions = {
        'samclip': comparison.run_samclip_baseline,
        'sclip': comparison.run_sclip_guided,
        'dual': comparison.run_dual_prompt,
        'adaptive': comparison.run_adaptive_threshold,
        'multiscale': comparison.run_multiscale,
        'combined': comparison.run_all_combined
    }

    for method_key in methods_to_run:
        if method_key in method_functions:
            result, stats = method_functions[method_key](image, args.vocabulary)
            results_dict[stats['method']] = result
            stats_dict[stats['method']] = stats

    # Visualize
    vis_path = output_dir / 'comprehensive_comparison.png'
    comparison.visualize_comparison(image, results_dict, stats_dict, str(vis_path))

    # Save results
    json_path = output_dir / 'results.json'
    comparison.save_results_json(stats_dict, str(json_path))

    # Print summary
    comparison.print_summary_table(stats_dict)

    print(f"\n✓ Comparison complete!")
    print(f"  Results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
