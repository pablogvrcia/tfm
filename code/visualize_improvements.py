#!/usr/bin/env python3
"""
Visualize Improvements: Interactive Analysis of Method Enhancements

This script creates detailed visualizations showing:
1. Prompt comparison (fixed vs adaptive thresholds)
2. Scale diversity (single vs multi-scale)
3. Feature fusion (SCLIP only vs dual-stream)
4. Performance metrics comparison

For Master's Thesis contribution analysis.

Usage:
    python visualize_improvements.py \
        --image examples/2007_000033.jpg \
        --vocabulary aeroplane person background \
        --output visualizations/

Author: Pablo García García
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from typing import List, Dict
import torch.nn.functional as F

from models.sclip_segmentor import SCLIPSegmentor
from models.adaptive_thresholding import AdaptiveThresholdCalculator, extract_prompts_with_adaptive_thresholds
from models.multiscale_prompting import MultiScalePrompter
from models.dual_prompt_strategy import DualPromptExtractor
from clip_guided_segmentation import extract_prompt_points_from_clip


def visualize_threshold_comparison(
    image: np.ndarray,
    class_names: List[str],
    output_path: str,
    device: str = None
):
    """
    Visualize fixed vs adaptive thresholds.

    Shows:
    - Fixed threshold prompts (0.7 for all)
    - Adaptive threshold prompts (per-class)
    - Threshold values per class
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[Threshold Comparison] Running...")

    # Get base prediction
    segmentor = SCLIPSegmentor(
        model_name="ViT-B/16",
        device=device,
        use_sam=False,
        verbose=False,
        has_background_class=True
    )

    seg_map, logits = segmentor.predict_dense(image, class_names, return_logits=True)
    probs = torch.softmax(logits, dim=0).cpu().numpy().transpose(1, 2, 0)

    # Extract prompts with fixed threshold
    prompts_fixed = extract_prompt_points_from_clip(
        seg_map, probs, class_names,
        min_confidence=0.7,
        min_region_size=100
    )

    # Extract prompts with adaptive thresholds
    threshold_calc = AdaptiveThresholdCalculator(
        base_threshold=0.7,
        adaptation_strength=0.3,
        verbose=True
    )

    prompts_adaptive = extract_prompts_with_adaptive_thresholds(
        seg_map, probs, class_names,
        threshold_calculator=threshold_calc,
        min_region_size=100
    )

    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.2)

    # Panel 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Fixed threshold prompts
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image)
    for prompt in prompts_fixed:
        px, py = prompt['point']
        ax2.scatter(px, py, c='red', s=100, alpha=0.8,
                   edgecolors='white', linewidths=2, marker='o')
    ax2.set_title(f'Fixed Threshold (0.7)\n{len(prompts_fixed)} prompts',
                 fontsize=14, fontweight='bold', color='darkred')
    ax2.axis('off')

    # Panel 3: Adaptive threshold prompts
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image)
    for prompt in prompts_adaptive:
        px, py = prompt['point']
        ax3.scatter(px, py, c='green', s=100, alpha=0.8,
                   edgecolors='white', linewidths=2, marker='o')
    ax3.set_title(f'Adaptive Thresholds\n{len(prompts_adaptive)} prompts',
                 fontsize=14, fontweight='bold', color='darkgreen')
    ax3.axis('off')

    # Panel 4: Threshold comparison table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    # Create threshold comparison table
    thresholds = threshold_calc.class_thresholds

    table_data = []
    for class_name in class_names:
        fixed_count = sum(1 for p in prompts_fixed if p['class_name'] == class_name)
        adaptive_count = sum(1 for p in prompts_adaptive if p['class_name'] == class_name)
        threshold = thresholds.get(class_name, 0.7)
        class_type = threshold_calc._classify_class_type(class_name)

        table_data.append([
            class_name,
            "0.700",
            f"{threshold:.3f}",
            str(fixed_count),
            str(adaptive_count),
            class_type.capitalize()
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Class', 'Fixed Thresh', 'Adaptive Thresh', 'Fixed Prompts', 'Adaptive Prompts', 'Type'],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0, 0.8, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows based on threshold change
    for i, row in enumerate(table_data, start=1):
        fixed_thresh = float(row[1])
        adaptive_thresh = float(row[2])

        if adaptive_thresh < fixed_thresh:
            color = '#E3F2FD'  # Blue - lower threshold
        elif adaptive_thresh > fixed_thresh:
            color = '#FFF3E0'  # Orange - higher threshold
        else:
            color = '#FFFFFF'  # White - same

        for j in range(6):
            table[(i, j)].set_facecolor(color)

    plt.suptitle('Adaptive vs Fixed Confidence Thresholds',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def visualize_multiscale_analysis(
    image: np.ndarray,
    class_names: List[str],
    output_path: str,
    device: str = None
):
    """
    Visualize multi-scale prompting benefits.

    Shows:
    - Single-scale prompts (1.0x only)
    - Multi-scale prompts (0.75x, 1.0x, 1.5x)
    - Scale distribution
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[Multi-Scale Analysis] Running...")

    # Initialize segmentor
    segmentor = SCLIPSegmentor(
        model_name="ViT-B/16",
        device=device,
        use_sam=False,
        verbose=False,
        has_background_class=True
    )

    # Single scale
    seg_map, logits = segmentor.predict_dense(image, class_names, return_logits=True)
    probs = torch.softmax(logits, dim=0).cpu().numpy().transpose(1, 2, 0)

    prompts_single = extract_prompt_points_from_clip(
        seg_map, probs, class_names,
        min_confidence=0.7
    )

    # Multi-scale
    ms_prompter = MultiScalePrompter(
        scales=[0.75, 1.0, 1.5],
        min_confidence=0.7,
        verbose=True
    )

    prompts_multi = ms_prompter.extract_multiscale_prompts(
        image, class_names, segmentor, use_nms=True
    )

    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, hspace=0.25, wspace=0.15)

    # Panel 1: Single scale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    for prompt in prompts_single:
        px, py = prompt['point']
        ax1.scatter(px, py, c='blue', s=100, alpha=0.8,
                   edgecolors='white', linewidths=2)
    ax1.set_title(f'Single Scale (1.0x)\n{len(prompts_single)} prompts',
                 fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Multi-scale (colored by scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image)

    scale_colors = {0.75: 'orange', 1.0: 'blue', 1.5: 'green'}

    for prompt in prompts_multi:
        px, py = prompt['point']
        scale = prompt['scale']
        color = scale_colors.get(scale, 'gray')
        ax2.scatter(px, py, c=color, s=100, alpha=0.8,
                   edgecolors='white', linewidths=2)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='orange', label='0.75x (large objects)'),
        mpatches.Patch(facecolor='blue', label='1.0x (balanced)'),
        mpatches.Patch(facecolor='green', label='1.5x (small objects)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax2.set_title(f'Multi-Scale (0.75x, 1.0x, 1.5x)\n{len(prompts_multi)} prompts',
                 fontsize=14, fontweight='bold', color='darkgreen')
    ax2.axis('off')

    # Panel 3: Scale distribution
    ax3 = fig.add_subplot(gs[1, :])

    scale_counts = {}
    for p in prompts_multi:
        scale = p['scale']
        scale_counts[scale] = scale_counts.get(scale, 0) + 1

    scales = sorted(scale_counts.keys())
    counts = [scale_counts[s] for s in scales]
    colors_bar = [scale_colors[s] for s in scales]

    bars = ax3.bar(range(len(scales)), counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(prompts_multi)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax3.set_xticks(range(len(scales)))
    ax3.set_xticklabels([f'{s}x' for s in scales], fontsize=12)
    ax3.set_ylabel('Number of Prompts', fontsize=12, fontweight='bold')
    ax3.set_title('Prompt Distribution Across Scales', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Add summary statistics
    improvement = len(prompts_multi) - len(prompts_single)
    pct_improvement = (improvement / len(prompts_single)) * 100 if len(prompts_single) > 0 else 0

    summary_text = f'Multi-Scale Improvement: +{improvement} prompts (+{pct_improvement:.1f}%)'
    ax3.text(0.5, 0.95, summary_text,
            transform=ax3.transAxes,
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8),
            ha='center', va='top')

    plt.suptitle('Single-Scale vs Multi-Scale Prompting',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def visualize_dual_stream_fusion(
    image: np.ndarray,
    class_names: List[str],
    output_path: str,
    device: str = None
):
    """
    Visualize dual-stream feature fusion.

    Shows:
    - SCLIP stream predictions
    - CLIP stream predictions
    - Fused predictions
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[Dual-Stream Fusion] Running...")

    # Initialize dual-prompt extractor
    dual_prompter = DualPromptExtractor(
        model_name="ViT-B/16",
        device=device,
        sclip_weight=0.6,
        clip_weight=0.4,
        verbose=True
    )

    # Get dual predictions
    fused_mask, fused_logits, sclip_logits, clip_logits = dual_prompter.compute_dual_predictions(
        image, class_names, logit_scale=40.0
    )

    # Get individual predictions
    sclip_mask = F.softmax(sclip_logits, dim=0).argmax(dim=0).cpu().numpy()
    clip_mask = F.softmax(clip_logits, dim=0).argmax(dim=0).cpu().numpy()

    # Create visualization (call the method from dual_prompter)
    dual_prompter.visualize_dual_streams(
        image,
        class_names,
        output_path=output_path
    )

    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize improvements in detail',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--vocabulary', type=str, nargs='+', required=True,
                       help='Class vocabulary')
    parser.add_argument('--output', type=str, default='visualizations/',
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"\nLoading image: {args.image}")
    image = np.array(Image.open(args.image).convert('RGB'))
    print(f"  Image size: {image.shape[1]}×{image.shape[0]}\n")

    # Run visualizations
    print("="*70)
    print("GENERATING DETAILED VISUALIZATIONS")
    print("="*70)

    # 1. Threshold comparison
    visualize_threshold_comparison(
        image, args.vocabulary,
        str(output_dir / 'threshold_comparison.png'),
        device=args.device
    )

    # 2. Multi-scale analysis
    visualize_multiscale_analysis(
        image, args.vocabulary,
        str(output_dir / 'multiscale_analysis.png'),
        device=args.device
    )

    # 3. Dual-stream fusion
    visualize_dual_stream_fusion(
        image, args.vocabulary,
        str(output_dir / 'dual_stream_fusion.png'),
        device=args.device
    )

    print("\n" + "="*70)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
