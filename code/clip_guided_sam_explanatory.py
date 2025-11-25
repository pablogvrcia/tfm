"""
CLIP-Guided SAM Explanatory Visualization

Educational visualization tool that demonstrates every step of the CLIP-guided SAM pipeline.
Generates 18 detailed step visualizations plus an interactive HTML report.

Usage:
    python clip_guided_sam_explanatory.py \
        --image examples/football_frame.png \
        --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background \
        --output explanatory_results/football \
        --min-confidence 0.3 \
        --points-per-cluster 1 \
        --negative-points-per-cluster 2 \
        --create-html \
        --per-class-details
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2
import json
from pathlib import Path
from scipy.ndimage import label, distance_transform_edt
from typing import List, Tuple, Dict, Optional
import colorsys

from models.sclip_segmentor import SCLIPSegmentor
from clip_guided_segmentation import (
    extract_prompt_points_from_clip,
    segment_with_guided_prompts,
    merge_overlapping_masks,
    generate_distinct_colors
)


class CLIPGuidedSAMVisualizer:
    """
    Complete educational visualization of CLIP-guided SAM pipeline.

    Generates 18 detailed step visualizations showing the complete process
    from raw image to final segmentation.
    """

    def __init__(
        self,
        image_path: str,
        vocabulary: List[str],
        output_dir: str,
        min_confidence: float = 0.3,
        min_region_size: int = 100,
        points_per_cluster: int = 1,
        negative_points_per_cluster: int = 0,
        negative_confidence_threshold: float = 0.8,
        iou_threshold: float = 0.8,
        create_html: bool = True,
        create_per_class_details: bool = False,
        save_intermediate_data: bool = True,
        device: str = None,
    ):
        self.image_path = image_path
        self.vocabulary = vocabulary
        self.output_dir = Path(output_dir)
        self.min_confidence = min_confidence
        self.min_region_size = min_region_size
        self.points_per_cluster = points_per_cluster
        self.negative_points_per_cluster = negative_points_per_cluster
        self.negative_confidence_threshold = negative_confidence_threshold
        self.iou_threshold = iou_threshold
        self.create_html = create_html
        self.create_per_class_details = create_per_class_details
        self.save_intermediate_data = save_intermediate_data
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.steps_dir = self.output_dir / "steps"
        self.per_class_dir = self.output_dir / "per_class"
        self.data_dir = self.output_dir / "data"

        self.steps_dir.mkdir(parents=True, exist_ok=True)
        if create_per_class_details:
            self.per_class_dir.mkdir(parents=True, exist_ok=True)
        if save_intermediate_data:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for classes
        self.class_colors = generate_distinct_colors(len(vocabulary))

        # Storage for intermediate results
        self.intermediate_data = {}
        self.statistics = {}

    def run(self):
        """Execute complete visualization pipeline."""
        print("="*80)
        print("CLIP-GUIDED SAM EXPLANATORY VISUALIZATION")
        print("="*80)
        print(f"Image: {self.image_path}")
        print(f"Classes: {', '.join(self.vocabulary)}")
        print(f"Output: {self.output_dir}")
        print()

        # Load image
        image = self.load_image()

        # Step 1: Raw Image
        print("[Step 1/18] Raw Image")
        self.visualize_step_01_raw_image(image)

        # Step 2: Text Prompt Processing
        print("[Step 2/18] Text Prompt Processing")
        self.visualize_step_02_text_prompts()

        # Initialize SCLIP
        print("\nInitializing SCLIP...")
        segmentor = SCLIPSegmentor(
            device=self.device,
            use_sam=False,
            use_pamr=False,
            verbose=False,
            slide_inference=True,
            has_background_class=False  # For custom vocabularies
        )

        # Get CLIP predictions
        print("Running SCLIP dense prediction...")
        seg_map, logits = segmentor.predict_dense(image, self.vocabulary, return_logits=True)
        probs = torch.softmax(logits, dim=0)
        probs_np = probs.cpu().numpy().transpose(1, 2, 0)  # (H, W, num_classes)

        # Store intermediate data
        self.intermediate_data['image'] = image
        self.intermediate_data['seg_map'] = seg_map
        self.intermediate_data['probs'] = probs_np
        self.intermediate_data['logits'] = logits.cpu().numpy()

        # Step 3: SCLIP Sliding Window Patches
        print("[Step 3/18] SCLIP Sliding Window Patches")
        self.visualize_step_03_sclip_patches(image)

        # Step 4: Visual-Text Similarity
        print("[Step 4/18] Visual-Text Similarity Computation")
        self.visualize_step_04_similarity(image, probs_np)

        # Step 5: Per-Class Confidence Maps
        print("[Step 5/18] Per-Class Confidence Maps")
        self.visualize_step_05_confidence_maps(image, probs_np)

        # Step 6: SCLIP Dense Prediction
        print("[Step 6/18] SCLIP Dense Prediction (argmax)")
        self.visualize_step_06_dense_prediction(image, seg_map)

        # Step 7: SCLIP Dense Prediction Confidences
        print("[Step 7/18] SCLIP Dense Prediction Confidences")
        self.visualize_step_07_prediction_confidences(image, seg_map, probs_np)

        # Step 8: Confidence Thresholding
        print("[Step 8/18] Confidence Thresholding")
        high_conf_mask = self.visualize_step_08_thresholding(image, seg_map, probs_np)

        # Step 9: Connected Components per Class
        print("[Step 9/18] Connected Components per Class")
        labeled_components = self.visualize_step_09_connected_components(image, seg_map, high_conf_mask)

        # Step 10: Min Region Size Filter
        print("[Step 10/18] Min Region Size Filter")
        filtered_components = self.visualize_step_10_size_filter(image, labeled_components)

        # Extract prompts
        print("\nExtracting prompt points...")
        prompts = extract_prompt_points_from_clip(
            seg_map, probs_np, self.vocabulary,
            min_confidence=self.min_confidence,
            min_region_size=self.min_region_size,
            points_per_cluster=self.points_per_cluster,
            negative_points_per_cluster=self.negative_points_per_cluster,
            negative_confidence_threshold=self.negative_confidence_threshold
        )

        self.intermediate_data['prompts'] = prompts

        # Step 11: Positive Point Prompts
        print("[Step 11/18] Positive Point Prompts Extraction")
        self.visualize_step_11_positive_prompts(image, seg_map, prompts)

        # Step 12: Negative Point Prompts
        print("[Step 12/18] Negative Point Prompts Extraction")
        self.visualize_step_12_negative_prompts(image, seg_map, prompts)

        # Step 13: SAM Prompting Visualization
        print("[Step 13/18] SAM 2 Prompting Visualization")
        self.visualize_step_13_sam_prompting(image, prompts)

        # Run SAM (with all candidates saved)
        print("\nRunning SAM 2 segmentation...")
        results, all_candidates = self.segment_with_all_candidates(image, prompts)

        self.intermediate_data['sam_results'] = results
        self.intermediate_data['sam_candidates'] = all_candidates

        # Step 14: SAM Multi-Mask Candidates
        print("[Step 14/18] SAM Multi-Mask Candidates")
        self.visualize_step_14_sam_candidates(image, prompts, results, all_candidates)

        # Step 15: Best Mask Selection
        print("[Step 15/18] Best Mask Selection")
        self.visualize_step_15_mask_selection(image, results)

        # Merge overlapping masks
        print("\nMerging overlapping masks...")
        merged_results = merge_overlapping_masks(results, iou_threshold=self.iou_threshold)

        self.intermediate_data['merged_results'] = merged_results

        # Step 16: Overlap Resolution
        print("[Step 16/18] Overlap Resolution (IoU Merging)")
        self.visualize_step_16_overlap_resolution(image, results, merged_results)

        # Generate final segmentation (use SCLIP seg_map as base, overlay SAM masks)
        final_seg_map = self.create_final_segmentation(image, merged_results, seg_map=seg_map)
        self.intermediate_data['final_seg_map'] = final_seg_map

        # Step 17: Final Segmentation
        print("[Step 17/18] Final Segmentation")
        self.visualize_step_17_final_segmentation(image, final_seg_map)

        # Step 18: Comparison Grid
        print("[Step 18/18] Comparison Grid")
        self.visualize_step_18_comparison_grid(image, seg_map, probs_np, final_seg_map)

        # Compute statistics
        self.compute_statistics()

        # Per-class detailed views
        if self.create_per_class_details:
            print("\nGenerating per-class detailed views...")
            self.generate_per_class_details()

        # Save intermediate data
        if self.save_intermediate_data:
            print("\nSaving intermediate data...")
            self.save_data()

        # Generate interactive HTML
        if self.create_html:
            print("\nGenerating interactive HTML visualization...")
            self.generate_html()

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print(f"Output directory: {self.output_dir}")
        if self.create_html:
            print(f"Open: {self.output_dir / 'index.html'}")
        print("="*80)

    def load_image(self) -> np.ndarray:
        """Load and return RGB image."""
        image = Image.open(self.image_path)
        return np.array(image.convert("RGB"))

    def segment_with_all_candidates(self, image: np.ndarray, prompts: List[Dict]):
        """
        Run SAM segmentation and return both best masks and all candidates.

        Returns:
            results: List of best masks (one per prompt)
            all_candidates: List of all 3 mask candidates per prompt
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Build SAM model
        sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        predictor = SAM2ImagePredictor(sam2_model)

        predictor.set_image(image)

        results = []
        all_candidates = []

        for i, prompt_info in enumerate(prompts):
            point = prompt_info['point']
            class_idx = prompt_info['class_idx']
            negative_points = prompt_info.get('negative_points', [])

            # Combine positive and negative points
            all_coords = [point]
            all_labels = [1]  # 1 = foreground

            if negative_points:
                for neg_point in negative_points:
                    all_coords.append(neg_point)
                    all_labels.append(0)  # 0 = background

            point_coords = np.array(all_coords)
            point_labels = np.array(all_labels)

            # Get 3 mask candidates from SAM
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True  # Returns 3 masks
            )

            # Store all 3 candidates
            candidates = []
            for j in range(3):
                candidates.append({
                    'mask': masks[j],
                    'score': float(scores[j]),
                    'prompt_idx': i
                })
            all_candidates.append(candidates)

            # Pick best mask (highest score)
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            results.append({
                'mask': best_mask,
                'score': float(best_score),
                'class_idx': class_idx,
                'class_name': prompt_info['class_name'],
                'confidence': prompt_info['confidence'],
                'point': point
            })

        return results, all_candidates

    # ========== Step Visualization Methods ==========

    def visualize_step_01_raw_image(self, image: np.ndarray):
        """Step 1: Display raw input image."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title("Step 1: Raw Input Image", fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add image info
        h, w = image.shape[:2]
        info_text = f"Resolution: {w}×{h}\nClasses: {len(self.vocabulary)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.steps_dir / "01_raw_image.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_02_text_prompts(self):
        """Step 2: Show text prompts and vocabulary."""
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 2])

        # Top: Vocabulary list
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')

        vocab_text = "Input Vocabulary:\n\n"
        for i, class_name in enumerate(self.vocabulary):
            color = self.class_colors[i]
            vocab_text += f"  {i}. {class_name}\n"

        ax1.text(0.5, 0.5, vocab_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                family='monospace')

        # Bottom: Explanation
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')

        explanation = """Text Prompt Processing:

1. Each class name is converted to text embeddings using CLIP's text encoder
2. Optional: Multi-descriptor expansion (e.g., "person" → "person in shirt", "person in jeans", ...)
3. Text embeddings will be compared with visual patch embeddings to compute similarity
4. Template wrapping: "a photo of a {class_name}" (ImageNet-80 templates)

These text embeddings serve as the semantic anchors for the visual similarity computation."""

        ax2.text(0.1, 0.9, explanation, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        fig.suptitle("Step 2: Text Prompt Processing", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "02_text_prompts.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_03_sclip_patches(self, image: np.ndarray):
        """Step 3: Show SCLIP sliding window patch grid with full coverage."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        h, w = image.shape[:2]
        crop_size = 224
        stride = 112

        # Left: Original image with grid overlay
        ax1.imshow(image)
        ax1.set_title("Sliding Window Grid (Full Coverage)", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Calculate patches with edge handling (SCLIP approach)
        # Generate positions to ensure FULL image coverage
        x_positions = list(range(0, w - crop_size + 1, stride))
        y_positions = list(range(0, h - crop_size + 1, stride))

        # Add edge patches if needed to cover the entire image
        if len(x_positions) == 0 or x_positions[-1] + crop_size < w:
            # Add patch at right edge (may overlap more than stride)
            x_positions.append(max(0, w - crop_size))
        if len(y_positions) == 0 or y_positions[-1] + crop_size < h:
            # Add patch at bottom edge (may overlap more than stride)
            y_positions.append(max(0, h - crop_size))

        # Remove duplicates and sort
        x_positions = sorted(set(x_positions))
        y_positions = sorted(set(y_positions))

        num_patches_x = len(x_positions)
        num_patches_y = len(y_positions)
        total_patches = num_patches_x * num_patches_y

        # Draw all patches
        edge_patches = []
        for y in y_positions:
            for x in x_positions:
                # Check if this is an edge patch (higher overlap)
                is_edge_x = (x == x_positions[-1] and x > x_positions[-2] + stride) if len(x_positions) > 1 else False
                is_edge_y = (y == y_positions[-1] and y > y_positions[-2] + stride) if len(y_positions) > 1 else False
                is_edge = is_edge_x or is_edge_y

                if is_edge:
                    # Edge patches in orange to show they have higher overlap
                    rect = mpatches.Rectangle((x, y), crop_size, crop_size,
                                             linewidth=1.5, edgecolor='orange',
                                             facecolor='none', alpha=0.7)
                    edge_patches.append((x, y))
                else:
                    # Regular patches in green
                    rect = mpatches.Rectangle((x, y), crop_size, crop_size,
                                             linewidth=1, edgecolor='lime',
                                             facecolor='none', alpha=0.5)
                ax1.add_patch(rect)

        # Highlight sample patches
        sample_patches = [(x_positions[0], y_positions[0]),
                         (x_positions[min(1, len(x_positions)-1)], y_positions[min(1, len(y_positions)-1)]),
                         (x_positions[min(2, len(x_positions)-1)], y_positions[min(2, len(y_positions)-1)])]

        for i, (x, y) in enumerate(sample_patches[:3]):
            rect = mpatches.Rectangle((x, y), crop_size, crop_size,
                                     linewidth=3, edgecolor='red',
                                     facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x + crop_size//2, y + crop_size//2, f"P{i+1}",
                    color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))

        # Mark edge patches if any
        if edge_patches:
            for x, y in edge_patches[:1]:  # Mark one edge patch as example
                ax1.text(x + 10, y + 10, 'Edge',
                        color='orange', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Right: Explanation
        ax2.axis('off')

        # Calculate coverage
        regular_patches = total_patches - len(edge_patches)
        edge_overlap_info = ""
        if edge_patches:
            edge_overlap_info = f"\n  • Edge patches: {len(edge_patches)} (higher overlap for full coverage)"

        explanation = f"""SCLIP Sliding Window Patches (Full Coverage):

Parameters:
  • Crop size: {crop_size}×{crop_size} pixels
  • Stride: {stride} pixels (regular patches)
  • Overlap: {crop_size - stride} pixels ({(crop_size-stride)/crop_size*100:.0f}%)

Image dimensions:
  • Width: {w}px, Height: {h}px
  • Requires edge handling: {'Yes' if edge_patches else 'No'}

Grid dimensions:
  • Horizontal patches: {num_patches_x}
  • Vertical patches: {num_patches_y}
  • Total patches: {total_patches}{edge_overlap_info}

Process:
  1. Extract {crop_size}×{crop_size} patches with {stride}-pixel stride
  2. Add edge patches to ensure FULL image coverage
  3. Each patch encoded by CLIP vision encoder (ViT-B/16)
  4. Features from multiple layers extracted and fused
  5. Overlapping regions merged using CSA (weighted averaging)

Color coding:
  • Green: Regular patches (stride={stride})
  • Orange: Edge patches (higher overlap for full coverage)
  • Red: Example patches (P1, P2, P3)"""

        ax2.text(0.05, 0.98, explanation, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

        fig.suptitle("Step 3: SCLIP Sliding Window Patch Extraction (Full Image Coverage)",
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "03_sclip_patches.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_04_similarity(self, image: np.ndarray, probs: np.ndarray):
        """Step 4: Visualize visual-text similarity computation - SCLIP pipeline with precise steps."""
        fig = plt.figure(figsize=(22, 14))
        gs = GridSpec(4, 4, height_ratios=[0.4, 1.2, 1.3, 0.9], width_ratios=[1, 1, 1, 1])

        # Row 0: Title and overview
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        title_text = """SCLIP Dense Prediction Pipeline: Text embeddings computed once, then each 224×224 patch encoded independently, overlaps merged via CSA, similarities computed per-pixel"""
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=11,
                     fontweight='bold', style='italic', wrap=True,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        # Row 1: Detailed 6-step pipeline
        ax_pipeline = fig.add_subplot(gs[1, :])
        ax_pipeline.axis('off')
        ax_pipeline.set_xlim(0, 12)
        ax_pipeline.set_ylim(0, 2)

        # Step 1: Text Encoding (ONCE for all classes)
        ax_pipeline.add_patch(mpatches.FancyBboxPatch((0.1, 0.7), 1.6, 1.0,
                             boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2.5))
        ax_pipeline.text(0.9, 1.45, '①  Text Encoding', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_pipeline.text(0.9, 1.15, 'CLIP Text Enc.', ha='center', va='center', fontsize=7)
        ax_pipeline.text(0.9, 0.9, '["road", "person"\n"car"]', ha='center', va='center', fontsize=6, family='monospace')
        ax_pipeline.text(0.9, 0.5, 'ONCE', ha='center', va='top', fontsize=6, style='italic', color='red', fontweight='bold')

        # Arrow 1→2
        ax_pipeline.annotate('', xy=(2.0, 1.2), xytext=(1.7, 1.2),
                            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        ax_pipeline.text(1.85, 1.4, '{t₁, t₂, ..., tₙ}', ha='center', fontsize=7, style='italic', family='monospace')

        # Step 2: Patch Extraction
        ax_pipeline.add_patch(mpatches.FancyBboxPatch((2.1, 0.7), 1.6, 1.0,
                             boxstyle="round,pad=0.1", facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2.5))
        ax_pipeline.text(2.9, 1.45, '②  Patch Extract', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_pipeline.text(2.9, 1.15, '224×224 window', ha='center', va='center', fontsize=7)
        ax_pipeline.text(2.9, 0.9, 'stride = 112', ha='center', va='center', fontsize=6)
        ax_pipeline.text(2.9, 0.5, 'Sliding', ha='center', va='top', fontsize=6, style='italic')

        # Arrow 2→3
        ax_pipeline.annotate('', xy=(4.0, 1.2), xytext=(3.7, 1.2),
                            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        ax_pipeline.text(3.85, 1.4, 'patches', ha='center', fontsize=7, style='italic')

        # Step 3: Vision Encoding (PER PATCH)
        ax_pipeline.add_patch(mpatches.FancyBboxPatch((4.1, 0.7), 1.6, 1.0,
                             boxstyle="round,pad=0.1", facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2.5))
        ax_pipeline.text(4.9, 1.45, '③  Vision Enc.', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_pipeline.text(4.9, 1.15, 'CLIP ViT-B/16', ha='center', va='center', fontsize=7)
        ax_pipeline.text(4.9, 0.9, 'per patch', ha='center', va='center', fontsize=6)
        ax_pipeline.text(4.9, 0.5, 'PER PATCH', ha='center', va='top', fontsize=6, style='italic', color='red', fontweight='bold')

        # Arrow 3→4
        ax_pipeline.annotate('', xy=(6.0, 1.2), xytext=(5.7, 1.2),
                            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        ax_pipeline.text(5.85, 1.4, 'features', ha='center', fontsize=7, style='italic')

        # Step 4: CSA (merge overlaps)
        ax_pipeline.add_patch(mpatches.FancyBboxPatch((6.1, 0.7), 1.6, 1.0,
                             boxstyle="round,pad=0.1", facecolor='#FFF9C4', edgecolor='#F57C00', linewidth=2.5))
        ax_pipeline.text(6.9, 1.45, '④  CSA Merge', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_pipeline.text(6.9, 1.15, 'Correlative', ha='center', va='center', fontsize=7)
        ax_pipeline.text(6.9, 0.9, 'Self-Attention', ha='center', va='center', fontsize=7)
        ax_pipeline.text(6.9, 0.5, 'per pixel', ha='center', va='top', fontsize=6, style='italic')

        # Arrow 4→5
        ax_pipeline.annotate('', xy=(8.0, 1.2), xytext=(7.7, 1.2),
                            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        ax_pipeline.text(7.85, 1.4, 'v(x,y)', ha='center', fontsize=7, style='italic', family='monospace')

        # Step 5: Cosine Similarity
        ax_pipeline.add_patch(mpatches.FancyBboxPatch((8.1, 0.7), 1.8, 1.0,
                             boxstyle="round,pad=0.1", facecolor='#E1BEE7', edgecolor='#6A1B9A', linewidth=2.5))
        ax_pipeline.text(9.0, 1.45, '⑤  Cosine Sim.', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_pipeline.text(9.0, 1.1, 'sim = v(x,y)·tᵢ', ha='center', va='center', fontsize=7, family='monospace')
        ax_pipeline.text(9.0, 0.85, '||v(x,y)|| · ||tᵢ||', ha='center', va='center', fontsize=7, family='monospace')
        ax_pipeline.text(9.0, 0.5, 'per class i', ha='center', va='top', fontsize=6, style='italic')

        # Arrow 5→6
        ax_pipeline.annotate('', xy=(10.2, 1.2), xytext=(9.9, 1.2),
                            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        ax_pipeline.text(10.05, 1.4, 'scores', ha='center', fontsize=7, style='italic')

        # Step 6: Softmax
        ax_pipeline.add_patch(mpatches.FancyBboxPatch((10.3, 0.7), 1.5, 1.0,
                             boxstyle="round,pad=0.1", facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2.5))
        ax_pipeline.text(11.05, 1.45, '⑥  Softmax', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_pipeline.text(11.05, 1.1, 'probabilities', ha='center', va='center', fontsize=7)
        ax_pipeline.text(11.05, 0.85, 'H×W×C', ha='center', va='center', fontsize=7, family='monospace')
        ax_pipeline.text(11.05, 0.5, 'OUTPUT', ha='center', va='top', fontsize=6, style='italic', fontweight='bold')

        # Row 2: Detailed per-patch vision encoding and CSA mechanism
        # Column 0: Per-patch vision encoding process
        ax_vision = fig.add_subplot(gs[2, 0])
        ax_vision.axis('off')
        ax_vision.set_xlim(0, 1)
        ax_vision.set_ylim(0, 1)

        ax_vision.text(0.5, 0.97, '③ Vision Encoding per Patch', ha='center', fontsize=10,
                      fontweight='bold', bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

        # Show a single patch going through ViT
        # Input patch
        patch_rect = mpatches.FancyBboxPatch((0.15, 0.72), 0.25, 0.18,
                                            boxstyle="round,pad=0.01",
                                            facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax_vision.add_patch(patch_rect)
        ax_vision.text(0.275, 0.81, '224×224\npatch', ha='center', va='center', fontsize=7, fontweight='bold')

        # Arrow to ViT
        ax_vision.annotate('', xy=(0.48, 0.81), xytext=(0.40, 0.81),
                          arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        # ViT encoder box
        vit_rect = mpatches.FancyBboxPatch((0.50, 0.68), 0.35, 0.26,
                                          boxstyle="round,pad=0.02",
                                          facecolor='#FFE082', edgecolor='#F57C00', linewidth=2.5)
        ax_vision.add_patch(vit_rect)
        ax_vision.text(0.675, 0.89, 'CLIP ViT-B/16', ha='center', fontsize=8, fontweight='bold')
        ax_vision.text(0.675, 0.84, '• Patch embed (16×16)', ha='left', fontsize=6, family='monospace')
        ax_vision.text(0.675, 0.80, '• 12 Transformer blocks', ha='left', fontsize=6, family='monospace')
        ax_vision.text(0.675, 0.76, '• Layer norm', ha='left', fontsize=6, family='monospace')
        ax_vision.text(0.675, 0.72, '• Projection head', ha='left', fontsize=6, family='monospace')

        # Arrow to output
        ax_vision.annotate('', xy=(0.675, 0.62), xytext=(0.675, 0.68),
                          arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax_vision.text(0.75, 0.65, 'vᵢ ∈ ℝ⁵¹²', ha='left', fontsize=7,
                      family='monospace', style='italic', fontweight='bold')

        # Output embedding
        embed_rect = mpatches.FancyBboxPatch((0.28, 0.50), 0.44, 0.08,
                                            boxstyle="round,pad=0.01",
                                            facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax_vision.add_patch(embed_rect)
        ax_vision.text(0.5, 0.54, 'Patch embedding vᵢ (512-dim vector)',
                      ha='center', va='center', fontsize=7, fontweight='bold')

        # Key points
        ax_vision.text(0.5, 0.40, 'Key points:', ha='center', fontsize=7, fontweight='bold', style='italic')
        ax_vision.text(0.08, 0.32, '• Each 224×224 patch encoded independently',
                      ha='left', fontsize=6.5, family='monospace')
        ax_vision.text(0.08, 0.26, '• ViT splits patch into 14×14 grid of 16×16 tokens',
                      ha='left', fontsize=6.5, family='monospace')
        ax_vision.text(0.08, 0.20, '• Self-attention within patch (not across patches)',
                      ha='left', fontsize=6.5, family='monospace')
        ax_vision.text(0.08, 0.14, '• Output: 512-dim embedding per patch',
                      ha='left', fontsize=6.5, family='monospace')
        ax_vision.text(0.08, 0.08, '• Repeated for EVERY sliding window position',
                      ha='left', fontsize=6.5, family='monospace', color='red', fontweight='bold')
        ax_vision.text(0.08, 0.02, '  Example: 500×375 image → ~12 patches to encode',
                      ha='left', fontsize=6, family='monospace', style='italic')

        # Column 1-2: CSA Mechanism - Detailed self-attention computation
        ax_csa = fig.add_subplot(gs[2, 1:3])
        ax_csa.axis('off')
        ax_csa.set_xlim(0, 2)
        ax_csa.set_ylim(0, 1)

        ax_csa.text(1.0, 0.97, '④ Cross-Spatial Attention (CSA): Merge Overlapping Patches',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

        # Left side: Show pixel P covered by 4 patches
        ax_csa.text(0.35, 0.88, 'Pixel P covered by 4 patches:', ha='center', fontsize=8, fontweight='bold')

        # Central pixel P
        center_circle = plt.Circle((0.35, 0.68), 0.055, color='#D32F2F', alpha=0.9, zorder=3)
        ax_csa.add_patch(center_circle)
        ax_csa.text(0.35, 0.68, 'P', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=4)

        # 4 overlapping patches
        positions = [(0.20, 0.78), (0.50, 0.78), (0.20, 0.58), (0.50, 0.58)]
        colors = ['#00BCD4', '#8BC34A', '#FFC107', '#E91E63']
        patch_labels = ['v₁', 'v₂', 'v₃', 'v₄']
        for i, (px, py) in enumerate(positions):
            patch_circle = plt.Circle((px, py), 0.045, color=colors[i], alpha=0.7, zorder=1)
            ax_csa.add_patch(patch_circle)
            ax_csa.text(px, py, patch_labels[i], ha='center', va='center', fontsize=7, fontweight='bold', zorder=2)
            ax_csa.annotate('', xy=(0.35, 0.68), xytext=(px, py),
                          arrowprops=dict(arrowstyle='->', lw=1.5, color=colors[i], alpha=0.8))

        # Right side: Correlative Self-Attention formula
        ax_csa.text(1.3, 0.88, 'Correlative Self-Attention Formula:', ha='center', fontsize=8, fontweight='bold')

        # Step-by-step CSA computation
        ax_csa.text(0.75, 0.78, '① Feature matrix for pixel P:', ha='left', fontsize=7, fontweight='bold')
        ax_csa.text(0.78, 0.74, 'X = [v₁(P), v₂(P), v₃(P), v₄(P)]ᵀ ∈ ℝ⁴ˣ⁵¹²',
                   ha='left', fontsize=6.5, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        ax_csa.text(0.75, 0.66, '② Compute attention scores:', ha='left', fontsize=7, fontweight='bold')
        ax_csa.text(0.78, 0.62, 'S = X·Wᵣ·Wᵣᵀ·Xᵀ / τ  ∈ ℝ⁴ˣ⁴',
                   ha='left', fontsize=6.5, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        ax_csa.text(0.78, 0.58, 'where Wᵣ ∈ ℝ⁵¹²ˣᵈ is learned projection',
                   ha='left', fontsize=6, family='monospace', style='italic')
        ax_csa.text(0.78, 0.54, 'τ is temperature parameter',
                   ha='left', fontsize=6, family='monospace', style='italic')

        ax_csa.text(0.75, 0.46, '③ Apply softmax to get attention weights:', ha='left', fontsize=7, fontweight='bold')
        ax_csa.text(0.78, 0.42, 'A = Softmax(S)  ∈ ℝ⁴ˣ⁴',
                   ha='left', fontsize=6.5, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        ax_csa.text(0.78, 0.38, 'Each row sums to 1.0 (normalized)',
                   ha='left', fontsize=6, family='monospace', style='italic', color='red')

        ax_csa.text(0.75, 0.30, '④ Compute weighted average:', ha='left', fontsize=7, fontweight='bold')
        ax_csa.text(0.78, 0.26, 'V = A·X  ∈ ℝ⁴ˣ⁵¹²',
                   ha='left', fontsize=6.5, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        ax_csa.text(0.78, 0.22, 'v(P) = V[i] for patch i containing P',
                   ha='left', fontsize=6, family='monospace', style='italic')

        ax_csa.text(0.75, 0.14, '⑤ Result: Merged feature for pixel P:', ha='left', fontsize=7, fontweight='bold')
        ax_csa.text(0.78, 0.10, 'v(P) = Σᵢ αᵢ·vᵢ(P)  ∈ ℝ⁵¹²',
                   ha='left', fontsize=6.5, family='monospace', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        ax_csa.text(0.78, 0.06, 'where αᵢ are learned attention weights (context-aware)',
                   ha='left', fontsize=6, family='monospace', style='italic')
        ax_csa.text(0.78, 0.02, 'This handles unequal coverage (1-4 patches per pixel)',
                   ha='left', fontsize=6, family='monospace', style='italic', color='blue')

        # Bottom note
        ax_csa.text(0.35, 0.48, 'Interior pixel\n(4 patches)', ha='center', fontsize=6.5, style='italic')
        ax_csa.text(0.35, 0.40, '↓', ha='center', fontsize=10)
        ax_csa.text(0.35, 0.35, 'v(P) merged\nfrom 4 vᵢ', ha='center', fontsize=6.5,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

        # Column 3: Cosine Similarity computation
        ax_cosine = fig.add_subplot(gs[2, 3])
        ax_cosine.axis('off')
        ax_cosine.set_xlim(0, 1)
        ax_cosine.set_ylim(0, 1)

        ax_cosine.text(0.5, 0.97, '⑤ Cosine Similarity', ha='center', fontsize=10,
                      fontweight='bold', bbox=dict(boxstyle='round', facecolor='#E1BEE7', alpha=0.8))

        # Visual representation
        # Merged visual feature
        v_rect = mpatches.FancyBboxPatch((0.15, 0.78), 0.28, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax_cosine.add_patch(v_rect)
        ax_cosine.text(0.29, 0.82, 'v(x,y) ∈ ℝ⁵¹²', ha='center', fontsize=7,
                      fontweight='bold', family='monospace')

        # Text embeddings
        t_rect = mpatches.FancyBboxPatch((0.57, 0.78), 0.28, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax_cosine.add_patch(t_rect)
        ax_cosine.text(0.71, 0.82, 'tᵢ ∈ ℝ⁵¹²', ha='center', fontsize=7,
                      fontweight='bold', family='monospace')

        # Dot product
        ax_cosine.text(0.5, 0.70, '↓ Cosine Similarity ↓', ha='center', fontsize=7, style='italic')

        # Formula box
        formula_rect = mpatches.FancyBboxPatch((0.08, 0.52), 0.84, 0.14,
                                              boxstyle="round,pad=0.02",
                                              facecolor='lightyellow', edgecolor='orange', linewidth=2.5)
        ax_cosine.add_patch(formula_rect)
        ax_cosine.text(0.5, 0.62, 'simᵢ(x,y) = v(x,y) · tᵢ', ha='center', fontsize=8,
                      fontweight='bold', family='monospace')
        ax_cosine.text(0.5, 0.58, '──────────────', ha='center', fontsize=8)
        ax_cosine.text(0.5, 0.54, '||v(x,y)|| · ||tᵢ||', ha='center', fontsize=8,
                      fontweight='bold', family='monospace')

        # Explanation
        ax_cosine.text(0.5, 0.44, 'Computed for each class i:', ha='center', fontsize=7, fontweight='bold')
        ax_cosine.text(0.08, 0.38, '• Measures alignment between visual', ha='left', fontsize=6.5)
        ax_cosine.text(0.08, 0.34, '  and text embeddings', ha='left', fontsize=6.5)
        ax_cosine.text(0.08, 0.30, '• Range: [-1, 1], normalized to [-1, 1]', ha='left', fontsize=6.5)
        ax_cosine.text(0.08, 0.26, '• Higher = better match', ha='left', fontsize=6.5, fontweight='bold', color='green')

        # Result
        ax_cosine.text(0.5, 0.18, 'Output per pixel (x,y):', ha='center', fontsize=7, fontweight='bold')
        result_rect = mpatches.FancyBboxPatch((0.15, 0.08), 0.70, 0.08,
                                             boxstyle="round,pad=0.01",
                                             facecolor='lightcoral', edgecolor='red', linewidth=2)
        ax_cosine.add_patch(result_rect)
        ax_cosine.text(0.5, 0.12, '[sim₁, sim₂, ..., simₙ]  (n classes)',
                      ha='center', fontsize=7, fontweight='bold', family='monospace')

        ax_cosine.text(0.5, 0.02, 'Then: Softmax → probabilities', ha='center',
                      fontsize=6.5, style='italic', color='red')

        # Row 3: Technical explanation
        ax_exp = fig.add_subplot(gs[3, :])
        ax_exp.axis('off')

        explanation = """Technical Details:
① Text Encoding: All class names encoded ONCE via CLIP text encoder → {t₁, t₂, ..., tₙ} ∈ ℝⁿˣᵈ (d=512 for ViT-B/16)
② Patch Extraction: Slide 224×224 window with 112-pixel stride → overlapping patches covering entire image
③ Vision Encoding: Each patch independently encoded via CLIP vision encoder (ViT-B/16) → patch features vᵢ(x,y)
④ Cross-Spatial Attention (CSA): Merge features from overlapping patches using correlative self-attention → v(x,y) = Σᵢ αᵢ·vᵢ(x,y)
⑤ Cosine Similarity: For each pixel (x,y) and each class i, compute similarity between v(x,y) and text embedding tᵢ
⑥ Softmax: Normalize similarities across classes → probability map P(class i | pixel x,y) → OUTPUT: H×W×C dense predictions"""

        ax_exp.text(0.02, 0.7, explanation, transform=ax_exp.transAxes,
                   fontsize=8.5, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=0.8))

        fig.suptitle("Step 4: SCLIP Visual-Text Similarity Computation", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "04_similarity.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_05_confidence_maps(self, image: np.ndarray, probs: np.ndarray):
        """Step 5: Show per-class confidence maps."""
        num_classes = len(self.vocabulary)
        cols = min(4, num_classes)
        rows = (num_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if num_classes == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_classes):
            row, col = i // cols, i % cols
            ax = axes[row, col]

            conf_map = probs[:, :, i]
            im = ax.imshow(conf_map, cmap='YlGnBu_r', vmin=0, vmax=1)
            ax.set_title(f"{self.vocabulary[i]}\nMax: {conf_map.max():.3f}",
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for i in range(num_classes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')

        fig.suptitle("Step 5: Per-Class Confidence Maps", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "05_confidence_maps.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_06_dense_prediction(self, image: np.ndarray, seg_map: np.ndarray):
        """Step 6: Show SCLIP dense prediction (argmax)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Colored segmentation
        colored_seg = self.apply_color_map(seg_map)
        ax1.imshow(colored_seg)
        ax1.set_title("Dense Prediction (argmax)", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Right: Overlay on image
        ax2.imshow(image)
        ax2.imshow(colored_seg, alpha=0.5)
        ax2.set_title("Prediction Overlay", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.class_colors[i], label=self.vocabulary[i])
            for i in range(len(self.vocabulary))
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(4, len(self.vocabulary)),
                  fontsize=10, framealpha=0.9)

        fig.suptitle("Step 6: SCLIP Dense Prediction (Class with Highest Confidence)",
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.steps_dir / "06_dense_prediction.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_07_prediction_confidences(self, image: np.ndarray,
                                                 seg_map: np.ndarray, probs: np.ndarray):
        """Step 7: Show prediction confidences (multi-view)."""
        max_probs = probs.max(axis=2)

        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3)

        # Left: Confidence heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(max_probs, cmap='YlGnBu_r', vmin=0, vmax=1)
        ax1.set_title("Confidence Heatmap\n(Max Probability)", fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Middle: Confidence-weighted prediction
        ax2 = fig.add_subplot(gs[0, 1])
        colored_seg = self.apply_color_map(seg_map)
        # Create RGBA with alpha = confidence
        rgba_pred = np.dstack([colored_seg, (max_probs * 255).astype(np.uint8)])
        ax2.imshow(image)
        ax2.imshow(rgba_pred)
        ax2.set_title("Confidence-Weighted Overlay\n(Opacity = Confidence)",
                     fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Right: Uncertain regions
        ax3 = fig.add_subplot(gs[0, 2])
        uncertain_mask = max_probs < self.min_confidence
        uncertain_viz = image.copy()
        uncertain_viz[uncertain_mask] = [255, 0, 0]  # Red for uncertain
        ax3.imshow(image)
        ax3.imshow(uncertain_viz, alpha=0.3)
        ax3.set_title(f"Uncertain Regions\n(Confidence < {self.min_confidence})",
                     fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Statistics
        stats_text = f"Mean confidence: {max_probs.mean():.3f}\n"
        stats_text += f"Median confidence: {np.median(max_probs):.3f}\n"
        stats_text += f"Uncertain pixels: {uncertain_mask.sum():,} ({uncertain_mask.sum()/uncertain_mask.size*100:.1f}%)"

        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle("Step 7: SCLIP Dense Prediction Confidences",
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(self.steps_dir / "07_prediction_confidences.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_08_thresholding(self, image: np.ndarray,
                                      seg_map: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Step 8: Show confidence thresholding."""
        max_probs = probs.max(axis=2)
        high_conf_mask = max_probs >= self.min_confidence

        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        # Top-left: Original prediction
        colored_seg = self.apply_color_map(seg_map)
        axes[0, 0].imshow(colored_seg)
        axes[0, 0].set_title("Before Thresholding\n(All Predictions)", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Top-right: After thresholding
        filtered_seg = seg_map.copy()
        filtered_seg[~high_conf_mask] = -1  # Mark uncertain as -1
        colored_filtered = self.apply_color_map(filtered_seg, mark_invalid=True)
        axes[0, 1].imshow(colored_filtered)
        axes[0, 1].set_title(f"After Thresholding (≥{self.min_confidence})\n(Uncertain = Gray)",
                           fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Bottom-left: Confidence histogram
        axes[1, 0].hist(max_probs.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.min_confidence, color='red', linestyle='--', linewidth=2,
                          label=f'Threshold = {self.min_confidence}')
        axes[1, 0].set_xlabel('Confidence', fontsize=10)
        axes[1, 0].set_ylabel('Pixel Count', fontsize=10)
        axes[1, 0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Bottom-right: Statistics
        axes[1, 1].axis('off')
        stats_text = f"""Confidence Thresholding Statistics:

Threshold: {self.min_confidence}

Before filtering:
  • Total pixels: {high_conf_mask.size:,}
  • All pixels have predictions

After filtering:
  • High confidence: {high_conf_mask.sum():,} pixels ({high_conf_mask.sum()/high_conf_mask.size*100:.1f}%)
  • Low confidence: {(~high_conf_mask).sum():,} pixels ({(~high_conf_mask).sum()/high_conf_mask.size*100:.1f}%)

Rationale:
  Low-confidence predictions are unreliable and will be excluded
  from prompt point extraction. This focuses SAM on regions where
  CLIP has strong semantic understanding."""

        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        fig.suptitle("Step 8: Confidence Thresholding", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "08_thresholding.png", dpi=150, bbox_inches='tight')
        plt.close()

        return high_conf_mask

    def visualize_step_09_connected_components(self, image: np.ndarray,
                                               seg_map: np.ndarray,
                                               high_conf_mask: np.ndarray) -> Dict:
        """Step 9: Show connected components per class."""
        labeled_components = {}

        num_classes = len(self.vocabulary)
        cols = min(3, num_classes)
        rows = (num_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if num_classes == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for class_idx in range(num_classes):
            row, col = class_idx // cols, class_idx % cols
            ax = axes[row, col]

            # Get mask for this class
            class_mask = (seg_map == class_idx) & high_conf_mask

            # Find connected components
            labeled, num_regions = label(class_mask)
            labeled_components[class_idx] = (labeled, num_regions)

            # Visualize
            ax.imshow(image, alpha=0.3)
            if num_regions > 0:
                # Color each component differently
                colored_components = np.zeros((*labeled.shape, 3))
                for region_id in range(1, num_regions + 1):
                    region_mask = labeled == region_id
                    hue = (region_id * 0.618033988749895) % 1.0  # Golden ratio
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    colored_components[region_mask] = rgb
                ax.imshow(colored_components, alpha=0.7)

                # Label each component
                for region_id in range(1, num_regions + 1):
                    region_mask = labeled == region_id
                    y_coords, x_coords = np.where(region_mask)
                    if len(x_coords) > 0:
                        cx, cy = int(x_coords.mean()), int(y_coords.mean())
                        ax.text(cx, cy, str(region_id), color='white', fontsize=10,
                               fontweight='bold', ha='center', va='center',
                               bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))

            ax.set_title(f"{self.vocabulary[class_idx]}\n{num_regions} regions",
                        fontsize=10, fontweight='bold')
            ax.axis('off')

        # Hide unused subplots
        for i in range(num_classes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')

        fig.suptitle("Step 9: Connected Components per Class", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "09_connected_components.png", dpi=150, bbox_inches='tight')
        plt.close()

        return labeled_components

    def visualize_step_10_size_filter(self, image: np.ndarray,
                                     labeled_components: Dict) -> Dict:
        """Step 10: Show min region size filtering."""
        filtered_components = {}

        num_classes = len(self.vocabulary)
        cols = min(3, num_classes)
        rows = (num_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if num_classes == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for class_idx in range(num_classes):
            row, col = class_idx // cols, class_idx % cols
            ax = axes[row, col]

            labeled, num_regions = labeled_components[class_idx]

            # Filter by size
            kept_regions = []
            removed_regions = []

            for region_id in range(1, num_regions + 1):
                region_mask = labeled == region_id
                region_size = region_mask.sum()

                if region_size >= self.min_region_size:
                    kept_regions.append(region_id)
                else:
                    removed_regions.append(region_id)

            filtered_components[class_idx] = (labeled, kept_regions)

            # Visualize
            ax.imshow(image, alpha=0.3)

            # Show removed regions in red
            for region_id in removed_regions:
                region_mask = labeled == region_id
                red_overlay = np.zeros((*labeled.shape, 4))
                red_overlay[region_mask] = [1, 0, 0, 0.5]
                ax.imshow(red_overlay)

            # Show kept regions in green
            for region_id in kept_regions:
                region_mask = labeled == region_id
                green_overlay = np.zeros((*labeled.shape, 4))
                green_overlay[region_mask] = [0, 1, 0, 0.5]
                ax.imshow(green_overlay)

            ax.set_title(f"{self.vocabulary[class_idx]}\nKept: {len(kept_regions)} / Removed: {len(removed_regions)}",
                        fontsize=10, fontweight='bold')
            ax.axis('off')

        # Hide unused subplots
        for i in range(num_classes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='green', alpha=0.5, label=f'Kept (≥{self.min_region_size} px)'),
            mpatches.Patch(facecolor='red', alpha=0.5, label=f'Removed (<{self.min_region_size} px)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10, framealpha=0.9)

        fig.suptitle("Step 10: Min Region Size Filter", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(self.steps_dir / "10_size_filter.png", dpi=150, bbox_inches='tight')
        plt.close()

        return filtered_components

    def visualize_step_11_positive_prompts(self, image: np.ndarray,
                                          seg_map: np.ndarray, prompts: List[Dict]):
        """Step 11: Show positive point prompts."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(image)

        # Draw all positive prompts
        for prompt_info in prompts:
            point = prompt_info['point']
            class_idx = prompt_info['class_idx']
            confidence = prompt_info['confidence']

            color = self.class_colors[class_idx]

            # Draw point
            ax.plot(point[0], point[1], '*', color=color, markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5)

            # Draw small label
            if confidence > 0.7:  # Only label high-confidence points
                ax.text(point[0], point[1]-10, f"{confidence:.2f}",
                       color='white', fontsize=8, ha='center',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

        ax.set_title(f"Step 11: Positive Point Prompts ({len(prompts)} points)",
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.class_colors[i], label=self.vocabulary[i])
            for i in range(len(self.vocabulary))
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

        # Statistics
        stats_text = f"Points per cluster: {self.points_per_cluster}\n"
        stats_text += f"Total prompts: {len(prompts)}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.steps_dir / "11_positive_prompts.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_12_negative_prompts(self, image: np.ndarray,
                                          seg_map: np.ndarray, prompts: List[Dict]):
        """Step 12: Show negative point prompts."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(image)

        total_negative = 0

        # Draw prompts with their negative points
        for prompt_info in prompts:
            point = prompt_info['point']
            class_idx = prompt_info['class_idx']
            negative_points = prompt_info.get('negative_points', [])

            color = self.class_colors[class_idx]

            # Draw positive point (green star)
            ax.plot(point[0], point[1], '*', color='lime', markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5, zorder=10)

            # Draw negative points (red X)
            for neg_point in negative_points:
                ax.plot(neg_point[0], neg_point[1], 'x', color='red', markersize=12,
                       markeredgewidth=3, zorder=10)
                # Draw line connecting to positive point
                ax.plot([point[0], neg_point[0]], [point[1], neg_point[1]],
                       'r--', alpha=0.3, linewidth=1)
                total_negative += 1

        ax.set_title(f"Step 12: Negative Point Prompts ({total_negative} negative points)",
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                      markersize=12, label='Positive (foreground)'),
            plt.Line2D([0], [0], marker='x', color='red', markersize=10,
                      markeredgewidth=3, label='Negative (background)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

        # Statistics
        stats_text = f"Negative points per cluster: {self.negative_points_per_cluster}\n"
        stats_text += f"Negative confidence threshold: {self.negative_confidence_threshold}\n"
        stats_text += f"Total positive: {len(prompts)}\n"
        stats_text += f"Total negative: {total_negative}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.steps_dir / "12_negative_prompts.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_13_sam_prompting(self, image: np.ndarray, prompts: List[Dict]):
        """Step 13: SAM prompting visualization."""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(image)

        # Draw all prompts with labels
        for i, prompt_info in enumerate(prompts):
            point = prompt_info['point']
            class_idx = prompt_info['class_idx']
            class_name = prompt_info['class_name']
            negative_points = prompt_info.get('negative_points', [])

            color = self.class_colors[class_idx]

            # Positive point
            ax.plot(point[0], point[1], '*', color=color, markersize=20,
                   markeredgecolor='white', markeredgewidth=2, zorder=10)
            ax.text(point[0], point[1]+20, f"{i+1}", color='white', fontsize=9,
                   ha='center', fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=color, alpha=0.9))

            # Negative points
            for neg_point in negative_points:
                ax.plot(neg_point[0], neg_point[1], 'x', color='red', markersize=12,
                       markeredgewidth=3, zorder=10)

        ax.set_title(f"Step 13: SAM 2 Prompting ({len(prompts)} prompt groups)",
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.class_colors[i], label=self.vocabulary[i])
            for i in range(len(self.vocabulary))
        ]
        legend_elements.extend([
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
                      markersize=12, label='Positive prompt'),
            plt.Line2D([0], [0], marker='x', color='red', markersize=10,
                      markeredgewidth=3, label='Negative prompt'),
        ])
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

        plt.tight_layout()
        plt.savefig(self.steps_dir / "13_sam_prompting.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_14_sam_candidates(self, image: np.ndarray,
                                        prompts: List[Dict], results: List[Dict],
                                        all_candidates: List[List[Dict]]):
        """Step 14: Show SAM multi-mask candidates with actual 3 masks per prompt."""
        # Show first 6 prompts as examples
        num_examples = min(6, len(prompts))

        fig, axes = plt.subplots(num_examples, 4, figsize=(18, 4*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_examples):
            prompt_info = prompts[i]
            point = prompt_info['point']
            class_name = prompt_info['class_name']
            negative_points = prompt_info.get('negative_points', [])

            # Get the 3 candidates for this prompt
            candidates = all_candidates[i]

            # Find which candidate is the best
            best_score = max(c['score'] for c in candidates)
            best_idx = next(j for j, c in enumerate(candidates) if c['score'] == best_score)

            # Original image with prompt
            axes[i, 0].imshow(image)
            axes[i, 0].plot(point[0], point[1], '*', color='lime', markersize=15,
                          markeredgecolor='white', markeredgewidth=1.5, zorder=10)
            # Draw negative points if present
            for neg_point in negative_points:
                axes[i, 0].plot(neg_point[0], neg_point[1], 'x', color='red',
                              markersize=10, markeredgewidth=2.5, zorder=10)
            axes[i, 0].set_title(f"Prompt {i+1}: {class_name}", fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')

            # Show all 3 mask candidates
            for j in range(3):
                candidate = candidates[j]
                mask = candidate['mask']
                score = candidate['score']
                is_best = (j == best_idx)

                axes[i, j+1].imshow(image)

                # Mask overlay with different colors/alpha for best vs others
                mask_overlay = np.zeros((*mask.shape, 4))
                mask_bool = mask.astype(bool)  # Ensure boolean type

                if is_best:
                    # Best mask: bright green with higher alpha
                    mask_overlay[mask_bool] = [0, 1, 0, 0.7]
                    title_color = 'green'
                    title = f"Mask {j+1} ★ BEST\nScore: {score:.3f}"
                else:
                    # Other masks: cyan with lower alpha
                    mask_overlay[mask_bool] = [0, 0.8, 1, 0.4]
                    title_color = 'darkblue'
                    title = f"Mask {j+1}\nScore: {score:.3f}"

                axes[i, j+1].imshow(mask_overlay)
                axes[i, j+1].set_title(title, fontsize=9, fontweight='bold', color=title_color)
                axes[i, j+1].axis('off')

        # Add explanation text
        explanation = """SAM Multi-Mask Output: For each point prompt, SAM generates 3 mask candidates with different granularities.
The mask with the highest confidence score (marked with ★) is selected as the final prediction.
Green = Best mask | Cyan = Alternative candidates"""

        fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle("Step 14: SAM Multi-Mask Candidates (First 6 Prompts)",
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.04, 1, 0.99])
        plt.savefig(self.steps_dir / "14_sam_candidates.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_15_mask_selection(self, image: np.ndarray, results: List[Dict]):
        """Step 15: Show best mask selection."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: All selected masks
        axes[0].imshow(image)
        for result in results:
            mask = result['mask']
            class_idx = result['class_idx']
            color = self.class_colors[class_idx]

            mask_overlay = np.zeros((*mask.shape, 4))
            mask_bool = mask.astype(bool)  # Ensure boolean type
            mask_overlay[mask_bool] = [*color, 0.5]
            axes[0].imshow(mask_overlay)

        axes[0].set_title(f"All Best Masks ({len(results)} masks)", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Right: Statistics
        axes[1].axis('off')

        # Compute statistics
        scores = [r['score'] for r in results]
        sizes = [r['mask'].sum() for r in results]

        stats_text = f"""Mask Selection Statistics:

Total masks generated: {len(results)}

SAM Scores:
  • Mean: {np.mean(scores):.3f}
  • Median: {np.median(scores):.3f}
  • Min: {np.min(scores):.3f}
  • Max: {np.max(scores):.3f}

Mask Sizes:
  • Mean: {np.mean(sizes):,.0f} pixels
  • Median: {np.median(sizes):,.0f} pixels
  • Min: {np.min(sizes):,} pixels
  • Max: {np.max(sizes):,} pixels

Selection Criterion:
  For each prompt, SAM generates 3 candidate masks.
  We select the mask with the highest IoU score.
  Higher scores indicate better mask quality."""

        axes[1].text(0.1, 0.9, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        fig.suptitle("Step 15: Best Mask Selection", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.steps_dir / "15_mask_selection.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_16_overlap_resolution(self, image: np.ndarray,
                                            results: List[Dict], merged_results: List[Dict]):
        """Step 16: Show overlap resolution."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Before merging (with overlaps)
        axes[0].imshow(image)
        for result in results:
            mask = result['mask']
            class_idx = result['class_idx']
            color = self.class_colors[class_idx]

            mask_overlay = np.zeros((*mask.shape, 4))
            mask_bool = mask.astype(bool)  # Ensure boolean type
            mask_overlay[mask_bool] = [*color, 0.5]
            axes[0].imshow(mask_overlay)

        axes[0].set_title(f"Before Merging ({len(results)} masks)", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Right: After merging
        axes[1].imshow(image)
        for result in merged_results:
            mask = result['mask']
            class_idx = result['class_idx']
            color = self.class_colors[class_idx]

            mask_overlay = np.zeros((*mask.shape, 4))
            mask_bool = mask.astype(bool)  # Ensure boolean type
            mask_overlay[mask_bool] = [*color, 0.5]
            axes[1].imshow(mask_overlay)

        axes[1].set_title(f"After Merging ({len(merged_results)} masks)", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Statistics
        removed = len(results) - len(merged_results)
        stats_text = f"IoU threshold: {self.iou_threshold}\n"
        stats_text += f"Removed: {removed} overlapping masks\n"
        stats_text += f"Kept: {len(merged_results)} masks"

        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        fig.suptitle("Step 16: Overlap Resolution (IoU-based Merging)",
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.steps_dir / "16_overlap_resolution.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_17_final_segmentation(self, image: np.ndarray, final_seg_map: np.ndarray):
        """Step 17: Show final segmentation."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Segmentation mask
        colored_seg = self.apply_color_map(final_seg_map)
        axes[1].imshow(colored_seg)
        axes[1].set_title("Final Segmentation", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(colored_seg, alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.class_colors[i], label=self.vocabulary[i])
            for i in range(len(self.vocabulary))
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(4, len(self.vocabulary)),
                  fontsize=10, framealpha=0.9)

        fig.suptitle("Step 17: Final Segmentation Output", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.steps_dir / "17_final_segmentation.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_step_18_comparison_grid(self, image: np.ndarray, sclip_seg: np.ndarray,
                                         probs: np.ndarray, final_seg: np.ndarray):
        """Step 18: Show comparison grid."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2)

        # Top-left: Original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Top-right: SCLIP prediction
        ax2 = fig.add_subplot(gs[0, 1])
        colored_sclip = self.apply_color_map(sclip_seg)
        ax2.imshow(colored_sclip)
        ax2.set_title("SCLIP Dense Prediction", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Bottom-left: SCLIP confidences
        ax3 = fig.add_subplot(gs[1, 0])
        max_probs = probs.max(axis=2)
        im = ax3.imshow(max_probs, cmap='YlGnBu_r', vmin=0, vmax=1)
        ax3.set_title("SCLIP Prediction Confidences", fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Bottom-right: Final SAM-refined
        ax4 = fig.add_subplot(gs[1, 1])
        colored_final = self.apply_color_map(final_seg)
        ax4.imshow(colored_final)
        ax4.set_title("SAM-Refined Final Output", fontsize=14, fontweight='bold')
        ax4.axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.class_colors[i], label=self.vocabulary[i])
            for i in range(len(self.vocabulary))
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(4, len(self.vocabulary)),
                  fontsize=10, framealpha=0.9)

        fig.suptitle("Step 18: Complete Pipeline Comparison", fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(self.steps_dir / "18_comparison_grid.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ========== Utility Methods ==========

    def apply_color_map(self, seg_map: np.ndarray, mark_invalid: bool = False) -> np.ndarray:
        """Convert segmentation map to RGB using class colors."""
        h, w = seg_map.shape
        colored = np.zeros((h, w, 3))

        for class_idx in range(len(self.vocabulary)):
            mask = seg_map == class_idx
            colored[mask] = self.class_colors[class_idx]

        if mark_invalid:
            # Mark invalid regions (-1) as gray
            invalid_mask = seg_map == -1
            colored[invalid_mask] = [0.5, 0.5, 0.5]

        return colored

    def create_final_segmentation(self, image: np.ndarray, results: List[Dict],
                                 seg_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Create final segmentation map from SAM results.

        Args:
            image: Input image
            results: SAM segmentation results
            seg_map: Optional SCLIP segmentation map to use as base layer.
                    If provided, SAM masks will only override high-confidence regions.
                    If None, background pixels will be class 0.

        Returns:
            Final segmentation map
        """
        H, W = image.shape[:2]

        # Use SCLIP segmentation as base if available, otherwise zeros
        if seg_map is not None:
            final_seg_map = seg_map.copy()
        else:
            final_seg_map = np.zeros((H, W), dtype=np.int32)

        # Sort by confidence (higher confidence overwrites lower)
        sorted_results = sorted(results, key=lambda x: x['confidence'])

        # Overlay SAM-refined masks on top of SCLIP base
        for result in sorted_results:
            mask = result['mask']
            class_idx = result['class_idx']
            mask_bool = mask.astype(bool)  # Ensure boolean type
            final_seg_map[mask_bool] = class_idx

        return final_seg_map

    def compute_statistics(self):
        """Compute and store pipeline statistics."""
        prompts = self.intermediate_data.get('prompts', [])
        sam_results = self.intermediate_data.get('sam_results', [])
        merged_results = self.intermediate_data.get('merged_results', [])

        self.statistics = {
            'num_classes': len(self.vocabulary),
            'vocabulary': self.vocabulary,
            'num_prompts': len(prompts),
            'num_sam_masks': len(sam_results),
            'num_final_masks': len(merged_results),
            'parameters': {
                'min_confidence': self.min_confidence,
                'min_region_size': self.min_region_size,
                'points_per_cluster': self.points_per_cluster,
                'negative_points_per_cluster': self.negative_points_per_cluster,
                'iou_threshold': self.iou_threshold,
            },
            'per_class': {}
        }

        # Per-class statistics
        for class_idx, class_name in enumerate(self.vocabulary):
            class_prompts = [p for p in prompts if p['class_idx'] == class_idx]
            class_results = [r for r in merged_results if r['class_idx'] == class_idx]

            self.statistics['per_class'][class_name] = {
                'num_prompts': len(class_prompts),
                'num_final_masks': len(class_results),
                'avg_confidence': np.mean([p['confidence'] for p in class_prompts]) if class_prompts else 0,
            }

    def generate_per_class_details(self):
        """Generate detailed per-class visualizations."""
        print("  Generating per-class detailed views...")
        # Implementation would go here
        # For now, placeholder
        pass

    def save_data(self):
        """Save intermediate data for analysis."""
        # Save confidences
        probs = self.intermediate_data.get('probs')
        if probs is not None:
            np.save(self.data_dir / 'confidences.npy', probs)

        # Save prompts
        prompts = self.intermediate_data.get('prompts', [])
        # Convert to serializable format
        prompts_serializable = []
        for p in prompts:
            p_copy = p.copy()
            # Convert tuples to lists
            p_copy['point'] = list(p_copy['point'])
            p_copy['negative_points'] = [list(np) for np in p_copy.get('negative_points', [])]
            prompts_serializable.append(p_copy)

        with open(self.data_dir / 'prompts.json', 'w') as f:
            json.dump(prompts_serializable, f, indent=2)

        # Save statistics
        with open(self.data_dir / 'statistics.json', 'w') as f:
            json.dump(self.statistics, f, indent=2)

    def generate_html(self):
        """Generate interactive HTML visualization."""
        html_content = self._create_html_content()

        with open(self.output_dir / 'index.html', 'w') as f:
            f.write(html_content)

    def _create_html_content(self) -> str:
        """Create HTML content for interactive visualization."""
        # Get list of step images
        step_images = sorted(self.steps_dir.glob("*.png"))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLIP-Guided SAM Pipeline Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        nav {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}

        .step-btn {{
            padding: 10px 15px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
            font-weight: 600;
        }}

        .step-btn:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102,126,234,0.4);
        }}

        .step-btn.active {{
            background: #667eea;
            color: white;
        }}

        .main-content {{
            padding: 30px;
        }}

        .step-container {{
            display: none;
            animation: fadeIn 0.5s;
        }}

        .step-container.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .step-image {{
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .step-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}

        .step-info h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}

        .navigation {{
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
        }}

        .nav-btn {{
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
        }}

        .nav-btn:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102,126,234,0.4);
        }}

        .nav-btn:disabled {{
            background: #dee2e6;
            cursor: not-allowed;
            transform: none;
        }}

        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 2px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CLIP-Guided SAM Pipeline</h1>
            <p class="subtitle">Complete Educational Visualization</p>
            <p style="margin-top: 10px;">Image: {Path(self.image_path).name} | Classes: {len(self.vocabulary)}</p>
        </header>

        <nav id="step-nav">
            <!-- Navigation buttons will be inserted by JavaScript -->
        </nav>

        <div class="main-content">
            <div id="steps-container">
                <!-- Step content will be inserted by JavaScript -->
            </div>

            <div class="navigation">
                <button class="nav-btn" id="prev-btn" onclick="navigateStep(-1)">← Previous</button>
                <button class="nav-btn" id="next-btn" onclick="navigateStep(1)">Next →</button>
            </div>
        </div>

        <footer>
            <p>Generated by CLIP-Guided SAM Explanatory Visualization</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Total Steps: {len(step_images)} |
                Parameters: confidence={self.min_confidence}, region_size={self.min_region_size},
                prompts={self.statistics.get('num_prompts', 0)}
            </p>
        </footer>
    </div>

    <script>
        const steps = {json.dumps([{'num': i+1, 'name': img.stem.replace('_', ' ').title(), 'image': f'steps/{img.name}'} for i, img in enumerate(step_images)])};

        const statistics = {json.dumps(self.statistics)};

        let currentStep = 0;

        function initializeNavigation() {{
            const nav = document.getElementById('step-nav');
            steps.forEach((step, idx) => {{
                const btn = document.createElement('button');
                btn.className = 'step-btn' + (idx === 0 ? ' active' : '');
                btn.textContent = `Step ${{step.num}}`;
                btn.onclick = () => goToStep(idx);
                btn.id = `step-btn-${{idx}}`;
                nav.appendChild(btn);
            }});
        }}

        function loadStep(index) {{
            const container = document.getElementById('steps-container');
            const step = steps[index];

            container.innerHTML = `
                <div class="step-container active">
                    <h2>${{step.name}}</h2>
                    <img src="${{step.image}}" alt="${{step.name}}" class="step-image">
                    <div class="step-info">
                        <h3>Step ${{step.num}} of ${{steps.length}}</h3>
                        <p>This step visualizes: <strong>${{step.name.toLowerCase()}}</strong></p>
                    </div>
                </div>
            `;

            updateNavButtons();
        }}

        function goToStep(index) {{
            if (index < 0 || index >= steps.length) return;

            // Update button states
            document.querySelectorAll('.step-btn').forEach((btn, idx) => {{
                btn.classList.toggle('active', idx === index);
            }});

            currentStep = index;
            loadStep(index);
        }}

        function navigateStep(direction) {{
            const newStep = currentStep + direction;
            if (newStep >= 0 && newStep < steps.length) {{
                goToStep(newStep);
            }}
        }}

        function updateNavButtons() {{
            document.getElementById('prev-btn').disabled = currentStep === 0;
            document.getElementById('next-btn').disabled = currentStep === steps.length - 1;
        }}

        // Initialize
        initializeNavigation();
        loadStep(0);

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') navigateStep(-1);
            if (e.key === 'ArrowRight') navigateStep(1);
        }});
    </script>
</body>
</html>
"""

        return html


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-Guided SAM Explanatory Visualization"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--vocabulary", nargs='+', required=True,
                       help="List of class names for CLIP")
    parser.add_argument("--output", required=True, help="Output directory")

    # SCLIP parameters
    parser.add_argument("--min-confidence", type=float, default=0.3,
                       help="Minimum CLIP confidence for prompts")
    parser.add_argument("--min-region-size", type=int, default=100,
                       help="Minimum region size (pixels) for prompts")

    # SAM prompt parameters
    parser.add_argument("--points-per-cluster", type=int, default=1,
                       help="Number of points per cluster")
    parser.add_argument("--negative-points-per-cluster", type=int, default=0,
                       help="Number of negative points per cluster")
    parser.add_argument("--negative-confidence-threshold", type=float, default=0.8,
                       help="Minimum confidence for negative point regions")
    parser.add_argument("--iou-threshold", type=float, default=0.8,
                       help="IoU threshold for merging overlapping masks")

    # Visualization options
    parser.add_argument("--create-html", action="store_true", default=True,
                       help="Create interactive HTML visualization")
    parser.add_argument("--per-class-details", action="store_true",
                       help="Generate per-class detailed views")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Create visualizer
    visualizer = CLIPGuidedSAMVisualizer(
        image_path=args.image,
        vocabulary=args.vocabulary,
        output_dir=args.output,
        min_confidence=args.min_confidence,
        min_region_size=args.min_region_size,
        points_per_cluster=args.points_per_cluster,
        negative_points_per_cluster=args.negative_points_per_cluster,
        negative_confidence_threshold=args.negative_confidence_threshold,
        iou_threshold=args.iou_threshold,
        create_html=args.create_html,
        create_per_class_details=args.per_class_details,
        device=args.device,
    )

    # Run visualization
    visualizer.run()


if __name__ == "__main__":
    main()
