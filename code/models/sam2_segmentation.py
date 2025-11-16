"""
SAM 2 Mask Generation Module

This module handles automatic mask generation using the Segment Anything Model 2 (SAM 2).
It provides high-quality, class-agnostic segmentation masks at multiple scales.

Reference: Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
import os


@dataclass
class MaskCandidate:
    """Represents a single mask candidate from SAM 2."""
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    predicted_iou: float
    stability_score: float
    point_coords: Optional[np.ndarray] = None


class SAM2MaskGenerator:
    """
    Generates segmentation masks using SAM 2 in automatic mode.

    Implements the mask generation strategy described in Chapter 3.2.2:
    - Automatic mask generation with point prompts
    - Multi-scale hierarchical coverage
    - IoU and stability filtering
    """

    def __init__(
        self,
        model_type: str = "sam2_hiera_tiny",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        points_per_side: int = 48,  # Increased from 32 for better small object detection
        pred_iou_thresh: float = 0.65,  # Lowered for more mask proposals
        stability_score_thresh: float = 0.80,  # Lowered for more mask proposals
        crop_n_layers: int = 0,
        crop_overlap_ratio: float = 512/1500,
        use_fp16: bool = True,  # Mixed precision for faster inference (inspired by SAM-Lightening 2024)
        use_compile: bool = False,  # torch.compile() for JIT optimization
        batch_prompts: bool = True,  # Batch processing of prompts for speedup
    ):
        """
        Initialize SAM 2 mask generator with 2025 performance optimizations.

        Args:
            model_type: SAM 2 checkpoint variant
            device: Device for computation
            points_per_side: Grid resolution for point prompts (32x32 = 1024 points)
            pred_iou_thresh: Minimum IoU confidence threshold
            stability_score_thresh: Minimum stability score threshold
            crop_n_layers: Number of crop layers for large images
            crop_overlap_ratio: Overlap ratio between crops
            use_fp16: Enable mixed precision (FP16) for faster inference
            use_compile: Enable torch.compile() for JIT optimization
            batch_prompts: Enable batch processing of prompts (faster than sequential)
        """
        self.device = device
        self.model_type = model_type
        self.use_fp16 = use_fp16 and device == "cuda"
        self.use_compile = use_compile
        self.batch_prompts = batch_prompts

        # Store parameters for automatic mask generation
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_overlap_ratio = crop_overlap_ratio

        # Try to load SAM 2
        self.mask_generator = None
        self._load_sam2()

    def _load_sam2(self):
        """Load SAM 2 model and create automatic mask generator."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            # Map model type to config file
            model_cfg_map = {
                "sam2_hiera_tiny": "sam2_hiera_t.yaml",
                "sam2_hiera_small": "sam2_hiera_s.yaml",
                "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
                "sam2_hiera_large": "sam2_hiera_l.yaml"
            }

            config_name = model_cfg_map.get(self.model_type, "sam2_hiera_l.yaml")
            checkpoint_path = f"checkpoints/{self.model_type}.pt"

            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                print(f"WARNING: SAM 2 checkpoint not found at {checkpoint_path}")
                print(f"Please download checkpoints using:")
                print(f"  python scripts/download_sam2_checkpoints.py --model {self.model_type}")
                print("Using mock implementation with superpixels for now.\n")
                return

            # Build SAM 2 model
            # SAM 2 uses Hydra config system - just pass the config name
            # It will look in sam2/configs/sam2/ directory automatically
            self.sam2_model = build_sam2(
                config_file=config_name,
                ckpt_path=checkpoint_path,
                device=self.device
            )

            # Note: We use autocast for FP16, NOT .half()
            # This allows PyTorch to automatically handle mixed precision
            if self.use_fp16:
                print(f"✓ SAM2: Enabled FP16 mixed precision (autocast) for 2x speedup")

            # Apply torch.compile() for JIT optimization
            if self.use_compile:
                try:
                    self.sam2_model = torch.compile(self.sam2_model, mode="reduce-overhead")
                    print(f"✓ SAM2: Enabled torch.compile() for JIT optimization")
                except Exception as e:
                    print(f"⚠ SAM2: torch.compile() failed: {e}")
                    self.use_compile = False

            # Create automatic mask generator
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_overlap_ratio=self.crop_overlap_ratio,
            )

            print(f"✓ SAM 2 loaded successfully: {self.model_type}")

        except ImportError as e:
            print(f"WARNING: SAM 2 not installed. Please install with:")
            print(f"  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            print("Using mock implementation with superpixels for now.\n")

        except Exception as e:
            print(f"WARNING: SAM 2 initialization failed: {type(e).__name__}: {str(e)}")
            print("Using mock implementation with superpixels for now.\n")

    def generate_masks(
        self,
        image: np.ndarray,
        return_raw: bool = False
    ) -> List[MaskCandidate]:
        """
        Generate comprehensive set of mask proposals for an image.

        Args:
            image: RGB image as numpy array (H, W, 3)
            return_raw: If True, return raw SAM 2 output

        Returns:
            List of MaskCandidate objects sorted by area (largest first)
        """
        if self.mask_generator is None:
            # Mock implementation for testing without SAM 2
            return self._generate_mock_masks(image)

        # Generate masks using SAM 2
        masks = self.mask_generator.generate(image)

        if return_raw:
            return masks

        # Convert to MaskCandidate objects
        candidates = []
        for mask_data in masks:
            candidate = MaskCandidate(
                mask=mask_data["segmentation"],
                bbox=self._mask_to_bbox(mask_data["segmentation"]),
                area=mask_data["area"],
                predicted_iou=mask_data["predicted_iou"],
                stability_score=mask_data["stability_score"],
                point_coords=mask_data.get("point_coords", None)
            )
            candidates.append(candidate)

        # Sort by area (largest first) for efficient processing
        candidates.sort(key=lambda x: x.area, reverse=True)

        return candidates

    def segment_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        point_labels: List[int]
    ) -> List[MaskCandidate]:
        """
        Segment image using point prompts.

        This enables CLIP-guided prompting where CLIP identifies likely
        object locations and we prompt SAM 2 at those locations.

        Args:
            image: RGB image array (H, W, 3)
            points: List of (x, y) point coordinates
            point_labels: List of labels (1=foreground, 0=background)

        Returns:
            List of MaskCandidate objects, one per point prompt
        """
        if not hasattr(self, 'sam2_model') or self.sam2_model is None:
            # Fallback to automatic mode if predictor not available
            return self.generate_masks(image)

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Create predictor if not exists
            if not hasattr(self, 'predictor'):
                self.predictor = SAM2ImagePredictor(self.sam2_model)

            # Set the image
            self.predictor.set_image(image)

            all_masks = []

            # Batch processing optimization (inspired by EfficientViT-SAM 2024)
            batch_success = False
            if self.batch_prompts and len(points) > 1:
                try:
                    # Process all points in batch for ~2-3x speedup
                    point_coords_batch = np.array([[p[0], p[1]] for p in points])  # Shape: (N, 2)
                    point_labels_batch = np.array(point_labels)  # Shape: (N,)

                    # Use autocast for mixed precision
                    with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
                        # Batch predict - process all points at once
                        masks_batch, scores_batch, _ = self.predictor.predict(
                            point_coords=point_coords_batch,
                            point_labels=point_labels_batch,
                            multimask_output=True  # Get 3 masks per point
                        )

                    # Convert batch results to MaskCandidate objects
                    for i, (masks, scores) in enumerate(zip(masks_batch, scores_batch)):
                        for mask, score in zip(masks, scores):
                            candidate = MaskCandidate(
                                mask=mask.astype(np.uint8),
                                bbox=self._mask_to_bbox(mask),
                                area=int(mask.sum()),
                                predicted_iou=float(score),
                                stability_score=float(score),
                                point_coords=point_coords_batch[i:i+1]
                            )
                            all_masks.append(candidate)
                    batch_success = True
                except Exception as batch_error:
                    print(f"  WARNING: Batch processing failed ({batch_error}), falling back to sequential")
                    all_masks = []  # Reset

            if not batch_success:
                # Sequential processing (original method)
                for point, label in zip(points, point_labels):
                    point_coords = np.array([[point[0], point[1]]])  # Shape: (1, 2)
                    point_labels_arr = np.array([label])  # Shape: (1,)

                    # Use autocast for mixed precision
                    with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
                        # Predict masks
                        masks, scores, _ = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels_arr,
                            multimask_output=True  # Get 3 masks per point
                        )

                    # Convert to MaskCandidate objects
                    for mask, score in zip(masks, scores):
                        candidate = MaskCandidate(
                            mask=mask.astype(np.uint8),
                            bbox=self._mask_to_bbox(mask),
                            area=int(mask.sum()),
                            predicted_iou=float(score),
                            stability_score=float(score),  # Use IoU as stability
                            point_coords=point_coords
                        )
                        all_masks.append(candidate)

            # Sort by predicted IoU (best first)
            all_masks.sort(key=lambda x: x.predicted_iou, reverse=True)

            return all_masks

        except Exception as e:
            print(f"WARNING: Point prompting failed: {e}")
            # Fallback to automatic mode
            return self.generate_masks(image)

    def segment_automatic(self, image: np.ndarray) -> List[MaskCandidate]:
        """Alias for generate_masks for consistent API."""
        return self.generate_masks(image)

    def segment_with_points_hierarchical(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        point_labels: List[int],
        point_classes: Optional[List[int]] = None,
        output_scales: List[float] = [0.25, 0.5, 1.0]
    ) -> Dict[str, any]:
        """
        Segment image using point prompts and return hierarchical multi-scale masks.

        This method generates masks at multiple scales and organizes them into
        a pyramid structure for hierarchical refinement.

        Args:
            image: RGB image array (H, W, 3)
            points: List of (x, y) point coordinates
            point_labels: List of labels (1=foreground, 0=background)
            point_classes: Optional class IDs for each point
            output_scales: Scales for mask pyramid (e.g., [0.25, 0.5, 1.0])

        Returns:
            Dictionary with:
                - 'masks_pyramid': Dict[scale -> (N, H_s, W_s) mask tensor]
                - 'scores': (N,) predicted IoU scores
                - 'point_coords': (N, 2) point coordinates
                - 'point_classes': (N,) class assignments (if provided)
        """
        if not hasattr(self, 'sam2_model') or self.sam2_model is None:
            # Fallback: return single-scale masks
            masks = self.segment_with_points(image, points, point_labels)
            H, W = image.shape[:2]
            masks_single = np.array([m.mask for m in masks])

            return {
                'masks_pyramid': {1.0: torch.from_numpy(masks_single).float()},
                'scores': torch.tensor([m.predicted_iou for m in masks]),
                'point_coords': np.array(points),
                'point_classes': np.array(point_classes) if point_classes else None
            }

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Create predictor if not exists
            if not hasattr(self, 'predictor'):
                self.predictor = SAM2ImagePredictor(self.sam2_model)

            # Set the image
            self.predictor.set_image(image)

            H, W = image.shape[:2]

            # Prepare batch inputs
            point_coords_batch = np.array([[p[0], p[1]] for p in points])
            point_labels_batch = np.array(point_labels)

            # Use autocast for mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
                # Get masks with multimask_output (3 masks per point)
                masks_batch, scores_batch, _ = self.predictor.predict(
                    point_coords=point_coords_batch,
                    point_labels=point_labels_batch,
                    multimask_output=True  # Returns 3 masks: coarse, medium, fine
                )

            # Organize masks into hierarchical pyramid
            # SAM2 multimask_output returns 3 masks per point:
            # - Mask 0: Coarse (larger region)
            # - Mask 1: Medium
            # - Mask 2: Fine (smaller, more precise)

            N = len(points)
            masks_pyramid = {}

            for scale_idx, scale in enumerate(output_scales):
                # Compute target size for this scale
                H_s = int(H * scale)
                W_s = int(W * scale)

                # For each point, select appropriate mask based on scale
                scale_masks = []

                for point_idx in range(N):
                    # Get the 3 masks for this point
                    point_masks = masks_batch[point_idx]  # (3, H, W)
                    point_scores = scores_batch[point_idx]  # (3,)

                    # Select mask based on scale:
                    # - Coarse scale (0.25): Use largest mask (index 0)
                    # - Medium scale (0.5): Use medium mask (index 1)
                    # - Fine scale (1.0): Use finest mask (index 2)
                    if scale <= 0.3:
                        mask_idx = 0  # Coarse
                    elif scale <= 0.7:
                        mask_idx = 1  # Medium
                    else:
                        mask_idx = 2  # Fine

                    selected_mask = point_masks[mask_idx]

                    # Resize to target scale
                    mask_tensor = torch.from_numpy(selected_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    mask_resized = F.interpolate(
                        mask_tensor,
                        size=(H_s, W_s),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    scale_masks.append(mask_resized)

                # Stack into tensor (N, H_s, W_s)
                masks_pyramid[scale] = torch.stack(scale_masks).to(self.device)

            # Collect best scores (use highest score among 3 masks per point)
            best_scores = []
            for point_idx in range(N):
                best_score = scores_batch[point_idx].max()
                best_scores.append(best_score)

            result = {
                'masks_pyramid': masks_pyramid,
                'scores': torch.tensor(best_scores).to(self.device),
                'point_coords': point_coords_batch,
                'point_classes': np.array(point_classes) if point_classes else None
            }

            return result

        except Exception as e:
            print(f"WARNING: Hierarchical segmentation failed: {e}")
            # Fallback to single-scale
            masks = self.segment_with_points(image, points, point_labels)
            H, W = image.shape[:2]
            masks_single = np.array([m.mask for m in masks[:N]])  # Limit to N

            return {
                'masks_pyramid': {1.0: torch.from_numpy(masks_single).float()},
                'scores': torch.tensor([m.predicted_iou for m in masks[:N]]),
                'point_coords': np.array(points),
                'point_classes': np.array(point_classes) if point_classes else None
            }

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box (x, y, w, h)."""
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return (0, 0, 0, 0)

        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()

        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def _generate_mock_masks(self, image: np.ndarray) -> List[MaskCandidate]:
        """
        Generate mock masks for testing without SAM 2 installed.
        Uses simple superpixel segmentation as a placeholder.
        """
        from skimage.segmentation import slic

        # Generate superpixels
        segments = slic(image, n_segments=200, compactness=10, start_label=0)

        candidates = []
        for segment_id in np.unique(segments):
            mask = (segments == segment_id).astype(np.uint8)
            area = mask.sum()

            # Skip very small regions
            if area < 100:
                continue

            candidate = MaskCandidate(
                mask=mask,
                bbox=self._mask_to_bbox(mask),
                area=int(area),
                predicted_iou=0.9,  # Mock confidence
                stability_score=0.95,
                point_coords=None
            )
            candidates.append(candidate)

        candidates.sort(key=lambda x: x.area, reverse=True)
        return candidates

    def visualize_masks(
        self,
        image: np.ndarray,
        masks: List[MaskCandidate],
        max_display: int = 50,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize masks overlaid on the image.

        Args:
            image: Original RGB image
            masks: List of mask candidates
            max_display: Maximum number of masks to display
            alpha: Transparency for overlay

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Generate random colors for each mask
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

        for i, mask_candidate in enumerate(masks[:max_display]):
            mask = mask_candidate.mask
            color = colors[i]

            # Create colored overlay
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color

            # Blend with original
            vis = cv2.addWeighted(vis, 1, colored_mask, alpha, 0)

            # Draw bounding box
            x, y, w, h = mask_candidate.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color.tolist(), 2)

            # Add confidence scores
            text = f"IoU:{mask_candidate.predicted_iou:.2f}"
            cv2.putText(vis, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1)

        return vis

    def filter_by_size(
        self,
        masks: List[MaskCandidate],
        min_area: int = 1024,  # 32x32 pixels
        max_area: Optional[int] = None
    ) -> List[MaskCandidate]:
        """
        Filter masks by size constraints.

        Args:
            masks: List of mask candidates
            min_area: Minimum mask area in pixels
            max_area: Maximum mask area (None = no limit)

        Returns:
            Filtered list of masks
        """
        filtered = []
        for mask in masks:
            if mask.area < min_area:
                continue
            if max_area is not None and mask.area > max_area:
                continue
            filtered.append(mask)

        return filtered

    def compute_hierarchical_groups(
        self,
        masks: List[MaskCandidate]
    ) -> Dict[str, List[MaskCandidate]]:
        """
        Group masks by scale (fine, medium, coarse).

        Returns:
            Dictionary with 'fine', 'medium', 'coarse' keys
        """
        if not masks:
            return {"fine": [], "medium": [], "coarse": []}

        areas = np.array([m.area for m in masks])
        percentile_33 = np.percentile(areas, 33)
        percentile_67 = np.percentile(areas, 67)

        groups = {
            "fine": [m for m in masks if m.area < percentile_33],
            "medium": [m for m in masks if percentile_33 <= m.area < percentile_67],
            "coarse": [m for m in masks if m.area >= percentile_67]
        }

        return groups
