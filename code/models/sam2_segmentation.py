"""
SAM 2 Mask Generation Module

This module handles automatic mask generation using the Segment Anything Model 2 (SAM 2).
It provides high-quality, class-agnostic segmentation masks at multiple scales.

Reference: Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2


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
        model_type: str = "sam2_hiera_large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 0,
        crop_overlap_ratio: float = 512/1500,
    ):
        """
        Initialize SAM 2 mask generator.

        Args:
            model_type: SAM 2 checkpoint variant
            device: Device for computation
            points_per_side: Grid resolution for point prompts (32x32 = 1024 points)
            pred_iou_thresh: Minimum IoU confidence threshold
            stability_score_thresh: Minimum stability score threshold
            crop_n_layers: Number of crop layers for large images
            crop_overlap_ratio: Overlap ratio between crops
        """
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

        # Import SAM 2 - users need to install: pip install segment-anything-2
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            # Build model
            self.sam2_model = build_sam2(model_type, device=device)

            # Create automatic mask generator
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_overlap_ratio=crop_overlap_ratio,
            )

        except (ImportError, Exception) as e:
            # Fall back to mock if SAM 2 not installed or model/config missing
            if isinstance(e, ImportError):
                print("Warning: SAM 2 not installed. Using mock implementation.")
            else:
                print(f"Warning: SAM 2 initialization failed ({type(e).__name__}). Using mock implementation.")
            self.mask_generator = None

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
