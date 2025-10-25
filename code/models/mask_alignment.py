"""
Mask-Text Alignment Module

This module implements the alignment between SAM 2 masks and text prompts
using dense CLIP features, as described in Chapter 3.2.3.

Scoring formula (Equation 3.1):
    S_i = (1/|M_i|) * Σ_{p ∈ M_i} sim(f_p, e_t)

With refinements:
- Background suppression
- Spatial weighting
- Multi-scale aggregation
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2

from .sam2_segmentation import MaskCandidate
from .clip_features import CLIPFeatureExtractor


@dataclass
class ScoredMask:
    """Mask candidate with alignment scores."""
    mask_candidate: MaskCandidate
    similarity_score: float
    background_score: float
    final_score: float
    rank: int


class MaskTextAligner:
    """
    Aligns SAM 2 masks with text prompts using CLIP features.

    Implements the scoring strategy from Chapter 3:
    - Compute similarity for each mask region
    - Apply background suppression
    - Use spatial weighting to reduce boundary noise
    - Return top-K most relevant masks
    """

    def __init__(
        self,
        clip_extractor: CLIPFeatureExtractor,
        background_weight: float = 0.3,
        use_spatial_weighting: bool = True,
        similarity_threshold: float = 0.25,
    ):
        """
        Initialize mask-text aligner.

        Args:
            clip_extractor: CLIP feature extractor instance
            background_weight: Weight for background suppression (α in Eq 3.2)
            use_spatial_weighting: Whether to weight center pixels more
            similarity_threshold: Minimum score threshold
        """
        self.clip_extractor = clip_extractor
        self.background_weight = background_weight
        self.use_spatial_weighting = use_spatial_weighting
        self.similarity_threshold = similarity_threshold

    def align_masks_with_text(
        self,
        masks: List[MaskCandidate],
        text_prompt: str,
        image: np.ndarray,
        top_k: int = 5,
        return_similarity_maps: bool = False
    ) -> Tuple[List[ScoredMask], Optional[dict]]:
        """
        Score and rank masks according to text prompt.

        Args:
            masks: List of SAM 2 mask candidates
            text_prompt: User text query
            image: Original image (H, W, 3)
            top_k: Number of top masks to return
            return_similarity_maps: Whether to return visualization data

        Returns:
            - List of top-K scored masks
            - Optional: dictionary with similarity maps for visualization
        """
        h, w = image.shape[:2]

        # Extract CLIP features
        _, dense_features = self.clip_extractor.extract_image_features(image)
        text_embedding = self.clip_extractor.extract_text_features(
            text_prompt,
            use_prompt_ensemble=True
        )

        # Compute full similarity map
        similarity_map = self.clip_extractor.compute_similarity_map(
            dense_features,
            text_embedding,
            target_size=(h, w),
            aggregation="mean"
        )

        # Compute background suppression
        background_map = self.clip_extractor.compute_background_suppression(
            dense_features,
            target_size=(h, w)
        )

        # Score each mask
        scored_masks = []
        for mask_candidate in masks:
            # Compute scores
            sim_score = self._compute_mask_score(
                mask_candidate.mask,
                similarity_map,
                use_spatial_weighting=self.use_spatial_weighting
            )

            bg_score = self._compute_mask_score(
                mask_candidate.mask,
                background_map,
                use_spatial_weighting=False
            )

            # Final score with background suppression (Equation 3.2)
            final_score = sim_score - self.background_weight * bg_score

            scored_mask = ScoredMask(
                mask_candidate=mask_candidate,
                similarity_score=sim_score,
                background_score=bg_score,
                final_score=final_score,
                rank=0  # Will be assigned after sorting
            )

            scored_masks.append(scored_mask)

        # Filter by threshold
        scored_masks = [m for m in scored_masks if m.final_score > self.similarity_threshold]

        # Sort by final score (descending)
        scored_masks.sort(key=lambda x: x.final_score, reverse=True)

        # Assign ranks
        for rank, scored_mask in enumerate(scored_masks, start=1):
            scored_mask.rank = rank

        # Select top-K
        top_masks = scored_masks[:top_k]

        # Prepare visualization data if requested
        vis_data = None
        if return_similarity_maps:
            vis_data = {
                "similarity_map": similarity_map,
                "background_map": background_map,
                "text_prompt": text_prompt,
                "all_scored_masks": scored_masks
            }

        return top_masks, vis_data

    def _compute_mask_score(
        self,
        mask: np.ndarray,
        score_map: np.ndarray,
        use_spatial_weighting: bool = True
    ) -> float:
        """
        Compute average score within a mask region.

        Implements Equation 3.1 with optional spatial weighting.

        Args:
            mask: Binary mask (H, W)
            score_map: Similarity/score map (H, W)
            use_spatial_weighting: Apply spatial weighting

        Returns:
            Average score within mask
        """
        if mask.sum() == 0:
            return 0.0

        if use_spatial_weighting:
            # Create spatial weights (higher at center, lower at boundaries)
            weights = self._compute_spatial_weights(mask)
            weighted_scores = score_map[mask > 0] * weights[mask > 0]
            score = weighted_scores.sum() / weights[mask > 0].sum()
        else:
            # Simple average
            score = score_map[mask > 0].mean()

        return float(score)

    def _compute_spatial_weights(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute spatial weights that emphasize center pixels.

        Uses distance transform to give higher weight to pixels
        far from the mask boundary.

        Args:
            mask: Binary mask

        Returns:
            Weight map (same shape as mask)
        """
        # Distance transform (distance from each pixel to nearest boundary)
        dist_transform = cv2.distanceTransform(
            mask.astype(np.uint8),
            cv2.DIST_L2,
            5
        )

        # Normalize to [0.5, 1.0] (boundaries get 0.5, center gets 1.0)
        if dist_transform.max() > 0:
            weights = 0.5 + 0.5 * (dist_transform / dist_transform.max())
        else:
            weights = np.ones_like(dist_transform)

        return weights

    def visualize_scored_masks(
        self,
        image: np.ndarray,
        scored_masks: List[ScoredMask],
        max_display: int = 5
    ) -> np.ndarray:
        """
        Visualize top scored masks with scores.

        Args:
            image: Original image
            scored_masks: List of scored masks
            max_display: Maximum number to display

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Color gradient from red (high score) to yellow (low score)
        colors = [
            (255, 0, 0),    # Red
            (255, 128, 0),  # Orange
            (255, 255, 0),  # Yellow
            (128, 255, 0),  # Yellow-green
            (0, 255, 0),    # Green
        ]

        for i, scored_mask in enumerate(scored_masks[:max_display]):
            mask = scored_mask.mask_candidate.mask
            color = colors[min(i, len(colors) - 1)]

            # Create colored overlay
            overlay = vis.copy()
            overlay[mask > 0] = color

            # Blend
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

            # Draw bounding box
            x, y, w, h = scored_mask.mask_candidate.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # Add rank and score
            text = f"#{scored_mask.rank} Score:{scored_mask.final_score:.3f}"
            cv2.putText(
                vis, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            cv2.putText(
                vis, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return vis

    def create_comparison_grid(
        self,
        image: np.ndarray,
        scored_masks: List[ScoredMask],
        vis_data: dict,
        num_masks: int = 3
    ) -> np.ndarray:
        """
        Create a comparison grid showing:
        - Original image
        - Similarity map
        - Top masks overlaid
        - Individual top masks

        Args:
            image: Original image
            scored_masks: Scored masks
            vis_data: Visualization data from align_masks_with_text
            num_masks: Number of top masks to show

        Returns:
            Grid visualization
        """
        # Visualize similarity map
        sim_vis = self.clip_extractor.visualize_similarity_map(
            image,
            vis_data["similarity_map"],
            alpha=0.6
        )

        # Visualize scored masks
        masks_vis = self.visualize_scored_masks(image, scored_masks, max_display=num_masks)

        # Create individual mask visualizations
        individual_vis = []
        for i, scored_mask in enumerate(scored_masks[:num_masks]):
            mask_img = image.copy()
            mask = scored_mask.mask_candidate.mask

            # Apply mask
            mask_img[mask == 0] = mask_img[mask == 0] * 0.3

            # Add text
            text = f"Rank #{i+1}: {scored_mask.final_score:.3f}"
            cv2.putText(
                mask_img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            individual_vis.append(mask_img)

        # Arrange in grid
        # Row 1: Original | Similarity map | Top masks
        # Row 2: Individual top masks
        h, w = image.shape[:2]

        row1 = np.hstack([
            cv2.resize(image, (w, h)),
            cv2.resize(sim_vis, (w, h)),
            cv2.resize(masks_vis, (w, h))
        ])

        # Pad individual_vis to have at least 3 images
        while len(individual_vis) < 3:
            individual_vis.append(np.zeros_like(image))

        row2 = np.hstack([cv2.resize(img, (w, h)) for img in individual_vis[:3]])

        grid = np.vstack([row1, row2])

        return grid
