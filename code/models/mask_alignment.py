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
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2
from PIL import Image as PILImage

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
        similarity_threshold: float = 0.02,  # Lowered to match actual CLIP similarity scores
        use_multiscale: bool = True,  # Multi-scale CLIP voting
        multiscale_weights: Tuple[float, float, float] = (0.2, 0.5, 0.3),  # Weights for [224, 336, 512]
        verbose: bool = True,  # Enable logging for debugging
    ):
        """
        Initialize mask-text aligner.

        Args:
            clip_extractor: CLIP feature extractor instance
            background_weight: Weight for background suppression (α in Eq 3.2)
            use_spatial_weighting: Whether to weight center pixels more
            similarity_threshold: Minimum score threshold
            use_multiscale: Whether to use multi-scale CLIP voting
            multiscale_weights: Weights for [224px, 336px, 512px] scales
            verbose: Enable logging for debugging
        """
        self.clip_extractor = clip_extractor
        self.background_weight = background_weight
        self.use_spatial_weighting = use_spatial_weighting
        self.similarity_threshold = similarity_threshold
        self.use_multiscale = use_multiscale
        self.multiscale_weights = multiscale_weights
        self.multiscale_sizes = [224, 336, 512]
        self.verbose = verbose

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
            # Method 1: Direct CLIP embedding of masked region (better!)
            # Extract mask region and get its CLIP embedding
            mask_img = self._extract_masked_region(image, mask_candidate.mask)
            if mask_img is not None:
                if self.use_multiscale:
                    # Multi-scale CLIP voting: extract at multiple resolutions
                    sim_score = self._compute_multiscale_similarity(
                        mask_img, text_embedding
                    )
                    bg_embedding = self.clip_extractor.extract_text_features(
                        ["background", "nothing", "empty space"],
                        use_prompt_ensemble=False
                    ).mean(dim=0)
                    bg_score = self._compute_multiscale_similarity_to_embedding(
                        mask_img, bg_embedding
                    )

                    # For confuser score, use single scale at 336px (current default)
                    mask_embedding, _ = self.clip_extractor.extract_image_features(mask_img)
                    confuser_score = self._compute_confuser_score(
                        mask_embedding, text_prompt
                    )
                else:
                    # Single-scale (original method)
                    mask_embedding, _ = self.clip_extractor.extract_image_features(mask_img)
                    # Cosine similarity with text
                    sim_score = float(F.cosine_similarity(
                        mask_embedding.unsqueeze(0),
                        text_embedding.unsqueeze(0)
                    ).item())

                    # Background score (similarity to background concepts)
                    bg_embedding = self.clip_extractor.extract_text_features(
                        ["background", "nothing", "empty space"],
                        use_prompt_ensemble=False
                    ).mean(dim=0)
                    bg_score = float(F.cosine_similarity(
                        mask_embedding.unsqueeze(0),
                        bg_embedding.unsqueeze(0)
                    ).item())

                    # FIX: Add negative/confuser scoring for common mismatches
                    # This helps distinguish tires from grilles, license plates, etc.
                    confuser_score = self._compute_confuser_score(
                        mask_embedding, text_prompt
                    )
            else:
                # Fallback to old method if mask extraction fails
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
                confuser_score = 0.0  # No confuser score for fallback method

            # Final score with background suppression and confuser penalty
            # Equation 3.2 (extended): S_final = S_sim - α*S_bg - β*S_confuser
            final_score = sim_score - self.background_weight * bg_score - 0.3 * confuser_score

            # Add size penalty: penalize masks that are too large (>20% of image)
            mask_area = mask_candidate.mask.sum()
            image_area = mask_candidate.mask.size
            area_ratio = mask_area / image_area
            if area_ratio > 0.2:  # More than 20% of image
                size_penalty = (area_ratio - 0.2) * 0.5  # Penalty scales with size
                final_score -= size_penalty

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

    def align_masks_with_multiple_texts(
        self,
        masks: List[MaskCandidate],
        text_prompts: List[str],
        image: np.ndarray,
        use_background_suppression: bool = True,
        score_threshold: float = 0.12,
        return_per_class: bool = True,
        prompt_denoising_threshold: float = 0.12,
        temperature: float = 100.0  # Score calibration (MaskCLIP/MasQCLIP)
    ) -> dict:
        """
        BATCH MODE: Score masks against multiple text prompts simultaneously.

        This is the core optimization for open-vocabulary segmentation:
        - Generate SAM masks once
        - Extract all text embeddings in batch
        - Score each mask against all classes simultaneously
        - Return masks grouped by class

        Args:
            masks: List of SAM 2 mask candidates
            text_prompts: List of text queries (e.g., all 171 COCO-Stuff classes)
            image: Original image (H, W, 3)
            use_background_suppression: Whether to apply background suppression
            score_threshold: Minimum score to keep a mask-class pair
            return_per_class: If True, return dict[class_name] = List[ScoredMask]
                             If False, return flat list of (class_name, ScoredMask) tuples
            prompt_denoising_threshold: Minimum max score to keep a class (default: 0.12, same as score_threshold)
            temperature: Score calibration factor (default: 100.0 from MaskCLIP/MasQCLIP)
                        Higher values increase score spread, making correct classes more confident

        Returns:
            Dictionary mapping class names to lists of scored masks:
            {
                "car": [ScoredMask1, ScoredMask2, ...],
                "road": [ScoredMask1, ...],
                ...
            }
        """
        if len(text_prompts) == 0:
            return {}

        # Extract all text embeddings in one batch
        text_embeddings = self.clip_extractor.extract_text_features(
            text_prompts,
            use_prompt_ensemble=True,
            normalize=True
        )  # Shape: (num_prompts, D)

        # Pre-compute background embedding if needed
        bg_embedding = None
        if use_background_suppression:
            bg_embedding = self.clip_extractor.extract_text_features(
                ["background", "nothing", "empty space"],
                use_prompt_ensemble=False
            ).mean(dim=0)  # Shape: (D,)

        # Store results: class_name -> List[ScoredMask]
        class_to_masks = {prompt: [] for prompt in text_prompts}

        # For each mask, compute similarity to all classes
        total_image_pixels = image.shape[0] * image.shape[1]

        for mask_candidate in masks:
            mask = mask_candidate.mask
            mask_size = mask.sum()

            # Skip tiny masks (< 0.1% of image)
            min_size = total_image_pixels * 0.001
            if mask_size < min_size:
                continue

            # CRITICAL FIX: Skip oversized masks (likely background)
            # MaskCLIP/MasQCLIP finding: Masks > 50% of image are usually sky/water/background
            # These score high because they include both object + large background region
            max_size = total_image_pixels * 0.5  # 50% threshold
            if mask_size > max_size:
                if self.verbose:
                    print(f"  [Mask Filtering] Skipping oversized mask: {mask_size/total_image_pixels*100:.1f}% of image")
                continue

            # Extract masked region
            masked_region = self._extract_masked_region(image, mask)
            if masked_region is None:
                continue

            # Multi-scale CLIP scoring (like MaskCLIP paper)
            if self.use_multiscale:
                # Compute multi-scale similarities to ALL classes at once
                scale_similarities_all = []
                pil_img = PILImage.fromarray(masked_region)

                for size in self.multiscale_sizes:
                    # Resize to target size
                    resized = pil_img.resize((size, size), PILImage.Resampling.BILINEAR)
                    resized_np = np.array(resized)

                    # Extract CLIP features at this scale
                    mask_embedding, _ = self.clip_extractor.extract_image_features(resized_np, normalize=True)

                    # Compute similarity to ALL classes at once (batched)
                    similarities = F.cosine_similarity(
                        mask_embedding.unsqueeze(0),  # (1, D)
                        text_embeddings,  # (num_prompts, D)
                        dim=1
                    )  # Shape: (num_prompts,)

                    scale_similarities_all.append(similarities.cpu().numpy())

                # Weighted average across scales for all classes
                multi_scale_scores = sum(
                    w * s for w, s in zip(self.multiscale_weights, scale_similarities_all)
                )  # Shape: (num_prompts,)

                # Background suppression (if enabled)
                if use_background_suppression and bg_embedding is not None:
                    # Use middle scale (336px) for background score
                    resized_336 = pil_img.resize((336, 336), PILImage.Resampling.BILINEAR)
                    resized_336_np = np.array(resized_336)
                    mask_embedding_336, _ = self.clip_extractor.extract_image_features(resized_336_np, normalize=True)

                    bg_score = F.cosine_similarity(
                        mask_embedding_336.unsqueeze(0),
                        bg_embedding.unsqueeze(0)
                    ).item()

                    multi_scale_scores -= self.background_weight * bg_score

                final_scores = multi_scale_scores
            else:
                # Single-scale scoring (faster but less accurate)
                mask_embedding, _ = self.clip_extractor.extract_image_features(masked_region, normalize=True)

                # Compute similarity to ALL classes at once
                similarities = F.cosine_similarity(
                    mask_embedding.unsqueeze(0),  # (1, D)
                    text_embeddings,  # (num_prompts, D)
                    dim=1
                )  # Shape: (num_prompts,)

                final_scores = similarities.cpu().numpy()

                # Background suppression (if enabled)
                if use_background_suppression and bg_embedding is not None:
                    bg_score = F.cosine_similarity(
                        mask_embedding.unsqueeze(0),
                        bg_embedding.unsqueeze(0)
                    ).item()
                    final_scores -= self.background_weight * bg_score

            # SCORE CALIBRATION (MaskCLIP/MasQCLIP paper):
            # Apply temperature scaling to expand compressed score ranges
            # This makes correct classes more confident relative to distractors
            # MaskCLIP uses temperature=100 before softmax (page 8)
            if temperature != 1.0:
                # Apply temperature scaling
                # Higher temperature = more spread in scores
                final_scores = final_scores * temperature

                # Apply softmax to convert to probabilities (optional but recommended)
                # This ensures scores sum to 1 and creates clearer distinction
                exp_scores = np.exp(final_scores - np.max(final_scores))  # Subtract max for numerical stability
                final_scores = exp_scores / exp_scores.sum()

            # MASK QUALITY PENALTY:
            # Penalize oversized masks (likely include too much background)
            # This addresses the airplane+sky, boat+water issue
            # Formula: quality_score = similarity_score * (1 - size_penalty)
            #
            # Aggressive penalty for masks > 15% of image:
            # - 15% image size: no penalty (1.0)
            # - 25% image size: small penalty (0.76)
            # - 35% image size: medium penalty (0.52)
            # - 45% image size: large penalty (0.28)
            # - 50%+ image size: very large penalty (0.15)
            mask_ratio = mask_size / total_image_pixels
            if mask_ratio > 0.15:
                # Apply aggressive penalty for large masks
                # penalty = (mask_ratio - 0.15) / 0.35  --> 0 at 15%, 1.0 at 50%
                size_penalty_factor = min((mask_ratio - 0.15) / 0.35, 1.0)
                # Reduce score by up to 85% for very large masks (more aggressive!)
                quality_multiplier = 1.0 - (0.85 * size_penalty_factor)
            else:
                quality_multiplier = 1.0  # No penalty for compact masks

            # Apply quality multiplier to all class scores
            final_scores = final_scores * quality_multiplier

            # Assign mask to classes based on scores
            # Strategy: Each mask can match multiple classes, but we assign to best match
            # and also keep other good matches (for multi-label scenarios)
            best_score_idx = np.argmax(final_scores)
            best_score = final_scores[best_score_idx]

            # Add to best matching class if above threshold
            if best_score >= score_threshold:
                best_class = text_prompts[best_score_idx]

                scored_mask = ScoredMask(
                    mask_candidate=mask_candidate,
                    similarity_score=best_score / quality_multiplier,  # Original similarity before penalty
                    background_score=bg_score if use_background_suppression else 0.0,
                    final_score=best_score,  # After quality penalty
                    rank=0  # Will be set later when sorting
                )

                class_to_masks[best_class].append(scored_mask)

        # Sort masks within each class by score (highest first) and assign ranks
        for class_name in class_to_masks:
            class_to_masks[class_name].sort(key=lambda x: x.final_score, reverse=True)
            for rank, scored_mask in enumerate(class_to_masks[class_name], start=1):
                scored_mask.rank = rank

        # PROMPT DENOISING (MaskCLIP paper p.8):
        # Remove classes where the max confidence across ALL masks is below threshold.
        # This filters out "distractor classes" that don't actually appear in the image.
        # Critical for open-vocabulary with many classes!
        denoised_classes = {}
        num_classes_before = len([c for c, m in class_to_masks.items() if len(m) > 0])

        # Collect max scores for all classes (for logging/debugging)
        class_max_scores = []
        for class_name, masks_list in class_to_masks.items():
            if len(masks_list) == 0:
                continue

            # Get max score for this class across all masks
            max_score = max(m.final_score for m in masks_list)
            class_max_scores.append((class_name, max_score))

        # ADAPTIVE DENOISING: Use both absolute threshold AND relative filtering
        # This handles cases where scores are very compressed (0.138-0.188)
        if len(class_max_scores) > 0:
            scores_array = np.array([s for _, s in class_max_scores])

            # Strategy 1: Absolute threshold (keep scores >= threshold)
            absolute_threshold = prompt_denoising_threshold

            # Strategy 2: Adaptive threshold (keep top 50% by default)
            # Use median as adaptive threshold, but ensure it's at least the absolute threshold
            adaptive_threshold = max(np.median(scores_array), absolute_threshold)

            # Use the adaptive threshold if we have many classes (open-vocab scenario)
            # This filters out bottom 50% of classes when scores are compressed
            effective_threshold = adaptive_threshold if len(class_max_scores) > 5 else absolute_threshold

            for class_name, max_score in class_max_scores:
                if max_score >= effective_threshold:
                    # Find the masks_list for this class
                    denoised_classes[class_name] = class_to_masks[class_name]
        else:
            denoised_classes = class_to_masks
            effective_threshold = prompt_denoising_threshold

        # Use denoised results instead of raw results
        num_classes_after = len(denoised_classes)
        num_filtered = num_classes_before - num_classes_after

        # Log the denoising effect (useful for debugging)
        if self.verbose:
            if len(class_max_scores) > 0:
                scores_array = [s for _, s in class_max_scores]
                print(f"  [Prompt Denoising] Absolute threshold: {prompt_denoising_threshold:.3f}")
                print(f"  [Prompt Denoising] Adaptive threshold: {effective_threshold:.3f} (median of {len(class_max_scores)} classes)")
                print(f"  [Prompt Denoising] Score range: {min(scores_array):.3f} - {max(scores_array):.3f}")
                print(f"  [Prompt Denoising] Score mean: {np.mean(scores_array):.3f} ± {np.std(scores_array):.3f}")
                print(f"  [Prompt Denoising] Classes before: {num_classes_before}, after: {num_classes_after} (filtered {num_filtered})")

                # Show top 5 kept classes
                kept_classes = sorted([(n, s) for n, s in class_max_scores if s >= effective_threshold],
                                     key=lambda x: x[1], reverse=True)
                if kept_classes:
                    print(f"  [Prompt Denoising] Top kept classes:")
                    for class_name, score in kept_classes[:5]:
                        print(f"    - {class_name}: {score:.3f}")

        class_to_masks = denoised_classes

        if return_per_class:
            return class_to_masks
        else:
            # Flatten to list of tuples
            flat_list = []
            for class_name, masks_list in class_to_masks.items():
                for scored_mask in masks_list:
                    flat_list.append((class_name, scored_mask))
            return flat_list

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

    def _extract_masked_region(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the masked region from the image for CLIP encoding.

        IMPROVED: Better handling for small objects like tires.

        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)

        Returns:
            Masked image region optimized for CLIP, or None if invalid
        """
        if mask.sum() == 0:
            return None

        # Get bounding box
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Crop to bounding box
        cropped_img = image[y_min:y_max+1, x_min:x_max+1].copy()
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

        # FIX 1: Set background to black (makes CLIP focus on foreground object)
        # Mean color can confuse CLIP by including context
        cropped_img[cropped_mask == 0] = [0, 0, 0]

        # FIX 2: Ensure minimum size for CLIP (needs ~224px to work well)
        # Small objects get poor features at tiny resolutions
        h, w = cropped_img.shape[:2]
        min_size = 224
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w) * 1.2  # 20% padding
            new_h, new_w = int(h * scale), int(w * scale)
            cropped_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return cropped_img

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

    def _compute_confuser_score(
        self,
        mask_embedding: torch.Tensor,
        text_prompt: str
    ) -> float:
        """
        Compute similarity to common confuser categories.

        This helps distinguish the target object from visually similar but
        semantically different objects (e.g., tire vs grille).

        Args:
            mask_embedding: CLIP embedding of masked region
            text_prompt: Original query text

        Returns:
            Maximum similarity to confuser categories (penalty score)
        """
        # Define confusers for common queries
        confuser_map = {
            "tire": ["grille", "license plate", "headlight", "road", "lane marking", "wheel rim"],
            "wheel": ["grille", "license plate", "headlight", "rim", "hubcap"],
            "window": ["door", "mirror", "windshield", "panel"],
            "door": ["window", "panel", "fender"],
            "person": ["mannequin", "statue", "reflection"],
            "car": ["truck", "van", "bus"],
        }

        # Find relevant confusers
        confusers = []
        prompt_lower = text_prompt.lower()
        for key, values in confuser_map.items():
            if key in prompt_lower:
                confusers = values
                break

        if not confusers:
            return 0.0  # No known confusers

        # Compute similarity to confusers
        confuser_embeddings = self.clip_extractor.extract_text_features(
            confusers,
            use_prompt_ensemble=False
        )

        # Max similarity to any confuser
        similarities = F.cosine_similarity(
            mask_embedding.unsqueeze(0),
            confuser_embeddings,
            dim=1
        )

        return float(similarities.max().item())

    def _compute_multiscale_similarity(
        self,
        mask_img: np.ndarray,
        text_embedding: torch.Tensor
    ) -> float:
        """
        Compute CLIP similarity at multiple scales and combine with weighted voting.

        Args:
            mask_img: Masked region image
            text_embedding: Text query embedding

        Returns:
            Weighted average similarity score
        """
        similarities = []
        pil_img = PILImage.fromarray(mask_img)

        for size in self.multiscale_sizes:
            # Resize to target size (maintaining aspect ratio with padding if needed)
            resized = pil_img.resize((size, size), PILImage.Resampling.BILINEAR)
            resized_np = np.array(resized)

            # Extract CLIP features at this scale
            embedding, _ = self.clip_extractor.extract_image_features(resized_np)

            # Compute similarity
            sim = float(F.cosine_similarity(
                embedding.unsqueeze(0),
                text_embedding.unsqueeze(0)
            ).item())

            similarities.append(sim)

        # Weighted average
        final_score = sum(w * s for w, s in zip(self.multiscale_weights, similarities))

        return final_score

    def _compute_multiscale_similarity_to_embedding(
        self,
        mask_img: np.ndarray,
        target_embedding: torch.Tensor
    ) -> float:
        """
        Compute CLIP similarity to a target embedding at multiple scales.

        Similar to _compute_multiscale_similarity but compares to an arbitrary embedding
        instead of text (used for background scoring).

        Args:
            mask_img: Masked region image
            target_embedding: Target embedding to compare against

        Returns:
            Weighted average similarity score
        """
        similarities = []
        pil_img = PILImage.fromarray(mask_img)

        for size in self.multiscale_sizes:
            # Resize to target size
            resized = pil_img.resize((size, size), PILImage.Resampling.BILINEAR)
            resized_np = np.array(resized)

            # Extract CLIP features at this scale
            embedding, _ = self.clip_extractor.extract_image_features(resized_np)

            # Compute similarity
            sim = float(F.cosine_similarity(
                embedding.unsqueeze(0),
                target_embedding.unsqueeze(0)
            ).item())

            similarities.append(sim)

        # Weighted average
        final_score = sum(w * s for w, s in zip(self.multiscale_weights, similarities))

        return final_score
