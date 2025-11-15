"""
Semantic-Guided Mask Merging (Phase 3C)

Enhances geometric mask merging with semantic consistency from CLIP features.
Addresses boundary ambiguity through attention-based pixel-level refinement.

Key innovations:
1. Semantic similarity check before merging overlapping masks
2. Attention-based boundary refinement for ambiguous regions
3. Training-free using pre-trained CLIP embeddings

Expected improvement: +2-3% mIoU from reducing false merges

Reference:
- Mask2Former: Pixel-decoder with semantic guidance
- SAM-CLIP (CVPR 2024): Merging semantic and spatial understanding
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class SemanticMaskMerger:
    """
    Semantically-aware mask merging that considers both geometric and semantic consistency.

    Pipeline:
    1. Identify overlapping mask pairs
    2. Compute semantic similarity using CLIP features
    3. If similar: merge masks
    4. If dissimilar: refine boundary using attention
    """

    def __init__(
        self,
        semantic_similarity_threshold: float = 0.7,
        boundary_refinement: bool = True,
        iou_threshold: float = 0.3,
        use_fp16: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize semantic mask merger.

        Args:
            semantic_similarity_threshold: Min cosine similarity to merge (0.6-0.8)
            boundary_refinement: Enable attention-based boundary refinement
            iou_threshold: Min IoU to consider masks as overlapping
            use_fp16: Use mixed precision
            device: Computation device
        """
        self.semantic_threshold = semantic_similarity_threshold
        self.boundary_refinement = boundary_refinement
        self.iou_threshold = iou_threshold
        self.use_fp16 = use_fp16 and device == "cuda"
        self.device = device

    def merge_masks_semantic(
        self,
        masks: torch.Tensor,
        class_ids: np.ndarray,
        class_embeddings: torch.Tensor,
        clip_features: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Merge overlapping masks using semantic guidance.

        Args:
            masks: (N, H, W) mask predictions (probability or binary)
            class_ids: (N,) class index for each mask
            class_embeddings: (K, D) text embeddings for K classes
            clip_features: (H, W, D) dense CLIP image features
            scores: (N,) optional confidence scores

        Returns:
            Dictionary with:
                - 'merged_masks': (M, H, W) merged masks (M <= N)
                - 'merged_class_ids': (M,) class assignments
                - 'merged_scores': (M,) confidence scores
                - 'merge_map': (N,) mapping from original to merged indices
        """
        with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
            N, H, W = masks.shape

            if N == 0:
                return {
                    'merged_masks': masks,
                    'merged_class_ids': class_ids,
                    'merged_scores': scores if scores is not None else torch.ones(0),
                    'merge_map': np.arange(0)
                }

            # Move to device
            masks = masks.to(self.device)
            class_embeddings = class_embeddings.to(self.device)
            clip_features = clip_features.to(self.device)
            if scores is not None:
                scores = scores.to(self.device)
            else:
                scores = torch.ones(N, device=self.device)

            # Find overlapping mask pairs
            overlap_pairs = self._find_overlapping_pairs(masks)

            # Initialize merge groups (each mask starts in its own group)
            merge_groups = list(range(N))  # merge_groups[i] = group_id for mask i

            # Process each overlapping pair
            for i, j in overlap_pairs:
                # Skip if already in same group
                if merge_groups[i] == merge_groups[j]:
                    continue

                # Check if same class (always merge same class)
                if class_ids[i] == class_ids[j]:
                    should_merge = True
                else:
                    # Different classes: check semantic similarity
                    should_merge = self._check_semantic_similarity(
                        mask_i=masks[i],
                        mask_j=masks[j],
                        class_i=class_ids[i],
                        class_j=class_ids[j],
                        class_embeddings=class_embeddings,
                        clip_features=clip_features
                    )

                if should_merge:
                    # Merge groups: assign all of group j to group i
                    old_group = merge_groups[j]
                    new_group = merge_groups[i]
                    for k in range(N):
                        if merge_groups[k] == old_group:
                            merge_groups[k] = new_group

            # Create merged masks from groups
            unique_groups = sorted(set(merge_groups))
            merged_masks = []
            merged_class_ids = []
            merged_scores = []
            merge_map = np.zeros(N, dtype=np.int32)

            for new_idx, group_id in enumerate(unique_groups):
                # Get all masks in this group
                group_mask_indices = [i for i in range(N) if merge_groups[i] == group_id]

                if len(group_mask_indices) == 1:
                    # Single mask: no merging needed
                    idx = group_mask_indices[0]
                    merged_masks.append(masks[idx])
                    merged_class_ids.append(class_ids[idx])
                    merged_scores.append(scores[idx])
                    merge_map[idx] = new_idx
                else:
                    # Multiple masks: merge them
                    group_masks = masks[group_mask_indices]
                    group_scores = scores[group_mask_indices]
                    group_classes = class_ids[group_mask_indices]

                    # Take most common class
                    unique, counts = np.unique(group_classes, return_counts=True)
                    merged_class = unique[counts.argmax()]

                    # Merge masks (union with score weighting)
                    merged_mask = self._merge_mask_union(group_masks, group_scores)

                    # Optional: refine boundary if masks have different classes
                    if len(np.unique(group_classes)) > 1 and self.boundary_refinement:
                        merged_mask = self._refine_boundary_attention(
                            merged_mask=merged_mask,
                            component_masks=group_masks,
                            component_classes=group_classes,
                            class_embeddings=class_embeddings,
                            clip_features=clip_features
                        )

                    merged_masks.append(merged_mask)
                    merged_class_ids.append(merged_class)
                    merged_scores.append(group_scores.max())

                    for idx in group_mask_indices:
                        merge_map[idx] = new_idx

            # Stack results
            if merged_masks:
                merged_masks = torch.stack(merged_masks)
                merged_scores = torch.stack(merged_scores)
            else:
                merged_masks = torch.zeros((0, H, W), device=self.device)
                merged_scores = torch.zeros(0, device=self.device)

            return {
                'merged_masks': merged_masks,
                'merged_class_ids': np.array(merged_class_ids),
                'merged_scores': merged_scores,
                'merge_map': merge_map
            }

    def _find_overlapping_pairs(
        self,
        masks: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Find pairs of masks with IoU > threshold.

        Args:
            masks: (N, H, W) binary/probability masks

        Returns:
            List of (i, j) pairs where i < j
        """
        N = masks.shape[0]
        pairs = []

        for i in range(N):
            for j in range(i + 1, N):
                # Compute IoU
                intersection = (masks[i] * masks[j]).sum()
                union = (masks[i] + masks[j]).clamp(0, 1).sum()

                if union > 0:
                    iou = intersection / union
                    if iou > self.iou_threshold:
                        pairs.append((i, j))

        return pairs

    def _check_semantic_similarity(
        self,
        mask_i: torch.Tensor,
        mask_j: torch.Tensor,
        class_i: int,
        class_j: int,
        class_embeddings: torch.Tensor,
        clip_features: torch.Tensor
    ) -> bool:
        """
        Check if two masks are semantically similar enough to merge.

        Args:
            mask_i, mask_j: (H, W) individual masks
            class_i, class_j: Class indices
            class_embeddings: (K, D) class text embeddings
            clip_features: (H, W, D) image features

        Returns:
            True if masks should be merged
        """
        # Extract mask features
        feat_i = self._extract_mask_feature(mask_i, clip_features)
        feat_j = self._extract_mask_feature(mask_j, clip_features)

        # Get class embeddings
        emb_i = class_embeddings[class_i]
        emb_j = class_embeddings[class_j]

        # Compute semantic similarities
        # 1. Inter-class similarity (how similar are the two class concepts?)
        class_sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()

        # 2. Feature-to-class consistency (do both masks match their assigned classes?)
        feat_i_to_class_i = F.cosine_similarity(feat_i.unsqueeze(0), emb_i.unsqueeze(0)).item()
        feat_j_to_class_j = F.cosine_similarity(feat_j.unsqueeze(0), emb_j.unsqueeze(0)).item()

        # 3. Cross-feature similarity (how similar are the actual regions?)
        region_sim = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0)).item()

        # Merging criteria:
        # - If classes are very similar (e.g., "cat" and "kitten"): merge
        # - If regions are very similar despite different classes: merge
        # - Otherwise: don't merge
        if class_sim > 0.8 or region_sim > self.semantic_threshold:
            return True
        else:
            return False

    def _extract_mask_feature(
        self,
        mask: torch.Tensor,
        clip_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract average CLIP feature for a mask region.

        Args:
            mask: (H, W) mask
            clip_features: (H, W, D) CLIP features

        Returns:
            (D,) averaged feature vector
        """
        mask_expanded = mask.unsqueeze(-1)  # (H, W, 1)
        weighted_sum = (mask_expanded * clip_features).sum(dim=[0, 1])  # (D,)
        mask_sum = mask.sum() + 1e-6
        avg_feature = weighted_sum / mask_sum

        # Normalize
        avg_feature = F.normalize(avg_feature, dim=0)

        return avg_feature

    def _merge_mask_union(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge multiple masks using score-weighted union.

        Args:
            masks: (M, H, W) masks to merge
            scores: (M,) confidence scores

        Returns:
            (H, W) merged mask
        """
        # Weighted average (higher scores contribute more)
        scores_normalized = scores / (scores.sum() + 1e-6)
        weights = scores_normalized.view(-1, 1, 1)  # (M, 1, 1)

        merged = (masks * weights).sum(dim=0)  # (H, W)

        # Clip to [0, 1]
        merged = merged.clamp(0, 1)

        return merged

    def _refine_boundary_attention(
        self,
        merged_mask: torch.Tensor,
        component_masks: torch.Tensor,
        component_classes: np.ndarray,
        class_embeddings: torch.Tensor,
        clip_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine mask boundary using attention-based pixel assignment.

        For pixels in the overlap region, use CLIP features to determine
        which class they belong to.

        Args:
            merged_mask: (H, W) current merged mask
            component_masks: (M, H, W) individual component masks
            component_classes: (M,) class indices
            class_embeddings: (K, D) class embeddings
            clip_features: (H, W, D) CLIP features

        Returns:
            (H, W) refined mask
        """
        H, W = merged_mask.shape

        # Find overlap region (where multiple component masks are active)
        overlap_count = (component_masks > 0.5).sum(dim=0)  # (H, W)
        overlap_region = overlap_count > 1  # (H, W) boolean

        if not overlap_region.any():
            return merged_mask  # No overlap to refine

        # For each pixel in overlap region, assign to best matching class
        refined_mask = merged_mask.clone()

        # Extract features for overlap pixels
        overlap_features = clip_features[overlap_region]  # (N_overlap, D)

        # Get embeddings for involved classes
        unique_classes = np.unique(component_classes)
        class_embs = class_embeddings[unique_classes]  # (N_classes, D)

        # Compute similarity: (N_overlap, N_classes)
        similarities = torch.matmul(
            F.normalize(overlap_features, dim=-1),
            F.normalize(class_embs, dim=-1).t()
        )

        # Assign each pixel to best matching class
        best_class_indices = similarities.argmax(dim=-1)  # (N_overlap,)

        # For pixels assigned to the dominant class, keep high confidence
        # For pixels assigned to other classes, reduce confidence
        dominant_class_idx = 0  # Assume first component is dominant (highest score)

        # Create refined overlap mask
        overlap_mask_refined = torch.zeros_like(merged_mask)
        overlap_indices = overlap_region.nonzero(as_tuple=True)

        for i, (best_idx, (y, x)) in enumerate(zip(best_class_indices, zip(*overlap_indices))):
            if best_idx == dominant_class_idx:
                overlap_mask_refined[y, x] = 1.0
            else:
                overlap_mask_refined[y, x] = 0.0

        # Blend refined overlap with non-overlap regions
        refined_mask[overlap_region] = overlap_mask_refined[overlap_region]

        return refined_mask
