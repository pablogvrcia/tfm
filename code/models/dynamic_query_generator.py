"""
Dynamic Multi-Scale Query Generator for MHQR (Phase 3A)

Implements adaptive query generation based on confidence scores and scene complexity.
Inspired by PSM-DIQ (2025) dynamic instance queries and Mask2Former architecture.

Key innovations:
1. Confidence-based adaptive query count (simple scenes: ~20, complex: ~150)
2. Multi-scale query pyramid (1/4, 1/2, 1, 2x base resolution)
3. Scene-adaptive threshold selection

Expected improvement: +5-8% mIoU from better small object detection

Reference:
- PSM-DIQ: "Panoptic Segmentation Method based on Dynamic Instance Queries", 2025
- Mask2Former: "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR 2022
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import label as connected_components
from scipy.ndimage import center_of_mass


class DynamicMultiScaleQueryGenerator:
    """
    Generates adaptive queries for SAM2 based on SCLIP confidence maps.

    Unlike static grid sampling (e.g., 64x64 = 4096 prompts), this module:
    - Analyzes confidence distribution to determine query count
    - Places queries at confident region centroids
    - Creates multi-scale queries for handling micro-to-macro objects
    """

    def __init__(
        self,
        base_resolution: Tuple[int, int] = (14, 14),  # SCLIP default
        scales: List[float] = [0.25, 0.5, 1.0, 2.0],  # Multi-scale pyramid
        min_queries: int = 10,
        max_queries: int = 200,
        confidence_thresholds: Dict[str, float] = None,
        use_adaptive_threshold: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize dynamic query generator.

        Args:
            base_resolution: Base feature map size (H, W)
            scales: Scale factors for multi-scale queries
            min_queries: Minimum queries per image (prevents empty scenes)
            max_queries: Maximum queries (computational limit)
            confidence_thresholds: Per-scale thresholds (auto-computed if None)
            use_adaptive_threshold: Adapt threshold to scene complexity
            device: Computation device
        """
        self.base_resolution = base_resolution
        self.scales = sorted(scales)
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.device = device
        self.use_adaptive_threshold = use_adaptive_threshold

        # Default confidence thresholds per scale
        # Lower threshold for finer scales (captures small objects)
        # Higher threshold for coarser scales (focuses on large confident regions)
        if confidence_thresholds is None:
            self.confidence_thresholds = {
                0.25: 0.7,  # Coarse: high confidence only
                0.5: 0.5,   # Medium-coarse
                1.0: 0.3,   # Base resolution
                2.0: 0.2    # Fine: lower threshold for small objects
            }
        else:
            self.confidence_thresholds = confidence_thresholds

    def generate_queries(
        self,
        confidence_maps: torch.Tensor,
        class_names: List[str],
        image_size: Tuple[int, int],
        return_metadata: bool = False
    ) -> Dict[str, any]:
        """
        Generate multi-scale queries from SCLIP confidence maps.

        Args:
            confidence_maps: (H, W, K) tensor of per-class confidence scores
            class_names: List of K class names
            image_size: Original image size (H_img, W_img)
            return_metadata: Return additional debugging info

        Returns:
            Dictionary with:
                - point_coords: (N, 2) array of query positions in image coordinates
                - point_labels: (N,) array of 1s (foreground points)
                - point_classes: (N,) array of class indices
                - scale_assignment: (N,) array of scale indices
                - (optional) metadata for debugging
        """
        H, W, K = confidence_maps.shape
        H_img, W_img = image_size

        all_queries = []
        all_class_ids = []
        all_scales = []
        metadata = {
            'queries_per_scale': {},
            'queries_per_class': {},
            'confidence_stats': {}
        }

        # Adaptive threshold adjustment based on global confidence distribution
        if self.use_adaptive_threshold:
            global_conf_mean = confidence_maps.max(dim=-1)[0].mean().item()
            threshold_adjustment = self._compute_threshold_adjustment(global_conf_mean)
        else:
            threshold_adjustment = 0.0

        # Generate queries at each scale
        for scale_idx, scale in enumerate(self.scales):
            scale_queries = []
            scale_class_ids = []

            # Adjust threshold for this scale
            base_threshold = self.confidence_thresholds[scale]
            adapted_threshold = max(0.1, min(0.9, base_threshold + threshold_adjustment))

            # Per-class query generation
            for class_idx in range(K):
                class_conf = confidence_maps[:, :, class_idx]

                # Extract confident regions
                confident_mask = class_conf > adapted_threshold

                if not confident_mask.any():
                    continue

                # Find connected components (separate instances)
                confident_mask_np = confident_mask.cpu().numpy()
                labeled_regions, num_regions = connected_components(confident_mask_np)

                # Generate query at centroid of each confident region
                for region_id in range(1, num_regions + 1):
                    region_mask = labeled_regions == region_id

                    # Skip tiny regions (likely noise)
                    region_size = region_mask.sum()
                    min_region_size = max(1, int(H * W * 0.001 * scale))  # Scale-adaptive
                    if region_size < min_region_size:
                        continue

                    # Compute centroid
                    centroid = center_of_mass(region_mask)
                    if np.isnan(centroid).any():
                        continue

                    y_feat, x_feat = centroid

                    # Convert to image coordinates
                    y_img = (y_feat / H) * H_img
                    x_img = (x_feat / W) * W_img

                    # Add jitter for multi-scale (denser sampling at finer scales)
                    if scale > 1.0:
                        jitter_radius = (2.0 / scale) * (H_img / H)
                        y_img += np.random.uniform(-jitter_radius, jitter_radius)
                        x_img += np.random.uniform(-jitter_radius, jitter_radius)

                    # Clip to image bounds
                    y_img = np.clip(y_img, 0, H_img - 1)
                    x_img = np.clip(x_img, 0, W_img - 1)

                    scale_queries.append([x_img, y_img])
                    scale_class_ids.append(class_idx)

            # Store scale-specific queries
            if scale_queries:
                all_queries.extend(scale_queries)
                all_class_ids.extend(scale_class_ids)
                all_scales.extend([scale_idx] * len(scale_queries))

                metadata['queries_per_scale'][scale] = len(scale_queries)

        # Enforce min/max query constraints
        num_queries = len(all_queries)

        if num_queries == 0:
            # Fallback: grid sampling at base resolution
            grid_queries = self._generate_fallback_grid(image_size)
            all_queries = grid_queries
            all_class_ids = [0] * len(grid_queries)  # Background class
            all_scales = [2] * len(grid_queries)  # Use base scale
            metadata['fallback_used'] = True
        elif num_queries > self.max_queries:
            # Subsample to max_queries (prioritize higher confidence)
            indices = self._subsample_queries(
                confidence_maps, all_queries, all_class_ids, self.max_queries
            )
            all_queries = [all_queries[i] for i in indices]
            all_class_ids = [all_class_ids[i] for i in indices]
            all_scales = [all_scales[i] for i in indices]
            metadata['subsampled'] = True
        elif num_queries < self.min_queries:
            # Add supplementary queries via low-threshold sampling
            supplement = self._generate_supplementary_queries(
                confidence_maps, image_size, self.min_queries - num_queries
            )
            all_queries.extend(supplement['queries'])
            all_class_ids.extend(supplement['class_ids'])
            all_scales.extend(supplement['scales'])
            metadata['supplemented'] = True

        # Convert to numpy arrays
        point_coords = np.array(all_queries, dtype=np.float32)
        point_labels = np.ones(len(all_queries), dtype=np.int32)  # All foreground
        point_classes = np.array(all_class_ids, dtype=np.int32)
        scale_assignment = np.array(all_scales, dtype=np.int32)

        # Compute metadata statistics
        for class_idx in range(K):
            count = (point_classes == class_idx).sum()
            if count > 0:
                metadata['queries_per_class'][class_names[class_idx]] = int(count)

        metadata['total_queries'] = len(point_coords)
        metadata['adaptive_threshold_adjustment'] = threshold_adjustment

        result = {
            'point_coords': point_coords,
            'point_labels': point_labels,
            'point_classes': point_classes,
            'scale_assignment': scale_assignment
        }

        if return_metadata:
            result['metadata'] = metadata

        return result

    def _compute_threshold_adjustment(self, global_conf_mean: float) -> float:
        """
        Adapt threshold based on global confidence distribution.

        Intuition:
        - High global confidence (>0.6): Scene is easy → raise threshold → fewer queries
        - Low global confidence (<0.4): Scene is hard → lower threshold → more queries

        Args:
            global_conf_mean: Mean of max confidence across all positions

        Returns:
            Threshold adjustment delta (range: -0.2 to +0.2)
        """
        if global_conf_mean > 0.6:
            # Easy scene: increase threshold (reduce queries)
            adjustment = (global_conf_mean - 0.6) * 0.5  # Max +0.2
        elif global_conf_mean < 0.4:
            # Hard scene: decrease threshold (increase queries)
            adjustment = (global_conf_mean - 0.4) * 0.5  # Max -0.2
        else:
            adjustment = 0.0

        return adjustment

    def _generate_fallback_grid(
        self,
        image_size: Tuple[int, int],
        grid_size: int = 16
    ) -> List[List[float]]:
        """
        Fallback grid sampling when no confident regions found.

        Args:
            image_size: (H, W)
            grid_size: Grid resolution (e.g., 16x16 = 256 queries)

        Returns:
            List of [x, y] query coordinates
        """
        H, W = image_size
        queries = []

        for i in range(grid_size):
            for j in range(grid_size):
                y = (i + 0.5) * (H / grid_size)
                x = (j + 0.5) * (W / grid_size)
                queries.append([x, y])

        return queries

    def _subsample_queries(
        self,
        confidence_maps: torch.Tensor,
        queries: List[List[float]],
        class_ids: List[int],
        target_count: int
    ) -> List[int]:
        """
        Subsample queries to target count, prioritizing high confidence.

        Args:
            confidence_maps: (H, W, K) confidence tensor
            queries: List of [x, y] coordinates
            class_ids: Corresponding class indices
            target_count: Desired number of queries

        Returns:
            Indices of selected queries
        """
        H, W, K = confidence_maps.shape
        H_img = queries[0][1]  # Infer from first query (rough estimate)

        # Compute confidence score for each query
        scores = []
        for (x, y), class_idx in zip(queries, class_ids):
            # Map back to feature coordinates
            x_feat = int((x / H_img) * W)
            y_feat = int((y / H_img) * H)
            x_feat = np.clip(x_feat, 0, W - 1)
            y_feat = np.clip(y_feat, 0, H - 1)

            conf = confidence_maps[y_feat, x_feat, class_idx].item()
            scores.append(conf)

        # Select top-k by confidence
        scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1][:target_count]

        return top_indices.tolist()

    def _generate_supplementary_queries(
        self,
        confidence_maps: torch.Tensor,
        image_size: Tuple[int, int],
        num_supplement: int
    ) -> Dict[str, List]:
        """
        Generate supplementary queries when below min_queries threshold.

        Uses lower confidence threshold to find additional candidates.

        Args:
            confidence_maps: (H, W, K)
            image_size: (H_img, W_img)
            num_supplement: Number of queries to add

        Returns:
            Dict with 'queries', 'class_ids', 'scales'
        """
        H, W, K = confidence_maps.shape
        H_img, W_img = image_size

        # Use very low threshold (0.1) to find any potential objects
        low_threshold = 0.1

        supplement_queries = []
        supplement_classes = []
        supplement_scales = []

        for class_idx in range(K):
            class_conf = confidence_maps[:, :, class_idx]
            candidates = (class_conf > low_threshold).nonzero(as_tuple=True)

            if len(candidates[0]) == 0:
                continue

            # Sample randomly from candidates
            num_candidates = len(candidates[0])
            sample_size = min(num_candidates, max(1, num_supplement // K))

            indices = np.random.choice(num_candidates, sample_size, replace=False)

            for idx in indices:
                y_feat = candidates[0][idx].item()
                x_feat = candidates[1][idx].item()

                y_img = (y_feat / H) * H_img
                x_img = (x_feat / W) * W_img

                supplement_queries.append([x_img, y_img])
                supplement_classes.append(class_idx)
                supplement_scales.append(2)  # Base scale

                if len(supplement_queries) >= num_supplement:
                    break

            if len(supplement_queries) >= num_supplement:
                break

        return {
            'queries': supplement_queries,
            'class_ids': supplement_classes,
            'scales': supplement_scales
        }
