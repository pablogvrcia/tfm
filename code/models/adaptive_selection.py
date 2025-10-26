"""
Adaptive Mask Selection Module

Solves the problem of determining how many masks to select based on
the semantic granularity of the query.

Examples:
- "car" → 1 mask (complete object)
- "tire" → 4 masks (parts of object)
- "mountain" → N masks (multiple instances)

Methods:
1. Hierarchical mask clustering (containment analysis)
2. Score gap detection (natural breaks in similarity scores)
3. Semantic granularity hints (plural detection, part-whole relationships)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import cv2
import re

from .sam2_segmentation import MaskCandidate
from .mask_alignment import ScoredMask


@dataclass
class MaskCluster:
    """Group of related masks representing the same semantic concept."""
    masks: List[ScoredMask]
    parent_mask: Optional[ScoredMask]  # Largest containing mask
    cluster_score: float
    semantic_type: str  # "singular", "parts", "instances"


class AdaptiveMaskSelector:
    """
    Automatically determines how many masks to select based on query semantics.

    Strategy:
    1. Analyze hierarchical mask relationships (containment/overlap)
    2. Detect score gaps to identify natural groupings
    3. Use linguistic cues from the text prompt
    4. Apply domain knowledge (common object parts, stuff vs. things)
    """

    def __init__(
        self,
        score_gap_threshold: float = 0.1,  # Minimum gap to separate clusters
        min_overlap_ratio: float = 0.8,    # Threshold for containment
        max_masks_per_query: int = 20,     # Safety limit
    ):
        """
        Initialize adaptive selector.

        Args:
            score_gap_threshold: Minimum score drop to consider a new cluster
            min_overlap_ratio: IoU threshold for parent-child relationships
            max_masks_per_query: Maximum masks to return
        """
        self.score_gap_threshold = score_gap_threshold
        self.min_overlap_ratio = min_overlap_ratio
        self.max_masks_per_query = max_masks_per_query

        # Known part-of relationships (could be expanded)
        self.part_of_relations = {
            "wheel", "tire", "door", "window", "headlight", "taillight",
            "leg", "arm", "head", "hand", "foot", "finger", "toe",
            "leaf", "branch", "petal", "stem",
            "button", "handle", "knob", "screen",
        }

        # Stuff categories (amorphous, can't be counted)
        # Note: "mountain" (singular) is stuff, but "mountains" (plural) is instances
        self.stuff_categories = {
            "sky", "grass", "water", "sand", "snow", "fog", "cloud",
            "road", "wall", "floor", "ceiling", "ground",
            "forest", "field", "ocean", "river", "sea",
        }

    def select_masks_adaptive(
        self,
        scored_masks: List[ScoredMask],
        text_prompt: str,
        image_shape: Tuple[int, int],
        max_masks: Optional[int] = None
    ) -> Tuple[List[ScoredMask], Dict]:
        """
        Adaptively select masks based on query semantics.

        Args:
            scored_masks: All scored masks (sorted by score)
            text_prompt: Original text query
            image_shape: (height, width) of image
            max_masks: Optional hard limit (overrides adaptive selection)

        Returns:
            - Selected masks
            - Debug info dictionary
        """
        if not scored_masks:
            return [], {"method": "empty", "reason": "No masks provided"}

        if max_masks is not None and max_masks <= 0:
            return [], {"method": "zero_limit", "reason": "max_masks <= 0"}

        # Step 1: Analyze text prompt for semantic cues
        semantic_type, prompt_info = self._analyze_prompt(text_prompt)

        # Step 2: Build hierarchical mask relationships
        hierarchy = self._build_mask_hierarchy(scored_masks, image_shape)

        # Step 3: Detect score gaps (natural clustering points)
        clusters = self._detect_score_clusters(scored_masks)

        # Step 4: Select based on semantic type
        if semantic_type == "singular":
            # Single complete object (e.g., "car", "person")
            selected = self._select_singular(scored_masks, hierarchy)

        elif semantic_type == "parts":
            # Multiple parts of an object (e.g., "tires", "windows")
            selected = self._select_parts(scored_masks, hierarchy, prompt_info)

        elif semantic_type == "stuff":
            # Stuff category (e.g., "sky", "grass")
            selected = self._select_stuff(scored_masks, hierarchy)

        elif semantic_type == "instances":
            # Multiple instances (e.g., "people", "cars", "mountains")
            selected = self._select_instances(scored_masks, clusters)

        else:  # "ambiguous"
            # Fall back to score gap detection
            selected = self._select_by_score_gap(scored_masks, clusters)

        # Apply hard limit if provided
        if max_masks is not None:
            selected = selected[:max_masks]

        # Safety limit
        selected = selected[:self.max_masks_per_query]

        # Debug info
        debug_info = {
            "method": semantic_type,
            "prompt_info": prompt_info,
            "num_clusters": len(clusters),
            "hierarchy_depth": max([h["level"] for h in hierarchy.values()]) if hierarchy else 0,
            "selected_count": len(selected),
            "score_range": (selected[0].final_score, selected[-1].final_score) if selected else (0, 0),
        }

        return selected, debug_info

    def _analyze_prompt(self, text_prompt: str) -> Tuple[str, Dict]:
        """
        Analyze text prompt to determine semantic type.

        Returns:
            - semantic_type: "singular", "parts", "stuff", "instances", "ambiguous"
            - prompt_info: Additional information dictionary
        """
        prompt_lower = text_prompt.lower().strip()
        words = prompt_lower.split()

        info = {
            "is_plural": False,
            "is_part": False,
            "is_stuff": False,
            "word_count": len(words),
        }

        # Check for stuff categories
        if any(stuff_word in prompt_lower for stuff_word in self.stuff_categories):
            info["is_stuff"] = True
            return "stuff", info

        # Check for parts
        if any(part_word in prompt_lower for part_word in self.part_of_relations):
            info["is_part"] = True

        # Check for plural forms (simple heuristic)
        last_word = words[-1] if words else ""
        if last_word.endswith('s') and not last_word.endswith('ss') and not last_word.endswith('us'):
            info["is_plural"] = True

        # Check for irregular plurals and always-plural words
        irregular_plurals = {"people", "children", "men", "women", "teeth", "feet"}
        always_plural = {"mountains"}  # Words ending in 's' that indicate plural
        if any(plural in words for plural in irregular_plurals | always_plural):
            info["is_plural"] = True

        # Check for explicit plural markers
        plural_markers = ["all", "multiple", "several", "many", "some", "both", "two", "three"]
        if any(marker in words for marker in plural_markers):
            info["is_plural"] = True

        # Decision logic
        if info["is_part"] and info["is_plural"]:
            return "parts", info
        elif info["is_part"]:
            return "parts", info  # Even singular parts might have multiple instances
        elif info["is_plural"]:
            return "instances", info
        elif len(words) == 1:
            return "singular", info
        elif "the" in words and len(words) <= 3:  # "the red car"
            return "singular", info
        else:
            return "ambiguous", info

    def _build_mask_hierarchy(
        self,
        masks: List[ScoredMask],
        image_shape: Tuple[int, int]
    ) -> Dict[int, Dict]:
        """
        Build hierarchical relationships between masks.

        Returns:
            Dictionary mapping mask index to hierarchy info:
            - "parent": Index of containing mask (or None)
            - "children": List of contained mask indices
            - "level": Depth in hierarchy (0 = top-level)
            - "overlap_scores": IoU with other masks
        """
        n_masks = len(masks)
        hierarchy = {}

        # Compute pairwise IoU and containment
        for i in range(n_masks):
            mask_i = masks[i].mask_candidate.mask
            area_i = mask_i.sum()

            hierarchy[i] = {
                "parent": None,
                "children": [],
                "level": 0,
                "overlap_scores": {},
            }

            for j in range(n_masks):
                if i == j:
                    continue

                mask_j = masks[j].mask_candidate.mask
                area_j = mask_j.sum()

                # Compute IoU
                intersection = (mask_i & mask_j).sum()
                union = (mask_i | mask_j).sum()
                iou = intersection / union if union > 0 else 0

                hierarchy[i]["overlap_scores"][j] = iou

                # Check if j contains i (i is inside j)
                if intersection / area_i >= self.min_overlap_ratio and area_j > area_i:
                    # j is a parent candidate
                    if hierarchy[i]["parent"] is None or area_j < masks[hierarchy[i]["parent"]].mask_candidate.mask.sum():
                        hierarchy[i]["parent"] = j

        # Compute levels (depth in hierarchy)
        def compute_level(idx, visited=None):
            if visited is None:
                visited = set()
            if idx in visited:
                return 0  # Cycle detection
            visited.add(idx)

            parent = hierarchy[idx]["parent"]
            if parent is None:
                return 0
            return 1 + compute_level(parent, visited)

        for i in range(n_masks):
            hierarchy[i]["level"] = compute_level(i)

        # Build children lists
        for i in range(n_masks):
            parent = hierarchy[i]["parent"]
            if parent is not None:
                hierarchy[parent]["children"].append(i)

        return hierarchy

    def _detect_score_clusters(
        self,
        masks: List[ScoredMask]
    ) -> List[List[int]]:
        """
        Detect natural clusters in similarity scores using gap detection.

        Returns:
            List of clusters, where each cluster is a list of mask indices
        """
        if len(masks) <= 1:
            return [[0]] if masks else []

        # Extract scores
        scores = np.array([m.final_score for m in masks])

        # Compute score gaps
        gaps = np.abs(np.diff(scores))

        # Find significant gaps
        mean_gap = gaps.mean()
        std_gap = gaps.std() if len(gaps) > 1 else 0
        threshold = max(self.score_gap_threshold, mean_gap + std_gap)

        # Split at gaps
        split_points = np.where(gaps >= threshold)[0] + 1

        # Create clusters
        clusters = []
        start = 0
        for split in split_points:
            if split > start:
                clusters.append(list(range(start, split)))
            start = split

        # Last cluster
        if start < len(masks):
            clusters.append(list(range(start, len(masks))))

        return clusters if clusters else [[0]]

    def _select_singular(
        self,
        masks: List[ScoredMask],
        hierarchy: Dict
    ) -> List[ScoredMask]:
        """
        Select for singular object queries (e.g., "car", "the person").

        Strategy: Return the highest-scoring mask.
        """
        return [masks[0]] if masks else []

    def _select_parts(
        self,
        masks: List[ScoredMask],
        hierarchy: Dict,
        prompt_info: Dict
    ) -> List[ScoredMask]:
        """
        Select for part queries (e.g., "wheels", "windows").

        Strategy:
        1. Find masks at the same hierarchy level (siblings)
        2. Filter by similar scores (within 1 std dev)
        3. Return all matching parts
        """
        if not masks:
            return []

        # Get scores
        scores = np.array([m.final_score for m in masks])
        top_score = scores[0]

        # Find masks with similar scores (likely same semantic level)
        score_threshold = top_score - 0.15  # Allow some variation

        selected_indices = []
        for i, mask in enumerate(masks):
            if mask.final_score >= score_threshold:
                selected_indices.append(i)

        # Filter to masks at similar hierarchy levels
        if hierarchy:
            levels = [hierarchy[i]["level"] for i in selected_indices]
            if levels:
                most_common_level = max(set(levels), key=levels.count)
                selected_indices = [i for i in selected_indices if hierarchy[i]["level"] == most_common_level]

        return [masks[i] for i in selected_indices[:10]]  # Cap at 10 parts

    def _select_stuff(
        self,
        masks: List[ScoredMask],
        hierarchy: Dict
    ) -> List[ScoredMask]:
        """
        Select for stuff categories (e.g., "sky", "grass").

        Strategy: Return the largest mask (stuff categories are typically large regions)
        """
        if not masks:
            return []

        # Find the largest mask among top scores
        top_masks = masks[:5]
        largest = max(top_masks, key=lambda m: m.mask_candidate.area)
        return [largest]

    def _select_instances(
        self,
        masks: List[ScoredMask],
        clusters: List[List[int]]
    ) -> List[ScoredMask]:
        """
        Select for instance queries (e.g., "people", "cars", "mountains").

        Strategy:
        1. Take the first cluster (highest scores)
        2. Filter out overlapping masks (keep non-overlapping instances)
        """
        if not masks:
            return []

        # Use first cluster
        first_cluster = clusters[0] if clusters else [0]
        candidate_masks = [masks[i] for i in first_cluster]

        # Filter out highly overlapping masks (keep diverse instances)
        selected = []
        for mask in candidate_masks:
            # Check overlap with already selected masks
            is_redundant = False
            for selected_mask in selected:
                iou = self._compute_iou(
                    mask.mask_candidate.mask,
                    selected_mask.mask_candidate.mask
                )
                if iou > 0.5:  # More than 50% overlap
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(mask)

        return selected[:15]  # Cap at 15 instances

    def _select_by_score_gap(
        self,
        masks: List[ScoredMask],
        clusters: List[List[int]]
    ) -> List[ScoredMask]:
        """
        Fallback: select based on score gap detection.

        Strategy: Return first cluster (before first significant score drop)
        """
        if not masks:
            return []

        if not clusters:
            return [masks[0]]

        # Return all masks in the first cluster
        first_cluster = clusters[0]
        return [masks[i] for i in first_cluster[:10]]

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return intersection / union if union > 0 else 0.0

    def visualize_hierarchy(
        self,
        image: np.ndarray,
        masks: List[ScoredMask],
        hierarchy: Dict,
        selected_indices: List[int]
    ) -> np.ndarray:
        """
        Visualize mask hierarchy and selection.

        Args:
            image: Original image
            masks: All masks
            hierarchy: Hierarchy info
            selected_indices: Indices of selected masks

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Draw all masks in gray
        for i, mask in enumerate(masks):
            if i not in selected_indices:
                vis[mask.mask_candidate.mask > 0] = [128, 128, 128]

        # Draw selected masks in color (by hierarchy level)
        level_colors = [
            (255, 0, 0),    # Red - level 0
            (0, 255, 0),    # Green - level 1
            (0, 0, 255),    # Blue - level 2
            (255, 255, 0),  # Yellow - level 3
        ]

        for i in selected_indices:
            mask = masks[i].mask_candidate.mask
            level = hierarchy.get(i, {}).get("level", 0)
            color = level_colors[min(level, len(level_colors) - 1)]

            overlay = vis.copy()
            overlay[mask > 0] = color
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

            # Draw bounding box
            x, y, w, h = masks[i].mask_candidate.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # Add text
            text = f"#{i} L{level} {masks[i].final_score:.2f}"
            cv2.putText(vis, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis
