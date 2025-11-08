"""
SCLIP-based Semantic Segmentation

This module implements dense semantic segmentation using SCLIP's approach:
- Cross-layer Self-Attention (CSA) for better dense features
- Direct pixel-wise classification (no SAM masks required)
- PAMR (Pixel-Adaptive Memory Refinement) for boundary refinement
- Optional SAM integration for region-based predictions

Performance: SCLIP achieves 22.77% mIoU on COCO-Stuff164k

Enhancement: LoFTup Integration
    - Optional LoFTup feature upsampling for improved spatial resolution
    - Better feature quality → improved segmentation accuracy
    - Especially beneficial for small objects and fine boundaries
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from PIL import Image
import cv2

from models.sclip_features import SCLIPFeatureExtractor
from models.pamr import PAMR
from models.sam2_segmentation import SAM2MaskGenerator


class SCLIPSegmentor:
    """
    SCLIP-based semantic segmentor with optional SAM integration.

    Two modes:
    1. Dense mode (use_sam=False): Pure SCLIP dense prediction (22.77% mIoU on COCO-Stuff)
    2. Hybrid mode (use_sam=True): SCLIP features + SAM masks for region proposals
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",  # SCLIP paper uses ViT-B/16
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_sam: bool = False,
        use_pamr: bool = False,  # SCLIP paper: PAMR disabled by default (pamr_steps=0)
        pamr_steps: int = 10,
        pamr_dilations: Tuple[int, int] = (8, 16),  # SCLIP paper uses (8, 16)
        logit_scale: float = 40.0,
        prob_threshold: float = 0.0,
        slide_inference: bool = True,  # SCLIP uses sliding window by default
        slide_crop: int = 224,  # SCLIP paper: crop=224
        slide_stride: int = 112,  # SCLIP paper: stride=112
        use_loftup: bool = False,  # Enable LoFTup feature upsampling
        loftup_adaptive: bool = True,  # Use adaptive upsampling factor
        loftup_upsample_factor: float = 2.0,  # Fixed upsampling factor (if not adaptive)
        verbose: bool = True,
    ):
        """
        Initialize SCLIP segmentor.

        Args:
            model_name: CLIP model (ViT-L/14@336px recommended by SCLIP)
            device: Computation device
            use_sam: Whether to use SAM for mask proposals (hybrid mode)
            use_pamr: Use PAMR for refinement
            pamr_steps: Number of PAMR iterations
            pamr_dilations: Dilation rates for PAMR
            logit_scale: Temperature for softmax (40.0 in SCLIP paper)
            prob_threshold: Probability threshold for predictions
            slide_inference: Use sliding window inference (slower but better)
            slide_crop: Crop size for sliding window
            slide_stride: Stride for sliding window
            use_loftup: Enable LoFTup feature upsampling (improves feature quality)
            loftup_adaptive: Use adaptive upsampling (adjusts based on feature size)
            loftup_upsample_factor: Fixed upsampling factor (if not adaptive)
            verbose: Print progress
        """
        self.device = device
        self.use_sam = use_sam
        self.use_pamr = use_pamr
        self.logit_scale = logit_scale
        self.prob_threshold = prob_threshold
        self.slide_inference = slide_inference
        self.slide_crop = slide_crop
        self.slide_stride = slide_stride
        self.use_loftup = use_loftup
        self.verbose = verbose

        # Text feature cache to avoid recomputing for same classes
        self._text_feature_cache = {}

        if verbose:
            print("[SCLIP Segmentor] Initializing...")
            print(f"  Mode: {'Hybrid (SAM + SCLIP)' if use_sam else 'Dense (SCLIP only)'}")
            print(f"  PAMR: {'Enabled' if use_pamr else 'Disabled'}")
            print(f"  Slide inference: {'Enabled' if slide_inference else 'Disabled'}")
            print(f"  LoFTup: {'Enabled' if use_loftup else 'Disabled'}")

        # Initialize SCLIP feature extractor
        self.clip_extractor = SCLIPFeatureExtractor(
            model_name=model_name,
            device=device,
            use_loftup=use_loftup,
            loftup_adaptive=loftup_adaptive,
            loftup_upsample_factor=loftup_upsample_factor,
            verbose=verbose
        )

        # Initialize PAMR if enabled
        if use_pamr and pamr_steps > 0:
            self.pamr = PAMR(num_iter=pamr_steps, dilations=list(pamr_dilations)).to(device)
            if verbose:
                print(f"  PAMR: {pamr_steps} steps, dilations={pamr_dilations}")
        else:
            self.pamr = None

        # Initialize SAM if hybrid mode
        if use_sam:
            self.sam_generator = SAM2MaskGenerator(device=device)
            if verbose:
                print("  SAM generator initialized")
        else:
            self.sam_generator = None

        if verbose:
            print("[SCLIP Segmentor] Ready!\n")

    def _get_text_features(self, class_names: List[str]) -> torch.Tensor:
        """
        Get text features with caching to avoid recomputing.

        Args:
            class_names: List of class names

        Returns:
            Text features tensor (num_classes, D)
        """
        # Create cache key from class names
        cache_key = tuple(class_names)

        # Return cached features if available
        if cache_key in self._text_feature_cache:
            return self._text_feature_cache[cache_key]

        # Compute text features
        text_features = self.clip_extractor.extract_text_features(
            class_names,
            use_prompt_ensemble=True,
            normalize=True
        )

        # Cache for future use
        self._text_feature_cache[cache_key] = text_features

        if self.verbose:
            print(f"[Cache] Encoded {len(class_names)} text prompts (cached for reuse)")

        return text_features

    @torch.no_grad()
    def predict_dense(
        self,
        image: np.ndarray,
        class_names: List[str],
        return_logits: bool = False
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """
        Dense semantic segmentation (SCLIP's original approach).

        This is the pure SCLIP method that achieves 22.77% mIoU.

        Args:
            image: Input image (H, W, 3) RGB numpy array
            class_names: List of class names to predict
            return_logits: Return raw logits before argmax

        Returns:
            - Segmentation mask (H, W) with class indices
            - Optional logits (num_classes, H, W)
        """
        # Store ORIGINAL resolution for final output
        orig_H, orig_W = image.shape[:2]

        # CRITICAL: SCLIP resizes images to 2048 on longer side BEFORE sliding window!
        # This is specified in SCLIP's config: Resize(scale=(2048, 448), keep_ratio=True)
        # Resize image to match SCLIP's preprocessing
        h, w = image.shape[:2]
        if max(h, w) != 2048:
            # Resize so longer side is 2048, keeping aspect ratio
            scale = 2048 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
            image = np.array(pil_img)
            if self.verbose:
                print(f"[SCLIP] Resized image from {w}x{h} to {new_w}x{new_h} (SCLIP standard)")

        # Current resolution after preprocessing
        H, W = image.shape[:2]

        if self.slide_inference:
            # Sliding window inference (slower but better for large images)
            logits = self._forward_slide(image, class_names)
        else:
            # Single forward pass
            logits = self._forward_single(image, class_names)

        # Apply PAMR refinement if enabled
        if self.pamr is not None:
            # PAMR needs image tensor: (1, 3, H, W)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            # Resize image to match logits
            if image_tensor.shape[-2:] != logits.shape[-2:]:
                image_tensor = F.interpolate(
                    image_tensor,
                    size=logits.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

            # Convert logits to same dtype as PAMR (float32)
            logits_dtype = logits.dtype
            logits = logits.float()

            # Apply PAMR: logits shape (1, num_classes, H_feat, W_feat)
            logits = self.pamr(image_tensor, logits.unsqueeze(0)).squeeze(0)

            # Convert back to original dtype
            logits = logits.to(logits_dtype)

        # Scale logits (temperature)
        logits = logits * self.logit_scale

        # Logits are now at resized resolution (H, W = 1363x2048)
        # Interpolate back to ORIGINAL resolution for evaluation
        if logits.shape[-2:] != (orig_H, orig_W):
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Convert to probabilities
        probs = F.softmax(logits, dim=0)

        # Get predictions
        pred_mask = probs.argmax(dim=0).cpu().numpy()

        # Apply probability threshold
        if self.prob_threshold > 0:
            max_probs = probs.max(dim=0)[0].cpu().numpy()
            pred_mask[max_probs < self.prob_threshold] = 0  # Background

        if return_logits:
            return pred_mask, logits
        else:
            return pred_mask, None

    def _extract_hierarchical_prompts(
        self,
        dense_pred: np.ndarray,
        logits: torch.Tensor,
        class_idx: int,
        num_positive: int = 10,
        num_negative: int = 5,
        min_distance: int = 20,
        confidence_threshold: float = 0.7
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Extract hierarchical prompts using SCLIP confidence scores.

        Strategy:
        - High-confidence regions (>threshold) → positive prompts (label=1)
        - Low-confidence regions from OTHER classes → negative prompts (label=0)
        - Helps SAM2 distinguish boundaries and suppress false positives

        Args:
            dense_pred: Dense prediction mask (H, W) with class indices
            logits: SCLIP logits (num_classes, H, W) with confidence scores
            class_idx: Target class index
            num_positive: Number of positive prompts to extract
            num_negative: Number of negative prompts to extract
            min_distance: Minimum distance between points
            confidence_threshold: Confidence threshold for positive prompts

        Returns:
            Tuple of (points, labels) where labels: 1=foreground, 0=background
        """
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=0)  # (num_classes, H, W)
        target_prob = probs[class_idx].cpu().numpy()  # (H, W)

        # Get binary mask for target class
        class_mask = (dense_pred == class_idx).astype(np.uint8)

        if class_mask.sum() == 0:
            return [], []

        points = []
        labels = []

        # 1. Extract POSITIVE prompts from high-confidence regions
        high_conf_mask = (target_prob > confidence_threshold) & (class_mask > 0)

        if high_conf_mask.sum() > 0:
            # Find connected components in high-confidence regions
            num_labels, label_map = cv2.connectedComponents(high_conf_mask.astype(np.uint8))

            # Extract centroid from each component
            for label_id in range(1, num_labels):
                component_mask = (label_map == label_id)

                # Skip very small components
                if component_mask.sum() < 100:
                    continue

                # Find weighted centroid using confidence scores
                y_coords, x_coords = np.where(component_mask)
                weights = target_prob[y_coords, x_coords]

                # Weighted average for centroid
                centroid_x = int(np.average(x_coords, weights=weights))
                centroid_y = int(np.average(y_coords, weights=weights))

                # Check minimum distance
                too_close = False
                for existing_x, existing_y, _ in points:
                    dist = np.sqrt((centroid_x - existing_x)**2 + (centroid_y - existing_y)**2)
                    if dist < min_distance:
                        too_close = True
                        break

                if not too_close:
                    points.append((centroid_x, centroid_y, target_prob[centroid_y, centroid_x]))
                    labels.append(1)  # Positive prompt

                    if len([l for l in labels if l == 1]) >= num_positive:
                        break

        # If still need more positive points, sample from medium-confidence regions
        if len([l for l in labels if l == 1]) < num_positive // 2 and class_mask.sum() > 0:
            medium_conf_mask = (target_prob > 0.5) & (target_prob <= confidence_threshold) & (class_mask > 0)

            if medium_conf_mask.sum() > 0:
                y_coords, x_coords = np.where(medium_conf_mask)
                weights = target_prob[y_coords, x_coords]

                # Sample points weighted by confidence
                num_samples = min(num_positive - len([l for l in labels if l == 1]), len(y_coords))
                if num_samples > 0:
                    probs_normalized = weights / weights.sum()
                    indices = np.random.choice(len(y_coords), num_samples, replace=False, p=probs_normalized)

                    for idx in indices:
                        x, y = int(x_coords[idx]), int(y_coords[idx])
                        points.append((x, y, target_prob[y, x]))
                        labels.append(1)

        # 2. Extract NEGATIVE prompts from competing classes
        # Find pixels where OTHER classes have higher confidence
        max_prob = probs.max(dim=0)[0].cpu().numpy()
        max_class = probs.argmax(dim=0).cpu().numpy()

        # Regions where another class is predicted with high confidence
        competing_mask = (max_class != class_idx) & (max_prob > 0.6) & (max_class != 0)  # Exclude background

        if competing_mask.sum() > 0:
            # Find regions near the target class (boundary confusion)
            kernel = np.ones((15, 15), np.uint8)
            dilated_class_mask = cv2.dilate(class_mask, kernel, iterations=1)

            # Negative prompts: near target class but assigned to other class
            negative_candidate_mask = competing_mask & (dilated_class_mask > 0)

            if negative_candidate_mask.sum() > 0:
                y_coords, x_coords = np.where(negative_candidate_mask)

                # Sample negative points
                num_samples = min(num_negative, len(y_coords))
                if num_samples > 0:
                    indices = np.random.choice(len(y_coords), num_samples, replace=False)

                    for idx in indices:
                        x, y = int(x_coords[idx]), int(y_coords[idx])

                        # Check minimum distance from existing points
                        too_close = False
                        for existing_x, existing_y, _ in points:
                            dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                            if dist < min_distance:
                                too_close = True
                                break

                        if not too_close:
                            points.append((x, y, 0.0))  # Confidence=0 for negative
                            labels.append(0)  # Negative prompt

        # Sort by confidence (positive prompts first, then by confidence)
        sorted_data = sorted(zip(points, labels), key=lambda x: (x[1], x[0][2]), reverse=True)

        if sorted_data:
            points_sorted, labels_sorted = zip(*sorted_data)
            # Remove confidence score from points
            points_sorted = [(x, y) for x, y, _ in points_sorted]
            return list(points_sorted), list(labels_sorted)

        return [], []

    def _extract_prompt_points(
        self,
        dense_pred: np.ndarray,
        class_idx: int,
        num_points: int = 16,
        min_distance: int = 20
    ) -> List[Tuple[int, int]]:
        """
        Extract point prompts from dense SCLIP prediction for a specific class.

        Uses connected components analysis to find representative points.

        Args:
            dense_pred: Dense prediction mask (H, W) with class indices
            class_idx: Target class index
            num_points: Target number of points to extract
            min_distance: Minimum distance between points

        Returns:
            List of (x, y) coordinates for SAM2 prompting
        """
        # Get binary mask for target class
        class_mask = (dense_pred == class_idx).astype(np.uint8)

        if class_mask.sum() == 0:
            return []

        # Find connected components
        num_labels, labels = cv2.connectedComponents(class_mask)

        points = []

        # For each component, find centroid
        for label_id in range(1, num_labels):  # Skip background (0)
            component_mask = (labels == label_id)

            # Skip very small components
            if component_mask.sum() < 100:
                continue

            # Find centroid using median for robustness
            y_coords, x_coords = np.where(component_mask)
            centroid_y = int(np.median(y_coords))
            centroid_x = int(np.median(x_coords))

            # Check minimum distance from existing points
            too_close = False
            for existing_x, existing_y in points:
                dist = np.sqrt((centroid_x - existing_x)**2 + (centroid_y - existing_y)**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                points.append((centroid_x, centroid_y))

        # If we have too few points, sample from high-confidence interior regions
        if len(points) < num_points // 2 and class_mask.sum() > 0:
            # Erode mask to get high-confidence interior points
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(class_mask, kernel, iterations=2)

            if eroded.sum() > 0:
                # Sample additional points from eroded mask
                y_coords, x_coords = np.where(eroded > 0)
                indices = np.random.choice(
                    len(y_coords),
                    min(num_points - len(points), len(y_coords)),
                    replace=False
                )

                for idx in indices:
                    x, y = int(x_coords[idx]), int(y_coords[idx])
                    points.append((x, y))

        return points[:num_points]

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _non_maximum_suppression(
        self,
        masks: List,
        iou_threshold: float = 0.7
    ) -> List:
        """
        Apply Non-Maximum Suppression to remove overlapping masks.

        Keeps masks with highest predicted_iou scores, removes heavily overlapping duplicates.

        Args:
            masks: List of MaskCandidate objects
            iou_threshold: IoU threshold for suppression (0.7 = remove if >70% overlap)

        Returns:
            Filtered list of masks after NMS
        """
        if not masks:
            return []

        # Sort by predicted_iou (best first)
        masks = sorted(masks, key=lambda x: x.predicted_iou, reverse=True)

        keep = []
        suppressed = [False] * len(masks)

        for i, mask_i in enumerate(masks):
            if suppressed[i]:
                continue

            keep.append(mask_i)

            # Suppress overlapping masks
            for j in range(i + 1, len(masks)):
                if suppressed[j]:
                    continue

                iou = self._compute_mask_iou(mask_i.mask > 0, masks[j].mask > 0)
                if iou > iou_threshold:
                    suppressed[j] = True

        return keep

    @torch.no_grad()
    def predict_with_sam(
        self,
        image: np.ndarray,
        class_names: List[str],
        use_prompted_sam: bool = True,
        use_hierarchical_prompts: bool = False,  # Disabled: standard prompting for best quality
        min_coverage: float = 0.6,
        min_iou_score: float = 0.0,  # No filtering by default (best quality)
        nms_iou_threshold: float = 1.0,  # No NMS by default (best coverage)
        use_best_mask_only: bool = False  # Use all masks for best coverage
    ) -> np.ndarray:
        """
        Hybrid: Dense SCLIP + SAM refinement.

        Two modes:
        1. Prompted SAM (default, faster): Extract points from SCLIP predictions
           and prompt SAM2 at those locations for targeted refinement
        2. Automatic SAM (legacy): Generate all masks and refine globally

        Strategy (prompted mode):
        1. Get dense SCLIP predictions first
        2. For each detected class, extract representative points
        3. Prompt SAM2 at those points for targeted mask generation
        4. Apply quality filtering and NMS to select best masks
        5. Use majority voting to assign class labels to refined masks

        Args:
            image: Input image (H, W, 3)
            class_names: List of class names
            use_prompted_sam: If True, use prompted segmentation (faster, more targeted)
            min_coverage: Minimum coverage threshold for majority voting (default: 0.6)
            min_iou_score: Minimum SAM2 predicted_iou score to keep mask (default: 0.70)
            nms_iou_threshold: IoU threshold for Non-Maximum Suppression (default: 0.8)
            use_best_mask_only: If True, use only best mask per point (default: True)

        Returns:
            Segmentation mask (H, W) with class indices
        """
        H, W = image.shape[:2]

        # Step 1: Get dense SCLIP predictions with confidence scores
        dense_pred, logits = self.predict_dense(image, class_names, return_logits=True)

        if use_prompted_sam:
            # NEW: Prompted SAM2 refinement (more efficient)
            if self.verbose:
                prompt_type = "hierarchical (confidence-based)" if use_hierarchical_prompts else "standard"
                print(f"[SAM Refinement] Using prompted SAM2 segmentation ({prompt_type})...")

            # Collect all prompt points from detected classes
            all_points = []
            all_labels = []
            point_to_class = {}  # Map point index to class index

            num_positive_total = 0
            num_negative_total = 0

            for class_idx in range(len(class_names)):
                if class_idx == 0:  # Skip background
                    continue

                if use_hierarchical_prompts:
                    # Extract hierarchical prompts with positive and negative points
                    class_points, class_labels = self._extract_hierarchical_prompts(
                        dense_pred,
                        logits,
                        class_idx,
                        num_positive=10,
                        num_negative=5,
                        min_distance=20,
                        confidence_threshold=0.7
                    )
                else:
                    # Extract simple positive points only
                    class_points = self._extract_prompt_points(
                        dense_pred,
                        class_idx,
                        num_points=16
                    )
                    class_labels = [1] * len(class_points)  # All foreground

                if class_points:
                    # Count positive and negative prompts
                    num_pos = sum(1 for l in class_labels if l == 1)
                    num_neg = sum(1 for l in class_labels if l == 0)
                    num_positive_total += num_pos
                    num_negative_total += num_neg

                    # Store mapping for later
                    for pt, label in zip(class_points, class_labels):
                        point_to_class[len(all_points)] = class_idx
                        all_points.append(pt)
                        all_labels.append(label)

                    if self.verbose:
                        if use_hierarchical_prompts:
                            print(f"  Class '{class_names[class_idx]}': {num_pos} positive, {num_neg} negative points")
                        else:
                            print(f"  Class '{class_names[class_idx]}': {len(class_points)} points")

            if not all_points:
                if self.verbose:
                    print("  No valid points found, returning dense prediction")
                return dense_pred

            if self.verbose:
                if use_hierarchical_prompts:
                    print(f"  Total: {num_positive_total} positive + {num_negative_total} negative = {len(all_points)} prompts across {len(class_names)-1} classes")
                else:
                    print(f"  Total: {len(all_points)} prompt points across {len(class_names)-1} classes")

            # Generate SAM masks with prompts
            sam_masks = self.sam_generator.segment_with_points(image, all_points, all_labels)

            if self.verbose:
                print(f"  Generated {len(sam_masks)} mask candidates from {len(all_points)} points")

        else:
            # OLD: Automatic SAM2 mask generation
            if self.verbose:
                print("[SAM Refinement] Using automatic SAM2 mask generation...")
            sam_masks = self.sam_generator.generate_masks(image)

        # Step 2: Quality filtering - keep high-quality masks
        filtered_masks = []
        num_points = len(all_points) if use_prompted_sam else len(sam_masks)

        if use_best_mask_only and use_prompted_sam:
            # Select best mask per point only
            for i in range(0, len(sam_masks), 3):
                point_masks = sam_masks[i:i+3]
                if point_masks:
                    # Select mask with highest predicted_iou
                    best = max(point_masks, key=lambda m: m.predicted_iou)
                    if best.predicted_iou >= min_iou_score:
                        filtered_masks.append(best)

            if self.verbose:
                print(f"  Selected best mask per point: {len(filtered_masks)}/{num_points} (IoU ≥ {min_iou_score})")
        else:
            # Keep top 2 masks per point (allows for multi-scale detection)
            if use_prompted_sam:
                for i in range(0, len(sam_masks), 3):
                    point_masks = sam_masks[i:i+3]
                    # Sort by IoU and take top 2
                    point_masks_sorted = sorted(point_masks, key=lambda m: m.predicted_iou, reverse=True)
                    for mask in point_masks_sorted[:2]:
                        if mask.predicted_iou >= min_iou_score:
                            filtered_masks.append(mask)

                if self.verbose:
                    print(f"  Selected top-2 masks per point: {len(filtered_masks)}/{num_points*2} (IoU ≥ {min_iou_score})")
            else:
                # For automatic mode, just filter by IoU
                filtered_masks = [m for m in sam_masks if m.predicted_iou >= min_iou_score]

        sam_masks = filtered_masks

        # Step 3: Apply Non-Maximum Suppression to remove overlapping duplicates
        if len(sam_masks) > 1:
            sam_masks_before_nms = len(sam_masks)
            sam_masks = self._non_maximum_suppression(sam_masks, iou_threshold=nms_iou_threshold)

            if self.verbose:
                print(f"  After NMS: {len(sam_masks)}/{sam_masks_before_nms} masks (removed {sam_masks_before_nms - len(sam_masks)} overlaps)")

        # Step 4: Refine predictions using SAM masks with majority voting
        output_mask = dense_pred.copy()
        refined_count = 0

        for mask_candidate in sam_masks:
            mask = mask_candidate.mask
            mask_region = mask > 0.5

            if mask_region.sum() == 0:
                continue

            # Get dense predictions within this SAM mask
            masked_predictions = dense_pred[mask_region]

            # Majority vote: find the most common class in this region
            unique_classes, counts = np.unique(masked_predictions, return_counts=True)
            majority_class = unique_classes[counts.argmax()]
            max_count = counts.max()  # FIXED: get actual max count, not index
            total_pixels = mask_region.sum()

            # Only refine if majority class has sufficient coverage
            coverage = max_count / total_pixels
            if coverage >= min_coverage:
                # Assign majority class to entire SAM mask (refines boundaries)
                output_mask[mask_region] = majority_class
                refined_count += 1

        if self.verbose:
            print(f"  Final: Refined {refined_count} regions with SAM2 masks")

        return output_mask

    def _forward_single(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> torch.Tensor:
        """
        Single forward pass to get dense logits.

        Returns:
            Logits (num_classes, H_feat, W_feat)
        """
        # Compute dense similarities
        similarities = self.clip_extractor.compute_dense_similarity(
            image,
            class_names,
            use_csa=True
        )  # (num_classes, H, W)

        return similarities

    def _forward_slide(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> torch.Tensor:
        """
        Sliding window inference matching SCLIP's implementation.

        SCLIP's approach:
        1. Convert whole high-res image to normalized tensor ONCE (NO resizing)
        2. Extract 224x224 crops from this tensor
        3. Process each crop through CLIP encoder
        4. Accumulate and average overlapping predictions

        Args:
            image: Input image (H, W, 3) at target resolution (e.g., 2048px)
            class_names: List of class names

        Returns:
            Logits (num_classes, H, W)
        """
        H, W = image.shape[:2]
        num_classes = len(class_names)
        crop_size = self.slide_crop
        stride = self.slide_stride

        # CRITICAL: Convert whole image to normalized tensor WITHOUT resizing
        # This preserves the high resolution
        image_tensor = self.clip_extractor.preprocess_without_resize(image)  # (3, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)

        # Get text features (cached for efficiency)
        text_features = self._get_text_features(class_names)  # (num_classes, D)

        # Initialize accumulators on CPU to save GPU memory
        logits_sum = torch.zeros((1, num_classes, H, W), dtype=torch.float32)
        count_mat = torch.zeros((1, 1, H, W), dtype=torch.float32)

        # Compute grid
        h_grids = max(H - crop_size + stride - 1, 0) // stride + 1
        w_grids = max(W - crop_size + stride - 1, 0) // stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * stride
                x1 = w_idx * stride
                y2 = min(y1 + crop_size, H)
                x2 = min(x1 + crop_size, W)
                y1 = max(y2 - crop_size, 0)
                x1 = max(x2 - crop_size, 0)

                # Extract crop from the HIGH-RES tensor
                crop_tensor = image_tensor[:, :, y1:y2, x1:x2]  # (1, 3, crop_h, crop_w)

                # Process crop through CLIP encoder with CSA
                with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
                    features = self.clip_extractor.model.encode_image(
                        crop_tensor.type(self.clip_extractor.model.dtype),
                        return_all=True,
                        csa=True
                    )  # (1, num_patches+1, D)

                # Remove CLS token and normalize
                features = features[:, 1:]  # (1, num_patches, D)
                features = features / features.norm(dim=-1, keepdim=True)

                # Compute logits: features @ text_features^T
                crop_logits = features @ text_features.T  # (1, num_patches, num_classes)

                # Reshape to spatial: (1, num_classes, h_patches, w_patches)
                patch_size = self.clip_extractor.model.visual.patch_size
                crop_h, crop_w = y2 - y1, x2 - x1
                h_patches = crop_h // patch_size
                w_patches = crop_w // patch_size
                crop_logits = crop_logits.permute(0, 2, 1).reshape(1, num_classes, h_patches, w_patches)

                # Upsample logits to crop size
                crop_logits = F.interpolate(
                    crop_logits,
                    size=(crop_h, crop_w),
                    mode='bilinear',
                    align_corners=False
                )

                # Accumulate (move to CPU to save GPU memory)
                logits_sum[:, :, y1:y2, x1:x2] += crop_logits.cpu()
                count_mat[:, :, y1:y2, x1:x2] += 1

        # Average overlapping predictions
        logits = logits_sum / count_mat

        return logits.squeeze(0).to(self.device)  # (num_classes, H, W)

    def _extract_masked_region(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract and crop masked region from image."""
        # Find bounding box
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Extract region
        region = image[y_min:y_max+1, x_min:x_max+1].copy()

        # Apply mask
        region_mask = mask[y_min:y_max+1, x_min:x_max+1]
        region[region_mask < 0.5] = 0

        return region

    def segment(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> np.ndarray:
        """
        Main segmentation interface.

        Automatically chooses between dense or hybrid mode based on use_sam setting.

        Args:
            image: Input image (H, W, 3)
            class_names: List of class names to predict

        Returns:
            Segmentation mask (H, W) with class indices
        """
        if self.use_sam:
            return self.predict_with_sam(image, class_names)
        else:
            pred_mask, _ = self.predict_dense(image, class_names)
            return pred_mask
