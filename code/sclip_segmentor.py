"""
SCLIP-based Semantic Segmentation

This module implements dense semantic segmentation using SCLIP's approach:
- Cross-layer Self-Attention (CSA) for better dense features
- Direct pixel-wise classification (no SAM masks required)
- PAMR (Pixel-Adaptive Memory Refinement) for boundary refinement
- Optional SAM integration for region-based predictions

Performance: SCLIP achieves 22.77% mIoU on COCO-Stuff164k
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
        self.verbose = verbose

        if verbose:
            print("[SCLIP Segmentor] Initializing...")
            print(f"  Mode: {'Hybrid (SAM + SCLIP)' if use_sam else 'Dense (SCLIP only)'}")
            print(f"  PAMR: {'Enabled' if use_pamr else 'Disabled'}")
            print(f"  Slide inference: {'Enabled' if slide_inference else 'Disabled'}")

        # Initialize SCLIP feature extractor
        self.clip_extractor = SCLIPFeatureExtractor(
            model_name=model_name,
            device=device
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

    @torch.no_grad()
    def predict_with_sam(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> np.ndarray:
        """
        Hybrid: SAM masks + SCLIP classification.

        Uses SAM to propose regions, then SCLIP to classify each region.

        Args:
            image: Input image (H, W, 3)
            class_names: List of class names

        Returns:
            Segmentation mask (H, W) with class indices
        """
        H, W = image.shape[:2]

        # Generate SAM masks
        sam_masks = self.sam_generator.generate_masks(image)

        # Get SCLIP text features for all classes
        text_features = self.clip_extractor.extract_text_features(
            class_names,
            use_prompt_ensemble=True,
            normalize=True
        )  # (num_classes, D)

        # Initialize output mask
        output_mask = np.zeros((H, W), dtype=np.int32)
        mask_scores = np.zeros((H, W), dtype=np.float32)

        # Classify each SAM mask
        for mask_candidate in sam_masks:
            mask = mask_candidate.mask

            # Extract masked region
            masked_region = self._extract_masked_region(image, mask)
            if masked_region is None:
                continue

            # Get SCLIP image features
            pil_img = Image.fromarray(masked_region)
            pil_img = pil_img.resize((336, 336), Image.Resampling.BILINEAR)
            resized_np = np.array(pil_img)

            mask_features, _ = self.clip_extractor.extract_image_features(
                resized_np,
                return_dense=False,
                use_csa=True,
                normalize=True
            )

            # Compute similarity to all classes
            similarities = F.cosine_similarity(
                mask_features.unsqueeze(0),
                text_features,
                dim=1
            )  # (num_classes,)

            # Get best class
            best_class_idx = similarities.argmax().item()
            best_score = similarities[best_class_idx].item()

            # Apply threshold
            if best_score < self.prob_threshold:
                continue

            # Update output mask (keep higher-scoring predictions)
            mask_region = mask > 0.5
            update_region = (best_score > mask_scores[mask_region])

            update_pixels = mask_region.copy()
            update_pixels[mask_region] = update_region

            output_mask[update_pixels] = best_class_idx
            mask_scores[update_pixels] = best_score

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

        # Get text features once
        text_features = self.clip_extractor.extract_text_features(
            class_names,
            use_prompt_ensemble=True,
            normalize=True
        )  # (num_classes, D)

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
