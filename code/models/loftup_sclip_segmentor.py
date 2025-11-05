"""
LoftUp-Enhanced SCLIP Segmentation

This module integrates LoftUp feature upsampling with your existing SCLIP+SAM2 pipeline.

Key improvements:
1. Upsamples SCLIP features from 14x downsampled to full resolution
2. Sharper semantic boundaries for better prompt extraction
3. More accurate prompt point localization (especially for small objects)
4. Maintains your 96% prompt reduction strategy
5. Keeps SAM2 for high-quality mask refinement

Architecture:
    CLIP → SCLIP (CSA) → LoftUp Upsampler → Prompt Extraction → SAM2 → Final Masks

    Instead of:
    CLIP → SCLIP (CSA) → Prompt Extraction → SAM2 → Final Masks
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from PIL import Image
import sys
import os

# Add LoftUp to path
loftup_path = os.path.join(os.path.dirname(__file__), '../../others/loftup')
if loftup_path not in sys.path:
    sys.path.insert(0, loftup_path)

from models.sclip_segmentor import SCLIPSegmentor
from scipy.ndimage import label, center_of_mass


class LoftUpSCLIPSegmentor(SCLIPSegmentor):
    """
    Enhanced SCLIP segmentor with LoftUp feature upsampling.

    This extends your existing SCLIPSegmentor with LoftUp upsampling to get:
    - Sharper semantic boundaries
    - More accurate prompt localization
    - Better small object detection

    Maintains compatibility with your existing clip_guided_segmentation.py workflow.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_loftup: bool = True,
        loftup_model_name: str = "loftup_clip",  # Pre-trained LoftUp for CLIP
        use_sam: bool = False,
        use_pamr: bool = False,
        pamr_steps: int = 10,
        pamr_dilations: Tuple[int, int] = (8, 16),
        logit_scale: float = 40.0,
        prob_threshold: float = 0.0,
        slide_inference: bool = True,
        slide_crop: int = 224,
        slide_stride: int = 112,
        verbose: bool = True,
    ):
        """
        Initialize LoftUp-enhanced SCLIP segmentor.

        Args:
            use_loftup: Enable LoftUp upsampling (default: True)
            loftup_model_name: Pre-trained LoftUp model name for torch.hub
                Options: 'loftup_clip', 'loftup_dinov2s', 'loftup_siglip2', etc.
            All other args same as SCLIPSegmentor
        """
        # Initialize parent SCLIP segmentor
        super().__init__(
            model_name=model_name,
            device=device,
            use_sam=use_sam,
            use_pamr=use_pamr,
            pamr_steps=pamr_steps,
            pamr_dilations=pamr_dilations,
            logit_scale=logit_scale,
            prob_threshold=prob_threshold,
            slide_inference=slide_inference,
            slide_crop=slide_crop,
            slide_stride=slide_stride,
            verbose=verbose
        )

        self.use_loftup = use_loftup
        self.loftup_upsampler = None

        # Load LoftUp model if enabled
        if use_loftup:
            if verbose:
                print(f"[LoftUp] Loading pre-trained upsampler: {loftup_model_name}")

            try:
                import torch.hub
                # Load pre-trained LoftUp model from torch hub
                self.loftup_upsampler = torch.hub.load(
                    'andrehuang/loftup',
                    loftup_model_name,
                    pretrained=True,
                    trust_repo=True
                )
                self.loftup_upsampler = self.loftup_upsampler.to(device)
                self.loftup_upsampler.eval()

                if verbose:
                    print(f"[LoftUp] Successfully loaded {loftup_model_name}")
                    print("[LoftUp] Feature upsampling enabled (14x → full resolution)")

            except Exception as e:
                print(f"[LoftUp] WARNING: Failed to load LoftUp model: {e}")
                print("[LoftUp] Falling back to standard SCLIP (no upsampling)")
                self.use_loftup = False
                self.loftup_upsampler = None

    def _upsample_features_with_loftup(
        self,
        low_res_features: torch.Tensor,
        image_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Upsample low-resolution CLIP features to full resolution using LoftUp.

        Args:
            low_res_features: CLIP features (1, D, H/14, W/14)
            image_tensor: Normalized image tensor (1, 3, H, W)

        Returns:
            Upsampled features (1, D, H, W)
        """
        if not self.use_loftup or self.loftup_upsampler is None:
            # Fallback to bilinear upsampling
            _, _, H, W = image_tensor.shape
            return F.interpolate(
                low_res_features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        with torch.no_grad():
            # Convert to float32 for LoftUp (it expects float32, not float16)
            original_dtype = low_res_features.dtype
            low_res_features_fp32 = low_res_features.float()
            image_tensor_fp32 = image_tensor.float()

            # LoftUp takes: (low_res_features, guidance_image)
            # Returns: (1, D, H, W) upsampled features
            upsampled_features = self.loftup_upsampler(low_res_features_fp32, image_tensor_fp32)

            # Convert back to original dtype if needed
            if original_dtype != torch.float32:
                upsampled_features = upsampled_features.to(original_dtype)

        return upsampled_features

    @torch.no_grad()
    def predict_dense(
        self,
        image: np.ndarray,
        class_names: List[str],
        return_logits: bool = False,
        return_features: bool = False
    ) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Dense semantic segmentation with optional LoftUp upsampling.

        Enhanced version that:
        1. Extracts low-res CLIP features
        2. Upsamples to full resolution with LoftUp
        3. Computes similarity with text features
        4. Applies PAMR refinement if enabled

        Args:
            image: Input image (H, W, 3) RGB numpy array
            class_names: List of class names to predict
            return_logits: Return raw logits before argmax
            return_features: Return upsampled features for analysis

        Returns:
            - Segmentation mask (H, W) with class indices
            - Optional logits (num_classes, H, W)
            - Optional upsampled features (D, H, W)
        """
        # Store original resolution
        orig_H, orig_W = image.shape[:2]

        # SCLIP preprocessing: resize to 2048px
        h, w = image.shape[:2]
        if max(h, w) != 2048:
            scale = 2048 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
            image = np.array(pil_img)
            if self.verbose:
                print(f"[SCLIP] Resized image from {w}x{h} to {new_w}x{new_h}")

        H, W = image.shape[:2]

        if self.use_loftup and self.loftup_upsampler is not None:
            # NEW: LoftUp-enhanced pipeline
            if self.verbose:
                print("[LoftUp] Using upsampled features for segmentation")

            # Get low-resolution CLIP features (before computing similarities)
            # We need to modify the forward to return features instead of logits
            logits, upsampled_features = self._forward_with_loftup(image, class_names)

        else:
            # Original SCLIP pipeline
            if self.slide_inference:
                logits = self._forward_slide(image, class_names)
            else:
                logits = self._forward_single(image, class_names)
            upsampled_features = None

        # Apply PAMR refinement if enabled
        if self.pamr is not None:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            if image_tensor.shape[-2:] != logits.shape[-2:]:
                image_tensor = F.interpolate(
                    image_tensor,
                    size=logits.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

            logits_dtype = logits.dtype
            logits = logits.float()
            logits = self.pamr(image_tensor, logits.unsqueeze(0)).squeeze(0)
            logits = logits.to(logits_dtype)

        # Scale logits
        logits = logits * self.logit_scale

        # Interpolate back to original resolution
        if logits.shape[-2:] != (orig_H, orig_W):
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Convert to probabilities and predictions
        probs = F.softmax(logits, dim=0)
        pred_mask = probs.argmax(dim=0).cpu().numpy()

        # Apply probability threshold
        if self.prob_threshold > 0:
            max_probs = probs.max(dim=0)[0].cpu().numpy()
            pred_mask[max_probs < self.prob_threshold] = 0

        if return_features:
            return pred_mask, logits if return_logits else None, upsampled_features
        elif return_logits:
            return pred_mask, logits, None
        else:
            return pred_mask, None, None

    def _forward_with_loftup(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoftUp upsampling.

        Pipeline:
        1. Extract low-res CLIP features (H/14, W/14)
        2. Upsample to full resolution with LoftUp (H, W)
        3. Compute cosine similarity with text features
        4. Return logits at full resolution

        Args:
            image: Input image (H, W, 3)
            class_names: List of class names

        Returns:
            - Logits (num_classes, H, W)
            - Upsampled features (D, H, W)
        """
        H, W = image.shape[:2]

        # For very large images, use sliding window with LoftUp on smaller crops
        # This prevents OOM errors
        max_size = 1024  # Maximum dimension for single LoftUp pass

        if max(H, W) > max_size:
            if self.verbose:
                print(f"[LoftUp] Image too large ({H}x{W}), using sliding window approach...")
            return self._forward_with_loftup_sliding(image, class_names, crop_size=max_size)

        # Preprocess image to normalized tensor
        image_tensor = self.clip_extractor.preprocess_without_resize(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Extract low-resolution CLIP features with CSA
        with torch.cuda.amp.autocast(enabled=False):
            clip_features = self.clip_extractor.model.encode_image(
                image_tensor.type(self.clip_extractor.model.dtype),
                return_all=True,
                csa=True
            )  # (1, num_patches+1, D)

        # Remove CLS token and reshape to spatial
        clip_features = clip_features[:, 1:]  # (1, num_patches, D)

        # Get dimensions
        num_patches = clip_features.shape[1]
        D = clip_features.shape[2]
        patch_size = self.clip_extractor.model.visual.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size

        # Reshape to spatial: (1, D, h_patches, w_patches)
        low_res_features = clip_features.permute(0, 2, 1).reshape(1, D, h_patches, w_patches)

        # Upsample features with LoftUp
        upsampled_features = self._upsample_features_with_loftup(
            low_res_features,
            image_tensor
        )  # (1, D, H, W)

        # Normalize upsampled features
        upsampled_features = F.normalize(upsampled_features, dim=1)

        # Get text features
        text_features = self._get_text_features(class_names)  # (num_classes, D)
        text_features = text_features.to(self.device)

        # Compute cosine similarity at full resolution
        # Reshape: (1, D, H, W) → (1, D, H*W) → (1, H*W, D)
        B, D, H_up, W_up = upsampled_features.shape
        features_flat = upsampled_features.reshape(B, D, -1).permute(0, 2, 1)  # (1, H*W, D)

        # Compute logits: (1, H*W, D) @ (D, num_classes) = (1, H*W, num_classes)
        logits_flat = features_flat @ text_features.T

        # Reshape back: (1, H*W, num_classes) → (1, num_classes, H, W)
        num_classes = len(class_names)
        logits = logits_flat.permute(0, 2, 1).reshape(1, num_classes, H_up, W_up)

        return logits.squeeze(0), upsampled_features.squeeze(0)  # (num_classes, H, W), (D, H, W)

    def _forward_with_loftup_sliding(
        self,
        image: np.ndarray,
        class_names: List[str],
        crop_size: int = 1024,
        stride: int = 768
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient sliding window approach for large images.

        Processes image in overlapping crops to avoid OOM.
        """
        H, W = image.shape[:2]
        num_classes = len(class_names)
        D = 512  # CLIP embedding dimension

        # Initialize accumulators on CPU to save GPU memory
        logits_sum = torch.zeros((num_classes, H, W), dtype=torch.float32)
        features_sum = torch.zeros((D, H, W), dtype=torch.float32)
        count_mat = torch.zeros((H, W), dtype=torch.float32)

        # Compute grid
        h_grids = max((H - crop_size) // stride + 1, 1)
        w_grids = max((W - crop_size) // stride + 1, 1)

        if self.verbose:
            print(f"  Processing {h_grids}x{w_grids} = {h_grids*w_grids} crops...")

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # Calculate crop boundaries
                y1 = h_idx * stride
                x1 = w_idx * stride
                y2 = min(y1 + crop_size, H)
                x2 = min(x1 + crop_size, W)
                y1 = max(y2 - crop_size, 0)
                x1 = max(x2 - crop_size, 0)

                # Extract crop
                crop = image[y1:y2, x1:x2]

                # Process crop (recursively calls _forward_with_loftup, but crop is small)
                crop_logits, crop_features = self._forward_with_loftup(crop, class_names)

                # Accumulate results (move to CPU to save GPU memory)
                logits_sum[:, y1:y2, x1:x2] += crop_logits.cpu()
                features_sum[:, y1:y2, x1:x2] += crop_features.cpu()
                count_mat[y1:y2, x1:x2] += 1

                # Clear GPU memory
                del crop_logits, crop_features
                torch.cuda.empty_cache()

                if (h_idx * w_grids + w_idx + 1) % 4 == 0:
                    print(f"    Processed {h_idx * w_grids + w_idx + 1}/{h_grids * w_grids} crops...")

        # Average overlapping predictions
        logits = logits_sum / count_mat.unsqueeze(0)
        features = features_sum / count_mat.unsqueeze(0)

        return logits.to(self.device), features.to(self.device)


def extract_prompt_points_from_upsampled(
    seg_map: np.ndarray,
    probs: np.ndarray,
    vocabulary: List[str],
    min_confidence: float = 0.7,
    min_region_size: int = 100
) -> List[dict]:
    """
    Extract prompt points from LoftUp-upsampled SCLIP predictions.

    This is a drop-in replacement for your existing prompt extraction function
    that works with full-resolution predictions from LoftUp.

    Args:
        seg_map: (H, W) predicted class indices at FULL resolution
        probs: (H, W, num_classes) probabilities at FULL resolution
        vocabulary: List of class names
        min_confidence: Minimum confidence to consider a region
        min_region_size: Minimum pixel area for a region

    Returns:
        List of prompt dictionaries with 'point', 'class_idx', 'class_name', etc.
    """
    H, W = seg_map.shape
    prompts = []

    print("\n[LoftUp] Extracting prompt points from upsampled predictions...")

    for class_idx, class_name in enumerate(vocabulary):
        # Get high-confidence regions for this class
        class_mask = (seg_map == class_idx)
        class_confidence = probs[:, :, class_idx]
        high_conf_mask = (class_mask & (class_confidence > min_confidence))

        # Find connected components
        labeled_regions, num_regions = label(high_conf_mask)

        if num_regions == 0:
            continue

        print(f"  {class_name}: found {num_regions} high-confidence regions")

        # For each region, extract centroid as prompt point
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Get centroid
            y_coords, x_coords = np.where(region_mask)
            centroid_x = int(x_coords.mean())
            centroid_y = int(y_coords.mean())

            # Get confidence at centroid
            confidence = class_confidence[centroid_y, centroid_x]

            prompts.append({
                'point': (centroid_x, centroid_y),
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': float(confidence),
                'region_size': int(region_size)
            })

    print(f"\n[LoftUp] Total prompt points extracted: {len(prompts)}")
    return prompts
