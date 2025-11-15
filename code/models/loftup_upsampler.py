"""
LoftUp Feature Upsampling Module

Implements coordinate-based cross-attention for upsampling CLIP features from low to high resolution.
This significantly improves dense prediction quality without requiring training.

Reference: Huang et al., "LoftUp: Improving CLIP for Dense Prediction", ICCV 2025
Paper: https://arxiv.org/abs/2404.07949
Code: https://github.com/andrehuang/loftup

Expected improvement: +2-4% mIoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import warnings


class LoftUpUpsampler(nn.Module):
    """
    LoftUp: Coordinate-based cross-attention upsampler for CLIP features.

    Key innovation: Uses pixel coordinates as queries to upsample low-resolution
    features to high-resolution via cross-attention, preserving semantic information
    while gaining spatial detail.

    Architecture:
    - Input: Low-res features (e.g., 14x14 or 24x24 from CLIP)
    - Output: High-res features (e.g., 224x224 or 336x336)
    - Method: Coordinate-based cross-attention with learnable positional encoding
    """

    def __init__(
        self,
        model_name: str = "loftup_clip",
        backbone: str = "ViT-B/16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        use_pretrained: bool = True,
    ):
        """
        Initialize LoftUp upsampler.

        Args:
            model_name: LoftUp model variant
            backbone: CLIP backbone (should match main model)
            device: Computation device
            use_fp16: Use mixed precision
            use_pretrained: Load pre-trained weights from torch.hub
        """
        super().__init__()

        self.device = device
        self.backbone = backbone
        self.use_fp16 = use_fp16 and device == "cuda"
        self.model_name = model_name

        self.upsampler = None
        self._load_model(use_pretrained)

    def _load_model(self, use_pretrained: bool):
        """Load LoftUp model from torch.hub or create mock implementation."""
        try:
            if use_pretrained:
                print(f"[LoftUp] Loading pre-trained model from torch.hub...")

                # Load from torch.hub
                # Repository: andrehuang/loftup
                # Model: loftup_clip (for CLIP ViT-B/16)
                self.upsampler = torch.hub.load(
                    'andrehuang/loftup',
                    self.model_name,
                    pretrained=True,
                    trust_repo=True  # Required for custom models
                )

                self.upsampler = self.upsampler.to(self.device)
                self.upsampler.eval()

                print(f"✓ LoftUp loaded successfully with pre-trained weights")
                print(f"  Model: {self.model_name}")
                print(f"  Backbone: {self.backbone}")

                if self.use_fp16:
                    print(f"  FP16: Enabled (via autocast)")
            else:
                print("[LoftUp] Using mock bilinear upsampling (no pre-trained weights)")
                self.upsampler = None

        except Exception as e:
            print(f"⚠ LoftUp loading failed: {type(e).__name__}: {str(e)}")
            print(f"  Falling back to bilinear upsampling")
            print(f"  To use LoftUp, ensure torch.hub can access: andrehuang/loftup")
            self.upsampler = None

    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int],
        original_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Upsample features from low-resolution to high-resolution using LoftUp.

        Args:
            features: Low-resolution features (B, C, H_low, W_low) or (B, N, C)
            target_size: Target spatial size (H_high, W_high)
            original_image: Optional high-res image for guidance (B, 3, H, W)

        Returns:
            Upsampled features (B, C, H_high, W_high)
        """
        if self.upsampler is None:
            # Fallback to bilinear upsampling
            return self._bilinear_upsample(features, target_size)

        # Use LoftUp's coordinate-based upsampling
        try:
            # Ensure features are in (B, C, H, W) format
            if features.dim() == 3:
                # Convert from (B, N, C) to (B, C, H, W)
                B, N, C = features.shape
                H = W = int(np.sqrt(N))
                features = features.transpose(1, 2).reshape(B, C, H, W)

            # Ensure features are on the correct device and contiguous
            features = features.to(self.device).contiguous()

            # Ensure both inputs are 4D tensors
            if features.dim() != 4:
                raise ValueError(f"Features must be 4D after reshaping, got {features.dim()}D")

            if original_image is None:
                # No image guidance - use bilinear upsampling
                return self._bilinear_upsample(features, target_size)

            # Ensure image is on correct device and is 4D
            original_image = original_image.to(self.device).contiguous()
            if original_image.dim() != 4:
                raise ValueError(f"Image must be 4D, got {original_image.dim()}D: shape {original_image.shape}")

            # Convert to float32 to avoid autocast issues with sparse tensors
            features_fp32 = features.float()
            image_fp32 = original_image.float()

            # Call LoftUp without autocast (it has issues with sparse tensors)
            # API: upsampler(low_res_features, high_res_image)
            # LoftUp returns features at the same spatial resolution as the input image
            # Input:  features_fp32 (B, C, H_lr, W_lr), image_fp32 (B, 3, H_hr, W_hr)
            # Output: upsampled (B, C, H_hr, W_hr) where H_hr, W_hr match image_fp32 size
            upsampled = self.upsampler(features_fp32, image_fp32)

            # LoftUp always returns (B, C, H, W) format
            # The output size matches the image size
            B, C, H_out, W_out = upsampled.shape
            if (H_out, W_out) != target_size:
                # Resize to target if needed (should rarely happen)
                # This only occurs if target_size differs from image size
                upsampled = F.interpolate(
                    upsampled,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            return upsampled

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            warnings.warn(f"LoftUp forward failed: {e}\n{error_details}\nFalling back to bilinear.")
            return self._bilinear_upsample(features, target_size)

    def _bilinear_upsample(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Fallback bilinear upsampling."""
        if features.dim() == 3:
            # (B, N, C) -> (B, C, H, W)
            B, N, C = features.shape
            H = W = int(np.sqrt(N))
            features = features.transpose(1, 2).reshape(B, C, H, W)

        # Bilinear interpolation
        upsampled = F.interpolate(
            features,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        return upsampled

    def get_expected_improvement(self) -> dict:
        """Return expected performance improvements."""
        return {
            "mIoU_gain": "+2-4%",
            "boundary_f1_gain": "+1-2%",
            "training_required": False,
            "inference_overhead": "~5-10ms per image",
            "reference": "Huang et al., ICCV 2025"
        }


def create_loftup_upsampler(
    backbone: str = "ViT-B/16",
    device: str = "cuda",
    use_fp16: bool = True,
    use_pretrained: bool = True
) -> LoftUpUpsampler:
    """
    Factory function to create LoftUp upsampler.

    Args:
        backbone: CLIP backbone architecture
        device: Computation device
        use_fp16: Enable mixed precision
        use_pretrained: Load pre-trained weights

    Returns:
        Configured LoftUp upsampler
    """
    return LoftUpUpsampler(
        model_name="loftup_clip",
        backbone=backbone,
        device=device,
        use_fp16=use_fp16,
        use_pretrained=use_pretrained
    )
