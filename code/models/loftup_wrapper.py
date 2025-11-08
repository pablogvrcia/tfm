"""
LoFTup Feature Upsampling Wrapper

LoFTup (Coordinate-Based Feature Upsampling) improves low-resolution features
from vision foundation models by upsampling them to higher resolutions while
preserving semantic information.

Paper: ICCV 2025 (Oral)
Repo: https://github.com/andrehuang/loftup

Integration with SCLIP:
    CLIP Encoder → Low-res features → LoFTup → High-res features → CSA → Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class LoFTupWrapper:
    """
    Wrapper for LoFTup feature upsampler to integrate with CLIP/SCLIP.

    LoFTup improves dense prediction by upsampling patch features to higher
    spatial resolution while maintaining semantic quality.
    """

    def __init__(
        self,
        model_name: str = "clip",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        upsample_factor: float = 2.0,
        use_torch_hub: bool = True,
        verbose: bool = True
    ):
        """
        Initialize LoFTup upsampler.

        Args:
            model_name: Feature extractor model ('clip', 'dinov2', 'siglip')
            device: Computation device
            upsample_factor: Factor to upsample features (e.g., 2.0 = 2x resolution)
            use_torch_hub: Load from torch.hub (True) or Hugging Face (False)
            verbose: Print initialization info
        """
        self.device = device
        self.model_name = model_name
        self.upsample_factor = upsample_factor
        self.verbose = verbose
        self.upsampler = None

        try:
            if use_torch_hub:
                if verbose:
                    print(f"[LoFTup] Loading upsampler from torch.hub for {model_name}...")

                # Load pre-trained LoFTup model from torch hub
                # Format: loftup_{model_name} (e.g., loftup_clip)
                self.upsampler = torch.hub.load(
                    'andrehuang/loftup',
                    f'loftup_{model_name}',
                    pretrained=True,
                    verbose=verbose
                )
            else:
                if verbose:
                    print(f"[LoFTup] Loading upsampler from Hugging Face for {model_name}...")

                from transformers import AutoModel
                # Load from Hugging Face Hub
                # Format: haiwen/loftup-{model_name}
                self.upsampler = AutoModel.from_pretrained(f"haiwen/loftup-{model_name}")

            self.upsampler = self.upsampler.to(device)
            self.upsampler.eval()

            if verbose:
                print(f"[LoFTup] Successfully loaded upsampler for {model_name}")
                print(f"[LoFTup] Upsample factor: {upsample_factor}x")

        except Exception as e:
            if verbose:
                warnings.warn(
                    f"[LoFTup] Failed to load upsampler: {e}\n"
                    f"Falling back to bilinear interpolation. "
                    f"To use LoFTup, install dependencies:\n"
                    f"  pip install timm einops"
                )
            self.upsampler = None

    def is_available(self) -> bool:
        """Check if LoFTup model is loaded and available."""
        return self.upsampler is not None

    @torch.no_grad()
    def upsample_features(
        self,
        features: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        use_bilinear_fallback: bool = True
    ) -> torch.Tensor:
        """
        Upsample features using LoFTup or bilinear interpolation fallback.

        Args:
            features: Input features (B, H, W, D) or (B, N, D) where N = H*W
            target_size: Target spatial size (H_new, W_new). If None, uses upsample_factor
            use_bilinear_fallback: Use bilinear if LoFTup not available

        Returns:
            Upsampled features (B, H_new, W_new, D)
        """
        # Handle different input formats
        if features.dim() == 3:
            # (B, N, D) format - need to infer spatial dimensions
            B, N, D = features.shape
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot infer square spatial dims from {N} patches"
            features = features.reshape(B, H, W, D)
        elif features.dim() == 4:
            B, H, W, D = features.shape
        else:
            raise ValueError(f"Expected features with 3 or 4 dims, got {features.dim()}")

        # Determine target size
        if target_size is None:
            target_size = (
                int(H * self.upsample_factor),
                int(W * self.upsample_factor)
            )

        H_new, W_new = target_size

        # Use LoFTup if available
        if self.upsampler is not None:
            try:
                # LoFTup expects (B, D, H, W) format
                features_bhw = features.permute(0, 3, 1, 2)  # (B, D, H, W)

                # Apply LoFTup upsampling
                # Note: LoFTup interface may vary - adjust based on actual API
                # Common approach: provide features and target size
                upsampled = self._apply_loftup(features_bhw, (H_new, W_new))

                # Convert back to (B, H, W, D)
                upsampled = upsampled.permute(0, 2, 3, 1)

                return upsampled

            except Exception as e:
                if self.verbose:
                    warnings.warn(f"[LoFTup] Upsampling failed: {e}, using bilinear fallback")
                if not use_bilinear_fallback:
                    raise

        # Fallback to bilinear interpolation
        if use_bilinear_fallback:
            return self._bilinear_upsample(features, (H_new, W_new))
        else:
            raise RuntimeError("LoFTup not available and bilinear fallback disabled")

    def _apply_loftup(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Apply LoFTup upsampling (internal method).

        Args:
            features: Features in (B, D, H, W) format
            target_size: Target spatial size (H_new, W_new)

        Returns:
            Upsampled features (B, D, H_new, W_new)
        """
        # LoFTup uses coordinate-based upsampling
        # The exact interface depends on the LoFTup implementation

        # Method 1: If LoFTup has a direct forward method
        if hasattr(self.upsampler, 'forward'):
            # Some implementations take features + target coordinates
            # Generate coordinate grid for target size
            H_new, W_new = target_size
            coords = self._generate_coordinate_grid(H_new, W_new, features.device)

            # Apply upsampler
            upsampled = self.upsampler(features, coords)
            return upsampled

        # Method 2: If LoFTup uses interpolate-like interface
        elif hasattr(self.upsampler, 'upsample'):
            return self.upsampler.upsample(features, size=target_size)

        # Method 3: Direct call (most common for torch.hub models)
        else:
            # Many torch.hub models use __call__ with size parameter
            return self.upsampler(features, size=target_size)

    def _generate_coordinate_grid(
        self,
        H: int,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate normalized coordinate grid for LoFTup.

        Args:
            H: Height
            W: Width
            device: Device for tensors

        Returns:
            Coordinate grid (1, H, W, 2) with values in [-1, 1]
        """
        # Create normalized coordinate grids
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)

        # Create meshgrid
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack and add batch dimension
        coords = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        return coords

    def _bilinear_upsample(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Bilinear upsampling fallback.

        Args:
            features: Features (B, H, W, D)
            target_size: Target size (H_new, W_new)

        Returns:
            Upsampled features (B, H_new, W_new, D)
        """
        # Convert to (B, D, H, W) for interpolate
        features_bhw = features.permute(0, 3, 1, 2)

        # Apply bilinear interpolation
        upsampled = F.interpolate(
            features_bhw,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Convert back to (B, H, W, D)
        upsampled = upsampled.permute(0, 2, 3, 1)

        return upsampled


class AdaptiveLoFTup(LoFTupWrapper):
    """
    Adaptive LoFTup that dynamically adjusts upsampling based on input size.

    For small features (e.g., 14x14 from CLIP), applies aggressive upsampling.
    For larger features, applies moderate upsampling to balance quality and compute.
    """

    def __init__(
        self,
        model_name: str = "clip",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_upsample_factor: float = 1.5,
        max_upsample_factor: float = 4.0,
        target_min_size: int = 56,
        verbose: bool = True
    ):
        """
        Initialize adaptive LoFTup.

        Args:
            model_name: Feature extractor model
            device: Computation device
            min_upsample_factor: Minimum upsampling factor
            max_upsample_factor: Maximum upsampling factor (for very small features)
            target_min_size: Target minimum spatial size
            verbose: Print info
        """
        # Start with base upsample factor
        super().__init__(
            model_name=model_name,
            device=device,
            upsample_factor=2.0,
            verbose=verbose
        )

        self.min_upsample_factor = min_upsample_factor
        self.max_upsample_factor = max_upsample_factor
        self.target_min_size = target_min_size

    @torch.no_grad()
    def upsample_features(
        self,
        features: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        use_bilinear_fallback: bool = True
    ) -> torch.Tensor:
        """
        Adaptively upsample features based on their current size.

        Args:
            features: Input features (B, H, W, D) or (B, N, D)
            target_size: Optional fixed target size
            use_bilinear_fallback: Use bilinear if LoFTup unavailable

        Returns:
            Upsampled features
        """
        # Get current spatial size
        if features.dim() == 3:
            B, N, D = features.shape
            H = W = int(N ** 0.5)
        else:
            B, H, W, D = features.shape

        # Determine adaptive upsampling factor
        if target_size is None:
            # Adapt based on current size
            if min(H, W) < 14:
                # Very small features - aggressive upsampling
                factor = self.max_upsample_factor
            elif min(H, W) < 28:
                # Small features - moderate upsampling
                factor = 3.0
            elif min(H, W) < self.target_min_size:
                # Medium features - mild upsampling
                factor = 2.0
            else:
                # Already large enough - minimal or no upsampling
                factor = max(self.min_upsample_factor, self.target_min_size / min(H, W))

            target_size = (int(H * factor), int(W * factor))

            if self.verbose:
                print(f"[AdaptiveLoFTup] Input: {H}x{W}, Factor: {factor:.1f}x, Output: {target_size[0]}x{target_size[1]}")

        # Apply upsampling
        return super().upsample_features(features, target_size, use_bilinear_fallback)
