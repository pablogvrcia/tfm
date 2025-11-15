"""
Hierarchical Mask Decoder with Cross-Scale Fusion (Phase 3B)

Implements coarse-to-fine mask refinement using cross-attention between
SAM2 mask embeddings and SCLIP semantic features.

Key innovations:
1. Multi-scale mask pyramid processing (coarse → fine)
2. Cross-attention between mask features and CLIP semantic features
3. Residual refinement (inspired by ResCLIP)
4. Training-free implementation

Expected improvement: +3-5% mIoU from boundary precision

Reference:
- Mask2Former: Multi-scale deformable attention
- ResCLIP: Residual attention for refinement
- PSM-DIQ: Multi-dimensional attention for feature extraction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class CrossScaleAttention(nn.Module):
    """
    Cross-attention module between mask embeddings and CLIP features.

    Query: Mask features (what we're trying to refine)
    Key/Value: CLIP semantic features (what provides semantic context)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        device: str = "cuda"
    ):
        """
        Initialize cross-scale attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            device: Computation device
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device

        # Note: We don't use learnable parameters for training-free approach
        # Instead, we use pre-computed similarity

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention (training-free version using cosine similarity).

        Args:
            query: (B, N_q, D) mask features
            key: (B, N_k, D) CLIP features
            value: (B, N_k, D) CLIP features
            mask: Optional attention mask

        Returns:
            (B, N_q, D) refined features
        """
        B, N_q, D = query.shape
        _, N_k, _ = key.shape

        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(key, dim=-1)

        # Compute attention scores: (B, N_q, N_k)
        attn_scores = torch.matmul(query_norm, key_norm.transpose(-2, -1))
        attn_scores = attn_scores * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, value)

        return output


class HierarchicalMaskDecoder:
    """
    Hierarchical mask decoder that refines masks through multi-scale processing.

    Pipeline:
    1. Start with coarse SAM2 masks
    2. For each scale (coarse to fine):
       - Apply cross-attention with CLIP features
       - Add residual connection
       - Upsample and fuse with next finer scale
    3. Output refined masks at original resolution
    """

    def __init__(
        self,
        scales: List[float] = [0.25, 0.5, 1.0, 2.0],
        embed_dim: int = 256,
        num_heads: int = 8,
        residual_weight: float = 0.3,
        use_fp16: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize hierarchical mask decoder.

        Args:
            scales: Scale factors for multi-scale processing
            embed_dim: Feature embedding dimension
            num_heads: Number of attention heads
            residual_weight: Weight for residual connection (0.2-0.4)
            use_fp16: Use mixed precision
            device: Computation device
        """
        self.scales = sorted(scales)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.residual_weight = residual_weight
        self.use_fp16 = use_fp16 and device == "cuda"
        self.device = device

        # Initialize cross-attention module
        self.cross_attn = CrossScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device
        ).to(device)

    def refine_masks_hierarchical(
        self,
        masks_pyramid: Dict[float, torch.Tensor],
        clip_features_pyramid: Dict[float, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Refine masks through hierarchical multi-scale processing.

        Args:
            masks_pyramid: Dict[scale -> masks] (B, N, H_s, W_s)
            clip_features_pyramid: Dict[scale -> features] (B, H_s, W_s, D)
            original_size: Target output size (H, W)

        Returns:
            Refined masks at original resolution (B, N, H, W)
        """
        with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
            # Process from coarse to fine
            refined_masks = None

            for scale_idx, scale in enumerate(self.scales):
                if scale not in masks_pyramid or scale not in clip_features_pyramid:
                    continue

                current_masks = masks_pyramid[scale]  # (B, N, H_s, W_s)
                current_features = clip_features_pyramid[scale]  # (B, H_s, W_s, D)

                # Refine current scale masks
                refined_current = self._refine_single_scale(
                    masks=current_masks,
                    clip_features=current_features,
                    previous_masks=refined_masks
                )

                # Upsample for next scale (if not the finest)
                if scale_idx < len(self.scales) - 1:
                    next_scale = self.scales[scale_idx + 1]
                    if next_scale in masks_pyramid:
                        next_size = masks_pyramid[next_scale].shape[-2:]
                        refined_masks = F.interpolate(
                            refined_current,
                            size=next_size,
                            mode='bilinear',
                            align_corners=False
                        )
                else:
                    # Final scale: upsample to original size
                    refined_masks = F.interpolate(
                        refined_current,
                        size=original_size,
                        mode='bilinear',
                        align_corners=False
                    )

            # Fallback if no scales processed
            if refined_masks is None:
                # Use highest resolution available
                finest_scale = max(masks_pyramid.keys())
                refined_masks = F.interpolate(
                    masks_pyramid[finest_scale],
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )

            return refined_masks

    def _refine_single_scale(
        self,
        masks: torch.Tensor,
        clip_features: torch.Tensor,
        previous_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Refine masks at a single scale using cross-attention.

        Args:
            masks: (B, N, H, W) current scale masks
            clip_features: (B, H, W, D) CLIP features at this scale
            previous_masks: (B, N, H, W) upsampled masks from coarser scale

        Returns:
            (B, N, H, W) refined masks
        """
        B, N, H, W = masks.shape
        _, _, _, D = clip_features.shape

        # Convert masks to features by pooling CLIP features within each mask
        mask_features = self._extract_mask_features(masks, clip_features)  # (B, N, D)

        # Reshape CLIP features for attention
        clip_features_flat = clip_features.view(B, H * W, D)  # (B, H*W, D)

        # Apply cross-attention: mask features attend to CLIP features
        attended_features = self.cross_attn(
            query=mask_features,
            key=clip_features_flat,
            value=clip_features_flat
        )  # (B, N, D)

        # Project attended features back to spatial domain
        # For each mask, compute similarity with CLIP features
        refined_masks = self._project_features_to_masks(
            attended_features,
            clip_features,
            original_masks=masks
        )  # (B, N, H, W)

        # Residual connection: blend refined with original
        refined_masks = (1 - self.residual_weight) * refined_masks + \
                        self.residual_weight * masks

        # Fuse with previous scale if available
        if previous_masks is not None:
            # Ensure same size
            if previous_masks.shape[-2:] != refined_masks.shape[-2:]:
                previous_masks = F.interpolate(
                    previous_masks,
                    size=refined_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Weighted fusion (favor finer scale)
            refined_masks = 0.7 * refined_masks + 0.3 * previous_masks

        return refined_masks

    def _extract_mask_features(
        self,
        masks: torch.Tensor,
        clip_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract feature representation for each mask by pooling CLIP features.

        Args:
            masks: (B, N, H, W) binary/probability masks
            clip_features: (B, H, W, D) CLIP features

        Returns:
            (B, N, D) mask feature embeddings
        """
        B, N, H, W = masks.shape
        _, _, _, D = clip_features.shape

        # Expand dimensions for broadcasting: (B, N, H, W, 1) x (B, 1, H, W, D)
        masks_expanded = masks.unsqueeze(-1)  # (B, N, H, W, 1)
        features_expanded = clip_features.unsqueeze(1)  # (B, 1, H, W, D)

        # Weighted pooling: sum of (mask * features) / sum of mask
        weighted_sum = (masks_expanded * features_expanded).sum(dim=[2, 3])  # (B, N, D)
        mask_sum = masks.sum(dim=[2, 3]).unsqueeze(-1) + 1e-6  # (B, N, 1)

        mask_features = weighted_sum / mask_sum  # (B, N, D)

        return mask_features

    def _project_features_to_masks(
        self,
        mask_features: torch.Tensor,
        clip_features: torch.Tensor,
        original_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Project mask features back to spatial masks using similarity.

        Args:
            mask_features: (B, N, D) refined mask embeddings
            clip_features: (B, H, W, D) spatial CLIP features
            original_masks: (B, N, H, W) original masks for structure

        Returns:
            (B, N, H, W) refined spatial masks
        """
        B, N, D = mask_features.shape
        _, H, W, _ = clip_features.shape

        # Normalize for cosine similarity
        mask_features_norm = F.normalize(mask_features, dim=-1)  # (B, N, D)
        clip_features_norm = F.normalize(clip_features, dim=-1)  # (B, H, W, D)

        # Compute similarity: (B, N, D) x (B, H, W, D) -> (B, N, H, W)
        similarity_maps = torch.einsum('bnd,bhwd->bnhw',
                                       mask_features_norm,
                                       clip_features_norm)

        # Apply sigmoid to convert to probabilities
        refined_masks = torch.sigmoid(similarity_maps * 10.0)  # Scale for sharpness

        return refined_masks

    def refine_masks_simple(
        self,
        masks: torch.Tensor,
        clip_features: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Simplified single-scale refinement (fallback for single-scale input).

        Args:
            masks: (B, N, H_m, W_m) input masks
            clip_features: (B, H_f, W_f, D) CLIP features
            target_size: Optional output size

        Returns:
            (B, N, H_t, W_t) refined masks
        """
        B, N, H_m, W_m = masks.shape
        _, H_f, W_f, D = clip_features.shape

        # Resize masks to match feature resolution
        if (H_m, W_m) != (H_f, W_f):
            masks_resized = F.interpolate(
                masks,
                size=(H_f, W_f),
                mode='bilinear',
                align_corners=False
            )
        else:
            masks_resized = masks

        # Apply single-scale refinement
        refined = self._refine_single_scale(
            masks=masks_resized,
            clip_features=clip_features,
            previous_masks=None
        )

        # Resize to target size if specified
        if target_size is not None and refined.shape[-2:] != target_size:
            refined = F.interpolate(
                refined,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        return refined
