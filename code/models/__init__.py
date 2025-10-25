"""
Models package for Open-Vocabulary Semantic Segmentation Pipeline.

Contains implementations of:
- SAM 2 mask generation
- CLIP dense feature extraction
- Mask-text alignment
- Stable Diffusion inpainting
"""

from .sam2_segmentation import SAM2MaskGenerator, MaskCandidate
from .clip_features import CLIPFeatureExtractor
from .mask_alignment import MaskTextAligner, ScoredMask
from .inpainting import StableDiffusionInpainter

__all__ = [
    'SAM2MaskGenerator',
    'MaskCandidate',
    'CLIPFeatureExtractor',
    'MaskTextAligner',
    'ScoredMask',
    'StableDiffusionInpainter',
]
