"""
Configuration file for the Open-Vocabulary Segmentation Pipeline.

Contains all hyperparameters and settings described in Chapter 3.3
(Implementation Details).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SAM2Config:
    """SAM 2 configuration (Chapter 3.3.2)"""
    model_type: str = "sam2_hiera_large"
    points_per_side: int = 32  # 32x32 grid = 1024 point prompts
    pred_iou_thresh: float = 0.88  # Minimum IoU confidence
    stability_score_thresh: float = 0.95  # Minimum stability score
    crop_n_layers: int = 0  # Number of crop layers for large images
    crop_overlap_ratio: float = 512/1500  # Overlap ratio between crops


@dataclass
class CLIPConfig:
    """CLIP configuration (Chapter 3.3.1)"""
    model_name: str = "ViT-L-14"  # ViT-L/14 variant
    pretrained: str = "openai"
    image_size: int = 336  # Input resolution (336x336)
    extract_layers: List[int] = field(default_factory=lambda: [6, 12, 18, 24])  # Multi-scale layers
    use_prompt_ensemble: bool = True  # Use prompt templates


@dataclass
class AlignmentConfig:
    """Mask-text alignment configuration (Chapter 3.3.3)"""
    background_weight: float = 0.3  # α in Equation 3.2
    use_spatial_weighting: bool = True  # Weight center pixels more
    similarity_threshold: float = 0.25  # Minimum score threshold
    top_k: int = 5  # Number of top masks to return


@dataclass
class InpaintingConfig:
    """Stable Diffusion inpainting configuration (Chapter 3.3.4)"""
    model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    num_inference_steps: int = 50  # Diffusion steps
    guidance_scale: float = 7.5  # Classifier-free guidance scale
    negative_prompt: str = "blurry, low quality, distorted, artifacts"
    mask_blur: int = 8  # Gaussian blur radius for mask edges (pixels)
    mask_dilation: int = 5  # Pixels to dilate mask


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    sam2: SAM2Config = field(default_factory=SAM2Config)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)

    # General settings
    device: str = "cuda"  # or "cpu"
    verbose: bool = True
    min_mask_area: int = 1024  # 32x32 pixels

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'sam2': self.sam2.__dict__,
            'clip': self.clip.__dict__,
            'alignment': self.alignment.__dict__,
            'inpainting': self.inpainting.__dict__,
            'device': self.device,
            'verbose': self.verbose,
            'min_mask_area': self.min_mask_area
        }


# Preset configurations for different use cases

def get_fast_config() -> PipelineConfig:
    """
    Fast configuration for quick prototyping.
    Reduces quality slightly but speeds up inference.
    """
    config = PipelineConfig()

    # Reduce SAM 2 mask count
    config.sam2.points_per_side = 16  # 16x16 = 256 prompts

    # Use fewer CLIP layers
    config.clip.extract_layers = [12, 24]

    # Reduce inpainting steps
    config.inpainting.num_inference_steps = 30

    return config


def get_quality_config() -> PipelineConfig:
    """
    High-quality configuration for best results.
    Slower but produces better segmentation and editing.
    """
    config = PipelineConfig()

    # More SAM 2 masks
    config.sam2.points_per_side = 64  # 64x64 = 4096 prompts
    config.sam2.pred_iou_thresh = 0.90

    # All CLIP layers
    config.clip.extract_layers = [3, 6, 9, 12, 15, 18, 21, 24]

    # More inpainting steps
    config.inpainting.num_inference_steps = 100
    config.inpainting.guidance_scale = 9.0

    return config


def get_balanced_config() -> PipelineConfig:
    """
    Balanced configuration (default).
    Good tradeoff between speed and quality.
    """
    return PipelineConfig()


# Dataset-specific configurations

def get_coco_config() -> PipelineConfig:
    """Configuration optimized for COCO-style images."""
    config = PipelineConfig()
    config.alignment.similarity_threshold = 0.20  # Lower threshold for diverse objects
    config.min_mask_area = 512  # Allow smaller objects
    return config


def get_pascal_voc_config() -> PipelineConfig:
    """Configuration optimized for PASCAL VOC."""
    config = PipelineConfig()
    config.alignment.similarity_threshold = 0.25
    config.min_mask_area = 1024
    return config


def get_ade20k_config() -> PipelineConfig:
    """Configuration optimized for ADE20K (scene parsing)."""
    config = PipelineConfig()
    config.sam2.points_per_side = 48  # More masks for complex scenes
    config.alignment.similarity_threshold = 0.22
    config.min_mask_area = 256  # Allow smaller scene elements
    return config
