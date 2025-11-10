"""
SCLIP Dense Feature Extraction Module

This module uses SCLIP's modified CLIP with Cross-layer Self-Attention (CSA)
for superior dense prediction performance.

SCLIP achieves 22.77% mIoU on COCO-Stuff164k (vs ~7% for standard approaches).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from PIL import Image
import sys

from models.clip import clip
from prompts.imagenet_template import openai_imagenet_template


class SCLIPFeatureExtractor:
    """
    Extracts dense vision-language features using SCLIP's CSA-enhanced CLIP.

    Key differences from standard CLIP:
    - Cross-layer Self-Attention (CSA) in final layer for better spatial features
    - Returns dense patch features (not just CLS token)
    - Optimized for segmentation (not classification)
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",  # SCLIP paper uses ViT-B/16, not ViT-L/14
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,  # Mixed precision for 2x speedup (inspired by TernaryCLIP 2025)
        use_compile: bool = False,  # torch.compile() for JIT optimization
    ):
        """
        Initialize SCLIP feature extractor with 2025 performance optimizations.

        Args:
            model_name: CLIP model variant (ViT-L/14@336px recommended by SCLIP)
            device: Computation device
            use_fp16: Enable mixed precision (FP16) for faster inference
            use_compile: Enable torch.compile() for JIT optimization
        """
        self.device = device
        self.model_name = model_name
        self.use_fp16 = use_fp16 and device == "cuda"
        self.use_compile = use_compile

        print(f"[SCLIP] Loading CLIP model with CSA: {model_name}")
        # Load SCLIP's modified CLIP (with CSA)
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()

        # Note: We use autocast for FP16, NOT .half()
        # This allows PyTorch to automatically handle mixed precision
        if self.use_fp16:
            print(f"[SCLIP] Enabled FP16 mixed precision (autocast) for 2x speedup")

        # Apply torch.compile() for JIT optimization (PyTorch 2.0+)
        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"[SCLIP] Enabled torch.compile() for JIT optimization")
            except Exception as e:
                print(f"[SCLIP] Warning: torch.compile() failed: {e}")
                self.use_compile = False

        # Get model dimensions
        self.patch_size = self.model.visual.patch_size
        self.embed_dim = self.model.visual.output_dim

        print(f"[SCLIP] Model loaded successfully")
        print(f"[SCLIP] Patch size: {self.patch_size}, Embedding dim: {self.embed_dim}")

        # Text embedding cache
        self.text_embedding_cache = {}

    def preprocess_without_resize(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CLIP WITHOUT resizing (for SCLIP's high-res approach).

        Applies only: RGB convert + ToTensor + Normalize
        Skips: Resize + CenterCrop

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            Preprocessed tensor (3, H, W)
        """
        # Convert to float32 and normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)

        # Normalize with CLIP's mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor.to(self.device)

    @torch.no_grad()
    def extract_image_features(
        self,
        image: np.ndarray,
        return_dense: bool = True,
        use_csa: bool = True,
        normalize: bool = True,
        preserve_resolution: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract image features using SCLIP's CSA with optimized inference.

        Args:
            image: Input image (RGB numpy array)
            return_dense: Return dense patch features (True for segmentation)
            use_csa: Use Cross-layer Self-Attention (SCLIP's key innovation)
            normalize: Whether to normalize features
            preserve_resolution: If True, don't resize to 224x224 (for SCLIP's high-res sliding window)

        Returns:
            - Global image embedding (D,) if return_dense=False
            - Dense patch features (H, W, D) if return_dense=True
        """
        # Preprocess image
        if preserve_resolution:
            # SCLIP's approach: preserve resolution, just normalize
            image_tensor = self.preprocess_without_resize(image).unsqueeze(0)
        else:
            # Standard CLIP approach: resize to 224x224
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Forward pass with CSA and mixed precision
        # autocast will automatically convert to FP16 where beneficial
        with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
            if return_dense:
                # Get dense features: (batch, num_patches, embed_dim)
                features = self.model.encode_image(image_tensor, return_all=True, csa=use_csa)

                if normalize:
                    features = F.normalize(features, dim=-1)

                # Remove CLS token and reshape to spatial grid
                # features shape: (1, num_patches + 1, embed_dim)
                patch_features = features[:, 1:, :]  # Remove CLS token

                # Calculate grid size
                num_patches = patch_features.shape[1]
                grid_size = int(np.sqrt(num_patches))

                # Reshape to (1, H, W, D)
                patch_features = patch_features.reshape(1, grid_size, grid_size, self.embed_dim)

                # Return as (H, W, D)
                return None, patch_features.squeeze(0)
            else:
                # Get global feature (CLS token only)
                features = self.model.encode_image(image_tensor, return_all=False, csa=use_csa)

                if normalize:
                    features = F.normalize(features, dim=-1)

                return features.squeeze(0), None

    @torch.no_grad()
    def extract_text_features(
        self,
        texts: List[str],
        use_prompt_ensemble: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract text embeddings using SCLIP's approach with optimized inference.

        SCLIP uses 80 ImageNet templates for robust text encoding.

        Args:
            texts: List of text prompts
            use_prompt_ensemble: Use 80 ImageNet templates (SCLIP's approach)
            normalize: Whether to normalize embeddings

        Returns:
            Text embeddings (N, D)
        """
        # Check cache
        cache_key = (tuple(texts), use_prompt_ensemble, normalize)
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', enabled=self.use_fp16):
            if use_prompt_ensemble:
                # SCLIP approach: Average 80 ImageNet templates per class
                all_embeddings = []

                for text in texts:
                    # Generate all templated prompts
                    templated_prompts = [template(text) for template in openai_imagenet_template]

                    # Tokenize all templates
                    tokens = clip.tokenize(templated_prompts).to(self.device)

                    # Encode all templates
                    template_features = self.model.encode_text(tokens)

                    if normalize:
                        template_features = F.normalize(template_features, dim=-1)

                    # Average across templates
                    text_embedding = template_features.mean(dim=0, keepdim=False)

                    if normalize:
                        text_embedding = F.normalize(text_embedding, dim=-1)

                    all_embeddings.append(text_embedding)

                result = torch.stack(all_embeddings, dim=0)
            else:
                # Direct encoding (no templates)
                tokens = clip.tokenize(texts).to(self.device)
                result = self.model.encode_text(tokens)

                if normalize:
                    result = F.normalize(result, dim=-1)

        # Cache result
        self.text_embedding_cache[cache_key] = result

        return result

    @torch.no_grad()
    def compute_dense_similarity(
        self,
        image: np.ndarray,
        texts: List[str],
        use_csa: bool = True,
        preserve_resolution: bool = True
    ) -> torch.Tensor:
        """
        Compute dense pixel-wise similarity between image and text prompts.

        This is SCLIP's core operation for semantic segmentation.

        Args:
            image: Input image (H, W, 3)
            texts: List of class names
            use_csa: Use Cross-layer Self-Attention
            preserve_resolution: Preserve image resolution (SCLIP's approach)

        Returns:
            Dense similarity map (num_classes, H, W)
        """
        # Extract dense image features
        _, dense_features = self.extract_image_features(
            image,
            return_dense=True,
            use_csa=use_csa,
            normalize=True,
            preserve_resolution=preserve_resolution
        )  # Shape: (H, W, D)

        # Extract text features
        text_features = self.extract_text_features(
            texts,
            use_prompt_ensemble=True,
            normalize=True
        )  # Shape: (num_classes, D)

        # Reshape for batch matrix multiplication
        H, W, D = dense_features.shape
        dense_flat = dense_features.reshape(H * W, D)  # (H*W, D)

        # Compute similarity: (H*W, D) @ (D, num_classes) = (H*W, num_classes)
        # Ensure both tensors have the same dtype (important for FP16 mixed precision)
        similarities = dense_flat @ text_features.to(dense_flat.dtype).T

        # Reshape back to spatial: (num_classes, H, W)
        similarities = similarities.T.reshape(len(texts), H, W)

        return similarities
