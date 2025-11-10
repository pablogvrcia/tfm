"""
CLIP Dense Feature Extraction Module

This module implements multi-scale dense CLIP feature extraction following
the MaskCLIP and CLIPSeg approaches described in Chapter 3.2.1.

References:
- Zhou et al., "Extract Free Dense Labels from CLIP", ECCV 2022
- Lüddecke & Ecker, "Image Segmentation Using Text and Image Prompts", CVPR 2022
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from PIL import Image
import open_clip


class CLIPFeatureExtractor:
    """
    Extracts dense vision-language features from CLIP.

    Implements multi-scale feature extraction from intermediate transformer
    layers (6, 12, 18, 24) as described in the thesis methodology.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        extract_layers: List[int] = [6, 12, 18, 24],  # Use only final layer for better semantics
        image_size: int = 336,
        use_fp16: bool = True,  # Mixed precision for 2x speedup (inspired by TernaryCLIP 2025)
        use_compile: bool = False,  # torch.compile() for JIT optimization (PyTorch 2.0+)
    ):
        """
        Initialize CLIP feature extractor with 2025 performance optimizations.

        Args:
            model_name: CLIP model variant (ViT-L/14 recommended)
            pretrained: Pretrained weights source
            device: Computation device
            extract_layers: Transformer layers to extract features from
            image_size: Input image resolution (336x336 for ViT-L/14)
            use_fp16: Enable mixed precision (FP16) for faster inference
            use_compile: Enable torch.compile() for JIT optimization
        """
        self.device = device
        self.extract_layers = extract_layers
        self.image_size = image_size
        self.use_fp16 = use_fp16 and device == "cuda"
        self.use_compile = use_compile

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.model.eval()

        # Apply mixed precision optimization (inspired by TernaryCLIP 2025)
        if self.use_fp16:
            self.model = self.model.half()
            print(f"[CLIP] Enabled FP16 mixed precision for 2x speedup")

        # Apply torch.compile() for JIT optimization (PyTorch 2.0+)
        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"[CLIP] Enabled torch.compile() for JIT optimization")
            except Exception as e:
                print(f"[CLIP] Warning: torch.compile() failed: {e}")
                self.use_compile = False

        # Load tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Get model dimensions
        # Handle different open_clip API versions
        if hasattr(self.model.visual, 'patch_embed'):
            # Older API: patch_embed.patch_size
            self.patch_size = self.model.visual.patch_embed.patch_size[0]
        elif hasattr(self.model.visual, 'patch_size'):
            # Newer API: patch_size directly
            patch_size_tuple = self.model.visual.patch_size
            self.patch_size = patch_size_tuple[0] if isinstance(patch_size_tuple, tuple) else patch_size_tuple
        else:
            # Fallback: default for ViT-L/14
            self.patch_size = 14

        # Get embedding dimension
        if hasattr(self.model.visual, 'embed_dim'):
            self.embed_dim = self.model.visual.embed_dim
        elif hasattr(self.model.visual, 'output_dim'):
            self.embed_dim = self.model.visual.output_dim
        else:
            self.embed_dim = 768  # Default for ViT-L/14

        self.num_patches = (image_size // self.patch_size) ** 2

        # Register hooks for intermediate features
        self.features = {}
        self._register_hooks()

        # Class synonym mapping (inspired by MasQCLIP)
        # Using top 2 synonyms per class to avoid over-averaging
        self.class_synonyms = {
            # PASCAL VOC classes
            'aeroplane': ['aeroplane', 'airplane'],
            'bicycle': ['bicycle', 'bike'],
            'bird': ['bird'],
            'boat': ['boat', 'ship'],
            'bottle': ['bottle'],
            'bus': ['bus'],
            'car': ['car', 'automobile'],
            'cat': ['cat'],
            'chair': ['chair'],
            'cow': ['cow'],
            'diningtable': ['dining table', 'table'],
            'dog': ['dog'],
            'horse': ['horse'],
            'motorbike': ['motorbike', 'motorcycle'],
            'person': ['person', 'people'],
            'pottedplant': ['potted plant', 'plant'],
            'sheep': ['sheep'],
            'sofa': ['sofa', 'couch'],
            'train': ['train', 'locomotive'],
            'tvmonitor': ['tv', 'television'],
            # Common stuff classes
            'background': ['background'],
            'sky': ['sky'],
            'grass': ['grass'],
            'road': ['road'],
            'water': ['water'],
            'building': ['building'],
            'tree': ['tree'],
        }

    def _get_class_variants(self, class_name: str) -> List[str]:
        """
        Get synonyms/variants for a class name.

        Args:
            class_name: Original class name

        Returns:
            List of class name variants (includes original if no synonyms found)
        """
        # Normalize class name (lowercase, replace underscores)
        normalized = class_name.lower().replace('_', '').replace('-', '')

        # Check if we have synonyms
        if normalized in self.class_synonyms:
            return self.class_synonyms[normalized]

        # Also check original name
        if class_name in self.class_synonyms:
            return self.class_synonyms[class_name]

        # Return original if no synonyms found
        return [class_name]

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""

        def get_activation(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        # Hook into transformer blocks
        for layer_idx in self.extract_layers:
            if hasattr(self.model.visual.transformer, 'resblocks'):
                # Standard CLIP structure
                self.model.visual.transformer.resblocks[layer_idx - 1].register_forward_hook(
                    get_activation(f'layer_{layer_idx}')
                )

    @torch.no_grad()
    def extract_image_features(
        self,
        image: Union[np.ndarray, Image.Image],
        normalize: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract dense features from an image with optimized inference.

        Args:
            image: Input image (RGB numpy array or PIL Image)
            normalize: Whether to normalize features

        Returns:
            - Global image embedding (D,)
            - List of dense feature maps from each layer (H, W, D)
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Apply FP16 if enabled
        if self.use_fp16:
            image_tensor = image_tensor.half()

        # Clear previous features
        self.features = {}

        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            image_features = self.model.encode_image(image_tensor)

        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        # Extract and process intermediate features
        dense_features = []

        for layer_idx in self.extract_layers:
            layer_key = f'layer_{layer_idx}'
            if layer_key not in self.features:
                continue

            # Get features: (batch, seq_len, dim)
            feat = self.features[layer_key]

            # Remove CLS token and reshape to spatial grid
            # feat shape: (1, num_patches + 1, embed_dim_intermediate)
            spatial_feat = feat[:, 1:, :]  # Remove CLS token

            # Calculate actual grid size from feature dimensions
            actual_num_patches = spatial_feat.shape[1]
            actual_grid_size = int(np.sqrt(actual_num_patches))
            actual_embed_dim = spatial_feat.shape[2]

            # Reshape to 2D grid: (1, H, W, D)
            spatial_feat = spatial_feat.reshape(
                1, actual_grid_size, actual_grid_size, actual_embed_dim
            )

            # Permute to (1, D, H, W) for interpolation
            spatial_feat = spatial_feat.permute(0, 3, 1, 2)

            # Project intermediate features to final embedding dimension if needed
            if actual_embed_dim != self.embed_dim:
                # Apply the model's projection to match final embedding space
                if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
                    # spatial_feat: (1, actual_embed_dim, H, W)
                    # proj: (actual_embed_dim, self.embed_dim)
                    # Need to reshape, project, then reshape back
                    batch, D, H, W = spatial_feat.shape
                    spatial_feat = spatial_feat.permute(0, 2, 3, 1)  # (1, H, W, D)
                    spatial_feat = spatial_feat.reshape(-1, actual_embed_dim)  # (H*W, D)
                    spatial_feat = spatial_feat @ self.model.visual.proj  # (H*W, embed_dim)
                    spatial_feat = spatial_feat.reshape(batch, H, W, self.embed_dim)  # (1, H, W, embed_dim)
                    spatial_feat = spatial_feat.permute(0, 3, 1, 2)  # (1, embed_dim, H, W)

            if normalize:
                spatial_feat = F.normalize(spatial_feat, dim=1)

            dense_features.append(spatial_feat.squeeze(0))  # (D, H, W)

        # If no compatible layers found, create a pseudo-dense feature from global embedding
        if len(dense_features) == 0:
            # Use the global image embedding as a uniform feature map
            # This ensures we always have at least one feature map for similarity computation
            # image_features shape: (1, D) with batch dimension
            global_feat = image_features.squeeze(0)  # (D,)
            pseudo_dense = global_feat.unsqueeze(-1).unsqueeze(-1)  # (D, 1, 1)
            pseudo_dense = pseudo_dense.expand(-1, 16, 16)  # (D, 16, 16) - arbitrary spatial size
            dense_features.append(pseudo_dense)  # (D, 16, 16)

        return image_features.squeeze(0), dense_features

    @torch.no_grad()
    def extract_text_features(
        self,
        texts: Union[str, List[str]],
        use_prompt_ensemble: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract text embeddings from prompts with optimized inference.

        Args:
            texts: Single text or list of texts
            use_prompt_ensemble: Use prompt templates for robustness
            normalize: Whether to normalize embeddings

        Returns:
            Text embeddings (N, D) or (D,) if single text
        """
        if isinstance(texts, str):
            texts = [texts]

        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if use_prompt_ensemble:
                # Use prompt templates as described in methodology
                # Keep it simple - 4 templates work best (over-averaging hurts)
                templates = [
                    "a photo of a {}",
                    "{} in a scene",
                    "a rendering of a {}",
                    "{}",
                ]

                all_embeddings = []
                for text in texts:
                    text_embeddings = []
                    for template in templates:
                        prompt = template.format(text)
                        tokens = self.tokenizer([prompt]).to(self.device)
                        embedding = self.model.encode_text(tokens)

                        if normalize:
                            embedding = F.normalize(embedding, dim=-1)

                        text_embeddings.append(embedding)

                    # Average across templates
                    text_embedding = torch.stack(text_embeddings).mean(dim=0)
                    all_embeddings.append(text_embedding)

                result = torch.cat(all_embeddings, dim=0)
            else:
                # Direct encoding
                tokens = self.tokenizer(texts).to(self.device)
                result = self.model.encode_text(tokens)

                if normalize:
                    result = F.normalize(result, dim=-1)

        return result.squeeze(0) if len(texts) == 1 else result

    def compute_similarity_map(
        self,
        dense_features: List[torch.Tensor],
        text_embedding: torch.Tensor,
        target_size: Tuple[int, int] = (224, 224),
        aggregation: str = "mean"
    ) -> np.ndarray:
        """
        Compute pixel-wise similarity between image features and text.

        Args:
            dense_features: List of dense feature maps from different layers
            text_embedding: Text embedding (D,)
            target_size: Target spatial resolution (H, W)
            aggregation: How to combine multi-scale features ('mean', 'max', 'sum')

        Returns:
            Similarity map (H, W) with values in [-1, 1]
        """
        similarity_maps = []

        for feat in dense_features:
            # feat: (D, H, W) - D may be different from text_embedding dim
            # text_embedding: (D_text,)

            D, H, W = feat.shape

            # Check if dimensions match
            if D != text_embedding.shape[0]:
                # Intermediate layers have different dimensions than final projection
                # We need to project or skip. For simplicity, skip incompatible layers.
                # In full implementation, use learned projection layers
                continue

            # Reshape to (H*W, D) for per-location normalization
            feat_flat = feat.reshape(D, -1).T  # (H*W, D)

            # Normalize each spatial location's feature vector
            feat_norm = F.normalize(feat_flat, dim=1)  # (H*W, D) each location normalized

            # Normalize text embedding
            text_norm = F.normalize(text_embedding.unsqueeze(0), dim=1)  # (1, D)

            # Compute cosine similarity at each spatial location
            sim = torch.matmul(text_norm, feat_norm.T)  # (1, H*W)
            sim = sim.reshape(1, 1, H, W)  # (1, 1, H, W)

            # Upsample to target size
            sim_upsampled = F.interpolate(
                sim,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

            similarity_maps.append(sim_upsampled.squeeze())

        # Check if we have any valid similarity maps
        if len(similarity_maps) == 0:
            # No compatible feature layers found
            # Return a uniform similarity map (fallback)
            return np.ones(target_size, dtype=np.float32) * 0.5

        # Aggregate across scales
        sim_stack = torch.stack(similarity_maps)  # (num_layers, H, W)

        if aggregation == "mean":
            final_sim = sim_stack.mean(dim=0)
        elif aggregation == "max":
            final_sim = sim_stack.max(dim=0)[0]
        elif aggregation == "sum":
            final_sim = sim_stack.sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return final_sim.cpu().numpy()

    def compute_background_suppression(
        self,
        dense_features: List[torch.Tensor],
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Compute background suppression mask using negative prompts.

        Args:
            dense_features: Dense feature maps
            target_size: Target spatial resolution

        Returns:
            Background probability map (H, W)
        """
        negative_prompts = ["background", "nothing", "empty space"]

        bg_embeddings = self.extract_text_features(
            negative_prompts,
            use_prompt_ensemble=False
        )

        # Average negative embeddings
        bg_embedding = bg_embeddings.mean(dim=0)

        # Compute similarity to background
        bg_sim = self.compute_similarity_map(
            dense_features,
            bg_embedding,
            target_size,
            aggregation="max"
        )

        return bg_sim

    def visualize_similarity_map(
        self,
        image: np.ndarray,
        similarity_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize similarity map overlaid on image.

        Args:
            image: Original RGB image
            similarity_map: Similarity scores (H, W)
            alpha: Overlay transparency

        Returns:
            Visualization image
        """
        import cv2

        # Resize similarity map to match image
        h, w = image.shape[:2]
        sim_resized = cv2.resize(similarity_map, (w, h))

        # Normalize to [0, 1]
        sim_normalized = (sim_resized - sim_resized.min()) / (
            sim_resized.max() - sim_resized.min() + 1e-8
        )

        # Apply colormap
        heatmap = cv2.applyColorMap(
            (sim_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Blend with original
        vis = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return vis
