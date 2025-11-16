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
from typing import List, Tuple, Optional, Dict
from PIL import Image
import cv2

from models.sclip_features import SCLIPFeatureExtractor
from models.pamr import PAMR
from models.sam2_segmentation import SAM2MaskGenerator
from utils.sclip_descriptors import parse_sclip_descriptors, map_logits_to_classes

# Phase 1 improvements (2025)
try:
    from models.resclip_attention import ResCLIPModule
    RESCLIP_AVAILABLE = True
except ImportError:
    RESCLIP_AVAILABLE = False

try:
    from models.densecrf_refine import DenseCRFRefiner
    DENSECRF_AVAILABLE = True
except ImportError:
    DENSECRF_AVAILABLE = False

# Phase 2A improvements (2025 - training-free human parsing)
try:
    from models.cliptrase import CLIPtraseRecalibration
    CLIPTRASE_AVAILABLE = True
except ImportError:
    CLIPTRASE_AVAILABLE = False

try:
    from models.clip_rc import RegionalCluesExtractor
    CLIP_RC_AVAILABLE = True
except ImportError:
    CLIP_RC_AVAILABLE = False


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
        # SCLIP multi-descriptor support
        descriptor_file: Optional[str] = None,  # Path to cls_voc21.txt or similar
        # 2025 optimization parameters
        use_fp16: bool = True,
        use_compile: bool = False,
        batch_prompts: bool = True,
        # Phase 1 improvements (ICCV/CVPR 2025 papers)
        use_loftup: bool = False,  # LoftUp feature upsampling (+2-4% mIoU)
        loftup_mode: str = "fast",  # "fast" (2x upsample) or "accurate" (full SCLIP approach)
        use_resclip: bool = False,  # ResCLIP residual attention (+8-13% mIoU)
        use_densecrf: bool = False,  # DenseCRF boundary refinement (+1-2% mIoU, +3-5% boundary F1)
        # Phase 2A improvements (2025 - training-free for human parsing)
        use_cliptrase: bool = False,  # CLIPtrase self-correlation recalibration (+5-10% mIoU person)
        use_clip_rc: bool = False,  # CLIP-RC regional clues extraction (+8-12% mIoU person)
        # Phase 2B improvements (2025 - prompt engineering)
        template_strategy: str = "imagenet80",  # Prompt template strategy (+2-5% mIoU, 3-4x faster)
        # Phase 2C improvements (2025 - confidence sharpening)
        use_confidence_sharpening: bool = False,  # Sharpen flat predictions (+5-8% mIoU)
        use_hierarchical_prediction: bool = False,  # Group similar classes (+3-5% mIoU)
        # Phase 3 improvements (MHQR - Multi-scale Hierarchical Query-based Refinement)
        use_mhqr: bool = False,  # Enable full MHQR pipeline (+8-15% mIoU expected)
        mhqr_dynamic_queries: bool = True,  # Use dynamic multi-scale query generation
        mhqr_hierarchical_decoder: bool = True,  # Use hierarchical mask decoder
        mhqr_semantic_merging: bool = True,  # Use semantic-guided mask merging
        mhqr_scales: List[float] = None,  # Multi-scale pyramid (default: [0.25, 0.5, 1.0])
    ):
        """
        Initialize SCLIP segmentor with 2025 performance optimizations.

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
            use_fp16: Enable FP16 mixed precision (2025 optimization)
            use_compile: Enable torch.compile() (2025 optimization)
            batch_prompts: Enable batch prompt processing for SAM (2025 optimization)
            use_loftup: Enable LoftUp feature upsampling (Phase 1)
            use_resclip: Enable ResCLIP residual attention (Phase 1)
            use_densecrf: Enable DenseCRF boundary refinement (Phase 1)
            use_cliptrase: Enable CLIPtrase self-correlation recalibration (Phase 2A)
            use_clip_rc: Enable CLIP-RC regional clues extraction (Phase 2A)
            template_strategy: Prompt template strategy (Phase 2B):
                - "imagenet80": Original 80 ImageNet templates (baseline)
                - "top7": Top-7 dense prediction templates (recommended, 3-4x faster, +2-3% mIoU)
                - "spatial": Spatial context templates (+1-2% mIoU)
                - "top3": Ultra-fast top-3 templates (5x faster)
                - "adaptive": Adaptive per-class selection (stuff vs thing, +3-5% mIoU)
            use_confidence_sharpening: Enable confidence sharpening for flat distributions (Phase 2C)
            use_hierarchical_prediction: Enable hierarchical class grouping (Phase 2C)
            use_mhqr: Enable full MHQR pipeline (Phase 3, +8-15% mIoU expected)
            mhqr_dynamic_queries: Use dynamic multi-scale query generation (Phase 3)
            mhqr_hierarchical_decoder: Use hierarchical mask decoder (Phase 3)
            mhqr_semantic_merging: Use semantic-guided mask merging (Phase 3)
            mhqr_scales: Multi-scale pyramid scales (Phase 3)
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
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        self.batch_prompts = batch_prompts
        self.use_loftup = use_loftup
        self.loftup_mode = loftup_mode if use_loftup else None
        self.use_resclip = use_resclip
        self.use_densecrf = use_densecrf
        self.use_cliptrase = use_cliptrase
        self.use_clip_rc = use_clip_rc
        self.template_strategy = template_strategy
        self.use_confidence_sharpening = use_confidence_sharpening
        self.use_hierarchical_prediction = use_hierarchical_prediction

        # Phase 3: MHQR parameters
        self.use_mhqr = use_mhqr
        self.mhqr_dynamic_queries = mhqr_dynamic_queries if use_mhqr else False
        self.mhqr_hierarchical_decoder = mhqr_hierarchical_decoder if use_mhqr else False
        self.mhqr_semantic_merging = mhqr_semantic_merging if use_mhqr else False
        self.mhqr_scales = mhqr_scales if mhqr_scales is not None else [0.25, 0.5, 1.0]

        # Multi-descriptor support (SCLIP's cls_voc21.txt approach)
        self.descriptor_file = descriptor_file
        self.query_words = None
        self.query_idx = None
        self.use_multi_descriptor = descriptor_file is not None

        if self.use_multi_descriptor:
            # Parse descriptor file
            self.query_words, query_idx_list = parse_sclip_descriptors(descriptor_file)
            self.query_idx = torch.tensor(query_idx_list, dtype=torch.long, device=device)
            self.num_classes = max(query_idx_list) + 1
            self.num_descriptors = len(self.query_words)

            if verbose:
                print(f"  Multi-descriptor mode: Loaded {self.num_descriptors} descriptors for {self.num_classes} classes")
                print(f"    Expansion ratio: {self.num_descriptors/self.num_classes:.2f}x")

        # Text feature cache to avoid recomputing for same classes
        self._text_feature_cache = {}

        if verbose:
            print("[SCLIP Segmentor] Initializing...")
            print(f"  Mode: {'Hybrid (SAM + SCLIP)' if use_sam else 'Dense (SCLIP only)'}")
            print(f"  PAMR: {'Enabled' if use_pamr else 'Disabled'}")
            print(f"  Slide inference: {'Enabled' if slide_inference else 'Disabled'}")
            loftup_status = 'Disabled'
            if use_loftup:
                loftup_status = f'Enabled ({loftup_mode} mode)'
            print(f"  Phase 1: LoftUp={loftup_status}, "
                  f"ResCLIP={'Enabled' if use_resclip else 'Disabled'}, "
                  f"DenseCRF={'Enabled' if use_densecrf else 'Disabled'}")
            if use_cliptrase or use_clip_rc:
                print(f"  Phase 2A: CLIPtrase={'Enabled' if use_cliptrase else 'Disabled'}, "
                      f"CLIP-RC={'Enabled' if use_clip_rc else 'Disabled'}")
            print(f"  Phase 2B: Template Strategy={template_strategy}")
            if use_confidence_sharpening or use_hierarchical_prediction:
                print(f"  Phase 2C: Confidence Sharpening={'Enabled' if use_confidence_sharpening else 'Disabled'}, "
                      f"Hierarchical={'Enabled' if use_hierarchical_prediction else 'Disabled'}")
            if use_mhqr:
                print(f"  Phase 3: MHQR Enabled - Dynamic Queries={'Yes' if mhqr_dynamic_queries else 'No'}, "
                      f"Hierarchical Decoder={'Yes' if mhqr_hierarchical_decoder else 'No'}, "
                      f"Semantic Merging={'Yes' if mhqr_semantic_merging else 'No'}")
                print(f"           Scales={mhqr_scales}")

        # Initialize SCLIP feature extractor with optimizations
        # Note: Feature extractor only uses LoftUp in "fast" mode (2x upsampling)
        # "accurate" mode runs LoftUp in sliding window (below)
        use_loftup_in_extractor = use_loftup and loftup_mode == "fast"

        self.clip_extractor = SCLIPFeatureExtractor(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            use_compile=use_compile,
            use_loftup=use_loftup_in_extractor,
            template_strategy=template_strategy,  # Phase 2B: Prompt engineering
        )

        # Initialize LoftUp upsampler for accurate mode (runs in sliding window)
        self.loftup_upsampler = None
        if use_loftup and loftup_mode == "accurate":
            try:
                from models.loftup_upsampler import LoftUpUpsampler
                self.loftup_upsampler = LoftUpUpsampler(
                    model_name="loftup_clip",
                    backbone=model_name,
                    device=device,
                    use_fp16=use_fp16,
                    use_pretrained=True
                )
                if verbose:
                    print(f"  LoftUp (accurate mode): Initialized for sliding window inference")
            except Exception as e:
                if verbose:
                    print(f"  WARNING: LoftUp accurate mode initialization failed: {e}")
                    print(f"  Falling back to fast mode")
                self.loftup_mode = "fast"
                self.use_loftup = False

        # Initialize PAMR if enabled
        if use_pamr and pamr_steps > 0:
            self.pamr = PAMR(num_iter=pamr_steps, dilations=list(pamr_dilations)).to(device)
            if verbose:
                print(f"  PAMR: {pamr_steps} steps, dilations={pamr_dilations}")
        else:
            self.pamr = None

        # Initialize SAM for hybrid mode
        if use_sam:
            self.sam_generator = SAM2MaskGenerator(
                device=device,
                use_fp16=use_fp16,
                use_compile=use_compile,
                batch_prompts=batch_prompts,
            )
            if verbose:
                print("  SAM generator initialized with 2025 optimizations")
        else:
            self.sam_generator = None

        # Initialize separate SAM for MHQR (if needed)
        self.mhqr_sam_generator = None
        if use_mhqr:
            # MHQR uses SAM differently - generates masks at CLIP feature resolution
            # Use large model for best quality (same as clip_guided_sam baseline)
            self.mhqr_sam_generator = SAM2MaskGenerator(
                model_type="sam2_hiera_large",  # Match clip_guided_sam baseline
                device=device,
                use_fp16=use_fp16,
                use_compile=use_compile,
                batch_prompts=batch_prompts,
            )
            if verbose:
                print("  MHQR SAM generator initialized (feature-resolution masking)")

        # Initialize ResCLIP if enabled (Phase 1)
        self.resclip_module = None
        if use_resclip:
            if not RESCLIP_AVAILABLE:
                if verbose:
                    print("  WARNING: ResCLIP requested but not available. Disabling.")
                self.use_resclip = False
            else:
                try:
                    self.resclip_module = ResCLIPModule(
                        use_rcs=True,  # Enable Residual Cross-correlation Self-attention
                        use_sfr=True,  # Enable Semantic Feedback Refinement
                        use_fp16=use_fp16,
                        device=device
                    )
                    if verbose:
                        print(f"  ResCLIP: Enabled (RCS + SFR, +8-13% mIoU expected)")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: ResCLIP initialization failed: {e}")
                    self.use_resclip = False
                    self.resclip_module = None

        # Initialize DenseCRF if enabled (Phase 1)
        self.densecrf_refiner = None
        if use_densecrf:
            if not DENSECRF_AVAILABLE:
                if verbose:
                    print("  WARNING: DenseCRF requested but not available. Disabling.")
                self.use_densecrf = False
            else:
                try:
                    self.densecrf_refiner = DenseCRFRefiner(
                        max_iterations=10,
                        pos_w=3.0,
                        bi_w=10.0
                    )
                    if verbose:
                        print(f"  DenseCRF: Enabled (+1-2% mIoU, +3-5% boundary F1 expected)")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: DenseCRF initialization failed: {e}")
                    self.use_densecrf = False
                    self.densecrf_refiner = None

        # Initialize CLIPtrase if enabled (Phase 2A - training-free human parsing)
        self.cliptrase_module = None
        if use_cliptrase:
            if not CLIPTRASE_AVAILABLE:
                if verbose:
                    print("  WARNING: CLIPtrase requested but not available. Disabling.")
                self.use_cliptrase = False
            else:
                try:
                    self.cliptrase_module = CLIPtraseRecalibration(
                        correlation_temperature=0.05,
                        recalibration_strength=0.5,
                        use_fp16=use_fp16,
                        device=device
                    )
                    if verbose:
                        print(f"  CLIPtrase: Enabled (self-correlation recalibration, +5-10% mIoU person expected)")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: CLIPtrase initialization failed: {e}")
                    self.use_cliptrase = False
                    self.cliptrase_module = None

        # Initialize CLIP-RC if enabled (Phase 2A - training-free human parsing)
        self.clip_rc_module = None
        if use_clip_rc:
            if not CLIP_RC_AVAILABLE:
                if verbose:
                    print("  WARNING: CLIP-RC requested but not available. Disabling.")
                self.use_clip_rc = False
            else:
                try:
                    self.clip_rc_module = RegionalCluesExtractor(
                        num_regions=4,
                        regional_weight=0.6,
                        use_fp16=use_fp16,
                        device=device
                    )
                    if verbose:
                        print(f"  CLIP-RC: Enabled (regional clues extraction, +8-12% mIoU person expected)")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: CLIP-RC initialization failed: {e}")
                    self.use_clip_rc = False
                    self.clip_rc_module = None

        # Initialize Phase 3 MHQR modules
        self.mhqr_query_generator = None
        self.mhqr_mask_decoder = None
        self.mhqr_mask_merger = None

        if use_mhqr:
            # Dynamic Multi-Scale Query Generator (REQUIRED for simplified MHQR)
            if mhqr_dynamic_queries:
                try:
                    from models.dynamic_query_generator import DynamicMultiScaleQueryGenerator
                    self.mhqr_query_generator = DynamicMultiScaleQueryGenerator(
                        scales=self.mhqr_scales,
                        min_queries=10,
                        max_queries=200,
                        use_adaptive_threshold=True,
                        device=device
                    )
                    if verbose:
                        print(f"  MHQR Query Generator: Initialized (adaptive, scales={self.mhqr_scales})")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: MHQR Query Generator failed to initialize: {e}")
                        print(f"  Will use fallback prompt extraction from clip_guided_segmentation")

            # Hierarchical Mask Decoder (OPTIONAL - not used in simplified MHQR)
            if mhqr_hierarchical_decoder:
                try:
                    from models.hierarchical_mask_decoder import HierarchicalMaskDecoder
                    self.mhqr_mask_decoder = HierarchicalMaskDecoder(
                        scales=self.mhqr_scales,
                        embed_dim=256,
                        num_heads=8,
                        residual_weight=0.3,
                        use_fp16=use_fp16,
                        device=device
                    )
                    if verbose:
                        print(f"  MHQR Mask Decoder: Initialized (hierarchical refinement)")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: MHQR Mask Decoder failed: {e} (not needed for simplified MHQR)")
                    self.mhqr_mask_decoder = None

            # Semantic-Guided Mask Merger (OPTIONAL - not used in simplified MHQR)
            if mhqr_semantic_merging:
                try:
                    from models.semantic_mask_merger import SemanticMaskMerger
                    self.mhqr_mask_merger = SemanticMaskMerger(
                        semantic_similarity_threshold=0.7,
                        boundary_refinement=True,
                        iou_threshold=0.3,
                        use_fp16=use_fp16,
                        device=device
                    )
                    if verbose:
                        print(f"  MHQR Mask Merger: Initialized (semantic-aware)")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: MHQR Mask Merger failed: {e} (not needed for simplified MHQR)")
                    self.mhqr_mask_merger = None

            # LoftUp Feature Upsampler (OPTIONAL - not used in simplified MHQR)
            self.mhqr_loftup = None
            try:
                from models.loftup_upsampler import LoftUpUpsampler
                self.mhqr_loftup = LoftUpUpsampler(
                    model_name="loftup_clip",
                    backbone=model_name,
                    device=device,
                    use_fp16=use_fp16,
                    use_pretrained=False  # Use bilinear fallback for now
                )
                if verbose:
                    print(f"  MHQR LoftUp: Initialized (14×14 → 56×56 upsampling)")
            except Exception as e:
                if verbose:
                    print(f"  WARNING: MHQR LoftUp not available: {e} (not needed for simplified MHQR)")
                self.mhqr_loftup = None

            if verbose:
                print(f"  MHQR Pipeline (Simplified): Ready - using dynamic queries + SAM + direct class assignment")

        if verbose:
            print("[SCLIP Segmentor] Ready!\n")

    def _get_text_features(self, class_names: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get text features with caching to avoid recomputing.

        In multi-descriptor mode, returns features for all descriptors.
        Otherwise, returns features for each class.

        Args:
            class_names: List of class names

        Returns:
            tuple: (text_features, query_idx)
                - text_features: (num_descriptors, D) if multi-descriptor, else (num_classes, D)
                - query_idx: Mapping to classes if multi-descriptor, else None
        """
        # Multi-descriptor mode: use query_words instead of class_names
        if self.use_multi_descriptor:
            cache_key = ('multi_descriptor', self.descriptor_file)

            if cache_key not in self._text_feature_cache:
                # Extract features for all descriptors
                text_features = self.clip_extractor.extract_text_features(
                    self.query_words,
                    use_prompt_ensemble=True,
                    template_strategy=self.template_strategy
                )
                self._text_feature_cache[cache_key] = text_features

            return self._text_feature_cache[cache_key], self.query_idx

        # Standard mode: one feature per class
        # Create cache key from class names
        cache_key = tuple(class_names)

        # Return cached features if available
        if cache_key in self._text_feature_cache:
            return self._text_feature_cache[cache_key], None

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

        return text_features, None

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

        # Apply Phase 2C: Confidence Sharpening (before temperature scaling)
        if self.use_confidence_sharpening or self.use_hierarchical_prediction:
            try:
                from prompts.confidence_sharpening import sharpen_predictions
                logits = sharpen_predictions(
                    logits,
                    class_names,
                    use_hierarchical=self.use_hierarchical_prediction,
                    use_calibration=self.use_confidence_sharpening,
                    use_adaptive_temp=False,  # We'll apply temperature manually
                    base_temperature=1.0  # No scaling here
                )
                if self.verbose:
                    print("[Phase 2C] Applied confidence sharpening")
            except Exception as e:
                if self.verbose:
                    print(f"[Phase 2C] Sharpening failed: {e}")

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

        # Apply DenseCRF boundary refinement if enabled (Phase 1)
        if self.use_densecrf and self.densecrf_refiner is not None:
            if self.verbose:
                print("[Phase 1] Applying DenseCRF boundary refinement...")

            # Resize image to match probability map
            image_for_crf = image.copy()
            if image_for_crf.shape[:2] != probs.shape[-2:]:
                image_for_crf = cv2.resize(
                    image_for_crf,
                    (probs.shape[2], probs.shape[1]),  # (W, H)
                    interpolation=cv2.INTER_LINEAR
                )

            # Apply DenseCRF refinement
            try:
                refined_probs = self.densecrf_refiner.refine_torch(
                    image=torch.from_numpy(image_for_crf),
                    probabilities=probs,
                    return_probs=True
                )
                probs = refined_probs
            except Exception as e:
                if self.verbose:
                    print(f"[Phase 1] DenseCRF failed: {e}. Using unrefined probabilities.")

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
        Single forward pass to get dense logits with Phase 1 & Phase 2A enhancements.

        Pipeline:
        1. Extract CLIP features (with CSA from SCLIP)
        2. Apply Phase 2A: CLIPtrase (self-correlation recalibration)
        3. Apply Phase 2A: CLIP-RC (regional clues extraction)
        4. Apply Phase 1: ResCLIP (RCS + SFR)
        5. Compute final similarities

        Returns:
            Logits (num_classes, H_feat, W_feat)
        """
        # Check if we need to extract features explicitly (for Phase 2A or ResCLIP)
        need_explicit_features = (
            self.use_cliptrase or self.use_clip_rc or self.use_resclip
        )

        if need_explicit_features:
            # Extract dense features explicitly
            _, dense_features = self.clip_extractor.extract_image_features(
                image,
                return_dense=True,
                use_csa=True,
                normalize=True,
                preserve_resolution=False
            )  # dense_features: (H, W, D)

            # Apply Phase 2A improvements (training-free human parsing)

            # Step 1: CLIPtrase - Self-correlation recalibration
            if self.use_cliptrase and self.cliptrase_module is not None:
                dense_features = self.cliptrase_module.forward(dense_features)

            # Step 2: CLIP-RC - Regional clues extraction
            if self.use_clip_rc and self.clip_rc_module is not None:
                dense_features = self.clip_rc_module.extract_regional_features(dense_features)

            # Apply Phase 1: ResCLIP if enabled
            if self.use_resclip and self.resclip_module is not None:
                # Apply RCS (Residual Cross-correlation Self-attention)
                enhanced_features = self.resclip_module.enhance_features(
                    dense_features,
                    residual_weight=0.3
                )  # (H, W, D)

                # Get text features
                text_features, query_idx = self._get_text_features(class_names)

                # Apply SFR (Semantic Feedback Refinement)
                # This computes multi-scale similarity maps
                similarities = self.resclip_module.refine_predictions(
                    enhanced_features,
                    text_features,
                    original_size=image.shape[:2]
                )  # (num_descriptors or num_classes, H, W)

                # Map multi-descriptor logits to classes if needed
                if query_idx is not None:
                    num_classes = len(class_names)
                    similarities = map_logits_to_classes(similarities, query_idx, num_classes)
            else:
                # Compute similarities directly from Phase 2A enhanced features
                text_features, query_idx = self._get_text_features(class_names)

                # Reshape features for similarity computation
                H, W, D = dense_features.shape
                features_flat = dense_features.reshape(H * W, D)
                features_norm = F.normalize(features_flat, dim=-1)
                text_norm = F.normalize(text_features, dim=-1)

                # Compute similarities
                similarities = features_norm @ text_norm.to(features_norm.dtype).T

                # Reshape based on multi-descriptor mode
                if query_idx is not None:
                    # Multi-descriptor: similarities shape is (H*W, num_descriptors)
                    num_descriptors = text_features.shape[0]
                    similarities = similarities.T.reshape(num_descriptors, H, W)
                    # Map to classes using max pooling
                    num_classes = len(class_names)
                    similarities = map_logits_to_classes(similarities, query_idx, num_classes)
                else:
                    # Standard: similarities shape is (H*W, num_classes)
                    similarities = similarities.T.reshape(len(class_names), H, W)
        else:
            # Standard SCLIP approach (no Phase 1 or Phase 2A)
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
        text_features, query_idx = self._get_text_features(class_names)

        # Determine number of outputs (descriptors in multi-descriptor mode, else classes)
        num_outputs = text_features.shape[0]  # num_descriptors or num_classes
        is_multi_descriptor = query_idx is not None

        # Initialize accumulators on CPU to save GPU memory
        logits_sum = torch.zeros((1, num_outputs, H, W), dtype=torch.float32)
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
                # Use autocast if FP16 is enabled for speedup
                with torch.amp.autocast(device_type='cuda', enabled=self.clip_extractor.use_fp16):
                    features = self.clip_extractor.model.encode_image(
                        crop_tensor,
                        return_all=True,
                        csa=True
                    )  # (1, num_patches+1, D)

                # Remove CLS token and normalize
                features = features[:, 1:]  # (1, num_patches, D)
                features = features / features.norm(dim=-1, keepdim=True)

                # Get crop dimensions and patch info
                patch_size = self.clip_extractor.model.visual.patch_size
                crop_h, crop_w = y2 - y1, x2 - x1
                h_patches = crop_h // patch_size
                w_patches = crop_w // patch_size

                # SCLIP's accurate LoftUp mode: Upsample features to full crop size BEFORE similarity
                if self.loftup_mode == "accurate" and self.loftup_upsampler is not None and crop_h == crop_size and crop_w == crop_size:
                    # Reshape features to spatial format for LoftUp
                    features_spatial = features.permute(0, 2, 1).reshape(1, -1, h_patches, w_patches)

                    # Apply LoftUp upsampling to FULL crop size (14x14 -> 224x224)
                    hr_features = self.loftup_upsampler(
                        features_spatial,
                        target_size=(crop_h, crop_w),
                        original_image=crop_tensor
                    )  # (1, D, crop_h, crop_w)

                    # Normalize upsampled features
                    hr_features = hr_features / hr_features.norm(dim=1, keepdim=True)

                    # Reshape to (H*W, D) for similarity computation
                    hr_features_flat = hr_features.flatten(2).permute(0, 2, 1)  # (1, H*W, D)

                    # Compute similarity at HIGH resolution
                    crop_logits = hr_features_flat @ text_features.to(hr_features.dtype).T  # (1, H*W, num_outputs)

                    # Reshape to spatial
                    crop_logits = crop_logits.permute(0, 2, 1).reshape(1, num_outputs, crop_h, crop_w)

                else:
                    # Standard or fast mode: Compute similarity at low res, then upsample
                    crop_logits = features @ text_features.to(features.dtype).T  # (1, num_patches, num_outputs)

                    # Reshape to spatial: (1, num_outputs, h_patches, w_patches)
                    crop_logits = crop_logits.permute(0, 2, 1).reshape(1, num_outputs, h_patches, w_patches)

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
        logits = logits_sum / count_mat  # (1, num_outputs, H, W)

        # Map multi-descriptor logits to classes if needed
        if is_multi_descriptor:
            logits = logits.squeeze(0)  # (num_outputs, H, W)
            logits = map_logits_to_classes(logits, query_idx, num_classes)
            return logits.to(self.device)  # (num_classes, H, W)
        else:
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

    def predict_with_mhqr(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> np.ndarray:
        """
        MHQR: Multi-scale Hierarchical Query-based Refinement (SIMPLIFIED).

        This is the CORRECT implementation that mirrors the working clip_guided_sam approach:
        1. Dense CLIP prediction
        2. Dynamic query generation (ONLY improvement over baseline)
        3. SAM at each query → best mask
        4. CRITICAL: Assign CLIP class from query location
        5. Merge overlaps (same class, high IoU threshold)
        6. Paint final map (confidence sorted)

        Args:
            image: Input image (H, W, 3)
            class_names: List of class names

        Returns:
            Segmentation mask (H, W) with class indices
        """
        H, W = image.shape[:2]
        num_classes = len(class_names)

        if self.verbose:
            print("[MHQR Pipeline] Starting (simplified, correct approach)...")

        # Step 1: Dense SCLIP prediction
        dense_pred, logits = self.predict_dense(image, class_names, return_logits=True)
        probs = torch.softmax(logits * self.logit_scale, dim=0)  # (K, H, W)

        if self.verbose:
            print(f"  Dense prediction: {H}×{W}, {num_classes} classes")

        # Step 2: Dynamic query generation (THE ONLY MHQR INNOVATION)
        if self.mhqr_query_generator is not None:
            query_result = self.mhqr_query_generator.generate_queries(
                confidence_maps=probs.permute(1, 2, 0),  # (H, W, K)
                class_names=class_names,
                image_size=(H, W),
                return_metadata=True
            )

            point_coords = query_result['point_coords']
            point_labels = query_result['point_labels']
            point_classes = query_result['point_classes']

            if self.verbose:
                print(f"  Dynamic queries: {len(point_coords)} points")
                if 'metadata' in query_result:
                    meta = query_result['metadata']
                    print(f"    Per-scale: {meta.get('queries_per_scale', {})}")
        else:
            # Fallback: extract prompts like clip_guided_sam does
            from clip_guided_segmentation import extract_prompt_points_from_clip
            probs_np = probs.permute(1, 2, 0).cpu().numpy()  # (H, W, K)
            prompts = extract_prompt_points_from_clip(
                dense_pred, probs_np, class_names,
                min_confidence=0.7,
                min_region_size=100
            )

            point_coords = np.array([p['point'] for p in prompts])
            point_labels = np.ones(len(prompts))
            point_classes = np.array([p['class_idx'] for p in prompts])

            if self.verbose:
                print(f"  Fallback queries: {len(point_coords)} points")

        if len(point_coords) == 0:
            if self.verbose:
                print("  No queries, returning dense prediction")
            return dense_pred

        # Step 3: SAM masks at each query (EXACTLY like clip_guided_sam)
        if self.mhqr_sam_generator is None:
            if self.verbose:
                print("  No SAM, returning dense prediction")
            return dense_pred

        # Use the proper segment_with_points API (returns MaskCandidate objects)
        try:
            points_list = [(int(x), int(y)) for x, y in point_coords]
            labels_list = [1] * len(points_list)  # All foreground points

            if self.verbose:
                print(f"  Prompting SAM with {len(points_list)} points...")

            # Get all masks at once (batch processing, more efficient)
            mask_candidates = self.mhqr_sam_generator.segment_with_points(
                image=image,
                points=points_list,
                point_labels=labels_list
            )

            if self.verbose:
                print(f"    [DEBUG] Received {len(mask_candidates)} MaskCandidates from SAM (expected {len(points_list) * 3})")

            # segment_with_points returns 3 MaskCandidate objects per point
            # BUT they're sorted by IoU, not in point order!
            # We need to group them by point_coords and pick the best per point
            results = []

            for i, (x, y) in enumerate(points_list):
                # Find all masks for this specific point by matching coordinates
                # Each mask has point_coords attribute: np.array([[x, y]])
                point_masks = []
                for mask_cand in mask_candidates:
                    if mask_cand.point_coords is not None:
                        px, py = mask_cand.point_coords[0]
                        if abs(px - x) < 1 and abs(py - y) < 1:  # Allow small floating point diff
                            point_masks.append(mask_cand)

                if len(point_masks) == 0:
                    if self.verbose and i < 3:
                        print(f"      [WARNING] No masks found for point {i} at ({x}, {y})")
                    continue

                # Pick mask with highest predicted_iou for this point
                best_mask = max(point_masks, key=lambda m: m.predicted_iou)

                # Get class assignment from query generator
                class_idx = point_classes[i]

                results.append({
                    'mask': best_mask.mask.astype(bool),
                    'class_idx': int(class_idx),
                    'score': float(best_mask.predicted_iou),
                    'confidence': float(probs[class_idx, y, x].cpu())
                })

            if self.verbose:
                print(f"    [DEBUG] Matched {len(results)} points to their best masks (expected {len(points_list)})")

        except Exception as e:
            if self.verbose:
                print(f"  WARNING: SAM point prompting failed: {e}")
                print("  Falling back to dense prediction")
            return dense_pred

        if len(results) == 0:
            if self.verbose:
                print("  No valid masks, returning dense prediction")
            return dense_pred

        if self.verbose:
            print(f"  Generated {len(results)} SAM masks")

        # Step 4: Merge overlapping masks (SAME class only, like clip_guided_sam)
        # Sort by confidence (keep highest)
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)

        kept_results = []
        iou_threshold = 0.8  # Only merge very high overlap

        for result in results:
            mask = result['mask']
            class_idx = result['class_idx']

            should_keep = True
            for kept in kept_results:
                if kept['class_idx'] != class_idx:
                    continue  # Only check same class

                # Calculate IoU
                kept_mask = kept['mask']
                intersection = (mask & kept_mask).sum()
                union = (mask | kept_mask).sum()
                iou = intersection / union if union > 0 else 0

                if iou > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept_results.append(result)

        if self.verbose:
            print(f"  After merging: {len(kept_results)} masks (removed {len(results) - len(kept_results)} overlaps)")

        # Step 5: Paint final segmentation (EXACTLY like clip_guided_sam)
        final_seg = np.zeros((H, W), dtype=np.int32)

        # Sort by confidence (lower first, so higher overwrites)
        kept_results = sorted(kept_results, key=lambda x: x['confidence'])

        for result in kept_results:
            mask = result['mask'].astype(bool)
            class_idx = result['class_idx']
            final_seg[mask] = class_idx

        if self.verbose:
            unique = np.unique(final_seg)
            print(f"[MHQR] Complete! Detected {len(unique)} classes")

        return final_seg

    def segment(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> np.ndarray:
        """
        Main segmentation interface.

        Automatically chooses between dense, hybrid, or MHQR mode based on settings.

        Args:
            image: Input image (H, W, 3)
            class_names: List of class names to predict

        Returns:
            Segmentation mask (H, W) with class indices
        """
        if self.use_mhqr:
            return self.predict_with_mhqr(image, class_names)
        elif self.use_sam:
            return self.predict_with_sam(image, class_names)
        else:
            pred_mask, _ = self.predict_dense(image, class_names)
            return pred_mask

    def _extract_all_points_simple(
        self,
        dense_pred: np.ndarray,
        class_names: List[str],
        num_points_per_class: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple fallback for query generation when MHQR query generator is not available.

        Args:
            dense_pred: (H, W) dense segmentation prediction
            class_names: List of class names
            num_points_per_class: Points to extract per class

        Returns:
            (point_coords, point_labels, point_classes) tuple
        """
        all_points = []
        all_labels = []
        all_classes = []

        for class_idx in range(len(class_names)):
            if class_idx == 0:  # Skip background
                continue

            # Extract points for this class
            class_points = self._extract_prompt_points(
                dense_pred,
                class_idx,
                num_points=num_points_per_class
            )

            if class_points:
                all_points.extend(class_points)
                all_labels.extend([1] * len(class_points))  # All foreground
                all_classes.extend([class_idx] * len(class_points))

        return (
            np.array(all_points, dtype=np.float32) if all_points else np.array([]),
            np.array(all_labels, dtype=np.int32) if all_labels else np.array([]),
            np.array(all_classes, dtype=np.int32) if all_classes else np.array([])
        )

    def _build_clip_features_pyramid(
        self,
        clip_features: torch.Tensor,
        scales: List[float]
    ) -> Dict[float, torch.Tensor]:
        """
        Build multi-scale CLIP feature pyramid.

        Args:
            clip_features: (H, W, D) dense CLIP features
            scales: List of scale factors

        Returns:
            Dict mapping scale to features at that scale
        """
        H, W, D = clip_features.shape
        pyramid = {}

        for scale in scales:
            H_s = int(H * scale)
            W_s = int(W * scale)

            if H_s == H and W_s == W:
                pyramid[scale] = clip_features
            else:
                # Resize features
                features_transposed = clip_features.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
                features_resized = F.interpolate(
                    features_transposed,
                    size=(H_s, W_s),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # (H_s, W_s, D)

                pyramid[scale] = features_resized

        return pyramid
