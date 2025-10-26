"""
Open-Vocabulary Semantic Segmentation and Editing Pipeline

This is the main pipeline that integrates all components:
1. SAM 2 mask generation (Chapter 3.2.2)
2. CLIP feature extraction (Chapter 3.2.1)
3. Mask-text alignment (Chapter 3.2.3)
4. Stable Diffusion inpainting (Chapter 3.2.4)

Usage:
    pipeline = OpenVocabSegmentationPipeline()
    result = pipeline.segment_and_edit(image, "red car", "remove")
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

from models.sam2_segmentation import SAM2MaskGenerator, MaskCandidate
from models.clip_features import CLIPFeatureExtractor
from models.mask_alignment import MaskTextAligner, ScoredMask
from models.inpainting import StableDiffusionInpainter


@dataclass
class PipelineResult:
    """Results from the complete pipeline."""
    original_image: np.ndarray
    segmentation_masks: List[ScoredMask]
    edited_image: Optional[Image.Image]
    timing: Dict[str, float]
    visualization_data: Optional[Dict] = None


class OpenVocabSegmentationPipeline:
    """
    Complete open-vocabulary segmentation and editing pipeline.

    Implements the methodology described in Chapter 3:
    - Automatic mask generation with SAM 2
    - Dense CLIP feature extraction
    - Mask-text alignment with scoring
    - Optional generative inpainting
    """

    def __init__(
        self,
        sam_model: str = "sam2_hiera_tiny",
        clip_model: str = "ViT-L-14",
        sd_model: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        """
        Initialize the complete pipeline.

        Args:
            sam_model: SAM 2 model variant
            clip_model: CLIP model variant
            sd_model: Stable Diffusion model
            device: Computation device
            verbose: Print progress information
        """
        self.device = device
        self.verbose = verbose
        self.sd_model = sd_model  # Store for lazy loading

        if verbose:
            print("Initializing Open-Vocabulary Segmentation Pipeline...")
            print(f"Device: {device}")

        # Initialize core components (always needed)
        if verbose:
            print("  [1/3] Loading SAM 2...")
        self.sam_generator = SAM2MaskGenerator(
            model_type=sam_model,
            device=device
        )

        if verbose:
            print("  [2/3] Loading CLIP...")
        self.clip_extractor = CLIPFeatureExtractor(
            model_name=clip_model,
            device=device
        )

        if verbose:
            print("  [3/3] Initializing mask aligner...")
        self.mask_aligner = MaskTextAligner(
            clip_extractor=self.clip_extractor
        )

        # Lazy-load inpainter (only when needed for editing)
        self._inpainter = None

        if verbose:
            print("Pipeline ready!\n")

    @property
    def inpainter(self):
        """Lazy-load the inpainter only when needed."""
        if self._inpainter is None:
            if self.verbose:
                print("  [Lazy-loading] Loading Stable Diffusion for editing...")
            self._inpainter = StableDiffusionInpainter(
                model_id=self.sd_model,
                device=self.device
            )
            if self.verbose:
                print("  Stable Diffusion ready!\n")
        return self._inpainter

    def segment(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_prompt: str,
        top_k: int = 5,
        min_mask_area: int = 1024,
        return_visualization: bool = True
    ) -> PipelineResult:
        """
        Segment objects based on text prompt.

        Args:
            image: Input image (array, PIL, or path)
            text_prompt: Text description of target object
            top_k: Number of top matches to return
            min_mask_area: Minimum mask size (32x32 = 1024 pixels)
            return_visualization: Include visualization data

        Returns:
            PipelineResult with segmentation masks and timing
        """
        timing = {}

        # Load image
        image_array = self._load_image(image)

        # Stage 1: Generate masks with SAM 2
        if self.verbose:
            print(f"Segmenting '{text_prompt}'...")
            print("  Stage 1: Generating masks with SAM 2...")

        t0 = time.time()
        all_masks = self.sam_generator.generate_masks(image_array)
        timing['sam2_generation'] = time.time() - t0

        if self.verbose:
            print(f"    Generated {len(all_masks)} mask candidates ({timing['sam2_generation']:.2f}s)")

        # Filter by size
        filtered_masks = self.sam_generator.filter_by_size(
            all_masks,
            min_area=min_mask_area
        )

        if self.verbose:
            print(f"    Filtered to {len(filtered_masks)} masks (min area: {min_mask_area})")

        # Stage 2 & 3: CLIP feature extraction + Mask-text alignment
        if self.verbose:
            print("  Stage 2-3: Aligning masks with text prompt...")

        t0 = time.time()
        scored_masks, vis_data = self.mask_aligner.align_masks_with_text(
            filtered_masks,
            text_prompt,
            image_array,
            top_k=top_k,
            return_similarity_maps=return_visualization
        )
        timing['clip_alignment'] = time.time() - t0

        if self.verbose:
            print(f"    Found {len(scored_masks)} matches ({timing['clip_alignment']:.2f}s)")
            for i, sm in enumerate(scored_masks[:3], 1):
                print(f"      #{i}: score={sm.final_score:.3f}, area={sm.mask_candidate.area}")

        # Prepare result
        result = PipelineResult(
            original_image=image_array,
            segmentation_masks=scored_masks,
            edited_image=None,
            timing=timing,
            visualization_data=vis_data if return_visualization else None
        )

        return result

    def segment_and_edit(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_prompt: str,
        edit_operation: str,
        edit_prompt: Optional[str] = None,
        top_k: int = 1,
        return_visualization: bool = True
    ) -> PipelineResult:
        """
        Segment and edit image based on text prompts.

        Args:
            image: Input image
            text_prompt: Description of object to segment
            edit_operation: One of ['remove', 'replace', 'style']
            edit_prompt: For 'replace' and 'style', describe the edit
            top_k: Number of masks to process (usually 1 for editing)
            return_visualization: Include visualization data

        Returns:
            PipelineResult with segmentation and edited image
        """
        # First, segment
        result = self.segment(
            image,
            text_prompt,
            top_k=top_k,
            return_visualization=return_visualization
        )

        if not result.segmentation_masks:
            if self.verbose:
                print("  No masks found! Cannot perform editing.")
            return result

        # Stage 4: Inpainting
        if self.verbose:
            print(f"  Stage 4: Inpainting ({edit_operation})...")

        t0 = time.time()

        # Get top mask
        top_mask = result.segmentation_masks[0]
        mask_array = top_mask.mask_candidate.mask

        # Perform inpainting based on operation
        if edit_operation == "remove":
            edited = self.inpainter.remove_object(
                result.original_image,
                mask_array
            )
        elif edit_operation == "replace":
            if edit_prompt is None:
                raise ValueError("edit_prompt required for 'replace' operation")
            edited = self.inpainter.replace_object(
                result.original_image,
                mask_array,
                edit_prompt
            )
        elif edit_operation == "style":
            if edit_prompt is None:
                raise ValueError("edit_prompt required for 'style' operation")
            edited = self.inpainter.style_transfer(
                result.original_image,
                mask_array,
                edit_prompt
            )
        else:
            raise ValueError(f"Unknown operation: {edit_operation}")

        result.timing['inpainting'] = time.time() - t0
        result.edited_image = edited

        if self.verbose:
            print(f"    Inpainting complete ({result.timing['inpainting']:.2f}s)")
            total_time = sum(result.timing.values())
            print(f"\n  Total pipeline time: {total_time:.2f}s")

        return result

    def visualize_results(
        self,
        result: PipelineResult,
        show_similarity: bool = True,
        show_masks: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Create comprehensive visualizations of results.

        Args:
            result: Pipeline result
            show_similarity: Include similarity maps
            show_masks: Include mask overlays

        Returns:
            Dictionary of visualization images
        """
        visualizations = {}

        # Original image
        visualizations['original'] = result.original_image

        # Scored masks overlay
        if show_masks and result.segmentation_masks:
            masks_vis = self.mask_aligner.visualize_scored_masks(
                result.original_image,
                result.segmentation_masks
            )
            visualizations['scored_masks'] = masks_vis

        # Similarity map
        if show_similarity and result.visualization_data:
            sim_vis = self.clip_extractor.visualize_similarity_map(
                result.original_image,
                result.visualization_data['similarity_map'],
                alpha=0.6
            )
            visualizations['similarity_map'] = sim_vis

        # Comparison grid
        if result.visualization_data and result.segmentation_masks:
            grid = self.mask_aligner.create_comparison_grid(
                result.original_image,
                result.segmentation_masks,
                result.visualization_data,
                num_masks=min(3, len(result.segmentation_masks))
            )
            visualizations['comparison_grid'] = grid

        # Edited result
        if result.edited_image is not None:
            visualizations['edited'] = np.array(result.edited_image)

            # Side-by-side comparison
            if result.segmentation_masks:
                mask_array = result.segmentation_masks[0].mask_candidate.mask
                comparison = self.inpainter.compare_results(
                    result.original_image,
                    result.edited_image,
                    mask_array
                )
                visualizations['edit_comparison'] = comparison

        return visualizations

    def _load_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Load image from various input types."""
        if isinstance(image, str):
            # Load from path
            img = Image.open(image).convert('RGB')
            return np.array(img)
        elif isinstance(image, Image.Image):
            # Convert PIL to numpy
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # Already numpy array
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def benchmark(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_prompts: List[str],
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark pipeline performance.

        Args:
            image: Test image
            text_prompts: List of prompts to test
            num_runs: Number of runs per prompt

        Returns:
            Average timing statistics
        """
        print(f"\nBenchmarking pipeline ({num_runs} runs per prompt)...\n")

        all_timings = []

        for prompt in text_prompts:
            print(f"Testing: '{prompt}'")

            for run in range(num_runs):
                result = self.segment(image, prompt, return_visualization=False)
                all_timings.append(result.timing)

        # Compute averages
        avg_timing = {}
        for key in all_timings[0].keys():
            values = [t[key] for t in all_timings]
            avg_timing[key] = np.mean(values)
            avg_timing[f'{key}_std'] = np.std(values)

        avg_timing['total'] = sum([avg_timing[k] for k in all_timings[0].keys()])

        print("\nBenchmark Results:")
        print(f"  SAM 2 generation: {avg_timing['sam2_generation']:.3f}s ± {avg_timing['sam2_generation_std']:.3f}s")
        print(f"  CLIP alignment:   {avg_timing['clip_alignment']:.3f}s ± {avg_timing['clip_alignment_std']:.3f}s")
        print(f"  Total:            {avg_timing['total']:.3f}s")

        return avg_timing
