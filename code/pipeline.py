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
from models.adaptive_selection import AdaptiveMaskSelector


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
        sam_model: str = "sam2_hiera_base_plus",
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
            print("  [3/4] Initializing mask aligner...")
        self.mask_aligner = MaskTextAligner(
            clip_extractor=self.clip_extractor
        )

        if verbose:
            print("  [4/4] Initializing adaptive selector...")
        self.adaptive_selector = AdaptiveMaskSelector()

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
        return_visualization: bool = True,
        use_adaptive_selection: bool = False
    ) -> PipelineResult:
        """
        Segment objects based on text prompt.

        Args:
            image: Input image (array, PIL, or path)
            text_prompt: Text description of target object
            top_k: Number of top matches to return (ignored if use_adaptive_selection=True)
            min_mask_area: Minimum mask size (32x32 = 1024 pixels)
            return_visualization: Include visualization data
            use_adaptive_selection: Use adaptive mask selection based on query semantics

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
        # Get all scored masks (we'll select adaptively if requested)
        all_scored_masks, vis_data = self.mask_aligner.align_masks_with_text(
            filtered_masks,
            text_prompt,
            image_array,
            top_k=50,  # Get more candidates for adaptive selection
            return_similarity_maps=return_visualization
        )
        timing['clip_alignment'] = time.time() - t0

        if self.verbose:
            print(f"    Found {len(all_scored_masks)} matches ({timing['clip_alignment']:.2f}s)")

        # Stage 3.5: Adaptive mask selection (optional)
        if use_adaptive_selection and all_scored_masks:
            if self.verbose:
                print("  Stage 3.5: Adaptive mask selection...")

            t0 = time.time()
            scored_masks, adaptive_info = self.adaptive_selector.select_masks_adaptive(
                all_scored_masks,
                text_prompt,
                image_shape=(image_array.shape[0], image_array.shape[1]),
                max_masks=None  # Let it decide
            )
            timing['adaptive_selection'] = time.time() - t0

            if self.verbose:
                print(f"    Method: {adaptive_info['method']}")
                print(f"    Selected {adaptive_info['selected_count']} masks ({timing['adaptive_selection']:.2f}s)")
                for i, sm in enumerate(scored_masks[:5], 1):
                    print(f"      #{i}: score={sm.final_score:.3f}, area={sm.mask_candidate.area}")

            # Store adaptive info in visualization data
            if vis_data is not None:
                vis_data['adaptive_info'] = adaptive_info
        else:
            # Traditional top-K selection
            scored_masks = all_scored_masks[:top_k]
            if self.verbose:
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

    def segment_batch(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_prompts: List[str],
        min_mask_area: int = 1024,
        use_background_suppression: bool = True,
        score_threshold: float = 0.12,
        top_k_per_class: Optional[int] = None,
        return_visualization: bool = False,
        prompt_denoising_threshold: float = 0.12,
        temperature: float = 100.0  # Score calibration from MaskCLIP/MasQCLIP
    ) -> Dict[str, List[ScoredMask]]:
        """
        BATCH MODE: Segment image with multiple text prompts simultaneously.

        This is optimized for open-vocabulary segmentation with many classes:
        - Generates SAM masks once (not once per class)
        - Scores all masks against all text prompts in parallel
        - Returns masks grouped by class

        Args:
            image: Input image (array, PIL, or path)
            text_prompts: List of text descriptions (e.g., all COCO-Stuff classes)
            min_mask_area: Minimum mask size (32x32 = 1024 pixels)
            use_background_suppression: Whether to suppress background matches
            score_threshold: Minimum similarity score to keep a mask
            top_k_per_class: If specified, limit to top K masks per class
            return_visualization: Include visualization data (not implemented yet)
            prompt_denoising_threshold: Min max-score to keep a class (default: 0.12, same as score_threshold)

        Returns:
            Dictionary mapping class names to lists of scored masks:
            {
                "car": [ScoredMask(score=0.85, ...), ScoredMask(score=0.72, ...)],
                "road": [ScoredMask(score=0.78, ...)],
                "sky": [ScoredMask(score=0.91, ...)],
                ...
            }

        Example:
            >>> pipeline = OpenVocabSegmentationPipeline()
            >>> classes = ["car", "road", "tree", "sky", "building"]
            >>> results = pipeline.segment_batch(image, classes)
            >>> # Get top car mask
            >>> if results["car"]:
            >>>     best_car_mask = results["car"][0]
            >>>     print(f"Car confidence: {best_car_mask.final_score}")
        """
        timing = {}

        # Load image
        image_array = self._load_image(image)

        if self.verbose:
            print(f"Batch segmentation with {len(text_prompts)} classes...")
            print("  Stage 1: Generating masks with SAM 2...")

        # Stage 1: Generate masks with SAM 2 (ONCE for all classes)
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

        if len(filtered_masks) == 0:
            if self.verbose:
                print("    No masks generated!")
            return {prompt: [] for prompt in text_prompts}

        # Stage 2: Batch CLIP alignment (score all masks against all classes)
        if self.verbose:
            print(f"  Stage 2: Scoring {len(filtered_masks)} masks against {len(text_prompts)} classes...")

        t0 = time.time()
        class_to_masks = self.mask_aligner.align_masks_with_multiple_texts(
            filtered_masks,
            text_prompts,
            image_array,
            use_background_suppression=use_background_suppression,
            score_threshold=score_threshold,
            return_per_class=True,
            prompt_denoising_threshold=prompt_denoising_threshold,
            temperature=temperature
        )
        timing['clip_alignment'] = time.time() - t0

        if self.verbose:
            # Count how many classes have at least one mask
            classes_with_masks = sum(1 for masks in class_to_masks.values() if len(masks) > 0)
            total_masks = sum(len(masks) for masks in class_to_masks.values())
            print(f"    Found {total_masks} mask-class pairs across {classes_with_masks} classes ({timing['clip_alignment']:.2f}s)")

        # Optionally limit to top K per class
        if top_k_per_class is not None:
            for class_name in class_to_masks:
                class_to_masks[class_name] = class_to_masks[class_name][:top_k_per_class]

        if self.verbose:
            print("Batch segmentation complete!\n")
            # Show top 10 classes by number of masks
            sorted_classes = sorted(
                class_to_masks.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            if sorted_classes:
                print("  Top detected classes:")
                for class_name, masks_list in sorted_classes[:10]:
                    if len(masks_list) > 0:
                        best_score = masks_list[0].final_score
                        print(f"    {class_name:20s}: {len(masks_list):2d} masks (best: {best_score:.3f})")

        return class_to_masks

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
