"""
Stable Diffusion Inpainting Module

This module handles generative inpainting using Stable Diffusion v2,
as described in Chapter 3.2.4.

Features:
- Mask-conditioned inpainting
- Prompt engineering for better results
- Mask refinement and blending
- Multiple inference strategies

Reference: Rombach et al., "High-Resolution Image Synthesis with
Latent Diffusion Models", CVPR 2022
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, Optional, Tuple
import cv2


class StableDiffusionInpainter:
    """
    Performs text-guided inpainting using Stable Diffusion.

    Implements the inpainting strategy from Chapter 3.2.4:
    - Latent space inpainting
    - High guidance scale for prompt adherence
    - Mask refinement for seamless blending
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        mask_blur: int = 8,
        mask_dilation: int = 5,
    ):
        """
        Initialize Stable Diffusion inpainting model.

        Args:
            model_id: HuggingFace model identifier
            device: Computation device
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            mask_blur: Gaussian blur radius for mask edges
            mask_dilation: Pixels to dilate mask for better coverage
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.mask_blur = mask_blur
        self.mask_dilation = mask_dilation

        # Load Stable Diffusion inpainting pipeline
        try:
            from diffusers import StableDiffusionInpaintPipeline

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self.pipe = self.pipe.to(device)

            # Enable memory optimizations
            if device == "cuda":
                self.pipe.enable_attention_slicing()
                # Optionally enable xformers for faster inference
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass

        except ImportError:
            print("Warning: diffusers not installed. Using mock inpainting.")
            self.pipe = None

    def inpaint(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, artifacts",
        num_samples: int = 1,
        seed: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, dict]]:
        """
        Perform inpainting on masked region.

        Args:
            image: Input RGB image
            mask: Binary mask (1 = inpaint, 0 = keep)
            prompt: Text description of desired content
            negative_prompt: What to avoid generating
            num_samples: Number of samples to generate (returns best)
            seed: Random seed for reproducibility
            return_intermediate: Return intermediate results

        Returns:
            Inpainted image, optionally with intermediate results
        """
        # Convert inputs to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Prepare mask
        mask_pil = self._prepare_mask(mask, image.size)

        # Engineer prompt for better results
        engineered_prompt = self._engineer_prompt(prompt, image)

        if self.pipe is None:
            # Mock inpainting for testing
            result = self._mock_inpaint(image, mask)
            return (result, {}) if return_intermediate else result

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate samples
        results = []
        for _ in range(num_samples):
            output = self.pipe(
                prompt=engineered_prompt,
                image=image,
                mask_image=mask_pil,
                negative_prompt=negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )

            results.append(output.images[0])

        # Select best result (in practice, could use CLIP score)
        best_result = results[0]

        if return_intermediate:
            intermediate = {
                "all_samples": results,
                "engineered_prompt": engineered_prompt,
                "mask": mask_pil,
                "original": image
            }
            return best_result, intermediate

        return best_result

    def _prepare_mask(
        self,
        mask: np.ndarray,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Prepare mask for inpainting:
        - Dilate to ensure complete coverage
        - Blur edges for smooth transitions
        - Resize to target size

        Args:
            mask: Binary mask (H, W)
            target_size: (width, height) for output

        Returns:
            Processed mask as PIL Image
        """
        # Ensure binary
        mask = (mask > 0.5).astype(np.uint8) * 255

        # Dilate mask by specified pixels
        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Blur edges for smooth transition
        if self.mask_blur > 0:
            mask = cv2.GaussianBlur(
                mask,
                (self.mask_blur * 2 + 1, self.mask_blur * 2 + 1),
                0
            )

        # Convert to PIL and resize
        mask_pil = Image.fromarray(mask).convert("L")
        mask_pil = mask_pil.resize(target_size, Image.LANCZOS)

        return mask_pil

    def _engineer_prompt(
        self,
        prompt: str,
        image: Image.Image
    ) -> str:
        """
        Engineer prompt for better inpainting results.

        Strategies:
        - Add quality descriptors
        - Add context about the scene
        - Ensure prompt clarity

        Args:
            prompt: Original user prompt
            image: Original image (for context analysis)

        Returns:
            Engineered prompt
        """
        # Basic prompt engineering
        quality_terms = "high quality, detailed, photorealistic"

        # Check if prompt already has quality terms
        if not any(term in prompt.lower() for term in ["quality", "detailed", "realistic"]):
            engineered = f"{prompt}, {quality_terms}"
        else:
            engineered = prompt

        return engineered

    def _mock_inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray
    ) -> Image.Image:
        """
        Mock inpainting using simple image processing.
        Used when Stable Diffusion is not available.

        Uses OpenCV inpainting as placeholder.
        """
        import cv2

        img_array = np.array(image)
        mask_array = (mask > 0.5).astype(np.uint8) * 255

        # Use OpenCV inpainting (Telea method)
        result = cv2.inpaint(
            img_array,
            mask_array,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )

        return Image.fromarray(result)

    def batch_inpaint(
        self,
        images: list,
        masks: list,
        prompts: list,
        batch_size: int = 4
    ) -> list:
        """
        Batch inpaint multiple images.

        Args:
            images: List of images
            masks: List of masks
            prompts: List of prompts
            batch_size: Batch size for processing

        Returns:
            List of inpainted images
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_masks = masks[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]

            for img, mask, prompt in zip(batch_images, batch_masks, batch_prompts):
                result = self.inpaint(img, mask, prompt)
                results.append(result)

        return results

    def remove_object(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: np.ndarray,
        context_prompt: Optional[str] = None
    ) -> Image.Image:
        """
        Remove object and fill with contextually appropriate content.

        Args:
            image: Input image
            mask: Mask of object to remove
            context_prompt: Optional context description

        Returns:
            Image with object removed
        """
        if context_prompt is None:
            # Generate generic fill prompt
            prompt = "natural background, seamless fill, matching surroundings"
        else:
            prompt = f"fill with {context_prompt}, matching surroundings"

        return self.inpaint(
            image,
            mask,
            prompt,
            negative_prompt="object, person, thing, blurry, artifacts"
        )

    def replace_object(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: np.ndarray,
        replacement_prompt: str
    ) -> Image.Image:
        """
        Replace masked object with new content.

        Args:
            image: Input image
            mask: Mask of object to replace
            replacement_prompt: Description of replacement

        Returns:
            Image with replaced object
        """
        return self.inpaint(
            image,
            mask,
            replacement_prompt,
            negative_prompt="blurry, distorted, low quality"
        )

    def style_transfer(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: np.ndarray,
        style_prompt: str
    ) -> Image.Image:
        """
        Apply style transfer to masked region.

        Args:
            image: Input image
            mask: Region to restyle
            style_prompt: Style description (e.g., "watercolor painting")

        Returns:
            Image with styled region
        """
        return self.inpaint(
            image,
            mask,
            f"in the style of {style_prompt}",
            negative_prompt="ugly, distorted, inconsistent"
        )

    def compare_results(
        self,
        original: np.ndarray,
        inpainted: Image.Image,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison visualization.

        Args:
            original: Original image
            inpainted: Inpainted result
            mask: Mask used

        Returns:
            Comparison grid
        """
        # Convert to arrays
        inpainted_array = np.array(inpainted)

        # Create mask visualization
        mask_vis = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

        # Create overlay showing edited region
        overlay = original.copy()
        overlay[mask > 0.5] = inpainted_array[mask > 0.5]

        # Concatenate
        comparison = np.hstack([
            original,
            mask_vis,
            inpainted_array,
            overlay
        ])

        return comparison
