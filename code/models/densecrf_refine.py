"""
DenseCRF Boundary Refinement Module

Implements Dense Conditional Random Fields for post-processing segmentation masks.
This refines segmentation boundaries by incorporating:
1. Appearance consistency (similar pixels should have similar labels)
2. Smoothness constraints (encourage spatial coherence)

Reference: Krähenbühl & Koltun, "Efficient Inference in Fully Connected CRFs", NIPS 2011
Paper: https://arxiv.org/abs/1210.5644

Expected improvement: +1-2% mIoU, +3-5% Boundary F1-score
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class DenseCRFRefiner:
    """
    Dense CRF for segmentation boundary refinement.

    Uses pydensecrf library for efficient mean-field inference.
    Falls back to bilateral filtering if pydensecrf not available.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        pos_w: float = 3.0,  # Weight for positional kernel
        pos_xy_std: float = 3.0,  # Std for positional kernel
        bi_w: float = 10.0,  # Weight for bilateral kernel
        bi_xy_std: float = 80.0,  # Std for bilateral spatial kernel
        bi_rgb_std: float = 13.0,  # Std for bilateral color kernel
        use_gpu: bool = False,  # CRF is CPU-only typically
    ):
        """
        Initialize DenseCRF refiner.

        Args:
            max_iterations: Number of mean-field iterations
            pos_w: Weight for positional (smoothness) kernel
            pos_xy_std: Standard deviation for positional kernel
            bi_w: Weight for bilateral (appearance) kernel
            bi_xy_std: Spatial standard deviation for bilateral kernel
            bi_rgb_std: Color standard deviation for bilateral kernel
            use_gpu: Whether to use GPU (if available)
        """
        self.max_iterations = max_iterations
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.use_gpu = use_gpu

        # Try to import pydensecrf
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
            self.dcrf = dcrf
            self.unary_from_softmax = unary_from_softmax
            self.create_pairwise_bilateral = create_pairwise_bilateral
            self.create_pairwise_gaussian = create_pairwise_gaussian
            self.crf_available = True
            print(f"✓ DenseCRF: Using pydensecrf for boundary refinement")
        except ImportError:
            self.crf_available = False
            print(f"⚠ DenseCRF: pydensecrf not available, using bilateral filtering fallback")
            print(f"  Install with: pip install pydensecrf")

    def refine(
        self,
        image: np.ndarray,
        probabilities: np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Refine segmentation using DenseCRF.

        Args:
            image: RGB image (H, W, 3) in range [0, 255], uint8
            probabilities: Class probabilities (num_classes, H, W) in range [0, 1]
            return_probs: If True, return refined probabilities; else return labels

        Returns:
            Refined segmentation: (H, W) labels or (num_classes, H, W) probabilities
        """
        if not self.crf_available:
            return self._bilateral_filter_fallback(image, probabilities, return_probs)

        # Ensure correct formats
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if probabilities.max() > 1.0:
            probabilities = probabilities / probabilities.max()

        num_classes, H, W = probabilities.shape

        # Create DenseCRF object
        d = self.dcrf.DenseCRF2D(W, H, num_classes)

        # Set unary potentials (negative log probabilities)
        unary = self.unary_from_softmax(probabilities)
        d.setUnaryEnergy(unary)

        # Add pairwise potentials

        # 1. Positional (smoothness) kernel
        # This encourages nearby pixels to have similar labels
        d.addPairwiseGaussian(
            sxy=(self.pos_xy_std, self.pos_xy_std),
            compat=self.pos_w,
            kernel=self.dcrf.DIAG_KERNEL,
            normalization=self.dcrf.NORMALIZE_SYMMETRIC
        )

        # 2. Bilateral (appearance) kernel
        # This encourages similar-looking pixels to have similar labels
        d.addPairwiseBilateral(
            sxy=(self.bi_xy_std, self.bi_xy_std),
            srgb=(self.bi_rgb_std, self.bi_rgb_std, self.bi_rgb_std),
            rgbim=image,
            compat=self.bi_w,
            kernel=self.dcrf.DIAG_KERNEL,
            normalization=self.dcrf.NORMALIZE_SYMMETRIC
        )

        # Perform inference
        Q = d.inference(self.max_iterations)

        # Reshape to (num_classes, H, W)
        Q = np.array(Q).reshape((num_classes, H, W))

        if return_probs:
            return Q
        else:
            # Return most likely label at each pixel
            return np.argmax(Q, axis=0).astype(np.uint8)

    def _bilateral_filter_fallback(
        self,
        image: np.ndarray,
        probabilities: np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Fallback refinement using bilateral filtering.

        This is less effective than DenseCRF but provides some boundary smoothing.
        """
        try:
            import cv2

            num_classes, H, W = probabilities.shape

            # Apply bilateral filter to each class probability map
            refined_probs = np.zeros_like(probabilities)

            for c in range(num_classes):
                prob_map = probabilities[c]

                # Convert to uint8 for bilateral filter
                prob_map_uint8 = (prob_map * 255).astype(np.uint8)

                # Apply bilateral filter (more aggressive parameters)
                # d: diameter of pixel neighborhood (increased for more smoothing)
                # sigmaColor: filter sigma in color space (higher = more smoothing)
                # sigmaSpace: filter sigma in coordinate space (higher = larger area)
                filtered = cv2.bilateralFilter(
                    prob_map_uint8,
                    d=9,
                    sigmaColor=75,
                    sigmaSpace=75
                )
                
                # filtered = cv2.bilateralFilter(
                #     prob_map_uint8,
                #     d=15,  # Increased from 9
                #     sigmaColor=150,  # Increased from 75
                #     sigmaSpace=150  # Increased from 75
                # )

                # Convert back to float
                refined_probs[c] = filtered.astype(np.float32) / 255.0

            # Renormalize probabilities
            refined_probs = refined_probs / (refined_probs.sum(axis=0, keepdims=True) + 1e-8)

            if return_probs:
                return refined_probs
            else:
                return np.argmax(refined_probs, axis=0).astype(np.uint8)

        except ImportError:
            warnings.warn("OpenCV not available for bilateral filtering. Returning original probabilities.")
            if return_probs:
                return probabilities
            else:
                return np.argmax(probabilities, axis=0).astype(np.uint8)

    def refine_torch(
        self,
        image: torch.Tensor,
        probabilities: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Refine segmentation using DenseCRF (PyTorch interface).

        Args:
            image: RGB image tensor (3, H, W) or (H, W, 3) in range [0, 1]
            probabilities: Class probabilities (num_classes, H, W) in range [0, 1]
            return_probs: If True, return probabilities; else return labels

        Returns:
            Refined segmentation tensor
        """
        # Convert to numpy
        if image.dim() == 3 and image.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()

        # Ensure uint8 range
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        probabilities_np = probabilities.cpu().numpy()

        # Refine
        refined = self.refine(image_np, probabilities_np, return_probs=return_probs)

        # Convert back to torch
        return torch.from_numpy(refined).to(probabilities.device)

    def get_expected_improvement(self) -> dict:
        """Return expected performance improvements."""
        return {
            "mIoU_gain": "+1-2%",
            "boundary_f1_gain": "+3-5%",
            "training_required": False,
            "inference_overhead": "~50-100ms per image",
            "reference": "Krähenbühl & Koltun, NIPS 2011"
        }


def create_densecrf_refiner(
    max_iterations: int = 10,
    pos_w: float = 3.0,
    bi_w: float = 10.0,
) -> DenseCRFRefiner:
    """
    Factory function to create DenseCRF refiner.

    Args:
        max_iterations: Number of inference iterations
        pos_w: Weight for positional kernel (smoothness)
        bi_w: Weight for bilateral kernel (appearance)

    Returns:
        Configured DenseCRF refiner
    """
    return DenseCRFRefiner(
        max_iterations=max_iterations,
        pos_w=pos_w,
        bi_w=bi_w
    )
