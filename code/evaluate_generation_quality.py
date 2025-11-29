#!/usr/bin/env python3
"""
Evaluate Generative Image Quality

Calculates:
- Fréchet Inception Distance (FID): Similarity between real and generated images
- CLIP Score: Alignment between generated image and text prompt

Usage:
    # Single image evaluation
    python evaluate_generation_quality.py \
        --generated-image output.png \
        --prompt "a red apple on a table" \
        --reference-image original.png

    # Batch evaluation
    python evaluate_generation_quality.py \
        --generated-dir results/ \
        --prompts-file prompts.txt \
        --reference-dir originals/
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import json

# FID calculation
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

# CLIP Score
import open_clip


class GenerationEvaluator:
    """Evaluates quality of generated images using FID and CLIP Score."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Load InceptionV3 for FID
        print(f"Loading InceptionV3 model on {device}...")
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        ).to(device)
        self.inception.eval()

        # Remove the final classification layer to get features
        self.inception.fc = torch.nn.Identity()

        # Load CLIP for CLIP Score
        print(f"Loading OpenCLIP model on {device}...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai', device=device
        )
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # InceptionV3 preprocessing
        self.inception_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def extract_inception_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract InceptionV3 features from a list of images."""
        features = []

        with torch.no_grad():
            for img in images:
                # Preprocess
                img_tensor = self.inception_transform(img).unsqueeze(0).to(self.device)

                # Extract features
                feat = self.inception(img_tensor)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def calculate_fid(self,
                     real_images: List[Image.Image],
                     generated_images: List[Image.Image]) -> float:
        """
        Calculate Fréchet Inception Distance between real and generated images.

        Args:
            real_images: List of PIL Images (reference/original)
            generated_images: List of PIL Images (generated/edited)

        Returns:
            FID score (lower is better, 0 = identical distributions)
        """
        print(f"Extracting features from {len(real_images)} real images...")
        real_features = self.extract_inception_features(real_images)

        print(f"Extracting features from {len(generated_images)} generated images...")
        gen_features = self.extract_inception_features(generated_images)

        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)

        # Calculate FID
        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        return fid

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet Distance between two Gaussian distributions."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_clip_score(self,
                            image: Image.Image,
                            prompt: str) -> float:
        """
        Calculate CLIP Score: cosine similarity between image and text embeddings.

        Args:
            image: PIL Image
            prompt: Text description

        Returns:
            CLIP score (0-100, higher is better)
        """
        with torch.no_grad():
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # Tokenize text
            text_input = self.clip_tokenizer([prompt]).to(self.device)

            # Get embeddings
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()

            # Scale to 0-100
            clip_score = max(0, similarity * 100)

        return clip_score

    def batch_clip_scores(self,
                         images: List[Image.Image],
                         prompts: List[str]) -> List[float]:
        """Calculate CLIP scores for multiple image-prompt pairs."""
        assert len(images) == len(prompts), "Number of images must match number of prompts"

        scores = []
        for img, prompt in zip(images, prompts):
            score = self.calculate_clip_score(img, prompt)
            scores.append(score)

        return scores


def load_image(path: str) -> Image.Image:
    """Load image from path."""
    return Image.open(path).convert('RGB')


def load_images_from_dir(dir_path: str) -> List[Image.Image]:
    """Load all images from directory."""
    dir_path = Path(dir_path)
    images = []

    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        for img_path in dir_path.glob(ext):
            images.append(load_image(img_path))

    return images


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from text file (one per line)."""
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(description='Evaluate generative image quality')

    # Single image mode
    parser.add_argument('--generated-image', type=str,
                       help='Path to generated/edited image')
    parser.add_argument('--reference-image', type=str,
                       help='Path to reference/original image')
    parser.add_argument('--prompt', type=str,
                       help='Text prompt used for generation')

    # Batch mode
    parser.add_argument('--generated-dir', type=str,
                       help='Directory containing generated images')
    parser.add_argument('--reference-dir', type=str,
                       help='Directory containing reference images')
    parser.add_argument('--prompts-file', type=str,
                       help='File containing prompts (one per line)')

    # Options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    parser.add_argument('--output-json', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--skip-fid', action='store_true',
                       help='Skip FID calculation (only CLIP score)')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = GenerationEvaluator(device=args.device)

    results = {}

    # ========================================================================
    # Single Image Mode
    # ========================================================================
    if args.generated_image:
        print("\n" + "="*70)
        print("Single Image Evaluation")
        print("="*70)

        gen_img = load_image(args.generated_image)
        print(f"✓ Loaded generated image: {args.generated_image}")

        # CLIP Score
        if args.prompt:
            print(f"\nCalculating CLIP Score...")
            print(f"Prompt: \"{args.prompt}\"")
            clip_score = evaluator.calculate_clip_score(gen_img, args.prompt)
            print(f"CLIP Score: {clip_score:.2f}/100")
            results['clip_score'] = clip_score

        # FID Score
        if args.reference_image and not args.skip_fid:
            ref_img = load_image(args.reference_image)
            print(f"✓ Loaded reference image: {args.reference_image}")

            print(f"\nCalculating FID...")
            fid = evaluator.calculate_fid([ref_img], [gen_img])
            print(f"FID Score: {fid:.2f}")
            results['fid_score'] = fid

    # ========================================================================
    # Batch Mode
    # ========================================================================
    elif args.generated_dir:
        print("\n" + "="*70)
        print("Batch Evaluation")
        print("="*70)

        gen_images = load_images_from_dir(args.generated_dir)
        print(f"✓ Loaded {len(gen_images)} generated images from {args.generated_dir}")

        # CLIP Scores
        if args.prompts_file:
            prompts = load_prompts_from_file(args.prompts_file)
            print(f"✓ Loaded {len(prompts)} prompts from {args.prompts_file}")

            if len(prompts) != len(gen_images):
                print(f"⚠ Warning: {len(prompts)} prompts but {len(gen_images)} images")
                # Use only matching pairs
                n = min(len(prompts), len(gen_images))
                prompts = prompts[:n]
                gen_images_clip = gen_images[:n]
            else:
                gen_images_clip = gen_images

            print(f"\nCalculating CLIP Scores for {len(prompts)} images...")
            clip_scores = evaluator.batch_clip_scores(gen_images_clip, prompts)

            mean_clip = np.mean(clip_scores)
            std_clip = np.std(clip_scores)

            print(f"Mean CLIP Score: {mean_clip:.2f} ± {std_clip:.2f}")
            print(f"Min: {min(clip_scores):.2f}, Max: {max(clip_scores):.2f}")

            results['clip_scores'] = {
                'mean': mean_clip,
                'std': std_clip,
                'min': min(clip_scores),
                'max': max(clip_scores),
                'all_scores': clip_scores
            }

        # FID Score
        if args.reference_dir and not args.skip_fid:
            ref_images = load_images_from_dir(args.reference_dir)
            print(f"✓ Loaded {len(ref_images)} reference images from {args.reference_dir}")

            print(f"\nCalculating FID between {len(ref_images)} reference and {len(gen_images)} generated images...")
            fid = evaluator.calculate_fid(ref_images, gen_images)
            print(f"FID Score: {fid:.2f}")

            results['fid_score'] = fid

    else:
        print("Error: Must provide either --generated-image or --generated-dir")
        return

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)

    if 'clip_score' in results:
        print(f"CLIP Score: {results['clip_score']:.2f}/100")
    elif 'clip_scores' in results:
        print(f"Mean CLIP Score: {results['clip_scores']['mean']:.2f}/100")

    if 'fid_score' in results:
        print(f"FID Score: {results['fid_score']:.2f}")

    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_json}")

    print("="*70)


if __name__ == "__main__":
    main()
