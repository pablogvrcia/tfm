"""
==================================================================================
DEMONSTRATION: SAM Mask Class Assignment Voting Policies
==================================================================================

This script provides an EXPLICIT and EXTENSIVE demonstration of how SAM masks
are assigned class labels in CLIP-guided SAM segmentation.

The CRITICAL QUESTION: When SAM generates a mask covering 10,000 pixels,
how do we decide which class to assign to that entire mask?

We explore 3 voting policies:
1. MAJORITY VOTING (simple, but buggy with noisy CLIP predictions)
2. CONFIDENCE-WEIGHTED VOTING (better, considers CLIP probabilities)
3. ARGMAX-ONLY VOTING (baseline, conservative)

Author: Claude + Pablo
Date: November 29, 2024
==================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple

# Set style for better visualizations
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10
plt.style.use('default')


# ==================================================================================
# SECTION 1: Simulate CLIP Predictions (Noisy Probability Maps)
# ==================================================================================

def create_synthetic_clip_predictions(
    height: int = 50,
    width: int = 50,
    num_classes: int = 5,
    noise_level: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create synthetic CLIP predictions that mimic real-world behavior.

    CLIP's behavior in dense prediction:
    - Predictions are NOISY at pixel level (common issue in COCO-Stuff)
    - Max probability often < 0.5 (uncertainty is high)
    - Neighboring pixels can have different argmax classes
    - Multiple classes can be "competitive" (similar probabilities)

    Args:
        height, width: Image dimensions
        num_classes: Number of semantic classes
        noise_level: Amount of noise (0.0 = clean, 1.0 = very noisy)

    Returns:
        probs: Probability map (H, W, num_classes) - CLIP's output
        seg_map: Argmax predictions (H, W) - class with max prob per pixel
        class_names: List of class names
    """
    print("="*80)
    print("STEP 1: Creating Synthetic CLIP Predictions")
    print("="*80)
    print(f"Image size: {height}x{width}")
    print(f"Number of classes: {num_classes}")
    print(f"Noise level: {noise_level:.1%}")
    print()

    class_names = ["sky", "grass", "tree", "person", "building"][:num_classes]

    # Create base "true" segmentation (clean)
    true_seg = np.zeros((height, width), dtype=int)

    # Create regions for each class
    # Sky (top 30%)
    true_seg[:int(height*0.3), :] = 0

    # Grass (bottom 40%)
    true_seg[int(height*0.6):, :] = 1

    # Tree (left middle)
    true_seg[int(height*0.3):int(height*0.6), :int(width*0.4)] = 2

    # Person (center)
    cy, cx = height//2, width//2
    for y in range(height):
        for x in range(width):
            if (y-cy)**2/100 + (x-cx)**2/50 < 1:  # Ellipse
                true_seg[y, x] = 3

    # Building (right middle)
    if num_classes >= 5:
        true_seg[int(height*0.3):int(height*0.6), int(width*0.6):] = 4

    # Initialize probability map
    probs = np.zeros((height, width, num_classes), dtype=np.float32)

    # For each pixel, create a probability distribution
    for y in range(height):
        for x in range(width):
            true_class = true_seg[y, x]

            # Base probabilities (ground truth class has high prob)
            base_prob = 0.6 - noise_level * 0.3  # Reduces with noise

            # Create probability distribution
            pixel_probs = np.random.rand(num_classes) * noise_level
            pixel_probs[true_class] = base_prob

            # Add spatial noise (neighboring pixels affect each other)
            if y > 0 and x > 0:
                # Borrow some probability from neighbors
                neighbor_probs = probs[y-1, x] * 0.2 + probs[y, x-1] * 0.2
                pixel_probs = 0.6 * pixel_probs + 0.4 * neighbor_probs

            # Normalize to sum to 1
            pixel_probs = pixel_probs / pixel_probs.sum()

            probs[y, x] = pixel_probs

    # Compute argmax (class with highest probability per pixel)
    seg_map = probs.argmax(axis=2)

    # Compute statistics
    max_probs = probs.max(axis=2)
    print("CLIP Prediction Statistics:")
    print(f"  Average max probability: {max_probs.mean():.3f}")
    print(f"  Min max probability: {max_probs.min():.3f}")
    print(f"  Max max probability: {max_probs.max():.3f}")
    print(f"  Std of max probability: {max_probs.std():.3f}")
    print()

    # Show how noisy argmax is
    argmax_matches_truth = (seg_map == true_seg).mean()
    print(f"Argmax accuracy vs ground truth: {argmax_matches_truth:.1%}")
    print(f"  → {(1-argmax_matches_truth)*100:.1f}% of pixels have WRONG argmax!")
    print()

    return probs, seg_map, class_names, true_seg


# ==================================================================================
# SECTION 2: Simulate SAM Mask Generation
# ==================================================================================

def create_synthetic_sam_mask(
    height: int,
    width: int,
    center: Tuple[int, int],
    size: int = 15,
    shape: str = "ellipse"
) -> np.ndarray:
    """
    Create a synthetic SAM mask.

    SAM's behavior:
    - Generates smooth, coherent regions
    - Often EXPANDS beyond the prompt point
    - Follows natural boundaries (edges, color changes)
    - But: Doesn't know about semantic classes!

    Args:
        height, width: Image dimensions
        center: (y, x) center of mask
        size: Approximate radius
        shape: "ellipse" or "rectangle"

    Returns:
        mask: Binary mask (H, W)
    """
    mask = np.zeros((height, width), dtype=bool)
    cy, cx = center

    if shape == "ellipse":
        for y in range(height):
            for x in range(width):
                # Ellipse equation
                if (y-cy)**2/(size**2) + (x-cx)**2/((size*1.5)**2) < 1:
                    mask[y, x] = True
    else:  # rectangle
        y1, y2 = max(0, cy-size), min(height, cy+size)
        x1, x2 = max(0, cx-size), min(width, cx+size)
        mask[y1:y2, x1:x2] = True

    return mask


def generate_sam_masks_from_prompts(
    probs: np.ndarray,
    seg_map: np.ndarray,
    num_prompts: int = 5
) -> List[Dict]:
    """
    Simulate SAM mask generation from CLIP-guided prompts.

    In real CLIP-guided SAM:
    1. CLIP identifies high-confidence regions for each class
    2. Sample prompt points from those regions
    3. SAM generates masks from each prompt
    4. Each mask needs a class label → THIS IS WHERE VOTING HAPPENS

    Args:
        probs: CLIP probability map (H, W, num_classes)
        seg_map: CLIP argmax map (H, W)
        num_prompts: Number of prompts to generate

    Returns:
        List of mask dictionaries with:
            - mask: Binary mask (H, W)
            - prompt_point: (y, x) where prompt was placed
            - prompt_class: Class index of the prompt
    """
    print("="*80)
    print("STEP 2: Generating SAM Masks from CLIP-guided Prompts")
    print("="*80)
    print(f"Number of prompts: {num_prompts}")
    print()

    height, width, num_classes = probs.shape
    masks = []

    # For each class, sample a prompt from high-confidence region
    for class_idx in range(min(num_prompts, num_classes)):
        # Find pixels where this class has high confidence
        class_confidence = probs[:, :, class_idx]
        high_conf_mask = class_confidence > 0.3

        if high_conf_mask.any():
            # Sample a prompt point
            high_conf_coords = np.argwhere(high_conf_mask)
            prompt_idx = np.random.randint(len(high_conf_coords))
            prompt_y, prompt_x = high_conf_coords[prompt_idx]

            # SAM generates a mask (simulated)
            mask = create_synthetic_sam_mask(
                height, width,
                center=(prompt_y, prompt_x),
                size=np.random.randint(10, 20)
            )

            masks.append({
                'mask': mask,
                'prompt_point': (prompt_y, prompt_x),
                'prompt_class': class_idx,
                'num_pixels': mask.sum()
            })

            print(f"Mask {len(masks)}: Prompt class={class_idx}, "
                  f"Point=({prompt_y},{prompt_x}), Pixels={mask.sum()}")

    print()
    print(f"Total masks generated: {len(masks)}")
    print()

    return masks


# ==================================================================================
# SECTION 3: Voting Policies - THE CRITICAL PART
# ==================================================================================

def voting_policy_majority(
    mask: np.ndarray,
    seg_map: np.ndarray,
    probs: np.ndarray
) -> Tuple[int, Dict]:
    """
    POLICY 1: MAJORITY VOTING (Simple Pixel Counting)

    Algorithm:
        1. Look at all pixels inside the SAM mask
        2. Check the argmax class (from CLIP) at each pixel
        3. Count how many pixels belong to each class
        4. Assign the class with the MOST pixels

    Problem:
        - Ignores CLIP confidence values!
        - A class with 1000 low-confidence pixels (0.25 prob) beats
          a class with 900 high-confidence pixels (0.80 prob)
        - Very sensitive to CLIP noise

    Example Bug:
        Mask covers a bear:
        - 5500 pixels predict "hair drier" (avg confidence 0.28)
        - 4500 pixels predict "bear" (avg confidence 0.45)
        → Majority voting chooses "hair drier" ❌

    Args:
        mask: SAM mask (H, W) boolean
        seg_map: CLIP argmax predictions (H, W)
        probs: CLIP probabilities (H, W, num_classes)

    Returns:
        assigned_class: Class index assigned to this mask
        stats: Dictionary with voting statistics
    """
    # Extract argmax predictions inside mask
    masked_predictions = seg_map[mask]

    # Count pixels for each class
    unique_classes, counts = np.unique(masked_predictions, return_counts=True)

    # Majority vote: class with most pixels wins
    winner_idx = counts.argmax()
    assigned_class = unique_classes[winner_idx]

    # Compute statistics
    stats = {
        'policy': 'majority',
        'unique_classes': unique_classes.tolist(),
        'pixel_counts': counts.tolist(),
        'winner': assigned_class,
        'winner_count': counts[winner_idx],
        'total_pixels': mask.sum(),
        'confidence': None  # Majority voting doesn't use confidence!
    }

    return assigned_class, stats


def voting_policy_confidence_weighted(
    mask: np.ndarray,
    seg_map: np.ndarray,
    probs: np.ndarray
) -> Tuple[int, Dict]:
    """
    POLICY 2: CONFIDENCE-WEIGHTED VOTING (Smart)

    Algorithm:
        1. Look at all pixels inside the SAM mask
        2. For each class, compute AVERAGE confidence across pixels
        3. Assign the class with the HIGHEST average confidence

    Advantages:
        - Considers CLIP's confidence values
        - A few high-confidence pixels can outweigh many low-confidence pixels
        - More robust to CLIP noise

    Example Fix:
        Same bear mask:
        - "hair drier": 5500 pixels, avg confidence 0.28
        - "bear": 4500 pixels, avg confidence 0.45
        → Confidence-weighted chooses "bear" ✅

    Args:
        mask: SAM mask (H, W) boolean
        seg_map: CLIP argmax predictions (H, W)
        probs: CLIP probabilities (H, W, num_classes)

    Returns:
        assigned_class: Class index assigned to this mask
        stats: Dictionary with voting statistics
    """
    num_classes = probs.shape[2]

    # For each class, compute average confidence inside mask
    class_avg_confidences = {}
    class_pixel_counts = {}

    # Get all predictions inside mask
    masked_predictions = seg_map[mask]
    unique_classes = np.unique(masked_predictions)

    for class_idx in unique_classes:
        # Get pixels where this class is argmax
        class_pixels = masked_predictions == class_idx

        # Get confidence values for this class at those pixels
        # Need to map back to original image coordinates
        mask_coords = np.argwhere(mask)
        class_mask_coords = mask_coords[class_pixels]

        confidences = []
        for y, x in class_mask_coords:
            confidences.append(probs[y, x, class_idx])

        avg_confidence = np.mean(confidences) if confidences else 0.0
        class_avg_confidences[class_idx] = avg_confidence
        class_pixel_counts[class_idx] = class_pixels.sum()

    # Choose class with HIGHEST average confidence
    assigned_class = max(class_avg_confidences.keys(),
                        key=lambda k: class_avg_confidences[k])

    # Compute statistics
    stats = {
        'policy': 'confidence_weighted',
        'unique_classes': list(class_avg_confidences.keys()),
        'pixel_counts': list(class_pixel_counts.values()),
        'avg_confidences': list(class_avg_confidences.values()),
        'winner': assigned_class,
        'winner_confidence': class_avg_confidences[assigned_class],
        'total_pixels': mask.sum()
    }

    return assigned_class, stats


def voting_policy_argmax_only(
    mask: np.ndarray,
    seg_map: np.ndarray,
    probs: np.ndarray,
    prompt_class: int
) -> Tuple[int, Dict]:
    """
    POLICY 3: ARGMAX-ONLY (Conservative Baseline)

    Algorithm:
        1. Simply trust the prompt class
        2. OR: Only consider pixels where prompt_class wins argmax
        3. If no such pixels, use prompt class anyway

    Advantages:
        - Simple, predictable
        - Guaranteed to match baseline SCLIP behavior
        - No risk of "class switching"

    Disadvantages:
        - Doesn't leverage SAM's spatial coherence
        - Misses opportunities to correct CLIP errors
        - Conservative, may not improve quality

    Args:
        mask: SAM mask (H, W) boolean
        seg_map: CLIP argmax predictions (H, W)
        probs: CLIP probabilities (H, W, num_classes)
        prompt_class: The class this prompt was generated for

    Returns:
        assigned_class: Class index assigned to this mask
        stats: Dictionary with voting statistics
    """
    # Simply assign the prompt class (most conservative)
    assigned_class = prompt_class

    # Count how many pixels actually agree
    masked_predictions = seg_map[mask]
    agrees = (masked_predictions == prompt_class).sum()

    stats = {
        'policy': 'argmax_only',
        'prompt_class': prompt_class,
        'winner': assigned_class,
        'agreeing_pixels': int(agrees),
        'total_pixels': mask.sum(),
        'agreement_rate': agrees / mask.sum() if mask.sum() > 0 else 0.0
    }

    return assigned_class, stats


# ==================================================================================
# SECTION 4: Comparison and Visualization
# ==================================================================================

def compare_voting_policies(
    masks: List[Dict],
    seg_map: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    true_seg: np.ndarray = None
) -> Dict:
    """
    Compare all three voting policies on the same masks.

    This is THE KEY EXPERIMENT that shows why voting matters!

    Args:
        masks: List of SAM masks
        seg_map: CLIP argmax predictions
        probs: CLIP probabilities
        class_names: List of class names
        true_seg: Ground truth segmentation (for accuracy)

    Returns:
        comparison_results: Dictionary with results for each policy
    """
    print("="*80)
    print("STEP 3: Comparing Voting Policies")
    print("="*80)
    print()

    results = {
        'majority': [],
        'confidence_weighted': [],
        'argmax_only': []
    }

    for mask_idx, mask_dict in enumerate(masks):
        mask = mask_dict['mask']
        prompt_class = mask_dict['prompt_class']

        print(f"\n{'='*80}")
        print(f"Mask #{mask_idx + 1}: Prompt class = {class_names[prompt_class]}")
        print(f"  Mask size: {mask.sum()} pixels")
        print(f"  Prompt point: {mask_dict['prompt_point']}")
        print(f"{'='*80}")

        # Apply each voting policy
        class_maj, stats_maj = voting_policy_majority(mask, seg_map, probs)
        class_conf, stats_conf = voting_policy_confidence_weighted(mask, seg_map, probs)
        class_argmax, stats_argmax = voting_policy_argmax_only(mask, seg_map, probs, prompt_class)

        # Display results
        print("\n1. MAJORITY VOTING:")
        print(f"   Winner: {class_names[class_maj]}")
        print(f"   Pixel counts:")
        for cls, count in zip(stats_maj['unique_classes'], stats_maj['pixel_counts']):
            pct = 100 * count / stats_maj['total_pixels']
            marker = " ← WINNER" if cls == class_maj else ""
            print(f"     {class_names[cls]}: {count} pixels ({pct:.1f}%){marker}")

        print("\n2. CONFIDENCE-WEIGHTED VOTING:")
        print(f"   Winner: {class_names[class_conf]}")
        print(f"   Average confidences:")
        for cls, conf, count in zip(stats_conf['unique_classes'],
                                   stats_conf['avg_confidences'],
                                   stats_conf['pixel_counts']):
            marker = " ← WINNER" if cls == class_conf else ""
            print(f"     {class_names[cls]}: {conf:.3f} avg confidence "
                  f"({count} pixels){marker}")

        print("\n3. ARGMAX-ONLY VOTING:")
        print(f"   Winner: {class_names[class_argmax]} (always = prompt class)")
        print(f"   Agreement: {stats_argmax['agreement_rate']:.1%} of pixels "
              f"have this class as argmax")

        # If ground truth available, compute accuracy
        if true_seg is not None:
            true_classes_in_mask = true_seg[mask]
            true_majority = np.bincount(true_classes_in_mask).argmax()

            print(f"\n   GROUND TRUTH: Most pixels belong to '{class_names[true_majority]}'")
            print(f"   Accuracy:")
            print(f"     Majority: {'✓ CORRECT' if class_maj == true_majority else '✗ WRONG'}")
            print(f"     Confidence: {'✓ CORRECT' if class_conf == true_majority else '✗ WRONG'}")
            print(f"     Argmax-only: {'✓ CORRECT' if class_argmax == true_majority else '✗ WRONG'}")

        # Check if policies disagree
        if class_maj != class_conf or class_maj != class_argmax:
            print(f"\n   ⚠️  POLICIES DISAGREE!")
            print(f"      Majority → {class_names[class_maj]}")
            print(f"      Confidence → {class_names[class_conf]}")
            print(f"      Argmax-only → {class_names[class_argmax]}")

        # Store results
        results['majority'].append((class_maj, stats_maj))
        results['confidence_weighted'].append((class_conf, stats_conf))
        results['argmax_only'].append((class_argmax, stats_argmax))

    return results


def visualize_voting_comparison(
    probs: np.ndarray,
    seg_map: np.ndarray,
    masks: List[Dict],
    results: Dict,
    class_names: List[str],
    true_seg: np.ndarray = None,
    save_path: str = None
):
    """
    Create comprehensive visualization comparing voting policies.

    Shows:
    - CLIP predictions (noisy)
    - SAM masks
    - Results from each voting policy
    - Where policies agree/disagree
    """
    height, width = seg_map.shape
    num_classes = len(class_names)

    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Define colors for classes
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Row 1: CLIP Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    max_probs = probs.max(axis=2)
    im1 = ax1.imshow(max_probs, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('CLIP Max Probability\n(Confidence)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(seg_map, cmap='tab10', vmin=0, vmax=num_classes-1)
    ax2.set_title('CLIP Argmax Predictions\n(Noisy!)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    if true_seg is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(true_seg, cmap='tab10', vmin=0, vmax=num_classes-1)
        ax3.set_title('Ground Truth\n(Clean)', fontsize=12, fontweight='bold')
        ax3.axis('off')

    # Show SAM masks
    ax4 = fig.add_subplot(gs[0, 3])
    mask_overlay = np.zeros((height, width, 3))
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['mask']
        color = colors[i % len(colors)][:3]
        mask_overlay[mask] = color
    ax4.imshow(mask_overlay)
    ax4.set_title(f'SAM Masks ({len(masks)} masks)\nOverlay', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Row 2-4: Results for each policy
    policy_names = ['majority', 'confidence_weighted', 'argmax_only']
    policy_titles = ['MAJORITY VOTING', 'CONFIDENCE-WEIGHTED', 'ARGMAX-ONLY']

    for row, (policy, title) in enumerate(zip(policy_names, policy_titles), start=1):
        # Create result segmentation
        result_seg = np.full((height, width), -1, dtype=int)

        for mask_idx, mask_dict in enumerate(masks):
            mask = mask_dict['mask']
            assigned_class, stats = results[policy][mask_idx]
            result_seg[mask] = assigned_class

        # Plot result
        ax = fig.add_subplot(gs[row, 0])
        masked_result = np.ma.masked_where(result_seg == -1, result_seg)
        im = ax.imshow(masked_result, cmap='tab10', vmin=0, vmax=num_classes-1)
        ax.set_title(f'{title}\nResult', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Plot difference vs ground truth (if available)
        if true_seg is not None:
            ax_diff = fig.add_subplot(gs[row, 1])
            correct = (result_seg == true_seg) & (result_seg != -1)
            wrong = (result_seg != true_seg) & (result_seg != -1)

            diff_map = np.zeros((height, width, 3))
            diff_map[correct] = [0, 1, 0]  # Green = correct
            diff_map[wrong] = [1, 0, 0]    # Red = wrong

            ax_diff.imshow(diff_map)
            accuracy = correct.sum() / (correct.sum() + wrong.sum()) if (correct.sum() + wrong.sum()) > 0 else 0
            ax_diff.set_title(f'Accuracy: {accuracy:.1%}\nGreen=Correct, Red=Wrong',
                            fontsize=12, fontweight='bold')
            ax_diff.axis('off')

        # Plot statistics
        ax_stats = fig.add_subplot(gs[row, 2:])
        ax_stats.axis('off')

        stats_text = f"{title} STATISTICS:\n\n"

        for mask_idx, (assigned_class, stats) in enumerate(results[policy]):
            stats_text += f"Mask {mask_idx+1} → {class_names[assigned_class]}\n"

            if policy == 'majority':
                stats_text += f"  Total pixels: {stats['total_pixels']}\n"
                for cls, count in zip(stats['unique_classes'], stats['pixel_counts']):
                    pct = 100 * count / stats['total_pixels']
                    stats_text += f"    {class_names[cls]}: {count} ({pct:.1f}%)\n"

            elif policy == 'confidence_weighted':
                stats_text += f"  Total pixels: {stats['total_pixels']}\n"
                for cls, conf, count in zip(stats['unique_classes'],
                                          stats['avg_confidences'],
                                          stats['pixel_counts']):
                    stats_text += f"    {class_names[cls]}: conf={conf:.3f}, pixels={count}\n"

            else:  # argmax_only
                stats_text += f"  Agreement: {stats['agreement_rate']:.1%}\n"

            stats_text += "\n"

        ax_stats.text(0.05, 0.95, stats_text,
                     transform=ax_stats.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=class_names[i])
                      for i in range(num_classes)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=num_classes,
              fontsize=12, frameon=True)

    fig.suptitle('VOTING POLICY COMPARISON: How SAM Masks Get Class Labels',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.tight_layout()
    return fig


# ==================================================================================
# SECTION 5: Main Demonstration
# ==================================================================================

def run_demonstration(
    noise_level: float = 0.4,
    num_prompts: int = 5,
    save_path: str = "voting_policies_demo.png"
):
    """
    Run the complete demonstration of voting policies.

    This creates a synthetic but realistic example showing:
    1. How CLIP predictions are noisy
    2. How SAM generates masks
    3. How different voting policies assign classes
    4. Why some policies work better than others

    Args:
        noise_level: How noisy CLIP predictions are (0.0-1.0)
        num_prompts: Number of SAM masks to generate
        save_path: Where to save visualization
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: SAM Mask Class Assignment Voting Policies")
    print("="*80)
    print()
    print("This demonstration shows how different voting policies assign")
    print("class labels to SAM masks in CLIP-guided segmentation.")
    print()
    print(f"Configuration:")
    print(f"  Noise level: {noise_level:.1%}")
    print(f"  Number of masks: {num_prompts}")
    print()

    # Step 1: Create synthetic CLIP predictions
    probs, seg_map, class_names, true_seg = create_synthetic_clip_predictions(
        height=50,
        width=50,
        num_classes=5,
        noise_level=noise_level
    )

    # Step 2: Generate SAM masks
    masks = generate_sam_masks_from_prompts(probs, seg_map, num_prompts=num_prompts)

    # Step 3: Compare voting policies
    results = compare_voting_policies(masks, seg_map, probs, class_names, true_seg)

    # Step 4: Visualize
    print("\n" + "="*80)
    print("STEP 4: Creating Visualization")
    print("="*80)
    fig = visualize_voting_comparison(
        probs, seg_map, masks, results, class_names, true_seg, save_path
    )

    # Step 5: Summary
    print("\n" + "="*80)
    print("SUMMARY: Key Takeaways")
    print("="*80)
    print()
    print("1. MAJORITY VOTING:")
    print("   - Simple: counts pixels, ignores confidence")
    print("   - Problem: sensitive to CLIP noise")
    print("   - Example bug: 'hair drier' wins over 'bear' due to more noisy pixels")
    print()
    print("2. CONFIDENCE-WEIGHTED VOTING:")
    print("   - Smart: considers CLIP confidence values")
    print("   - Better: fewer high-confidence pixels beat many low-confidence ones")
    print("   - Recommended for noisy CLIP predictions (COCO-Stuff)")
    print()
    print("3. ARGMAX-ONLY:")
    print("   - Conservative: trusts prompt class")
    print("   - Safe: guaranteed baseline quality")
    print("   - Limited: doesn't leverage SAM's spatial coherence")
    print()
    print("="*80)
    print()

    return fig, results


# ==================================================================================
# SECTION 6: Run the Demo
# ==================================================================================

if __name__ == "__main__":
    import sys

    # Configuration
    NOISE_LEVEL = 0.4  # 40% noise (typical for COCO-Stuff)
    NUM_PROMPTS = 5
    SAVE_PATH = "benchmarks/results/voting_policies_demo.png"

    print("Starting demonstration...")
    print(f"Output will be saved to: {SAVE_PATH}")
    print()

    # Run demonstration
    fig, results = run_demonstration(
        noise_level=NOISE_LEVEL,
        num_prompts=NUM_PROMPTS,
        save_path=SAVE_PATH
    )

    # Show plot
    plt.show()

    print("\nDemonstration complete!")
    print(f"Visualization saved to: {SAVE_PATH}")
