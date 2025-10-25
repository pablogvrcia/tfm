"""
Utility functions for the pipeline.

Includes:
- Image I/O helpers
- Visualization utilities
- Evaluation metrics
- Logging helpers
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List, Optional
import matplotlib.pyplot as plt


# ============================================================================
# Image I/O
# ============================================================================

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from path as RGB numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(image: Union[np.ndarray, Image.Image], path: Union[str, Path]):
    """Save image to path."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


def resize_image(
    image: np.ndarray,
    max_size: int = 1024,
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to fit within max_size.

    Args:
        image: Input image
        max_size: Maximum dimension
        keep_aspect: Maintain aspect ratio

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image

    if keep_aspect:
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
    else:
        new_h = new_w = max_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ============================================================================
# Visualization
# ============================================================================

def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """Create colored mask overlay on image."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Input image
        bbox: (x, y, w, h)
        color: Box color
        thickness: Line thickness
        label: Optional text label

    Returns:
        Image with bounding box
    """
    img = image.copy()
    x, y, w, h = bbox

    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    if label:
        # Add text background
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y), color, -1)
        cv2.putText(
            img, label, (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return img


def create_grid(images: List[np.ndarray], grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Arrange images in a grid.

    Args:
        images: List of images (must have same shape)
        grid_size: (rows, cols), auto-computed if None

    Returns:
        Grid image
    """
    if not images:
        raise ValueError("Empty image list")

    # Auto-compute grid size
    if grid_size is None:
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid_size = (rows, cols)

    rows, cols = grid_size
    h, w = images[0].shape[:2]

    # Pad images list if needed
    while len(images) < rows * cols:
        images.append(np.zeros_like(images[0]))

    # Create grid
    grid_rows = []
    for r in range(rows):
        row_images = images[r * cols:(r + 1) * cols]
        # Ensure all images in row have same height
        row_images = [cv2.resize(img, (w, h)) for img in row_images]
        grid_rows.append(np.hstack(row_images))

    return np.vstack(grid_rows)


def plot_results(
    images: dict,
    titles: Optional[dict] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot multiple images using matplotlib.

    Args:
        images: Dictionary of {name: image}
        titles: Optional dictionary of {name: title}
        figsize: Figure size
        save_path: Optional path to save figure
    """
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, (name, image) in enumerate(images.items()):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.max() > 1.0:  # Assume BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.axis('off')

        title = titles.get(name, name) if titles else name
        ax.set_title(title)

    # Hide extra subplots
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# Evaluation Metrics (Chapter 4.2)
# ============================================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU).

    Equation 4.1: IoU = |P ∩ G| / |P ∪ G|

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        IoU score [0, 1]
    """
    pred_mask = (pred_mask > 0.5).astype(np.bool_)
    gt_mask = (gt_mask > 0.5).astype(np.bool_)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute precision and recall.

    Equation 4.2:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        (precision, recall)
    """
    pred_mask = (pred_mask > 0.5).astype(np.bool_)
    gt_mask = (gt_mask > 0.5).astype(np.bool_)

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def compute_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute F1 score.

    Equation 4.3: F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        F1 score [0, 1]
    """
    precision, recall = compute_precision_recall(pred_mask, gt_mask)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_mean_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """
    Compute mean IoU across multiple masks.

    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks

    Returns:
        Mean IoU
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of predicted and ground truth masks must match")

    ious = [compute_iou(pred, gt) for pred, gt in zip(pred_masks, gt_masks)]
    return np.mean(ious)


def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: int = 2) -> float:
    """
    Compute boundary F1 score.

    Measures how well predicted boundaries match ground truth boundaries.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        threshold: Distance threshold for matching boundaries (pixels)

    Returns:
        Boundary F1 score
    """
    # Extract boundaries
    pred_boundary = cv2.Canny(
        (pred_mask > 0.5).astype(np.uint8) * 255, 100, 200
    )
    gt_boundary = cv2.Canny(
        (gt_mask > 0.5).astype(np.uint8) * 255, 100, 200
    )

    # Distance transforms
    pred_dist = cv2.distanceTransform(
        255 - pred_boundary, cv2.DIST_L2, 5
    )
    gt_dist = cv2.distanceTransform(
        255 - gt_boundary, cv2.DIST_L2, 5
    )

    # Count matches
    pred_coords = np.argwhere(pred_boundary > 0)
    gt_coords = np.argwhere(gt_boundary > 0)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0

    # Precision: predicted boundary points near GT
    pred_near_gt = (gt_dist[pred_coords[:, 0], pred_coords[:, 1]] < threshold).sum()
    precision = pred_near_gt / len(pred_coords)

    # Recall: GT boundary points near predicted
    gt_near_pred = (pred_dist[gt_coords[:, 0], gt_coords[:, 1]] < threshold).sum()
    recall = gt_near_pred / len(gt_coords)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


# ============================================================================
# Logging
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def print_timing_summary(timing: dict):
    """Print formatted timing summary."""
    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)

    for key, value in timing.items():
        if not key.endswith('_std'):
            print(f"  {key:20s}: {format_time(value)}")

    if 'total' in timing:
        print("-"*50)
        print(f"  {'Total':20s}: {format_time(timing['total'])}")

    print("="*50 + "\n")
