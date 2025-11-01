"""
Evaluation metrics for open-vocabulary semantic segmentation.

Implements standard metrics from the thesis:
- mIoU (mean Intersection over Union)
- Precision, Recall, F1
- Boundary F1
- Novel class mIoU (for COCO-Open split)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


def compute_miou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Compute mean Intersection over Union (mIoU).

    Args:
        pred_mask: Predicted segmentation (H, W) with class indices
        gt_mask: Ground truth segmentation (H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore (typically 255 for unlabeled)

    Returns:
        Dictionary with mIoU, per-class IoU, and other metrics
    """
    # Flatten masks
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()

    # Remove ignore index
    valid = gt_mask != ignore_index
    pred_mask = pred_mask[valid]
    gt_mask = gt_mask[valid]

    # Compute per-class IoU
    ious = []
    per_class_iou = {}

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if gt_cls.sum() == 0:
            # Class not present in ground truth - exclude from mIoU calculation
            iou = np.nan
        else:
            iou = intersection / union
            ious.append(iou)

        per_class_iou[cls] = iou

    # Mean IoU over classes that appear in GT
    miou = np.nanmean(ious) if len(ious) > 0 else 0.0

    # Pixel accuracy
    pixel_acc = (pred_mask == gt_mask).sum() / len(gt_mask)

    return {
        'miou': float(miou),
        'pixel_accuracy': float(pixel_acc),
        'per_class_iou': per_class_iou,
        'num_valid_classes': len(ious)
    }


def compute_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_id: Optional[int] = None,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        pred_mask: Predicted segmentation (H, W)
        gt_mask: Ground truth segmentation (H, W)
        class_id: Specific class to evaluate (None for binary)
        ignore_index: Index to ignore

    Returns:
        Dictionary with precision, recall, F1
    """
    # Flatten
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()

    # Remove ignore index
    valid = gt_mask != ignore_index
    pred_mask = pred_mask[valid]
    gt_mask = gt_mask[valid]

    # Binary case
    if class_id is not None:
        pred_mask = (pred_mask == class_id)
        gt_mask = (gt_mask == class_id)
    else:
        pred_mask = pred_mask > 0
        gt_mask = gt_mask > 0

    # True positives, false positives, false negatives
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_boundary_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.0075,
    ignore_index: int = 255
) -> float:
    """
    Compute boundary F1 score (measures boundary localization quality).

    Based on: Santosh et al., "Measuring the objectness of image windows", TPAMI 2012

    Args:
        pred_mask: Predicted segmentation (H, W)
        gt_mask: Ground truth segmentation (H, W)
        threshold: Distance threshold as fraction of image diagonal
        ignore_index: Index to ignore

    Returns:
        Boundary F1 score
    """
    # Remove ignore regions
    valid = gt_mask != ignore_index
    pred_mask = pred_mask.copy()
    pred_mask[~valid] = 0
    gt_mask = gt_mask.copy()
    gt_mask[~valid] = 0

    # Compute boundaries (Canny edge detection)
    pred_boundary = cv2.Canny((pred_mask > 0).astype(np.uint8) * 255, 50, 150) > 0
    gt_boundary = cv2.Canny((gt_mask > 0).astype(np.uint8) * 255, 50, 150) > 0

    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0  # Both empty
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0  # One empty

    # Distance threshold in pixels
    img_diag = np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2)
    dist_thresh = threshold * img_diag

    # Compute distance maps
    pred_dist = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    gt_dist = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5)

    # Precision: fraction of pred boundary points within threshold of GT
    pred_boundary_pts = pred_boundary.sum()
    pred_correct = (gt_dist[pred_boundary] <= dist_thresh).sum()
    precision = pred_correct / pred_boundary_pts if pred_boundary_pts > 0 else 0

    # Recall: fraction of GT boundary points within threshold of pred
    gt_boundary_pts = gt_boundary.sum()
    gt_correct = (pred_dist[gt_boundary] <= dist_thresh).sum()
    recall = gt_correct / gt_boundary_pts if gt_boundary_pts > 0 else 0

    # F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return float(f1)


def compute_novel_class_miou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    novel_classes: List[int],
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Compute mIoU specifically on novel classes (for COCO-Open split).

    Args:
        pred_masks: List of predicted segmentations
        gt_masks: List of ground truth segmentations
        novel_classes: List of novel class indices
        num_classes: Total number of classes
        ignore_index: Index to ignore

    Returns:
        mIoU on novel classes, base classes, and overall
    """
    base_classes = [c for c in range(num_classes) if c not in novel_classes]

    # Accumulate IoU for novel and base classes
    novel_ious = {cls: [] for cls in novel_classes}
    base_ious = {cls: [] for cls in base_classes}

    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        metrics = compute_miou(pred_mask, gt_mask, num_classes, ignore_index)

        for cls, iou in metrics['per_class_iou'].items():
            if not np.isnan(iou):
                if cls in novel_classes:
                    novel_ious[cls].append(iou)
                elif cls in base_classes:
                    base_ious[cls].append(iou)

    # Compute mean IoU for novel and base
    novel_miou_list = []
    for cls in novel_classes:
        if len(novel_ious[cls]) > 0:
            novel_miou_list.append(np.mean(novel_ious[cls]))

    base_miou_list = []
    for cls in base_classes:
        if len(base_ious[cls]) > 0:
            base_miou_list.append(np.mean(base_ious[cls]))

    novel_miou = np.mean(novel_miou_list) if len(novel_miou_list) > 0 else 0.0
    base_miou = np.mean(base_miou_list) if len(base_miou_list) > 0 else 0.0
    overall_miou = np.mean(novel_miou_list + base_miou_list) if len(novel_miou_list + base_miou_list) > 0 else 0.0

    return {
        'novel_miou': float(novel_miou),
        'base_miou': float(base_miou),
        'overall_miou': float(overall_miou),
        'num_novel_classes': len(novel_miou_list),
        'num_base_classes': len(base_miou_list)
    }


def compute_all_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        pred_mask: Predicted segmentation
        gt_mask: Ground truth segmentation
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        Dictionary with all metrics
    """
    miou_metrics = compute_miou(pred_mask, gt_mask, num_classes, ignore_index)
    f1_metrics = compute_f1(pred_mask, gt_mask, ignore_index=ignore_index)
    boundary_f1 = compute_boundary_f1(pred_mask, gt_mask, ignore_index=ignore_index)

    return {
        **miou_metrics,
        **f1_metrics,
        'boundary_f1': boundary_f1
    }
