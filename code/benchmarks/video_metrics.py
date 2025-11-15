"""
Video Object Segmentation Metrics

Implements standard VOS metrics:
- J (Region Similarity / IoU)
- F (Contour Accuracy / Boundary F-measure)
- J&F (Mean of J and F)
- Temporal Stability

Based on DAVIS evaluation protocol:
https://github.com/davisvideochallenge/davis-2017
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.ndimage import binary_erosion, binary_dilation


def db_eval_iou(annotation: np.ndarray, segmentation: np.ndarray) -> float:
    """
    Compute region similarity (J / IoU) for a single frame.

    Args:
        annotation: Ground truth mask (H, W), binary or multi-object
        segmentation: Predicted mask (H, W), binary or multi-object

    Returns:
        IoU score (Jaccard index)
    """
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1.0  # Both empty

    intersection = np.logical_and(annotation, segmentation).sum()
    union = np.logical_or(annotation, segmentation).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def db_eval_boundary(
    annotation: np.ndarray,
    segmentation: np.ndarray,
    bound_th: float = 0.008
) -> float:
    """
    Compute boundary F-measure (F) for a single frame.

    Args:
        annotation: Ground truth mask (H, W)
        segmentation: Predicted mask (H, W)
        bound_th: Boundary threshold as fraction of image diagonal

    Returns:
        Boundary F-measure score
    """
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    # Handle empty cases
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1.0
    if np.sum(annotation) == 0 or np.sum(segmentation) == 0:
        return 0.0

    # Get boundaries
    fg_boundary = _seg2bmap(annotation)
    gt_boundary = _seg2bmap(segmentation)

    # Compute boundary distance
    img_diag = np.sqrt(annotation.shape[0]**2 + annotation.shape[1]**2)
    bound_pix = bound_th * img_diag

    # Distance transform
    if np.sum(gt_boundary) == 0:
        precision = 1.0
    else:
        gt_dist = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5)
        precision = np.sum(gt_dist[fg_boundary] <= bound_pix) / np.sum(fg_boundary)

    if np.sum(fg_boundary) == 0:
        recall = 1.0
    else:
        fg_dist = cv2.distanceTransform((~fg_boundary).astype(np.uint8), cv2.DIST_L2, 5)
        recall = np.sum(fg_dist[gt_boundary] <= bound_pix) / np.sum(gt_boundary)

    # F-measure
    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def _seg2bmap(seg: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Convert segmentation mask to boundary map.

    Args:
        seg: Segmentation mask (H, W)
        width: Boundary width in pixels

    Returns:
        Boundary map (H, W)
    """
    seg = seg.astype(bool)

    # Morphological gradient
    eroded = binary_erosion(seg, iterations=width)
    boundary = seg ^ eroded

    return boundary.astype(bool)


def compute_j_metric(
    annotations: List[np.ndarray],
    predictions: List[np.ndarray],
    object_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute J metric (region similarity) for a video sequence.

    Args:
        annotations: List of ground truth masks per frame
        predictions: List of predicted masks per frame
        object_ids: List of object IDs to evaluate (None = all)

    Returns:
        Dictionary with J metrics per object and mean
    """
    if len(annotations) != len(predictions):
        raise ValueError(
            f"Number of annotations ({len(annotations)}) != "
            f"number of predictions ({len(predictions)})"
        )

    num_frames = len(annotations)

    # Get object IDs if not provided
    if object_ids is None:
        object_ids = []
        for annot in annotations:
            object_ids.extend(np.unique(annot).tolist())
        object_ids = sorted(list(set(object_ids)))
        if 0 in object_ids:
            object_ids.remove(0)  # Remove background

    # Compute J for each object
    j_per_object = {}

    for obj_id in object_ids:
        j_scores = []

        for annot, pred in zip(annotations, predictions):
            # Extract masks for this object
            annot_mask = (annot == obj_id)
            pred_mask = (pred == obj_id)

            j = db_eval_iou(annot_mask, pred_mask)
            j_scores.append(j)

        j_per_object[obj_id] = {
            'mean': float(np.mean(j_scores)),
            'recall': float(np.mean([j > 0.5 for j in j_scores])),  # % frames with J > 0.5
            'decay': float(np.mean(j_scores[:len(j_scores)//4]) - np.mean(j_scores[-len(j_scores)//4:])),
            'std': float(np.std(j_scores))
        }

    # Compute mean across objects
    j_mean = float(np.mean([obj['mean'] for obj in j_per_object.values()]))

    return {
        'J_mean': j_mean,
        'J_per_object': j_per_object,
        'num_objects': len(object_ids),
        'num_frames': num_frames
    }


def compute_f_metric(
    annotations: List[np.ndarray],
    predictions: List[np.ndarray],
    object_ids: Optional[List[int]] = None,
    bound_th: float = 0.008
) -> Dict[str, float]:
    """
    Compute F metric (boundary accuracy) for a video sequence.

    Args:
        annotations: List of ground truth masks per frame
        predictions: List of predicted masks per frame
        object_ids: List of object IDs to evaluate (None = all)
        bound_th: Boundary threshold

    Returns:
        Dictionary with F metrics per object and mean
    """
    if len(annotations) != len(predictions):
        raise ValueError(
            f"Number of annotations ({len(annotations)}) != "
            f"number of predictions ({len(predictions)})"
        )

    num_frames = len(annotations)

    # Get object IDs if not provided
    if object_ids is None:
        object_ids = []
        for annot in annotations:
            object_ids.extend(np.unique(annot).tolist())
        object_ids = sorted(list(set(object_ids)))
        if 0 in object_ids:
            object_ids.remove(0)

    # Compute F for each object
    f_per_object = {}

    for obj_id in object_ids:
        f_scores = []

        for annot, pred in zip(annotations, predictions):
            # Extract masks for this object
            annot_mask = (annot == obj_id)
            pred_mask = (pred == obj_id)

            f = db_eval_boundary(annot_mask, pred_mask, bound_th)
            f_scores.append(f)

        f_per_object[obj_id] = {
            'mean': float(np.mean(f_scores)),
            'recall': float(np.mean([f > 0.5 for f in f_scores])),
            'decay': float(np.mean(f_scores[:len(f_scores)//4]) - np.mean(f_scores[-len(f_scores)//4:])),
            'std': float(np.std(f_scores))
        }

    # Compute mean across objects
    f_mean = float(np.mean([obj['mean'] for obj in f_per_object.values()]))

    return {
        'F_mean': f_mean,
        'F_per_object': f_per_object,
        'num_objects': len(object_ids),
        'num_frames': num_frames
    }


def compute_temporal_stability(
    predictions: List[np.ndarray],
    object_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute temporal stability (smoothness) of predictions.

    Measures how consistent predictions are across consecutive frames.

    Args:
        predictions: List of predicted masks per frame
        object_ids: List of object IDs to evaluate

    Returns:
        Dictionary with stability metrics
    """
    if len(predictions) < 2:
        return {'T_mean': 1.0}

    # Get object IDs
    if object_ids is None:
        object_ids = []
        for pred in predictions:
            object_ids.extend(np.unique(pred).tolist())
        object_ids = sorted(list(set(object_ids)))
        if 0 in object_ids:
            object_ids.remove(0)

    # Compute temporal stability for each object
    t_per_object = {}

    for obj_id in object_ids:
        stability_scores = []

        for i in range(len(predictions) - 1):
            mask_t = (predictions[i] == obj_id)
            mask_t1 = (predictions[i+1] == obj_id)

            # IoU between consecutive frames
            iou = db_eval_iou(mask_t, mask_t1)
            stability_scores.append(iou)

        t_per_object[obj_id] = float(np.mean(stability_scores))

    # Mean across objects
    t_mean = float(np.mean(list(t_per_object.values())))

    return {
        'T_mean': t_mean,
        'T_per_object': t_per_object
    }


def compute_all_video_metrics(
    annotations: List[np.ndarray],
    predictions: List[np.ndarray],
    object_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute all video metrics (J, F, J&F, T).

    Args:
        annotations: List of ground truth masks
        predictions: List of predicted masks
        object_ids: List of object IDs to evaluate

    Returns:
        Dictionary with all metrics
    """
    # Compute J metric
    j_metrics = compute_j_metric(annotations, predictions, object_ids)

    # Compute F metric
    f_metrics = compute_f_metric(annotations, predictions, object_ids)

    # Compute temporal stability
    t_metrics = compute_temporal_stability(predictions, object_ids)

    # Compute J&F (mean of J and F)
    jf_mean = (j_metrics['J_mean'] + f_metrics['F_mean']) / 2.0

    return {
        'J': j_metrics['J_mean'],
        'F': f_metrics['F_mean'],
        'J&F': jf_mean,
        'T': t_metrics['T_mean'],
        'J_per_object': j_metrics['J_per_object'],
        'F_per_object': f_metrics['F_per_object'],
        'T_per_object': t_metrics['T_per_object'],
        'num_objects': j_metrics['num_objects'],
        'num_frames': j_metrics['num_frames']
    }


def aggregate_video_metrics(
    all_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple videos.

    Args:
        all_metrics: List of metric dictionaries (one per video)

    Returns:
        Aggregated metrics
    """
    if len(all_metrics) == 0:
        return {}

    # Aggregate mean metrics
    j_scores = [m['J'] for m in all_metrics]
    f_scores = [m['F'] for m in all_metrics]
    jf_scores = [m['J&F'] for m in all_metrics]
    t_scores = [m['T'] for m in all_metrics]

    return {
        'J_mean': float(np.mean(j_scores)),
        'J_std': float(np.std(j_scores)),
        'F_mean': float(np.mean(f_scores)),
        'F_std': float(np.std(f_scores)),
        'J&F_mean': float(np.mean(jf_scores)),
        'J&F_std': float(np.std(jf_scores)),
        'T_mean': float(np.mean(t_scores)),
        'T_std': float(np.std(t_scores)),
        'num_videos': len(all_metrics)
    }
