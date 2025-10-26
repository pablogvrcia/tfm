"""
Benchmark evaluation system for open-vocabulary segmentation.

Supports evaluation on:
- COCO-Stuff 164K
- PASCAL VOC 2012
- ADE20K
- COCO-Open vocabulary split (48 base + 17 novel classes)
"""

from .metrics import compute_miou, compute_f1, compute_boundary_f1, compute_all_metrics

__all__ = [
    'compute_miou',
    'compute_f1',
    'compute_boundary_f1',
    'compute_all_metrics',
]
