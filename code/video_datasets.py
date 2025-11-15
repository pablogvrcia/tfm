"""
Video Dataset Loaders for Open-Vocabulary Video Semantic Segmentation Benchmarks

This module contains dataset loaders for:
- DAVIS 2016/2017 (Densely Annotated VIdeo Segmentation)
- YouTube-VOS (YouTube Video Object Segmentation)
- MOSE (Multiple Object Segmentation in Entertainment)

These datasets are used for video object segmentation evaluation.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict, Tuple
import json
import cv2


class DAVISDataset:
    """
    DAVIS 2016/2017 Dataset Loader for Video Object Segmentation.

    DAVIS provides high-quality video sequences with pixel-accurate annotations.

    DAVIS 2016: Single object segmentation (50 sequences)
    DAVIS 2017: Multi-object segmentation (90 train + 30 val sequences)

    Website: https://davischallenge.org/
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'val',
        year: str = '2017',
        resolution: str = '480p',
        max_samples: Optional[int] = None
    ):
        """
        Initialize DAVIS dataset.

        Args:
            data_dir: Root directory containing DAVIS dataset
            split: Dataset split ('train', 'val', 'test-dev', 'test-challenge')
            year: DAVIS version ('2016' or '2017')
            resolution: Video resolution ('480p' or '1080p')
            max_samples: Maximum number of videos to load (None = all)
        """
        self.data_dir = Path(data_dir) / f"DAVIS-{year}" / "DAVIS"
        self.split = split
        self.year = year
        self.resolution = resolution

        # Paths
        self.images_dir = self.data_dir / "JPEGImages" / resolution
        self.annotations_dir = self.data_dir / "Annotations" / resolution
        self.imagesets_dir = self.data_dir / "ImageSets" / year

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"DAVIS images directory not found: {self.images_dir}\n"
                f"Please download DAVIS from https://davischallenge.org/davis{year}/code.html"
            )

        # Load sequence list from ImageSets
        split_file = self.imagesets_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines() if line.strip()]

        if max_samples is not None:
            self.sequences = self.sequences[:max_samples]

        print(f"[DAVIS-{year}] Loaded {len(self.sequences)} sequences from {split} split")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Dict:
        """
        Get a video sequence with all frames and annotations.

        Returns:
            Dictionary containing:
                - video_name: str
                - frames: List of RGB frames (numpy arrays)
                - annotations: List of annotation masks (numpy arrays)
                - num_frames: int
                - num_objects: int (for DAVIS 2017)
                - object_ids: List of object IDs
        """
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self))
            new_dataset = DAVISDataset.__new__(DAVISDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.year = self.year
            new_dataset.resolution = self.resolution
            new_dataset.images_dir = self.images_dir
            new_dataset.annotations_dir = self.annotations_dir
            new_dataset.imagesets_dir = self.imagesets_dir
            new_dataset.sequences = self.sequences[idx]
            return new_dataset

        video_name = self.sequences[idx]

        # Get frame paths
        video_frames_dir = self.images_dir / video_name
        video_annots_dir = self.annotations_dir / video_name

        frame_files = sorted(list(video_frames_dir.glob("*.jpg")))
        annot_files = sorted(list(video_annots_dir.glob("*.png")))

        if len(frame_files) == 0:
            raise ValueError(f"No frames found for sequence: {video_name}")

        # Load frames
        frames = []
        for frame_file in frame_files:
            frame = np.array(Image.open(frame_file).convert('RGB'))
            frames.append(frame)

        # Load annotations
        annotations = []
        for annot_file in annot_files:
            annot = np.array(Image.open(annot_file))
            annotations.append(annot)

        # Get object IDs from first annotation
        if len(annotations) > 0:
            object_ids = sorted(list(np.unique(annotations[0])))
            # Remove background (0)
            if 0 in object_ids:
                object_ids.remove(0)
        else:
            object_ids = []

        return {
            'video_name': video_name,
            'frames': frames,
            'annotations': annotations,
            'num_frames': len(frames),
            'num_objects': len(object_ids),
            'object_ids': object_ids,
            'frame_files': [str(f) for f in frame_files],
            'annot_files': [str(f) for f in annot_files]
        }

    def get_video_path(self, idx: int) -> Optional[str]:
        """
        Get video path if available (DAVIS provides frames, not videos).
        Returns None as DAVIS uses frame sequences.
        """
        return None


class YouTubeVOSDataset:
    """
    YouTube-VOS Dataset Loader for Video Object Segmentation.

    YouTube-VOS is a large-scale benchmark with 4,453 videos and 94 object categories.

    Website: https://youtube-vos.org/
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'valid',
        year: str = '2019',
        max_samples: Optional[int] = None
    ):
        """
        Initialize YouTube-VOS dataset.

        Args:
            data_dir: Root directory containing YouTube-VOS dataset
            split: Dataset split ('train', 'valid', 'test')
            year: Dataset version ('2018' or '2019')
            max_samples: Maximum number of videos to load (None = all)
        """
        self.data_dir = Path(data_dir) / f"youtube-vos-{year}"
        self.split = split
        self.year = year

        # Paths
        if split == 'train':
            self.images_dir = self.data_dir / "train" / "JPEGImages"
            self.annotations_dir = self.data_dir / "train" / "Annotations"
        else:
            self.images_dir = self.data_dir / split / "JPEGImages"
            self.annotations_dir = self.data_dir / split / "Annotations"

        # Load metadata
        meta_file = self.data_dir / split / "meta.json"

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"YouTube-VOS images directory not found: {self.images_dir}\n"
                f"Please download YouTube-VOS from https://youtube-vos.org/dataset/"
            )

        # Load metadata if available
        self.metadata = {}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.metadata = json.load(f)
            self.sequences = list(self.metadata['videos'].keys())
        else:
            # Fallback: list directories
            self.sequences = [d.name for d in self.images_dir.iterdir() if d.is_dir()]

        if max_samples is not None:
            self.sequences = self.sequences[:max_samples]

        print(f"[YouTube-VOS-{year}] Loaded {len(self.sequences)} sequences from {split} split")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Dict:
        """
        Get a video sequence with frames and annotations.

        Returns:
            Dictionary containing video data
        """
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self))
            new_dataset = YouTubeVOSDataset.__new__(YouTubeVOSDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.year = self.year
            new_dataset.images_dir = self.images_dir
            new_dataset.annotations_dir = self.annotations_dir
            new_dataset.metadata = self.metadata
            new_dataset.sequences = self.sequences[idx]
            return new_dataset

        video_name = self.sequences[idx]

        # Get frame paths
        video_frames_dir = self.images_dir / video_name
        video_annots_dir = self.annotations_dir / video_name

        frame_files = sorted(list(video_frames_dir.glob("*.jpg")))

        if len(frame_files) == 0:
            raise ValueError(f"No frames found for sequence: {video_name}")

        # Load frames
        frames = []
        for frame_file in frame_files:
            frame = np.array(Image.open(frame_file).convert('RGB'))
            frames.append(frame)

        # Load annotations (may not exist for all frames)
        annotations = []
        annot_files = []
        if video_annots_dir.exists():
            annot_files = sorted(list(video_annots_dir.glob("*.png")))
            for annot_file in annot_files:
                annot = np.array(Image.open(annot_file))
                annotations.append(annot)

        # Get object IDs from metadata or first annotation
        object_ids = []
        if video_name in self.metadata.get('videos', {}):
            object_ids = list(self.metadata['videos'][video_name]['objects'].keys())
        elif len(annotations) > 0:
            object_ids = sorted(list(np.unique(annotations[0])))
            if 0 in object_ids:
                object_ids.remove(0)

        return {
            'video_name': video_name,
            'frames': frames,
            'annotations': annotations,
            'num_frames': len(frames),
            'num_objects': len(object_ids),
            'object_ids': object_ids,
            'frame_files': [str(f) for f in frame_files],
            'annot_files': [str(f) for f in annot_files]
        }


def load_video_dataset(
    dataset_name: str,
    data_dir: Path,
    split: str = 'val',
    max_samples: Optional[int] = None,
    **kwargs
) -> object:
    """
    Factory function to load any video dataset by name.

    Args:
        dataset_name: Name of dataset ('davis-2016', 'davis-2017', 'youtube-vos')
        data_dir: Root directory containing datasets
        split: Dataset split
        max_samples: Maximum number of samples to load
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_map = {
        'davis-2016': lambda: DAVISDataset(data_dir, split=split, year='2016', max_samples=max_samples, **kwargs),
        'davis-2017': lambda: DAVISDataset(data_dir, split=split, year='2017', max_samples=max_samples, **kwargs),
        'youtube-vos': lambda: YouTubeVOSDataset(data_dir, split=split, max_samples=max_samples, **kwargs),
    }

    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unknown video dataset: {dataset_name}. "
            f"Available: {list(dataset_map.keys())}"
        )

    return dataset_map[dataset_name]()
