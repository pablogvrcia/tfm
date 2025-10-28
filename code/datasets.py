"""
Dataset Loaders for Open-Vocabulary Semantic Segmentation Benchmarks

This module contains dataset loaders for:
- PASCAL VOC 2012
- COCO-Stuff 164K
- ADE20K (placeholder)
- COCO-Open split (placeholder)
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List


class PASCALVOCDataset:
    """
    PASCAL VOC 2012 Segmentation Dataset Loader.

    Dataset: 21 classes (background + 20 object classes)
    """

    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]

    def __init__(self, data_dir: Path, split='val', max_samples: Optional[int] = None):
        """
        Initialize PASCAL VOC dataset.

        Args:
            data_dir: Root directory containing datasets (e.g., ./data/benchmarks)
            split: Dataset split ('train', 'val', 'trainval')
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_dir = Path(data_dir) / "pascal_voc" / "VOCdevkit" / "VOC2012"
        self.split = split
        self.num_classes = 21
        self.class_names = self.CLASSES

        # Load image IDs from split file
        split_file = self.data_dir / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing by creating new dataset with subset
            start, stop, step = idx.indices(len(self))
            new_dataset = PASCALVOCDataset.__new__(PASCALVOCDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.num_classes = self.num_classes
            new_dataset.class_names = self.class_names
            new_dataset.image_ids = self.image_ids[idx]
            return new_dataset

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        img_id = self.image_ids[idx]

        # Load image
        img_path = self.data_dir / "JPEGImages" / f"{img_id}.jpg"
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load segmentation mask
        mask_path = self.data_dir / "SegmentationClass" / f"{img_id}.png"
        mask = np.array(Image.open(mask_path))

        return {
            'image': image,
            'mask': mask,
            'class_names': self.class_names,
            'image_id': img_id
        }


class COCOStuffDataset:
    """
    COCO-Stuff 164K Dataset Loader.

    COCO-Stuff extends COCO with stuff annotations (171 classes total):
    - 80 thing classes (from COCO)
    - 91 stuff classes (backgrounds, materials, etc.)

    Paper: https://arxiv.org/abs/1612.03716
    """

    # COCO-Stuff 171 classes (indexed 0-170, with 0 reserved for unlabeled)
    STUFF_CLASSES = [
        'unlabeled',  # 0
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  # 1-9 things
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',  # 10-14
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',  # 15-24
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',  # 25-29
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',  # 30-36
        'skateboard', 'surfboard', 'tennis racket',  # 37-39
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',  # 40-46
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  # 47-56
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',  # 57-62
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',  # 63-73
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',  # 74-80 (end of things)
        # Stuff classes start here (81-170)
        'banner', 'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet', 'cage', 'cardboard',
        'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard',
        'curtain', 'desk', 'dirt', 'door', 'fence', 'floor', 'wood floor', 'tile floor', 'flower',
        'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel', 'ground', 'hill', 'house',
        'leaves', 'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform', 'playingfield', 'railing',
        'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
        'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural', 'table',
        'tent', 'textile', 'towel', 'tree', 'vegetable', 'wall', 'brick wall', 'concrete wall',
        'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'window',
        'wood'
    ]

    def __init__(self, data_dir: Path, split='val2027', max_samples: Optional[int] = None):
        """
        Initialize COCO-Stuff dataset.

        Args:
            data_dir: Root directory containing datasets (e.g., ./data/benchmarks)
            split: Dataset split ('val2027' for validation set)
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_dir = Path(data_dir) / "coco_stuff"
        self.split = split
        self.num_classes = 171
        self.class_names = self.STUFF_CLASSES

        # Paths - COCO-Stuff has different structure
        self.images_dir = self.data_dir / split / "images"
        self.annotations_dir = self.data_dir / split / "annotations"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")

        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
            
    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return self.STUFF_CLASSES

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self))
            new_dataset = COCOStuffDataset.__new__(COCOStuffDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.num_classes = self.num_classes
            new_dataset.class_names = self.class_names
            new_dataset.images_dir = self.images_dir
            new_dataset.annotations_dir = self.annotations_dir
            new_dataset.image_files = self.image_files[idx]
            return new_dataset

        # Load image
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load annotation mask
        img_id = img_path.stem  # e.g., "000000000139"
        mask_path = self.annotations_dir / f"{img_id}.png"

        if not mask_path.exists():
            # If mask doesn't exist, create empty mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.array(Image.open(mask_path))

        return {
            'image': image,
            'mask': mask,
            'class_names': self.class_names,
            'image_id': img_id
        }


class ADE20KDataset:
    """
    ADE20K Dataset Loader (Placeholder).

    Dataset: 150 classes
    Paper: https://arxiv.org/abs/1608.05442
    """

    def __init__(self, data_dir: Path, split='validation', max_samples: Optional[int] = None):
        """
        Initialize ADE20K dataset.

        Args:
            data_dir: Root directory containing datasets
            split: Dataset split ('training', 'validation')
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir) / "ade20k"
        self.split = split
        self.num_classes = 150
        self.class_names = [f'class_{i}' for i in range(self.num_classes)]  # TODO: Add actual class names

        # TODO: Implement ADE20K loading
        raise NotImplementedError("ADE20K dataset loader not yet implemented")

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class COCOOpenDataset:
    """
    COCO-Open Split Dataset Loader (Placeholder).

    Dataset: 48 base + 17 novel classes
    Used for evaluating generalization to novel categories.
    """

    def __init__(self, data_dir: Path, split='val', max_samples: Optional[int] = None):
        """
        Initialize COCO-Open dataset.

        Args:
            data_dir: Root directory containing datasets
            split: Dataset split ('val', 'test')
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir) / "coco_open"
        self.split = split
        self.num_classes = 65  # 48 base + 17 novel
        self.class_names = [f'class_{i}' for i in range(self.num_classes)]  # TODO: Add actual class names

        # TODO: Implement COCO-Open loading
        raise NotImplementedError("COCO-Open dataset loader not yet implemented")

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class MockDataset:
    """
    Mock dataset for testing and development.

    Generates random images and masks for quick prototyping.
    """

    def __init__(self, name: str, data_dir: Path, max_samples: int = 10):
        """
        Initialize mock dataset.

        Args:
            name: Dataset name (for compatibility)
            data_dir: Not used (for compatibility)
            max_samples: Number of samples to generate
        """
        self.name = name
        self.data_dir = data_dir
        self.num_classes = 21 if name == "pascal-voc" else 171
        self.max_samples = max_samples
        self.class_names = [f'class_{i}' for i in range(self.num_classes)]

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            num_samples = len(range(start, stop, step))
            return MockDataset(self.name, self.data_dir, max_samples=num_samples)

        if idx >= self.max_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.max_samples}")

        return {
            'image': np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            'mask': np.random.randint(0, self.num_classes, (512, 512), dtype=np.uint8),
            'class_names': self.class_names,
            'image_id': f'mock_{idx:04d}'
        }


def load_dataset(dataset_name: str, data_dir: Path, max_samples: Optional[int] = None):
    """
    Factory function to load any dataset by name.

    Args:
        dataset_name: Name of dataset ('pascal-voc', 'coco-stuff', 'ade20k', 'coco-open')
        data_dir: Root directory containing datasets
        max_samples: Maximum number of samples to load

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_map = {
        'pascal-voc': PASCALVOCDataset,
        'coco-stuff': COCOStuffDataset,
        'ade20k': ADE20KDataset,
        'coco-open': COCOOpenDataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")

    dataset_class = dataset_map[dataset_name]

    # Use different default splits for different datasets
    if dataset_name == 'pascal-voc':
        return dataset_class(data_dir, split='val', max_samples=max_samples)
    elif dataset_name == 'coco-stuff':
        return dataset_class(data_dir, split='val2027', max_samples=max_samples)
    elif dataset_name == 'ade20k':
        return dataset_class(data_dir, split='validation', max_samples=max_samples)
    elif dataset_name == 'coco-open':
        return dataset_class(data_dir, split='val', max_samples=max_samples)
    else:
        return dataset_class(data_dir, max_samples=max_samples)
