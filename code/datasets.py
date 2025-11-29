"""
Dataset Loaders for Open-Vocabulary Semantic Segmentation Benchmarks

This module contains dataset loaders for:
- PASCAL VOC 2012
- COCO-Stuff 164K
- Cityscapes
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

    Note: GT masks use COCO's original class IDs (0-181 with gaps).
    We convert them to contiguous train IDs (0-170) using MaskCLIP's mapping.
    """

    # Mapping from COCO class IDs to train IDs (0-170)
    # This is the standard COCO-Stuff 164K mapping used by MaskCLIP
    COCO_ID_TO_TRAIN_ID = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
        12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
        21: 20, 22: 21, 23: 22, 24: 23, 26: 24, 27: 25, 30: 26, 31: 27, 32: 28,
        33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37,
        42: 38, 43: 39, 45: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46,
        52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55,
        61: 56, 62: 57, 63: 58, 64: 59, 66: 60, 69: 61, 71: 62, 72: 63, 73: 64,
        74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 83: 73,
        84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 91: 80, 92: 81, 93: 82,
        94: 83, 95: 84, 96: 85, 97: 86, 98: 87, 99: 88, 100: 89, 101: 90, 102: 91,
        103: 92, 104: 93, 105: 94, 106: 95, 107: 96, 108: 97, 109: 98, 110: 99,
        111: 100, 112: 101, 113: 102, 114: 103, 115: 104, 116: 105, 117: 106,
        118: 107, 119: 108, 120: 109, 121: 110, 122: 111, 123: 112, 124: 113,
        125: 114, 126: 115, 127: 116, 128: 117, 129: 118, 130: 119, 131: 120,
        132: 121, 133: 122, 134: 123, 135: 124, 136: 125, 137: 126, 138: 127,
        139: 128, 140: 129, 141: 130, 142: 131, 143: 132, 144: 133, 145: 134,
        146: 135, 147: 136, 148: 137, 149: 138, 150: 139, 151: 140, 152: 141,
        153: 142, 154: 143, 155: 144, 156: 145, 157: 146, 158: 147, 159: 148,
        160: 149, 161: 150, 162: 151, 163: 152, 164: 153, 165: 154, 166: 155,
        167: 156, 168: 157, 169: 158, 170: 159, 171: 160, 172: 161, 173: 162,
        174: 163, 175: 164, 176: 165, 177: 166, 178: 167, 179: 168, 180: 169,
        181: 170, 255: 255  # Ignore label
    }

    # COCO-Stuff 171 classes (indexed 0-170)
    # This matches MaskCLIP's standard COCO-Stuff 164k class list exactly
    STUFF_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood'
    ]

    def __init__(self, data_dir: Path, split='val2017', max_samples: Optional[int] = None):
        """
        Initialize COCO-Stuff dataset.

        Args:
            data_dir: Root directory containing datasets (e.g., ./data/benchmarks)
            split: Dataset split ('val2017' for validation set)
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
            # Load mask with COCO class IDs
            mask_coco = np.array(Image.open(mask_path))

            # Convert from COCO IDs to train IDs (0-170)
            mask = np.zeros_like(mask_coco, dtype=np.uint8)
            for coco_id, train_id in self.COCO_ID_TO_TRAIN_ID.items():
                mask[mask_coco == coco_id] = train_id

        return {
            'image': image,
            'mask': mask,
            'class_names': self.class_names,
            'image_id': img_id
        }


class ADE20KDataset:
    """
    ADE20K Dataset Loader.

    Dataset: 150 classes (standard benchmark uses top 150 most frequent classes)
    Paper: https://arxiv.org/abs/1608.05442

    The full ADE20K dataset has 3688 object classes, but the standard
    benchmark evaluation uses the 150 most frequent classes.
    """

    # Top 150 classes from ADE20K (0-indexed, 0 is usually background/unlabeled)
    # Based on the objects.txt file and standard ADE20K benchmarks
    CLASSES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror',
        'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock',
        'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base',
        'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand',
        'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
        'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
    ]

    def __init__(self, data_dir: Path, split='validation', max_samples: Optional[int] = None):
        """
        Initialize ADE20K dataset.

        Args:
            data_dir: Root directory containing datasets (e.g., ./data/benchmarks)
            split: Dataset split ('training', 'validation')
            max_samples: Maximum number of samples to load (None = all)
        """
        import glob
        import pickle

        self.data_dir = Path(data_dir) / "ADE20K_2021_17_01"
        self.split = split
        self.num_classes = 150
        self.class_names = self.CLASSES

        # Validate split
        assert self.split in ["training", "validation"], \
            f"Split must be one of ['training', 'validation'], got {self.split}"

        # ADE20K directory structure:
        # ADE20K_2021_17_01/images/ADE/validation/**/*.jpg
        # ADE20K_2021_17_01/images/ADE/training/**/*.jpg

        images_base = self.data_dir / "images" / "ADE" / self.split

        if not images_base.exists():
            raise FileNotFoundError(f"Images directory not found: {images_base}")

        # Find all images (they're in nested subdirectories)
        self.image_files = []
        for img_path in sorted(glob.glob(str(images_base / "**" / "*.jpg"), recursive=True)):
            # Check if corresponding segmentation mask exists
            seg_path = img_path.replace('.jpg', '_seg.png')
            if Path(seg_path).exists():
                self.image_files.append(img_path)

        if len(self.image_files) == 0:
            raise ValueError(f"No images with segmentation masks found in {images_base}")

        # Load the index file for class name mapping
        index_path = self.data_dir / "index_ade20k.pkl"
        if index_path.exists():
            with open(index_path, 'rb') as f:
                self.index_data = pickle.load(f)
                # Get the object names mapping (index -> name)
                self.full_objectnames = self.index_data.get('objectnames', [])
        else:
            self.index_data = None
            self.full_objectnames = []

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]

    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return self.CLASSES

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self))
            new_dataset = ADE20KDataset.__new__(ADE20KDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.num_classes = self.num_classes
            new_dataset.class_names = self.class_names
            new_dataset.index_data = self.index_data
            new_dataset.full_objectnames = self.full_objectnames
            new_dataset.image_files = self.image_files[idx]
            return new_dataset

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Load image
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load segmentation mask
        # ADE20K masks are encoded as RGB where:
        # - Object class index = R * 256 + G (from full 3688-class taxonomy)
        # - Instance ID = B channel
        seg_path = img_path.replace('.jpg', '_seg.png')
        mask_rgb = np.array(Image.open(seg_path))

        if len(mask_rgb.shape) == 3:
            # Decode object indices using standard ADE20K encoding
            # object_index = R * 256 + G
            r = mask_rgb[:, :, 0].astype(np.int32)
            g = mask_rgb[:, :, 1].astype(np.int32)
            object_indices = r * 256 + g
        else:
            # Already decoded (grayscale)
            object_indices = mask_rgb.astype(np.int32)

        # For now, use a simple mapping: take object_index modulo 150
        # TODO: Implement proper mapping from full ADE20K taxonomy to 150-class benchmark
        # The proper way would be to map known object indices to their 150-class equivalents
        # For this initial implementation, we map 0 to 0 (background/unlabeled)
        # and all other indices to their position mod 150
        mask = np.zeros_like(object_indices, dtype=np.uint8)
        mask[object_indices == 0] = 0  # Background stays 0
        mask[object_indices > 0] = ((object_indices[object_indices > 0] - 1) % 149) + 1  # Map 1-3688 to 1-149

        # Mark very large indices (likely errors) as ignore
        mask[object_indices > 10000] = 255

        # Extract image ID from filename
        img_id = Path(img_path).stem

        return {
            'image': image,
            'mask': mask,
            'class_names': self.class_names,
            'image_id': img_id
        }


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


class CityscapesDataset:
    """
    Cityscapes Dataset Loader.

    Dataset: 19 classes for semantic segmentation
    Paper: https://arxiv.org/abs/1604.01685

    Supports two directory structures:

    1. Standard Cityscapes:
        cityscapes/
            leftImg8bit/
                train/cityname/*.png
                val/cityname/*.png
            gtFine/
                train/cityname/*_labelTrainIds.png
                val/cityname/*_labelTrainIds.png

    2. Simplified structure:
        cityscapes/
            train/
                img/*.png
                label/*.png
            val/
                img/*.png
                label/*.png
    """

    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'trafficlight', 'trafficsign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]

    # Standard Cityscapes ID to trainID mapping
    # Source: https://github.com/mcordts/cityscapesScripts
    ID_TO_TRAINID = {
        7: 0,   # road
        8: 1,   # sidewalk
        11: 2,  # building
        12: 3,  # wall
        13: 4,  # fence
        17: 5,  # pole
        19: 6,  # traffic light
        20: 7,  # traffic sign
        21: 8,  # vegetation
        22: 9,  # terrain
        23: 10, # sky
        24: 11, # person
        25: 12, # rider
        26: 13, # car
        27: 14, # truck
        28: 15, # bus
        31: 16, # train
        32: 17, # motorcycle
        33: 18, # bicycle
        # All other IDs map to 255 (ignore)
    }

    def __init__(self, data_dir: Path, split='val', max_samples: Optional[int] = None):
        """
        Initialize Cityscapes dataset.

        Args:
            data_dir: Root directory containing datasets (e.g., ./data/benchmarks)
            split: Dataset split ('train', 'val', 'train+val')
            max_samples: Maximum number of samples to load (None = all)
        """
        import glob

        self.data_dir = Path(data_dir) / "cityscapes"
        self.split = split
        self.num_classes = 19
        self.class_names = self.CLASSES

        # Validate split
        assert self.split in ["train", "val", "train+val"], \
            f"Split must be one of ['train', 'val', 'train+val'], got {self.split}"

        split_dirs = {
            "train": ["train"],
            "val": ["val"],
            "train+val": ["train", "val"]
        }

        # Collect image and label paths
        self.image_files = []
        self.label_files = []

        for split_dir in split_dirs[self.split]:
            # Check for two possible directory structures:
            # 1. Standard Cityscapes: leftImg8bit/val/cityname/*.png
            # 2. Simplified structure: val/img/*.png

            standard_images_path = self.data_dir / "leftImg8bit" / split_dir
            simplified_images_path = self.data_dir / split_dir / "img"

            if simplified_images_path.exists():
                # Simplified structure: val/img/ and val/label/
                images_path = simplified_images_path
                labels_path = self.data_dir / split_dir / "label"

                if not labels_path.exists():
                    raise FileNotFoundError(f"Labels directory not found: {labels_path}")

                # Find all images (no subdirectories in simplified structure)
                for img_path in sorted(glob.glob(str(images_path / "*.png"))):
                    self.image_files.append(img_path)
                    # Corresponding label has same filename
                    img_filename = Path(img_path).name
                    label_path = str(labels_path / img_filename)
                    self.label_files.append(label_path)

            elif standard_images_path.exists():
                # Standard Cityscapes structure: leftImg8bit/val/cityname/*.png
                images_path = standard_images_path

                # Find all images in subdirectories (cityname folders)
                for img_path in sorted(glob.glob(str(images_path / "*" / "*.png"))):
                    self.image_files.append(img_path)
                    # Convert image path to label path
                    # e.g., leftImg8bit/val/lindau/lindau_000000_000019_leftImg8bit.png
                    # ->    gtFine/val/lindau/lindau_000000_000019_gtFine_labelIds.png
                    label_path = img_path.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    label_path = label_path.replace("leftImg8bit", "gtFine")
                    self.label_files.append(label_path)
            else:
                raise FileNotFoundError(
                    f"Images directory not found. Tried:\n"
                    f"  Standard: {standard_images_path}\n"
                    f"  Simplified: {simplified_images_path}"
                )

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
            self.label_files = self.label_files[:max_samples]

    def get_num_classes(self):
        return self.num_classes

    def get_class_names(self):
        return self.CLASSES

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self))
            new_dataset = CityscapesDataset.__new__(CityscapesDataset)
            new_dataset.data_dir = self.data_dir
            new_dataset.split = self.split
            new_dataset.num_classes = self.num_classes
            new_dataset.class_names = self.class_names
            new_dataset.image_files = self.image_files[idx]
            new_dataset.label_files = self.label_files[idx]
            return new_dataset

        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Load image
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load label mask
        label_path = self.label_files[idx]
        if not Path(label_path).exists():
            # If label doesn't exist, create empty mask with ignore label
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        else:
            # Load mask (could be RGB or grayscale)
            mask_pil = Image.open(label_path)
            mask = np.array(mask_pil)

            # If mask is RGB, convert to grayscale by taking first channel
            # (Cityscapes labels are sometimes stored as color-coded RGB)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            # Convert Cityscapes IDs to trainIDs (0-18)
            # Check if mask needs conversion (has values outside 0-18 range)
            if mask.max() > 18 or not np.all(np.isin(mask[mask <= 18], list(range(19)))):
                # Create output mask with ignore label (255) as default
                trainid_mask = np.full(mask.shape, 255, dtype=np.uint8)

                # Apply mapping for valid IDs
                for cityscapes_id, train_id in self.ID_TO_TRAINID.items():
                    trainid_mask[mask == cityscapes_id] = train_id

                mask = trainid_mask

        # Extract image ID from filename
        # e.g., lindau_000000_000019_leftImg8bit.png -> lindau_000000_000019
        img_id = Path(img_path).stem.replace('_leftImg8bit', '')

        return {
            'image': image,
            'mask': mask,
            'class_names': self.class_names,
            'image_id': img_id
        }


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
        dataset_name: Name of dataset ('pascal-voc', 'coco-stuff', 'cityscapes', 'ade20k', 'coco-open')
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
        'cityscapes': CityscapesDataset,
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
        return dataset_class(data_dir, split='val2017', max_samples=max_samples)
    elif dataset_name == 'cityscapes':
        return dataset_class(data_dir, split='val', max_samples=max_samples)
    elif dataset_name == 'ade20k':
        return dataset_class(data_dir, split='validation', max_samples=max_samples)
    elif dataset_name == 'coco-open':
        return dataset_class(data_dir, split='val', max_samples=max_samples)
    else:
        return dataset_class(data_dir, max_samples=max_samples)
