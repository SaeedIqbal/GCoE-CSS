"""
Pascal VOC 2012 dataset for continual semantic segmentation.
"""

import os
from .base import ContinualDataset


class VOCIncremental(ContinualDataset):
    """
    Pascal VOC 2012 dataset with incremental learning setup.
    """
    # Pascal VOC class names and IDs
    CLASS_INFO = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_all_possible_labels(self):
        return list(range(21))  # 0-20

    def _load_dataset(self):
        """Load Pascal VOC image and label paths."""
        img_dir = os.path.join(self.root, "JPEGImages")
        label_dir = os.path.join(self.root, "SegmentationClass")

        split_file = os.path.join(self.root, "ImageSets", "Segmentation", f"{self.split}.txt")
        with open(split_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]

        images = [os.path.join(img_dir, name + ".jpg") for name in file_names]
        labels = [os.path.join(label_dir, name + ".png") for name in file_names]

        return images, labels

    def _generate_pseudo_labels(self, image, label):
        """
        In overlapped setting, old classes are annotated in new stage.
        Here, we simulate by not modifying the label.
        """
        if not self.use_overlap:
            return label
        return label


__all__ = ['VOCIncremental']