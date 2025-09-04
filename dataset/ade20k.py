# dataset/ade20k.py
"""
ADE20K dataset for continual semantic segmentation.
"""

import os
import json
from .base import ContinualDataset


class ADE20KIncremental(ContinualDataset):
    """
    ADE20K dataset with incremental learning setup.
    """
    def __init__(self, *args, **kwargs):
        # ADE20K has 150 classes + background
        self.class_names = self._load_class_names()
        super().__init__(*args, **kwargs)

    def _load_class_names(self):
        """Load ADE20K class names."""
        # Simulate class list; in practice, load from metadata
        return [f"class_{i}" for i in range(151)]

    def _get_all_possible_labels(self):
        return list(range(151))  # 0-150

    def _load_dataset(self):
        """Load ADE20K image and label paths."""
        img_dir = os.path.join(self.root, "images", self.split)
        label_dir = os.path.join(self.root, "annotations", self.split)

        images, labels = [], []

        for room in os.listdir(img_dir):
            room_img_dir = os.path.join(img_dir, room)
            room_label_dir = os.path.join(label_dir, room)
            for file in os.listdir(room_img_dir):
                if file.endswith(".jpg"):
                    img_path = os.path.join(room_img_dir, file)
                    lbl_path = os.path.join(room_label_dir, file.replace(".jpg", ".png"))
                    if os.path.exists(lbl_path):
                        images.append(img_path)
                        labels.append(lbl_path)

        return images, labels

    def _generate_pseudo_labels(self, image, label):
        """
        In overlapped setting, use old model to generate pseudo-labels.
        Here, we simulate by preserving old class regions.
        """
        if not self.use_overlap:
            return label
        return label


__all__ = ['ADE20KIncremental']