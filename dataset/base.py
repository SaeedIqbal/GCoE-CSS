"""
Base dataset class for Continual Semantic Segmentation.
Implements incremental learning logic and label masking.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import json


class ContinualDataset(Dataset):
    """
    Base class for continual semantic segmentation datasets.
    Handles:
    - Incremental class setup
    - Label remapping
    - Pseudo-labeling (overlapped setting)
    - Memory buffer management
    """
    def __init__(self, root, split, transform=None, current_labels=None,
                 old_labels=None, indices_path=None, use_overlap=False, mask_unseen=True):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.current_labels = current_labels or []
        self.old_labels = old_labels or []
        self.use_overlap = use_overlap
        self.mask_unseen = mask_unseen

        # All known classes at this stage
        self.present_labels = self.old_labels + self.current_labels
        self.ignore_label = 255

        # Load image/label paths
        self.images, self.labels = self._load_dataset()
        self.indices = self._load_or_create_indices(indices_path)

        # Create label mapping: original label -> training label
        self.label_map = self._create_label_mapping()

    def _load_dataset(self):
        """Load image and label file paths. Must be implemented by subclasses."""
        raise NotImplementedError

    def _create_label_mapping(self):
        """Map original labels to training indices (0 to num_classes-1)."""
        label_map = {label: idx for idx, label in enumerate(self.present_labels)}
        if self.mask_unseen:
            # Map unseen classes to ignore_label
            all_labels = self._get_all_possible_labels()
            for label in all_labels:
                if label not in self.present_labels:
                    label_map[label] = self.ignore_label
        return label_map

    def _get_all_possible_labels(self):
        """Return all possible class labels in the dataset. Implemented by subclasses."""
        raise NotImplementedError

    def _load_or_create_indices(self, indices_path):
        """Load pre-defined indices or create random split."""
        if indices_path and os.path.exists(indices_path):
            return np.load(indices_path)
        else:
            indices = np.random.permutation(len(self.images))
            if indices_path:
                os.makedirs(os.path.dirname(indices_path), exist_ok=True)
                np.save(indices_path, indices)
            return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.images[self.indices[idx]]
        lbl_path = self.labels[self.indices[idx]]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)

        # Convert label using mapping
        label = self._remap_label(label)

        # Apply transforms
        if self.transform is not None:
            image, label = self.transform(image, label)

        # Generate pseudo-labels for old classes in overlapped setting
        if self.use_overlap and self.old_labels:
            label = self._generate_pseudo_labels(image, label)

        return image, label

    def _remap_label(self, label):
        """Remap label using label_map."""
        label = np.array(label)
        remapped = np.full_like(label, self.ignore_label)
        for orig, train in self.label_map.items():
            remapped[label == orig] = train
        return Image.fromarray(remapped, mode='L')

    def _generate_pseudo_labels(self, image, label):
        """Generate pseudo-labels for old classes (simulated in base class)."""
        # In practice, this would use a frozen old model
        # Here, we simulate by preserving old class regions
        return label

    def get_memory_samples(self, num_samples, sampler=None):
        """Sample from current dataset for memory buffer."""
        if sampler is None:
            indices = np.random.choice(len(self), num_samples, replace=False)
        else:
            indices = sampler.sample(self, num_samples)
        return [self[i] for i in indices]


__all__ = ['ContinualDataset']