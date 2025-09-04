# dataset/__init__.py
"""
Initialization for the dataset package.
"""
from .base import ContinualDataset
from .voc import VOCIncremental
from .ade20k import ADE20KIncremental

__version__ = "1.0.0"
__author__ = "Saeed Iqbal"

__all__ = [
    "ContinualDataset",
    "VOCIncremental",
    "ADE20KIncremental"
]