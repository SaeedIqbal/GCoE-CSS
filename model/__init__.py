"""
Initialization for the model package.
"""
from .segmentation import GCoeSegmentationModel, ExpertModule, LatentOutputODE, DeepLabV3Plus

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    "GCoeSegmentationModel",
    "ExpertModule",
    "LatentOutputODE",
    "DeepLabV3Plus"
]